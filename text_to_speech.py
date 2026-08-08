"""
Step 2 of 2 — chapter text files -> chapter audio files (XTTS-v2 / Coqui TTS)

Reads every .txt file in ./output/text/ (produced by extract_text.py) and
generates one matching .wav file per chapter in ./output/audio/, so each
input chapter file maps 1:1 to an output audio file with the same base name.

Resumable at two levels:
    - chapter-level: if a chapter's final .wav already exists, it's skipped entirely
    - chunk-level: within an interrupted chapter, already-generated chunk
      wavs are reused instead of regenerated

Usage:
    python generate_audio.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # silence torch.load deprecation noise

import re
import os
import glob
from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

TEXT_INPUT_DIR = "output/text"
AUDIO_OUTPUT_DIR = "output/audio"
CHUNKS_DIR = "output/audio/_chunks"
SPEAKER_WAV = "reference_voice.wav"
LANGUAGE = "ru"
MAX_CHARS = 180  # must stay at/under XTTS's actual per-language limit for Russian (182 chars) —
                 # this is a hard per-sentence cap, not just a grouping size, so raising it doesn't
                 # reduce seams on long natural sentences, it just risks the truncation warning
CROSSFADE_MS = 40  # smooths the stitch point between chunks instead of a hard cut
SILENCE_THRESH_DB = -42  # anything quieter than this (relative) counts as "silence" for trimming

os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)


# ============================================================
# Chunking (sentence-based, hard-capped to avoid XTTS's per-language
# char limit even on punctuation-free runs of text like leftover tables)
# ============================================================
def hard_split(s, max_chars):
    words = s.split()
    parts = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current += (" " if current else "") + w
        else:
            if current:
                parts.append(current)
            current = w
    if current:
        parts.append(current)
    return parts


def chunk_text(text, max_chars=MAX_CHARS):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(s) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(hard_split(s, max_chars))
            continue
        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks


def trim_silence(audio_segment):
    """XTTS pads a bit of silence at the start and end of every chunk it
    generates. Left in place, that silence adds up to an audible pause at
    every single seam once chunks are stitched together. Trimming it here
    (independently at each end) removes that without affecting the speech."""
    start_trim = detect_leading_silence(audio_segment, silence_threshold=SILENCE_THRESH_DB)
    end_trim = detect_leading_silence(audio_segment.reverse(), silence_threshold=SILENCE_THRESH_DB)
    duration = len(audio_segment)
    return audio_segment[start_trim:duration - end_trim]


def looks_like_glitch(audio_segment, text, expected_chars_per_sec=15):
    """XTTS occasionally produces a garbled/looping artifact on an otherwise
    ordinary chunk, usually showing up as audio that's much longer than the
    text could plausibly take to read. Flagging that lets us retry once
    instead of baking a glitch into the final file."""
    expected_min_sec = max(0.5, len(text) / expected_chars_per_sec / 2.5)
    actual_sec = len(audio_segment) / 1000
    return actual_sec > expected_min_sec * 3.5  # generous margin — only catches real outliers


def generate_chunk_with_retry(tts, text, out_path):
    tts.tts_to_file(
        text=text,
        speaker_wav=SPEAKER_WAV,
        language=LANGUAGE,
        file_path=out_path,
        temperature=0.7,            # slightly lower than before — fewer random artifacts, still natural
        repetition_penalty=5.0,
    )
    audio = AudioSegment.from_wav(out_path)
    if looks_like_glitch(audio, text):
        tts.tts_to_file(
            text=text,
            speaker_wav=SPEAKER_WAV,
            language=LANGUAGE,
            file_path=out_path,
            temperature=0.7,
            repetition_penalty=5.0,
        )
        audio = AudioSegment.from_wav(out_path)
    trimmed = trim_silence(audio)
    trimmed.export(out_path, format="wav")


# ============================================================
# Generate audio for one chapter's chunks (resumable, fault-tolerant)
# ============================================================
def generate_chapter_audio(tts, chapter_stem, chunks):
    chapter_chunk_dir = os.path.join(CHUNKS_DIR, chapter_stem)
    os.makedirs(chapter_chunk_dir, exist_ok=True)

    paths = []
    failed = []
    for i, chunk in enumerate(chunks):
        out_path = os.path.join(chapter_chunk_dir, f"chunk_{i:05d}.wav")
        if os.path.exists(out_path):
            paths.append(out_path)  # resume: skip already-generated chunks
            continue
        try:
            generate_chunk_with_retry(tts, chunk, out_path)
            paths.append(out_path)
        except Exception as e:
            print(f"    [SKIPPED chunk {i}] {e}")
            failed.append((i, chunk))
        print(f"    chunk {i+1}/{len(chunks)}")

    if failed:
        fail_log = os.path.join(chapter_chunk_dir, "failed_chunks.txt")
        with open(fail_log, "w", encoding="utf-8") as f:
            for i, c in failed:
                f.write(f"{i}\t{c}\n")
        print(f"    {len(failed)} chunk(s) failed — see {fail_log}")

    return paths


def stitch_audio(paths, output_path):
    combined = AudioSegment.from_wav(paths[0])
    for p in paths[1:]:
        combined = combined.append(AudioSegment.from_wav(p), crossfade=CROSSFADE_MS)
    combined.export(output_path, format="wav")


if __name__ == "__main__":
    text_files = sorted(glob.glob(os.path.join(TEXT_INPUT_DIR, "*.txt")))
    if not text_files:
        raise SystemExit(f"No .txt files found in {TEXT_INPUT_DIR}/ — run extract_text.py first.")

    print(f"Found {len(text_files)} chapter file(s).")
    tts = None  # load lazily, only if there's at least one chapter left to do

    for text_path in text_files:
        chapter_stem = os.path.splitext(os.path.basename(text_path))[0]
        audio_out_path = os.path.join(AUDIO_OUTPUT_DIR, f"{chapter_stem}.wav")

        if os.path.exists(audio_out_path):
            print(f"[SKIP] {chapter_stem} — audio already exists")
            continue

        if tts is None:
            print("Loading XTTS-v2 model...")
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

        with open(text_path, encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"[{chapter_stem}] {len(chunks)} chunks")

        chunk_paths = generate_chapter_audio(tts, chapter_stem, chunks)

        print(f"[{chapter_stem}] stitching -> {audio_out_path}")
        stitch_audio(chunk_paths, audio_out_path)

    print(f"\nDone. Chapter audio files are in {AUDIO_OUTPUT_DIR}/")