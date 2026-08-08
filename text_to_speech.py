"""
Step 2 of 2 — cleaned text -> audiobook (XTTS-v2 / Coqui TTS)

Reads book_clean.txt (produced by extract_text.py), chunks it, generates
Russian speech per chunk, and stitches everything into one final wav.

Resumable: if interrupted, rerun and it picks up where it left off
(already-generated chunk files are skipped).

Usage:
    python generate_audio.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # silence torch.load deprecation noise

import re
import os
from TTS.api import TTS
from pydub import AudioSegment

CLEAN_TEXT_PATH = "book_clean.txt"
OUTPUT_DIR = "chunks"
FINAL_OUTPUT = "audiobook.wav"
SPEAKER_WAV = "reference_voice.wav"
LANGUAGE = "ru"
MAX_CHARS = 180  # XTTS-v2's actual internal limit for Russian is 182 chars;
                 # keeping chunks under this avoids silent mid-sentence truncation

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Chunking (sentence-based, hard-capped to avoid XTTS's per-language
# char limit even on punctuation-free runs of text like tables)
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


# ============================================================
# Generate audio per chunk (resumable, fault-tolerant)
# ============================================================
def generate_audio(chunks):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    paths = []
    failed = []
    for i, chunk in enumerate(chunks):
        out_path = os.path.join(OUTPUT_DIR, f"chunk_{i:05d}.wav")
        if os.path.exists(out_path):
            paths.append(out_path)  # resume: skip already-generated chunks
            continue
        try:
            tts.tts_to_file(
                text=chunk,
                speaker_wav=SPEAKER_WAV,
                language=LANGUAGE,
                file_path=out_path,
                temperature=0.75,          # more natural intonation variation
                repetition_penalty=5.0,
            )
            paths.append(out_path)
        except Exception as e:
            print(f"[SKIPPED chunk {i}] {e}")
            failed.append((i, chunk))
        print(f"Generated {i+1}/{len(chunks)}")

    if failed:
        with open("failed_chunks.txt", "w", encoding="utf-8") as f:
            for i, c in failed:
                f.write(f"{i}\t{c}\n")
        print(f"{len(failed)} chunks failed and were skipped — see failed_chunks.txt")

    return paths


# ============================================================
# Stitch all chunk wavs into one final audiobook file
# ============================================================
def stitch_audio(paths, output_path):
    combined = AudioSegment.empty()
    for p in paths:
        combined += AudioSegment.from_wav(p)
    combined.export(output_path, format="wav")


if __name__ == "__main__":
    if not os.path.exists(CLEAN_TEXT_PATH):
        raise SystemExit(f"{CLEAN_TEXT_PATH} not found — run extract_text.py first.")

    with open(CLEAN_TEXT_PATH, encoding="utf-8") as f:
        text = f.read()

    print("Chunking...")
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    print("Generating audio (this will take a while, safe to resume if interrupted)...")
    paths = generate_audio(chunks)

    print("Stitching final audiobook...")
    stitch_audio(paths, FINAL_OUTPUT)

    print(f"Done. Output saved to {FINAL_OUTPUT}")