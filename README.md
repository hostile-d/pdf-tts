# PDF Audiobook Maker

A small, personal-use **AI audiobook generator** for creating an audio version of an older Russian-language book when an official audiobook is unavailable.

It takes a PDF, extracts readable prose, and uses the AI text-to-speech model Coqui XTTS-v2 to narrate it in the voice represented by a short reference recording. The generated pieces are kept separately while processing, so a long run can resume after an interruption.

## AI used in this project

This project uses **Coqui XTTS-v2**, a neural text-to-speech model. It generates spoken Russian audio from the cleaned book text and conditions the narration on `reference_voice.wav`, allowing the output to resemble that reference voice. It requires an NVIDIA GPU with CUDA in the current configuration.

## Development note

This code was written with assistance from **Claude** and **Codex** by a developer who is not a Python specialist. Treat it as an experimental personal project: review the code, dependencies, and generated output before relying on it or adapting it for another use.

## How it works

```
book.pdf  ->  pdf_to_txt.py  ->  book_clean.txt  ->  text_to_speech.py  ->  audiobook.wav
```

1. `pdf_to_txt.py` extracts text with PyMuPDF and removes common PDF artefacts such as page numbers, repeated headers/footers, table-of-contents pages, and tables.
2. Review and, if necessary, edit `book_clean.txt`. This step is useful for correcting OCR or layout mistakes before speech generation.
3. `text_to_speech.py` splits the text into short, sentence-aware chunks and synthesizes each chunk with XTTS-v2 using `reference_voice.wav`.
4. The chunks are joined into `audiobook.wav`.

## Requirements

- Python 3.11 (the included environment was created with it)
- NVIDIA GPU and CUDA: the synthesis script explicitly uses CUDA
- FFmpeg, required by `pydub` to export audio
- Python packages in `requirements.txt`

## Usage

Place these files in the project directory:

- `book.pdf` — the source book
- `reference_voice.wav` — a clear recording of a voice you are authorized to use

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Extract and review the text:

```bash
python pdf_to_txt.py
```

After reviewing `book_clean.txt`, generate the audiobook:

```bash
python text_to_speech.py
```

Audio chunks are written to `chunks/`. If generation stops, run the last command again; existing chunks are skipped. On completion, the combined result is written to `audiobook.wav`.

## Responsible use

This project is intended for personal use with material you are allowed to reproduce and with voices you have permission to use. The code license applies only to this repository's code; it does not grant rights to a source book, generated audio, a voice recording, or any third-party model or dependency.

## License

The code is released under the [MIT License](LICENSE).
