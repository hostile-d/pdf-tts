"""
Step 1 of 2 — PDF -> cleaned, chapter-split text files

Extracts text from a Russian PDF, cleans it (strips running headers/footers,
page numbers, TOC dot-leaders, data tables), and splits it into one file per
chapter based on detected "Глава N" / "Часть N" markers in the page layout.

Usage:
    python extract_text.py

Output:
    ./output/text/00_<title>.txt, 01_<title>.txt, ... one file per chapter
    (chapter 00 is front matter — dedication/preface/intro before "Глава 1")
"""

import fitz  # pymupdf
import re
import os
from collections import Counter

PDF_PATH = "book.pdf"
TEXT_OUTPUT_DIR = "output/text"

MARKER_RE = re.compile(r'^(Часть|Глава)\s+([IVXLCDM]+|\d+)\.?\s*$')
MIN_CHAPTER_CHARS = 300  # chapters shorter than this (e.g. a bare Part divider) get merged into the next one


def normalize(line):
    s = re.sub(r'\d+', '', line)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def is_title_block(block_text):
    """True if a block's letters are entirely uppercase — used to identify
    chapter/section title blocks (as opposed to running headers, which mix case)."""
    stripped = re.sub(r'[\d\s.,:;\-–—«»!?]', '', block_text)
    if len(stripped) < 4:
        return False
    return stripped == stripped.upper() and any(c.isalpha() for c in stripped)


def looks_like_toc_entry(line):
    s = line.strip()
    if not s or len(s) > 90:
        return False
    return bool(re.match(r'^.{3,80}?\s\d{1,4}$', s)) and not s.endswith(('.', '!', '?'))


def is_toc_page(lines):
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 5:
        return False
    toc_like = sum(1 for l in non_empty if looks_like_toc_entry(l))
    return toc_like / len(non_empty) > 0.5


def strip_part_title_runs(text):
    marker_re = re.compile(r'Часть\s+(?:\d{1,2}|[IVXLCDM]+)\.?\s+')
    matches = list(marker_re.finditer(text))
    if len(matches) < 2:
        return text
    to_remove = []
    i = 0
    while i < len(matches) - 1:
        j = i
        while j + 1 < len(matches):
            gap = text[matches[j].end():matches[j + 1].start()]
            if len(gap) <= 80 and not re.search(r'[.!?]', gap):
                j += 1
            else:
                break
        if j > i:
            to_remove.append((matches[i].start(), matches[j].end()))
            i = j + 1
        else:
            i += 1
    for start, end in sorted(to_remove, reverse=True):
        text = text[:start] + text[end:]
    text = re.sub(r'\s+([.,!?])', r'\1', text)  # cleanup stray space before punctuation left by removal
    return text


def clean_block_text_list(blocks_text_list, running_heads):
    """Apply line-level junk filtering (headers, page numbers, TOC dots)
    and paragraph/dehyphenation cleanup to a list of raw block texts."""
    page_num_re = re.compile(r'^\d{1,4}$')
    toc_dots_re = re.compile(r'\.\s*\.\s*\.\s*\.')

    def is_junk_line(line):
        s = line.strip()
        if not s:
            return True
        if page_num_re.match(s):
            return True
        if toc_dots_re.search(s):
            return True
        if normalize(s) in running_heads:
            return True
        return False

    raw = "\n".join(blocks_text_list)
    raw = fix_title_boundaries(raw)
    lines = raw.split("\n")
    kept = [l for l in lines if not is_junk_line(l)]
    text = "\n".join(kept)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'\n{2,}', '<PARA>', text)
    text = re.sub(r'\n', ' ', text)
    text = text.replace('<PARA>', '\n\n')
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = strip_part_title_runs(text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = normalize_for_tts(text)
    return text.strip()


def has_table_row(blocks):
    """A real table has 2+ blocks sharing the same vertical row (side-by-side
    columns). A centered epigraph or chapter marker is just offset horizontally
    while stacked vertically — no row overlap — so this avoids false positives
    that a simple 'x0 is far from the margin' check would trigger on."""
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            b1, b2 = blocks[i], blocks[j]
            y_overlap = min(b1[3], b2[3]) - max(b1[1], b2[1])
            if y_overlap > 5:
                return True
    return False


TITLE_START = "\uE000"
TITLE_END = "\uE001"


def format_block_for_body(block_text):
    """Standalone section titles (ПРЕДИСЛОВИЕ, ВВЕДЕНИЕ, subheadings, etc.)
    have no natural sentence punctuation, so without this they get fused
    into the surrounding prose and TTS reads them as one run-on blob with
    no pause on either side. Wrap them in sentinels here; fix_title_boundaries()
    later inserts a period before AND after, so the sentence-splitter gets
    real boundaries on both sides."""
    if is_title_block(block_text):
        clean = re.sub(r'\s+', ' ', block_text).strip().rstrip(':')
        return f"{TITLE_START}{clean}.{TITLE_END}"
    return block_text


def fix_title_boundaries(text):
    """Ensure a sentence-ending period precedes every title-marked span
    (not just follows it), so titles never get glued onto the end of the
    previous sentence."""
    result = []
    i = 0
    while True:
        idx = text.find(TITLE_START, i)
        if idx == -1:
            result.append(text[i:])
            break
        before = text[i:idx]
        result.append(before)
        stripped = before.rstrip()
        if stripped and stripped[-1] not in '.!?':
            result.append('.\n\n')
        else:
            result.append('\n\n')
        end_idx = text.find(TITLE_END, idx)
        title_text = text[idx + 1:end_idx]
        result.append(title_text)
        result.append('\n\n')
        i = end_idx + 1
    return ''.join(result)


def normalize_for_tts(text):
    """Strip characters that carry no spoken sound but that XTTS sometimes
    tries to vocalize anyway, producing a click/glitch — every quote-mark
    style (Russian guillemets, German-style low quotes, smart/curly quotes,
    straight quotes). Just deleting them (not replacing with a pause) is
    the standard audiobook-narration convention: a narrator doesn't speak
    the quotation marks themselves."""
    return re.sub(r'[«»„‚‹›‟"\u2018\u2019\u201c\u201d"\'`]', '', text)


def slugify(title, max_len=50):
    s = title.strip().lower()
    s = re.sub(r'[^\w\s-]', '', s, flags=re.UNICODE)
    s = re.sub(r'\s+', '_', s)
    return s[:max_len].strip('_') or "untitled"


def extract_chapters(pdf_path):
    doc = fitz.open(pdf_path)

    # --- pass 1: detect running headers/footers and standard column width (for table stripping) ---
    pos_counter = Counter()
    widths = Counter()
    margins = Counter()
    for page in doc:
        lines = page.get_text().split("\n")
        top, bottom = lines[:3], lines[-2:]
        for l in set(top + bottom):
            norm = normalize(l)
            if norm and len(norm) < 90:
                pos_counter[norm] += 1
        for b in page.get_text("blocks"):
            if b[6] != 0:
                continue
            widths[round((b[2] - b[0]) / 5) * 5] += 1
            margins[round(b[0] / 5) * 5] += 1
    running_heads = {norm for norm, c in pos_counter.items() if c >= 6}
    standard_width = widths.most_common(1)[0][0]
    standard_margin = margins.most_common(1)[0][0]

    # --- pass 2: walk pages, split into chapters at Часть/Глава markers, strip tables ---
    chapters = []  # list of (title, [block_text, ...])
    current_title = "Введение"
    current_blocks = []

    for page in doc:
        raw_blocks = [b for b in page.get_text("blocks") if b[6] == 0]
        raw_blocks.sort(key=lambda b: (b[1], b[0]))

        # table detection: page has 2+ blocks sharing the same row (side-by-side columns)
        if has_table_row(raw_blocks):
            body_blocks = [b for b in raw_blocks if (b[2] - b[0]) >= 0.85 * standard_width]
        else:
            body_blocks = list(raw_blocks)

        # TOC page check (uses full page text, not just filtered blocks)
        all_lines = page.get_text().split("\n")
        if is_toc_page(all_lines):
            continue  # skip whole page

        # chapter/part marker detection — only meaningful at top of page, before other body content
        if body_blocks and MARKER_RE.match(body_blocks[0][4].strip()):
            marker_text = body_blocks[0][4].strip()
            consumed = 1
            title = marker_text
            if len(body_blocks) > 1 and is_title_block(body_blocks[1][4]):
                title = body_blocks[1][4].strip().replace("\n", " ")
                consumed = 2
            # flush current chapter, start a new one
            chapters.append((current_title, current_blocks))
            current_title = title
            current_blocks = [format_block_for_body(b[4]) for b in body_blocks[consumed:]]
        else:
            current_blocks.extend(format_block_for_body(b[4]) for b in body_blocks)

    chapters.append((current_title, current_blocks))

    # --- pass 3: clean each chapter's text, merge tiny chapters (e.g. bare Part dividers) forward ---
    cleaned = []
    for title, blocks_text_list in chapters:
        text = clean_block_text_list(blocks_text_list, running_heads)
        cleaned.append((title, text))

    merged = []
    pending_title = None
    for title, text in cleaned:
        if len(text) < MIN_CHAPTER_CHARS:
            pending_title = title if pending_title is None else f"{pending_title} — {title}"
            continue
        if pending_title:
            title = f"{pending_title} — {title}"
            pending_title = None
        merged.append((title, text))
    if pending_title and merged:
        # trailing tiny chapter with nothing after it: attach to the last real chapter
        last_title, last_text = merged[-1]
        merged[-1] = (f"{last_title} — {pending_title}", last_text)

    return merged


if __name__ == "__main__":
    os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

    print(f"Extracting and splitting {PDF_PATH} into chapters...")
    chapters = extract_chapters(PDF_PATH)

    for i, (title, text) in enumerate(chapters):
        slug = slugify(title)
        out_path = os.path.join(TEXT_OUTPUT_DIR, f"{i:02d}_{slug}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  {out_path}  ({len(text)} chars)  — {title[:60]}")

    print(f"\nDone. {len(chapters)} chapter file(s) written to {TEXT_OUTPUT_DIR}/")
    print("Review/edit them if needed, then run: python generate_audio.py")