"""
Step 1 of 2 — PDF -> cleaned text extraction

Extracts text from a Russian PDF and cleans it up for TTS:
    - strips repeating running headers/footers (detected by position)
    - strips bare page numbers
    - strips table-of-contents dot-leaders ( . . . . )
    - rejoins hyphenated line breaks
    - preserves paragraph breaks

Usage:
    python extract_text.py

Output:
    book_clean.txt — review/edit this before running generate_audio.py
"""

import fitz  # pymupdf
import re
from collections import Counter

PDF_PATH = "book.pdf"
CLEAN_TEXT_PATH = "book_clean.txt"


def normalize(line):
    s = re.sub(r'\d+', '', line)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def is_table_line(line):
    """Table rows extracted from PDFs sometimes contain multiple runs of 2+
    spaces (column gaps) — a secondary, weaker signal alongside block detection."""
    gaps = re.findall(r' {2,}', line)
    return len(gaps) >= 2 and len(line.strip()) > 30


def looks_like_toc_entry(line):
    """A line that's mostly a short title ending in a page number,
    e.g. 'Глава 3. В ловушке контрзависимости 84' — common in TOC pages
    even without dot-leaders."""
    s = line.strip()
    if not s or len(s) > 90:
        return False
    return bool(re.match(r'^.{3,80}?\s\d{1,4}$', s)) and not s.endswith(('.', '!', '?'))


def strip_part_title_runs(text):
    """Some front-matter pages list 'Часть I ... Часть II ... Часть III ...'
    as a single full-width block (not a real two-column table), so the
    block-position table detector misses it. Find chains of 2+ consecutive
    'Часть <N/roman>' markers with short title text between them and strip
    the whole chain, including the trailing title after the last marker."""
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
        if j > i:  # found a chain of 2+ consecutive markers
            to_remove.append((matches[i].start(), matches[j].end()))
            i = j + 1
        else:
            i += 1

    for start, end in sorted(to_remove, reverse=True):
        text = text[:start] + text[end:]
    return text


def is_toc_page(lines):
    """Flag a whole page as table-of-contents if most of its non-empty
    lines look like TOC entries."""
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 5:
        return False
    toc_like = sum(1 for l in non_empty if looks_like_toc_entry(l))
    return toc_like / len(non_empty) > 0.5


def get_page_text_with_tables_stripped(doc):
    """Use block positions (not just text) to find and drop data tables.
    A real prose paragraph block spans close to the page's full text-column
    width; table cells are narrow, irregular-width blocks, and a genuine
    multi-column table also has at least one block shifted right into a
    second column. Detecting that shift flags the page as containing a
    table, and on that page we drop every block that isn't full-width
    prose (which safely removes cell text, row/column headers, and labels
    without touching normal paragraphs elsewhere)."""
    from collections import Counter
    widths = Counter()
    margins = Counter()
    for page in doc:
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, bno, btype = b
            if btype != 0:
                continue
            widths[round((x1 - x0) / 5) * 5] += 1
            margins[round(x0 / 5) * 5] += 1
    standard_width = widths.most_common(1)[0][0]
    standard_margin = margins.most_common(1)[0][0]

    pages_lines = []
    for page in doc:
        blocks = [b for b in page.get_text("blocks") if b[6] == 0]
        has_second_column = any(
            b[0] > standard_margin + 50 for b in blocks
        )
        if has_second_column:
            # table page: keep only genuinely full-width paragraph blocks
            blocks = [b for b in blocks if (b[2] - b[0]) >= 0.85 * standard_width]
        blocks.sort(key=lambda b: (b[1], b[0]))  # reading order: top-to-bottom, left-to-right
        text = "\n".join(b[4] for b in blocks)
        pages_lines.append(text.split("\n"))
    return pages_lines


def extract_clean_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages_lines = get_page_text_with_tables_stripped(doc)

    # detect running headers/footers by position (top 3 / bottom 2 lines of each page)
    pos_counter = Counter()
    for lines in pages_lines:
        top = lines[:3]
        bottom = lines[-2:]
        for l in set(top + bottom):
            norm = normalize(l)
            if norm and len(norm) < 90:
                pos_counter[norm] += 1

    running_heads = {norm for norm, c in pos_counter.items() if c >= 6}

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
        if is_table_line(line):
            return True
        if normalize(s) in running_heads:
            return True
        return False

    cleaned_pages = []
    dropped_toc_pages = 0
    for lines in pages_lines:
        if is_toc_page(lines):
            dropped_toc_pages += 1
            continue  # skip the whole page, it's a table of contents
        kept = [l for l in lines if not is_junk_line(l)]
        cleaned_pages.append("\n".join(kept))

    if dropped_toc_pages:
        print(f"Dropped {dropped_toc_pages} page(s) detected as table of contents.")

    full_text = "\n".join(cleaned_pages)
    full_text = re.sub(r'(\w)-\n(\w)', r'\1\2', full_text)          # dehyphenate
    full_text = re.sub(r'\n{2,}', '<PARA>', full_text)               # mark paragraph breaks
    full_text = re.sub(r'\n', ' ', full_text)                        # join remaining line breaks
    full_text = full_text.replace('<PARA>', '\n\n')
    full_text = re.sub(r'[ \t]{2,}', ' ', full_text)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = strip_part_title_runs(full_text)
    full_text = re.sub(r'[ \t]{2,}', ' ', full_text)  # cleanup any double-space left by the strip
    return full_text.strip()


if __name__ == "__main__":
    print(f"Extracting and cleaning text from {PDF_PATH}...")
    text = extract_clean_text(PDF_PATH)

    with open(CLEAN_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Done. Clean text saved to {CLEAN_TEXT_PATH} ({len(text)} chars)")
    print("Review/edit it if needed, then run: python generate_audio.py")