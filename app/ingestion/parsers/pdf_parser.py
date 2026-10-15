"""PDF parser (pdfplumber), with per-page OCR fallback.

Per page:
  1. If the text layer has fewer than ``MIN_TEXT_CHARS`` non-space characters
     the page is OCR'd (see ocr.py). The decision looks only at the page.
  2. Otherwise tables are extracted and rendered as row-wise text, and the
     remaining text is read line by line with its font size and weight.

Headings are lines set noticeably larger than the document's body text, or
short, fully bold lines at body size. Body size is the most common font size
of non-table text across the whole document, so no sizes are hard-coded.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pdfplumber

from app.ingestion.parsers import ocr as ocr_module
from app.ingestion.parsers.builder import SectionBuilder, clean
from app.ingestion.parsers.models import ParsedDocument
from app.ingestion.parsers.tables import TableData, render_table, split_stacked_row
from app.logging_setup import get_logger

log = get_logger(__name__)

MIN_TEXT_CHARS = 20  # fewer non-space characters than this -> no usable text layer

HEADING_SIZE_RATIO = 1.15  # this much larger than body text -> heading
BOLD_HEADING_MAX_CHARS = 90
PARAGRAPH_GAP_RATIO = 0.5  # vertical gap (x font size) that starts a new paragraph
OCR_HEADING_HEIGHT_RATIO = 1.3
OCR_HEADING_MAX_CHARS = 100
PAGE_MARGIN_RATIO = 0.08  # outer band of the page where header/footer lines live

# \ufffd is how a bullet appears when the PDF font has no Unicode mapping for it.
_BULLET_GLYPHS = "\u2022\u00b7\u25aa\u25cf\u25e6\u25a0\u25a1\ufffd"
# pdfplumber writes "(cid:N)" for a glyph the font cannot map to Unicode. At the
# start of a line that is a list bullet; anywhere else it is left as is (and
# counted in a warning) rather than silently deleting a character.
_UNMAPPED_GLYPH = re.compile(r"\(cid:\d+\)")
_LEADING_UNMAPPED_GLYPH = re.compile(r"^\(cid:\d+\)\s*")
_PAGE_NUMBER = re.compile(r"(?:^|\s)page\s+\d+(?:\s*(?:of|/)\s*\d+)?\s*$|^\s*\d+\s*$", re.I)
_NON_SPACE = re.compile(r"\S")

# Replaced in tests; same signature as ocr_module.ocr_pdf_page.
OcrFunction = Callable[[Path, int], "list[ocr_module.OcrLine]"]


@dataclass
class _Line:
    text: str
    top: float
    bottom: float
    size: float
    bold: bool


@dataclass
class _Block:
    kind: str  # "heading" | "para" | "table"
    text: str
    page: int
    rows: tuple = ()  # tables only: structured rows (see parsers/tables.py)


def _is_bold(fontname: str) -> bool:
    name = fontname.lower()
    return "bold" in name or "black" in name or "heavy" in name


def _center_inside(obj: dict, box: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = box
    cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


def _is_page_furniture(text: str, top: float, bottom: float, page_height: float) -> bool:
    """A page-number footer/header ("Page 3", "3 of 9") in the page margin."""
    in_margin = top > page_height * (1 - PAGE_MARGIN_RATIO) or bottom < page_height * PAGE_MARGIN_RATIO
    return in_margin and bool(_PAGE_NUMBER.search(text.strip()))


def _line_from_chars(text: str, chars: list[dict], top: float, bottom: float) -> _Line | None:
    glyphs = [c for c in chars if c["text"].strip()]
    if not glyphs:
        return None
    size = Counter(round(c["size"], 1) for c in glyphs).most_common(1)[0][0]
    return _Line(
        text=_LEADING_UNMAPPED_GLYPH.sub("\u2022 ", clean(text)),
        top=top,
        bottom=bottom,
        size=size,
        bold=all(_is_bold(c["fontname"]) for c in glyphs),
    )


def _tables_text(page, tables) -> list[tuple[float, TableData]]:
    """(top, row-wise text + structured rows) for each table found on the page."""
    found = []
    for table in tables:
        rows = split_stacked_row(table.extract())
        first_row = table.rows[0].bbox if table.rows else table.bbox
        glyphs = [
            c for c in page.chars if c["text"].strip() and _center_inside(c, first_row)
        ]
        has_header = bool(glyphs) and all(_is_bold(c["fontname"]) for c in glyphs)
        data = render_table(rows, has_header)
        if data.text:
            found.append((table.bbox[1], data))
    return found


def _page_lines(page, table_boxes: list[tuple[float, float, float, float]]) -> list[_Line]:
    """Text lines outside tables, without page-number furniture."""
    if table_boxes:
        page = page.filter(
            lambda o: not (
                o.get("object_type") == "char" and any(_center_inside(o, b) for b in table_boxes)
            )
        )
    lines = []
    for raw in page.extract_text_lines(return_chars=True):
        line = _line_from_chars(raw["text"], raw["chars"], raw["top"], raw["bottom"])
        if line and line.text and not _is_page_furniture(
            line.text, line.top, line.bottom, float(page.height)
        ):
            lines.append(line)
    return lines


def _body_size(lines: list[_Line]) -> float | None:
    sizes: Counter[float] = Counter()
    for line in lines:
        sizes[line.size] += len(line.text)
    return sizes.most_common(1)[0][0] if sizes else None


def _is_heading(line: _Line, body: float | None) -> bool:
    if body is None or len(line.text) > 150:
        return False
    if line.size >= body * HEADING_SIZE_RATIO:
        return True
    return (
        line.bold
        and line.size >= body * 0.95
        and len(line.text) <= BOLD_HEADING_MAX_CHARS
        and not line.text.rstrip().endswith((".", ",", ";", ":"))
    )


def _starts_bullet(text: str) -> bool:
    """A list item: a bullet glyph, or a dash/asterisk followed by a space."""
    if not text:
        return False
    if text[0] in _BULLET_GLYPHS:
        return True
    return text[0] in "-*" and text[1:2] == " "


def _blocks_for_page(
    page_no: int, items: list[tuple[float, str, object]], body: float | None
) -> list[_Block]:
    """Turn a page's lines and tables (sorted top to bottom) into blocks."""
    blocks: list[_Block] = []
    paragraph: list[str] = []
    prev: _Line | None = None  # previous body or heading line
    prev_was_heading = False

    def close_paragraph() -> None:
        if paragraph:
            blocks.append(_Block("para", " ".join(paragraph), page_no))
            paragraph.clear()

    for _top, kind, payload in items:
        if kind == "table":
            close_paragraph()
            blocks.append(_Block("table", payload.text, page_no, payload.rows))
            prev, prev_was_heading = None, False
            continue
        line: _Line = payload  # type: ignore[assignment]
        gap = line.top - prev.bottom if prev else None
        if _is_heading(line, body):
            close_paragraph()
            wrapped = (
                prev_was_heading
                and prev is not None
                and prev.size == line.size
                and gap is not None
                and gap < 0.8 * line.size
            )
            if wrapped:  # a long heading wrapped onto a second line
                blocks[-1].text += " " + line.text
            else:
                blocks.append(_Block("heading", line.text, page_no))
            prev, prev_was_heading = line, True
            continue
        new_paragraph = (
            prev is None
            or prev_was_heading
            or _starts_bullet(line.text)
            or (gap is not None and gap > PARAGRAPH_GAP_RATIO * line.size)
        )
        if new_paragraph:
            close_paragraph()
        paragraph.append(line.text)
        prev, prev_was_heading = line, False
    close_paragraph()
    return blocks


_SENTENCE_END = (".", "!", "?", ":", ";", ")", '"', "\u201d")


def _join_page_breaks(blocks_by_page: dict[int, list[_Block]]) -> None:
    """Re-join a paragraph that a page break cut in the middle of a sentence.

    Joined only when the last block of a page is a paragraph that does not end
    like a sentence AND the next page starts with a paragraph (before any
    heading or table) that begins with a lowercase letter. The joined text
    keeps the page where the paragraph started. Both conditions are needed so
    a bullet list or a new section is never glued onto the previous one.
    """
    pages = sorted(blocks_by_page)
    for earlier, later in zip(pages, pages[1:]):
        if later != earlier + 1:
            continue
        before, after = blocks_by_page[earlier], blocks_by_page[later]
        if not before or not after:
            continue
        tail, head = before[-1], after[0]
        if (
            tail.kind == "para"
            and head.kind == "para"
            and not tail.text.rstrip().endswith(_SENTENCE_END)
            and head.text[:1].islower()
        ):
            tail.text += " " + head.text
            after.pop(0)


def _blocks_from_ocr(page_no: int, lines: list["ocr_module.OcrLine"]) -> list[_Block]:
    """Group OCR lines into paragraphs; unusually tall short lines are headings."""
    if not lines:
        return []
    typical = sorted(l.height for l in lines)[len(lines) // 2]
    blocks: list[_Block] = []
    current_key = None
    for line in lines:
        is_heading = (
            line.height >= typical * OCR_HEADING_HEIGHT_RATIO
            and len(line.text) <= OCR_HEADING_MAX_CHARS
        )
        if is_heading:
            if blocks and blocks[-1].kind == "heading" and current_key == line.paragraph:
                blocks[-1].text += " " + line.text
            else:
                blocks.append(_Block("heading", line.text, page_no))
            current_key = line.paragraph
            continue
        if blocks and blocks[-1].kind == "para" and current_key == line.paragraph:
            blocks[-1].text += " " + line.text
        else:
            blocks.append(_Block("para", line.text, page_no))
        current_key = line.paragraph
    return blocks


def parse_pdf(path: str | Path, ocr_function: OcrFunction | None = None) -> ParsedDocument:
    path = Path(path)
    ocr_page = ocr_function or ocr_module.ocr_pdf_page
    doc = ParsedDocument(path=path, parser="pdf")

    pages_content: list[tuple[int, list[tuple[float, str, object]], list[_Line]]] = []
    ocr_blocks: dict[int, list[_Block]] = {}

    with pdfplumber.open(path) as pdf:
        doc.page_count = len(pdf.pages)
        for page_no, page in enumerate(pdf.pages, start=1):
            layer = page.extract_text() or ""
            if len(_NON_SPACE.findall(layer)) < MIN_TEXT_CHARS:
                lines = _ocr_page(doc, ocr_page, page_no)
                if lines is not None:
                    ocr_blocks[page_no] = _blocks_from_ocr(page_no, lines)
                continue
            found = page.find_tables()  # once per page: text and boxes must agree
            lines = _page_lines(page, [t.bbox for t in found])
            items: list[tuple[float, str, object]] = [(l.top, "line", l) for l in lines]
            items += [(top, "table", data) for top, data in _tables_text(page, found)]
            items.sort(key=lambda item: item[0])
            pages_content.append((page_no, items, lines))

    body = _body_size([l for _, _, lines in pages_content for l in lines])
    blocks_by_page: dict[int, list[_Block]] = dict(ocr_blocks)
    for page_no, items, _lines in pages_content:
        blocks_by_page[page_no] = _blocks_for_page(page_no, items, body)
    _join_page_breaks(blocks_by_page)

    builder = SectionBuilder()
    for page_no in sorted(blocks_by_page):
        for block in blocks_by_page[page_no]:
            if block.kind == "heading":
                builder.heading(block.text, block.page)
            elif block.kind == "table":
                builder.table(block.text, block.page, block.rows)
            else:
                builder.paragraph(block.text, block.page)
    doc.sections = builder.finish()

    unmapped = sum(len(_UNMAPPED_GLYPH.findall(s.text)) for s in doc.sections)
    if unmapped:
        log.warning("unmapped glyphs left in text doc=%s count=%d", path.name, unmapped)
    if doc.ocr_pages:
        log.info(
            "ocr doc=%s pages=%s chars=%d",
            path.name,
            doc.ocr_pages,
            doc.ocr_chars,
        )
    if doc.ocr_failed_pages and not doc.sections:
        doc.error = "OcrFailed"  # nothing readable at all: skip the document
    elif not doc.sections:
        log.warning("no text extracted doc=%s", path.name)
    return doc


def _ocr_page(
    doc: ParsedDocument, ocr_page: OcrFunction, page_no: int
) -> list["ocr_module.OcrLine"] | None:
    """OCR one page. On failure log the error type only and carry on."""
    try:
        lines = ocr_page(doc.path, page_no)
    except Exception as exc:  # noqa: BLE001 - tesseract/poppler missing, bad render, ...
        doc.ocr_failed_pages.append(page_no)
        log.error("ocr failed doc=%s page=%d error=%s", doc.path.name, page_no, type(exc).__name__)
        return None
    doc.ocr_pages.append(page_no)
    log.info(
        "ocr page doc=%s page=%d chars=%d", doc.path.name, page_no, sum(len(l.text) for l in lines)
    )
    return lines
