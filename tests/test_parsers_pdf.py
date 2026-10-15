import logging

import pytest

from app.ingestion.parsers import parse_document
from app.ingestion.parsers.ocr import OcrLine
from app.ingestion.parsers.pdf_parser import MIN_TEXT_CHARS, parse_pdf
from tests.parser_fixtures import (
    HAVE_OCR_TOOLS,
    INJECTION,
    make_mixed_pdf,
    make_page_break_pdf,
    make_pdf_from_pages,
    make_scanned_pdf,
    make_text_pdf,
)

requires_ocr = pytest.mark.skipif(not HAVE_OCR_TOOLS, reason="tesseract / poppler not installed")


def _no_ocr(path, page):  # a parser given this must never need OCR
    raise AssertionError(f"OCR must not run for page {page}")


def _all_text(doc):
    return "\n".join(s.text for s in doc.sections)


# --- text layer: headings, paragraphs, tables --------------------------------


def test_text_pdf_headings_and_sections(tmp_path):
    doc = parse_pdf(make_text_pdf(tmp_path / "t.pdf"), ocr_function=_no_ocr)
    assert doc.ok and doc.parser == "pdf" and doc.page_count == 1
    assert doc.ocr_pages == []
    headings = []
    for s in doc.sections:
        if s.heading not in headings:
            headings.append(s.heading)
    assert headings == ["Sample Leave Policy", "1. Annual leave", "2. Accrual"]
    assert all(s.page == 1 for s in doc.sections)


def test_wrapped_lines_become_one_paragraph_and_bullets_stay_separate(tmp_path):
    doc = parse_pdf(make_text_pdf(tmp_path / "t.pdf"), ocr_function=_no_ocr)
    leave = next(s for s in doc.sections if s.heading == "1. Annual leave" and s.kind == "text")
    paragraphs = leave.text.split("\n")
    assert paragraphs[0].startswith("Employees earn annual leave each month")
    assert paragraphs[0].endswith("respond promptly.")  # two visual lines, one paragraph
    assert paragraphs[1] == "Second paragraph under annual leave."
    assert paragraphs[2] == "\u2022 Leave year runs from January to December."
    assert paragraphs[3] == "\u2022 Working days are Monday to Friday."


def test_table_becomes_row_text_with_repeated_headers(tmp_path):
    doc = parse_pdf(make_text_pdf(tmp_path / "t.pdf"), ocr_function=_no_ocr)
    tables = [s for s in doc.sections if s.kind == "table"]
    assert len(tables) == 1 and doc.table_count == 1
    assert tables[0].heading == "2. Accrual"
    assert tables[0].text.splitlines() == [
        "Row: Tenure 0-2 years | Accrual 1.5 days/month",
        "Row: Tenure 3-5 years | Accrual 2 days/month",
    ]


def test_table_text_is_not_duplicated_in_the_text_flow(tmp_path):
    doc = parse_pdf(make_text_pdf(tmp_path / "t.pdf"), ocr_function=_no_ocr)
    assert _all_text(doc).count("1.5 days/month") == 1
    text_only = "\n".join(s.text for s in doc.sections if s.kind == "text")
    assert "0-2 years" not in text_only


def test_page_number_footer_is_dropped_but_instruction_like_text_is_kept(tmp_path):
    doc = parse_pdf(make_text_pdf(tmp_path / "t.pdf"), ocr_function=_no_ocr)
    text = _all_text(doc)
    assert "Page 1" not in text and "TEST-DOC" not in text
    assert INJECTION in text  # document content is data, never filtered


def test_unmapped_glyph_at_line_start_is_a_bullet_but_midline_ones_are_kept():
    from app.ingestion.parsers.pdf_parser import _line_from_chars

    glyphs = [{"text": "x", "size": 10.0, "fontname": "Helvetica"}]
    assert _line_from_chars("(cid:127) Item one", glyphs, 0, 10).text == "\u2022 Item one"
    # not deleted: a mid-line unmapped glyph might be a real character
    assert _line_from_chars("Fee is (cid:128)5", glyphs, 0, 10).text == "Fee is (cid:128)5"


# --- per-page OCR decision ---------------------------------------------------


def test_text_free_page_triggers_ocr_and_returns_its_text(tmp_path, caplog):
    path = make_scanned_pdf(tmp_path / "scan.pdf", ["unused: the OCR function is faked"])
    calls = []

    def fake_ocr(p, page):
        calls.append(page)
        return [
            OcrLine("INTERNAL MEMORANDUM", height=60, paragraph=(1, 1)),
            OcrLine("Employees may work from home", height=30, paragraph=(2, 1)),
            OcrLine("up to three days per week.", height=30, paragraph=(2, 1)),
            OcrLine("Please contact HR.", height=30, paragraph=(3, 1)),
        ]

    with caplog.at_level(logging.INFO):
        doc = parse_pdf(path, ocr_function=fake_ocr)

    assert calls == [1]
    assert doc.ocr_pages == [1] and doc.ocr_failed_pages == [] and doc.ok
    assert [(s.heading, s.page, s.text) for s in doc.sections] == [
        (
            "INTERNAL MEMORANDUM",
            1,
            "Employees may work from home up to three days per week.\nPlease contact HR.",
        )
    ]
    assert doc.ocr_chars == len(doc.sections[0].text)
    logged = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("ocr doc=scan.pdf pages=[1] chars=") and str(doc.ocr_chars) in m for m in logged)


def test_ocr_is_decided_per_page(tmp_path):
    path = make_mixed_pdf(tmp_path / "mixed.pdf", ["Page one has a real text layer, with plenty of words."], ["x"])
    calls = []

    def fake_ocr(p, page):
        calls.append(page)
        return [OcrLine("Text read from the scanned second page.", height=30, paragraph=(1, 1))]

    doc = parse_pdf(path, ocr_function=fake_ocr)
    assert calls == [2]  # page 1 has a text layer: no OCR
    assert doc.ocr_pages == [2]
    by_page = {s.page: s.text for s in doc.sections}
    assert by_page[1].startswith("Page one has a real text layer")
    assert by_page[2] == "Text read from the scanned second page."


def test_ocr_threshold_is_about_twenty_characters(tmp_path):
    assert MIN_TEXT_CHARS == 20
    short = make_pdf_from_pages(tmp_path / "short.pdf", [["Only 13 chars"]])  # 11 non-space
    longer = make_pdf_from_pages(tmp_path / "long.pdf", [["This line has thirty-plus characters"]])
    seen = []

    def fake(p, page):
        seen.append(p.name)
        return []

    parse_pdf(short, ocr_function=fake)
    parse_pdf(longer, ocr_function=fake)
    assert seen == ["short.pdf"]


def test_ocr_failure_on_one_page_keeps_the_rest(tmp_path, caplog):
    path = make_mixed_pdf(tmp_path / "mixed.pdf", ["Page one has a real text layer, with plenty of words."], ["x"])

    def broken_ocr(p, page):
        raise RuntimeError("tesseract exploded with secret page content")

    with caplog.at_level(logging.INFO):
        doc = parse_pdf(path, ocr_function=broken_ocr)
    assert doc.ok  # page 1 is still usable
    assert doc.ocr_failed_pages == [2] and doc.ocr_pages == []
    assert [s.page for s in doc.sections] == [1]
    assert any("ocr failed doc=mixed.pdf page=2 error=RuntimeError" in r.getMessage() for r in caplog.records)
    assert "secret page content" not in caplog.text  # error type only, never the message


def test_document_with_only_failed_ocr_pages_is_marked_skipped(tmp_path):
    path = make_scanned_pdf(tmp_path / "scan.pdf", ["whatever"])

    def broken_ocr(p, page):
        raise OSError("poppler missing")

    doc = parse_pdf(path, ocr_function=broken_ocr)
    assert doc.error == "OcrFailed" and not doc.ok and doc.sections == []


@requires_ocr
def test_real_ocr_reads_a_scanned_pdf(tmp_path):
    path = make_scanned_pdf(tmp_path / "scan.pdf", ["Hybrid work policy update", "Employees may work from home"])
    doc = parse_document(path)
    assert doc.ok and doc.ocr_pages == [1] and doc.ocr_failed_pages == []
    text = _all_text(doc).lower()
    assert "hybrid" in text and "employees" in text and "home" in text
    assert doc.ocr_chars > MIN_TEXT_CHARS


@requires_ocr
def test_real_ocr_only_for_the_image_page_of_a_mixed_pdf(tmp_path):
    path = make_mixed_pdf(
        tmp_path / "mixed.pdf",
        ["Page one carries a genuine text layer with several words."],
        ["Scanned page about carry forward"],
    )
    doc = parse_document(path)
    assert doc.ocr_pages == [2]
    by_page = {s.page: s.text.lower() for s in doc.sections}
    assert "genuine text layer" in by_page[1]
    assert "scanned" in by_page[2] and "forward" in by_page[2]


# --- a sentence cut by a page break ------------------------------------------


def test_sentence_split_by_a_page_break_is_rejoined(tmp_path):
    doc = parse_pdf(
        make_page_break_pdf(tmp_path / "pb.pdf", "at least ten working days of leave."), ocr_function=_no_ocr
    )
    paragraphs = _all_text(doc).split("\n")
    assert "Employees are encouraged to take at least ten working days of leave." in paragraphs
    assert "A later paragraph that stands on its own." in paragraphs
    joined = next(s for s in doc.sections if "encouraged" in s.text)
    assert joined.page == 1  # attributed to the page where the sentence started


def test_new_paragraph_on_the_next_page_is_not_glued_on(tmp_path):
    doc = parse_pdf(make_page_break_pdf(tmp_path / "pb.pdf", "Another paragraph starts here."), ocr_function=_no_ocr)
    paragraphs = _all_text(doc).split("\n")
    assert "Employees are encouraged to take" in paragraphs
    assert "Another paragraph starts here." in paragraphs
    assert {s.page for s in doc.sections} == {1, 2}
