import logging
from pathlib import Path

import pytest

from app.ingestion.parsers import PARSERS, parse_document
from tests.conftest import REAL_DB
from tests.parser_fixtures import HAVE_OCR_TOOLS, HTML_SAMPLE, MARKDOWN_SAMPLE, make_docx, make_text_pdf

CORPUS = REAL_DB.parent / "corpus"
SECRET = "SECRET-CONTENT-DO-NOT-LOG-12345"


# --- choosing a parser -------------------------------------------------------


@pytest.mark.parametrize(
    "name, parser",
    [("a.pdf", "pdf"), ("a.PDF", "pdf"), ("a.docx", "docx"), ("a.DocX", "docx"),
     ("a.md", "markdown"), ("a.markdown", "markdown"), ("a.html", "html"), ("a.HTM", "html")],
)
def test_parser_is_chosen_by_extension_case_insensitively(tmp_path, name, parser):
    source = {
        "pdf": make_text_pdf, "docx": make_docx,
    }.get(parser)
    path = tmp_path / name
    if source:
        source(tmp_path / "built")
        (tmp_path / "built").rename(path)
    else:
        path.write_text(HTML_SAMPLE if parser == "html" else MARKDOWN_SAMPLE, encoding="utf-8")
    doc = parse_document(path)
    assert doc.ok and doc.parser == parser and doc.sections


def test_supported_extensions():
    assert set(PARSERS) == {".pdf", ".docx", ".md", ".markdown", ".html", ".htm"}


# --- bad files never raise ----------------------------------------------------


def _assert_skipped(doc, error_type):
    assert not doc.ok and doc.error == error_type
    assert doc.sections == [] and doc.total_chars == 0 and doc.table_count == 0


def test_corrupt_pdf_is_skipped_without_raising(tmp_path, caplog):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 " + SECRET.encode() + b" this is not a real pdf \x00\x01\x02")
    with caplog.at_level(logging.INFO):
        doc = parse_document(path)
    assert not doc.ok and doc.sections == []
    assert doc.parser == "pdf"
    assert any(
        r.getMessage() == f"skipped doc=broken.pdf error={doc.error}" and r.levelno == logging.ERROR
        for r in caplog.records
    )


def test_corrupt_docx_is_skipped_without_raising(tmp_path, caplog):
    path = tmp_path / "broken.docx"
    path.write_bytes(SECRET.encode() + b" not a zip archive")
    with caplog.at_level(logging.INFO):
        doc = parse_document(path)
    _assert_skipped(doc, "PackageNotFoundError")
    assert doc.parser == "docx"


@pytest.mark.parametrize("suffix", [".pdf", ".docx"])
def test_zero_byte_file_is_skipped_without_raising(tmp_path, suffix):
    path = tmp_path / f"empty{suffix}"
    path.write_bytes(b"")
    doc = parse_document(path)
    assert not doc.ok and doc.sections == []


def test_unsupported_extension_is_skipped(tmp_path, caplog):
    path = tmp_path / "notes.xyz"
    path.write_text(SECRET, encoding="utf-8")
    with caplog.at_level(logging.INFO):
        doc = parse_document(path)
    _assert_skipped(doc, "UnsupportedFormat")
    assert doc.parser == "none"
    assert "skipped doc=notes.xyz error=UnsupportedFormat" in caplog.text


def test_missing_file_is_skipped(tmp_path):
    doc = parse_document(tmp_path / "gone.pdf")
    _assert_skipped(doc, "FileNotFoundError")


def test_only_doc_name_and_error_type_are_logged_never_content(tmp_path, caplog):
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(SECRET.encode())
    bad_docx = tmp_path / "bad.docx"
    bad_docx.write_bytes(SECRET.encode())
    odd = tmp_path / "odd.xyz"
    odd.write_text(SECRET, encoding="utf-8")
    with caplog.at_level(logging.DEBUG):
        for p in (bad_pdf, bad_docx, odd):
            parse_document(p)
    assert caplog.records
    assert SECRET not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)  # no tracebacks (they quote content)
    assert str(tmp_path) not in caplog.text  # file name only, not the directory


def test_one_bad_file_does_not_stop_the_others(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"garbage")
    good = tmp_path / "good.md"
    good.write_text(MARKDOWN_SAMPLE, encoding="utf-8")
    results = [parse_document(p) for p in sorted(tmp_path.iterdir())]
    assert [d.ok for d in results] == [False, True]


def test_markdown_with_invalid_utf8_still_parses(tmp_path):
    path = tmp_path / "latin.md"
    path.write_bytes(b"# Title\n\nCaf\xe9 menu, not valid utf-8.\n")
    doc = parse_document(path)
    assert doc.ok and "menu" in doc.sections[0].text


# --- the real corpus (only if it is present) --------------------------------


@pytest.mark.skipif(not CORPUS.is_dir(), reason="data/corpus not present")
def test_every_real_corpus_file_yields_text_and_nothing_crashes():
    files = [p for p in sorted(CORPUS.iterdir()) if p.is_file()]
    assert files
    for path in files:
        doc = parse_document(path)
        if not doc.ok:
            # the only acceptable failure: a scanned page when OCR tools are absent
            assert doc.error == "OcrFailed" and not HAVE_OCR_TOOLS, f"{path.name}: {doc.error}"
            continue
        assert doc.sections and doc.total_chars > 100, f"{path.name} extracted almost nothing"
        for section in doc.sections:
            assert section.text.strip()
            assert section.kind in ("text", "table")
            if section.kind == "table":
                assert all(line.startswith("Row: ") for line in section.text.splitlines()), path.name


@pytest.mark.skipif(not CORPUS.is_dir() or not HAVE_OCR_TOOLS, reason="needs data/corpus and OCR tools")
def test_real_corpus_scanned_document_is_read_by_ocr_without_special_cases():
    docs = [parse_document(p) for p in sorted(CORPUS.iterdir()) if p.suffix.lower() == ".pdf"]
    scanned = [d for d in docs if d.ocr_pages]
    assert scanned, "expected at least one PDF page to need OCR"
    for doc in scanned:
        assert doc.ok and doc.ocr_chars > 100 and doc.ocr_failed_pages == []
