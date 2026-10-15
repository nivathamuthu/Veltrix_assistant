import subprocess
import sys
from pathlib import Path

import pytest

from app.ingestion import report
from app.ingestion.parsers import ParsedDocument, Section
from app.ingestion.parsers import ocr as ocr_module
from tests.parser_fixtures import MARKDOWN_SAMPLE, make_docx, make_scanned_pdf, make_text_pdf


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def folder(tmp_path):
    make_text_pdf(tmp_path / "policy.pdf")
    make_docx(tmp_path / "handbook.docx")
    (tmp_path / "rules.md").write_text(MARKDOWN_SAMPLE, encoding="utf-8")
    (tmp_path / "tiny.md").write_text("# T\n\nhi\n", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "other.xyz").write_text("data", encoding="utf-8")
    return tmp_path


def _rows(out):
    lines = out.splitlines()
    header = lines[0].split()
    assert header[:2] == ["file", "parser"] and "chars" in header and "note" in lines[0]
    return {line.split()[0]: line for line in lines[2:] if line and not line.startswith(" ")}


def test_report_prints_one_row_per_file_with_the_required_columns(folder, capsys):
    assert report.main(["--dir", str(folder)]) == 0
    out = capsys.readouterr().out
    first = out.splitlines()[0]
    for column in ("file", "parser", "sections", "tables", "ocr pages", "chars"):
        assert column in first
    rows = _rows(out)
    assert {"policy.pdf", "handbook.docx", "rules.md", "tiny.md", "broken.pdf", "other.xyz"} <= set(rows)
    assert " pdf " in rows["policy.pdf"] and " docx " in rows["handbook.docx"] and " markdown " in rows["rules.md"]


def test_report_flags_failures_and_low_character_counts(folder, capsys):
    report.main(["--dir", str(folder)])
    rows = _rows(capsys.readouterr().out)
    assert "FAILED (" in rows["broken.pdf"]
    assert "FAILED (UnsupportedFormat)" in rows["other.xyz"]
    assert "LOW" in rows["tiny.md"]
    assert "LOW" not in rows["policy.pdf"] and "FAILED" not in rows["policy.pdf"]


def test_report_counts_tables_and_ends_with_a_summary(folder, capsys):
    report.main(["--dir", str(folder)])
    out = capsys.readouterr().out
    policy = _rows(out)["policy.pdf"].split()
    assert policy[policy.index("pdf") + 2] == "1"  # sections, then tables
    assert "6 files (5 supported)" in out and "flagged" in out


def test_cli_entry_point_prints_the_table_on_stdout_and_logs_on_stderr(folder):
    """Run it exactly as documented: python -m app.ingestion.report."""
    result = subprocess.run(
        [sys.executable, "-m", "app.ingestion.report", "--dir", str(folder)],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "policy.pdf" in result.stdout and "6 files (5 supported)" in result.stdout
    assert "ERROR" not in result.stdout  # the table stays clean
    assert "skipped doc=broken.pdf" in result.stderr and "skipped doc=other.xyz" in result.stderr


def test_report_missing_folder_exits_with_error(tmp_path, capsys):
    assert report.main(["--dir", str(tmp_path / "nope")]) == 2
    assert "folder not found" in capsys.readouterr().err


def test_report_dump_prints_every_section_of_one_file(folder, capsys):
    assert report.main(["--dir", str(folder), "--dump", "rules.md"]) == 0
    out = capsys.readouterr().out
    assert "===== rules.md =====" in out and "kind=table" in out
    assert "Row: Tenure 0-2 years | Accrual 1.5 days/month" in out
    assert report.main(["--dir", str(folder), "--dump", "absent.md"]) == 2


def test_report_show_ocr_prints_the_text_ocr_produced(tmp_path, capsys, monkeypatch):
    make_scanned_pdf(tmp_path / "scan.pdf", ["x"])
    monkeypatch.setattr(
        ocr_module,
        "ocr_pdf_page",
        lambda path, page: [ocr_module.OcrLine("Scanned memo text here, long enough to count.", 30, (1, 1))],
    )
    assert report.main(["--dir", str(tmp_path), "--show-ocr"]) == 0
    out = capsys.readouterr().out
    assert "scan.pdf" in out and "1 (" in out  # "ocr pages" column shows page 1 and a char count
    assert "===== OCR text: scan.pdf (pages [1]) =====" in out
    assert "Scanned memo text here, long enough to count." in out


def test_flag_thresholds():
    ok = ParsedDocument(path=report.Path("a.pdf"), parser="pdf", page_count=2,
                        sections=[Section(None, 1, "x" * 1500)])
    assert report.flag(ok) == ""
    thin_pages = ParsedDocument(path=report.Path("a.pdf"), parser="pdf", page_count=10,
                                sections=[Section(None, 1, "x" * 1500)])
    assert report.flag(thin_pages) == "LOW: few characters per page"
    empty = ParsedDocument(path=report.Path("a.pdf"), parser="pdf")
    assert report.flag(empty) == "LOW: no text"
    ocr_failed = ParsedDocument(path=report.Path("a.pdf"), parser="pdf", ocr_failed_pages=[2],
                                sections=[Section(None, 1, "x" * 1500)])
    assert report.flag(ocr_failed) == "OCR FAILED on pages [2]"
