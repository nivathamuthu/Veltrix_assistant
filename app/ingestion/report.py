"""Parse every file in CORPUS_DIR and print what the parsers extracted.

    python -m app.ingestion.report                  # table for CORPUS_DIR
    python -m app.ingestion.report --show-ocr       # also print OCR'd page text
    python -m app.ingestion.report --dump FILE      # print every section of FILE

A developer tool for checking extraction quality; the service does not use it.
The table flags documents whose character count looks too low to be complete.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import load_settings
from app.ingestion.parsers import PARSERS, ParsedDocument, parse_document
from app.logging_setup import configure_logging

# A PDF page of real text holds well over this many characters; any document
# shorter than LOW_TOTAL_CHARS is suspect too.
LOW_CHARS_PER_PAGE = 300
LOW_TOTAL_CHARS = 400

HEADERS = ("file", "parser", "sections", "tables", "ocr pages", "chars", "note")


def flag(doc: ParsedDocument) -> str:
    if doc.error:
        return f"FAILED ({doc.error})"
    if doc.ocr_failed_pages:
        return f"OCR FAILED on pages {doc.ocr_failed_pages}"
    if not doc.sections:
        return "LOW: no text"
    if doc.total_chars < LOW_TOTAL_CHARS:
        return "LOW: few characters"
    if doc.page_count and doc.total_chars / doc.page_count < LOW_CHARS_PER_PAGE:
        return "LOW: few characters per page"
    return ""


def row_for(doc: ParsedDocument) -> tuple[str, ...]:
    ocr = ",".join(str(p) for p in doc.ocr_pages) or "-"
    if doc.ocr_pages:
        ocr += f" ({doc.ocr_chars} chars)"
    return (
        doc.path.name,
        doc.parser,
        str(len(doc.sections)),
        str(doc.table_count),
        ocr,
        str(doc.total_chars),
        flag(doc),
    )


def format_table(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(HEADERS[i]), *(len(r[i]) for r in rows)) for i in range(len(HEADERS))]
    numeric = {2, 3, 5}

    def fmt(cells: tuple[str, ...]) -> str:
        out = [c.rjust(widths[i]) if i in numeric else c.ljust(widths[i]) for i, c in enumerate(cells)]
        return "  ".join(out).rstrip()

    rule = "  ".join("-" * w for w in widths)
    return "\n".join([fmt(HEADERS), rule, *(fmt(r) for r in rows)])


def corpus_files(directory: Path) -> list[Path]:
    """Every regular file in the folder (unsupported types are reported, not hidden)."""
    return sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))


def print_sections(doc: ParsedDocument, only_pages: list[int] | None = None) -> None:
    for number, section in enumerate(doc.sections, start=1):
        if only_pages is not None and section.page not in only_pages:
            continue
        where = f"p.{section.page}" if section.page else "-"
        print(f"\n[{number}] kind={section.kind} page={where} heading={section.heading!r}")
        print(section.text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ingestion.report", description=__doc__)
    parser.add_argument("--dir", type=Path, help="folder to scan (default: CORPUS_DIR)")
    parser.add_argument("--show-ocr", action="store_true", help="print the text OCR produced")
    parser.add_argument("--dump", metavar="FILE", help="print every section of one file")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    settings = load_settings()
    configure_logging(settings.log_level, stream=sys.stderr)  # keep stdout for the table
    directory = args.dir or settings.corpus_dir
    if not directory.is_dir():
        print(f"folder not found: {directory}", file=sys.stderr)
        return 2

    docs = [parse_document(path) for path in corpus_files(directory)]
    print(format_table([row_for(d) for d in docs]))

    supported = sum(1 for d in docs if d.path.suffix.lower() in PARSERS)
    flagged = [d for d in docs if flag(d)]
    print(f"\n{len(docs)} files ({supported} supported), {len(docs) - len(flagged)} clean, "
          f"{len(flagged)} flagged, {sum(d.total_chars for d in docs)} characters in total")

    if args.show_ocr:
        for doc in docs:
            if doc.ocr_pages:
                print(f"\n===== OCR text: {doc.path.name} (pages {doc.ocr_pages}) =====")
                print_sections(doc, only_pages=doc.ocr_pages)
    if args.dump:
        matches = [d for d in docs if d.path.name == args.dump]
        if not matches:
            print(f"\nno such file in folder: {args.dump}", file=sys.stderr)
            return 2
        print(f"\n===== {args.dump} =====")
        print_sections(matches[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
