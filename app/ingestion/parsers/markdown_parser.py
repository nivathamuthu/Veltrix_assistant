"""Markdown parser: ``#`` headings, pipe tables, everything else as text.

Written by hand rather than with a Markdown library because we need the
source text (not rendered HTML) split on headings, and tables as row text.
Text that reads like instructions is kept: it is document content.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.parsers.builder import SectionBuilder, clean
from app.ingestion.parsers.models import ParsedDocument
from app.ingestion.parsers.tables import rows_to_text

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(```|~~~)")
_HORIZONTAL_RULE = re.compile(r"^ {0,3}([-*_])([ \t]*\1){2,}[ \t]*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def _split_row(line: str) -> list[str]:
    """Cells of a pipe-table row (``\\|`` is a literal pipe)."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    return [cell.replace("\\|", "|").strip() for cell in re.split(r"(?<!\\)\|", line)]


def _is_table_start(lines: list[str], i: int) -> bool:
    return (
        "|" in lines[i]
        and i + 1 < len(lines)
        and bool(_TABLE_SEPARATOR.match(lines[i + 1]))
    )


def parse_markdown(path: str | Path) -> ParsedDocument:
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    builder = SectionBuilder()
    paragraph: list[str] = []
    in_fence = False

    def close() -> None:
        if paragraph:
            builder.paragraph(clean(" ".join(paragraph)))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        if _FENCE.match(line):  # code fence: contents are text, never headings
            close()
            in_fence = not in_fence
            i += 1
            continue
        if in_fence:
            builder.paragraph(line.strip())
            i += 1
            continue

        if not line.strip():
            close()
            i += 1
            continue

        heading = _ATX_HEADING.match(line)
        if heading:
            close()
            builder.heading(heading.group(2))
            i += 1
            continue

        # Setext heading: a text line underlined with === or ---
        if (
            i + 1 < len(lines)
            and _SETEXT_UNDERLINE.match(lines[i + 1])
            and not paragraph
            and not _LIST_ITEM.match(line)
            and "|" not in line
        ):
            builder.heading(line)
            i += 2
            continue

        if _HORIZONTAL_RULE.match(line):  # a divider, not content
            close()
            i += 1
            continue

        if _is_table_start(lines, i):
            close()
            rows = [_split_row(line)]
            i += 2  # header + separator
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                rows.append(_split_row(lines[i]))
                i += 1
            builder.table(rows_to_text(rows, has_header=True))
            continue

        if _LIST_ITEM.match(line):
            close()
        paragraph.append(line.strip())
        i += 1

    close()
    return ParsedDocument(path=path, parser="markdown", sections=builder.finish())
