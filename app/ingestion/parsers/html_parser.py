"""HTML parser (BeautifulSoup).

Only ``script``, ``style`` and ``nav`` are removed. Everything else in the
body is kept, including footers, notes and anything that reads like an
instruction to an AI: that is document content and is ingested as data.
``h1``..``h6`` are headings; tables become row-wise text.
"""
from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Comment, Doctype, NavigableString, ProcessingInstruction, Tag

from app.ingestion.parsers.builder import SectionBuilder, clean
from app.ingestion.parsers.models import ParsedDocument
from app.ingestion.parsers.tables import rows_to_text

REMOVED_TAGS = ("script", "style", "nav")
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCKS = {
    "p", "div", "section", "article", "main", "header", "footer", "aside", "ul", "ol",
    "li", "blockquote", "pre", "form", "fieldset", "figure", "figcaption", "address",
    "details", "summary", "dl", "dt", "dd", "caption", "br", "hr",
}
_SKIPPED_STRINGS = (Comment, Doctype, ProcessingInstruction)


def _span(cell: Tag, attribute: str) -> int:
    try:
        return max(1, min(int(cell.get(attribute, 1)), 50))
    except (TypeError, ValueError):
        return 1


def _table_rows(table: Tag) -> tuple[list[list[str]], bool]:
    """Rows of a table (not of tables nested inside it) and whether it has a header."""
    rows: list[list[str]] = []
    pending: dict[int, tuple[str, int]] = {}  # column -> (text, rows still to fill)
    has_header = False
    for index, tr in enumerate(table.find_all("tr")):
        if tr.find_parent("table") is not table:
            continue
        cells = tr.find_all(["th", "td"], recursive=False)
        if not rows and cells:
            has_header = all(c.name == "th" for c in cells) or tr.find_parent("thead") is not None
        row: list[str] = []
        column = 0
        queue = list(cells)
        while queue or column in pending:
            if column in pending:  # a cell from above spans down into this row
                text, remaining = pending[column]
                row.append(text)
                if remaining > 1:
                    pending[column] = (text, remaining - 1)
                else:
                    del pending[column]
                column += 1
                continue
            cell = queue.pop(0)
            text = clean(cell.get_text(" "))
            for _ in range(_span(cell, "colspan")):
                row.append(text)
                if _span(cell, "rowspan") > 1:
                    pending[column] = (text, _span(cell, "rowspan") - 1)
                column += 1
        rows.append(row)
    return rows, has_header


def parse_html(path: str | Path) -> ParsedDocument:
    path = Path(path)
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for tag in soup.find_all(list(REMOVED_TAGS)):
        tag.decompose()
    root = soup.body or soup
    builder = SectionBuilder()
    inline: list[str] = []

    def flush() -> None:
        # Text nodes already carry their own spaces ("a <b>b</b>."), so join
        # them as written, exactly as a browser would, then collapse whitespace.
        text = clean("".join(inline))
        inline.clear()
        if text:
            builder.paragraph(text)

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                if not isinstance(child, _SKIPPED_STRINGS):
                    inline.append(str(child))
                continue
            name = (child.name or "").lower()
            if name in _HEADINGS:
                flush()
                builder.heading(child.get_text(" "))
            elif name == "table":
                flush()
                rows, has_header = _table_rows(child)
                builder.table(rows_to_text(rows, has_header))
            elif name in _BLOCKS:
                flush()
                walk(child)
                flush()
            else:  # inline element: a, span, b, em, ...
                walk(child)

    walk(root)
    flush()
    return ParsedDocument(path=path, parser="html", sections=builder.finish())
