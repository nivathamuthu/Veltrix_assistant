"""Tables -> row-wise text.

Retrieval works on text, so a table is turned into one line per data row with
the column names repeated in every row. A single row then still says what each
value means even when it is retrieved on its own:

    Row: Tenure 0-2 years | Accrual 1.5 days/month

Tables with no header row (for example a "Document ID | ..." key-value box)
become ``Row: Document ID | ...`` lines instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from app.ingestion.parsers.builder import clean

ROW_PREFIX = "Row:"
CELL_SEPARATOR = " | "

# One table row as ``(column header, value)`` per non-empty cell. The header is
# "" for a table without a header row. Consumers that need to know which value
# belongs to which column (Phase 3c holiday extraction) read this instead of
# trying to split the rendered text back apart.
TableRow = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TableData:
    text: str  # row-wise text, as indexed for retrieval
    rows: tuple[TableRow, ...]  # the same rows, structured


def render_table(rows: Iterable[Sequence[str | None]], has_header: bool) -> TableData:
    """Both forms of a table, built from one pass so they can never disagree.

    ``has_header``: treat the first row as column names (repeated in every
    following row). Needs at least one more row; otherwise the table is
    rendered without headers. Cell whitespace (including wrapped lines inside
    a cell) is collapsed. Empty cells are omitted and fully empty rows dropped.
    """
    cleaned = [[clean(cell) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return TableData("", ())

    structured: list[TableRow] = []
    if has_header and len(cleaned) >= 2:
        header, body = cleaned[0], cleaned[1:]
        for row in body:
            cells = tuple(
                (header[index] if index < len(header) else "", value)
                for index, value in enumerate(row)
                if value
            )
            if cells:
                structured.append(cells)
    else:
        structured = [tuple(("", v) for v in row if v) for row in cleaned]

    lines = [
        f"{ROW_PREFIX} {CELL_SEPARATOR.join(f'{name} {value}'.strip() for name, value in row)}"
        for row in structured
    ]
    return TableData("\n".join(lines), tuple(structured))


def rows_to_text(rows: Iterable[Sequence[str | None]], has_header: bool) -> str:
    """Render table rows as text, one ``Row: ...`` line per row."""
    return render_table(rows, has_header).text


def split_stacked_row(rows: list[list[str | None]]) -> list[list[str | None]]:
    """Undo a common PDF extraction artefact.

    For boxes drawn without row separators (typically key-value boxes) the PDF
    library returns ONE row whose cells each hold several lines::

        [["Document ID\\nVersion\\nOwner", "POL-X\\n2.0\\nPeople Team"]]

    When a table has exactly one row, two or more columns and every cell holds
    the same number (> 1) of lines, split it back into one row per line.
    """
    if len(rows) != 1 or len(rows[0]) < 2:
        return rows
    columns = [(cell or "").split("\n") for cell in rows[0]]
    height = len(columns[0])
    if height < 2 or any(len(col) != height for col in columns):
        return rows
    return [[col[i] for col in columns] for i in range(height)]
