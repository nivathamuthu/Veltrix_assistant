"""Tables -> row-wise text.

Retrieval works on text, so a table is turned into one line per data row with
the column names repeated in every row. A single row then still says what each
value means even when it is retrieved on its own:

    Row: Tenure 0-2 years | Accrual 1.5 days/month

Tables with no header row (for example a "Document ID | ..." key-value box)
become ``Row: Document ID | ...`` lines instead.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from app.ingestion.parsers.builder import clean

ROW_PREFIX = "Row:"
CELL_SEPARATOR = " | "


def rows_to_text(rows: Iterable[Sequence[str | None]], has_header: bool) -> str:
    """Render table rows as text, one ``Row: ...`` line per row.

    ``has_header``: treat the first row as column names (repeated in every
    following row). Needs at least one more row; otherwise the table is
    rendered without headers. Cell whitespace (including wrapped lines inside
    a cell) is collapsed. Empty cells are omitted and fully empty rows dropped.
    """
    cleaned = [[clean(cell) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""

    lines: list[str] = []
    if has_header and len(cleaned) >= 2:
        header, body = cleaned[0], cleaned[1:]
        for row in body:
            parts = []
            for index, value in enumerate(row):
                if not value:
                    continue
                name = header[index] if index < len(header) else ""
                parts.append(f"{name} {value}".strip())
            if parts:
                lines.append(f"{ROW_PREFIX} {CELL_SEPARATOR.join(parts)}")
    else:
        for row in cleaned:
            lines.append(f"{ROW_PREFIX} {CELL_SEPARATOR.join(v for v in row if v)}")
    return "\n".join(lines)


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
