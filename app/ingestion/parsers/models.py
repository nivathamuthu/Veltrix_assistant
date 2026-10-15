"""Parser output types.

A parser turns one file into an ordered list of ``Section`` objects. Chunking,
metadata and indexing happen later (Phase 3b); parsers only extract text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Kind = Literal["text", "table"]


@dataclass(frozen=True)
class Section:
    heading: str | None  # nearest heading above this text, if any
    page: int | None  # 1-based page for PDFs, None for other formats
    text: str
    kind: Kind = "text"
    # Table sections only: the same rows as ``text`` but as (column header, value)
    # pairs. Not part of equality or repr; ``text`` stays what gets indexed.
    rows: tuple = field(default=(), repr=False, compare=False)


@dataclass
class ParsedDocument:
    path: Path
    parser: str  # "pdf", "docx", "markdown", "html" or "none"
    sections: list[Section] = field(default_factory=list)
    page_count: int | None = None  # PDFs only
    ocr_pages: list[int] = field(default_factory=list)  # pages read by OCR
    ocr_failed_pages: list[int] = field(default_factory=list)
    # Exception class name (or a short code) if the file was skipped. Never
    # contains message text, which could echo document content.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def table_count(self) -> int:
        return sum(1 for s in self.sections if s.kind == "table")

    @property
    def total_chars(self) -> int:
        return sum(len(s.text) for s in self.sections)

    @property
    def ocr_chars(self) -> int:
        return sum(len(s.text) for s in self.sections if s.page in self.ocr_pages)
