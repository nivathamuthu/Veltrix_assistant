"""Shared helpers that turn a stream of headings / paragraphs / tables into
``Section`` objects, so every format behaves the same way."""
from __future__ import annotations

import re

from app.ingestion.parsers.models import Section

_WHITESPACE = re.compile(r"\s+")


def clean(text: str | None) -> str:
    """Collapse all runs of whitespace (including newlines) to single spaces."""
    return _WHITESPACE.sub(" ", text or "").strip()


class SectionBuilder:
    """Collects content under the current heading.

    * ``heading()`` starts a new section; a heading with no content under it
      produces no section.
    * ``paragraph()`` adds a paragraph; a change of page closes the section so
      every section sits on one page.
    * ``table()`` emits a table section immediately, under the current heading.
    """

    def __init__(self) -> None:
        self.sections: list[Section] = []
        self._heading: str | None = None
        self._page: int | None = None
        self._paragraphs: list[str] = []

    def heading(self, text: str, page: int | None = None) -> None:
        self.flush()
        self._heading = clean(text) or None
        self._page = page

    def paragraph(self, text: str, page: int | None = None) -> None:
        text = text.strip()
        if not text:
            return
        if self._paragraphs and page != self._page:
            self.flush()
        self._page = page
        self._paragraphs.append(text)

    def table(self, text: str, page: int | None = None, rows: tuple = ()) -> None:
        text = text.strip()
        if not text:
            return
        self.flush()
        self.sections.append(Section(self._heading, page, text, "table", rows))
        self._page = page

    def flush(self) -> None:
        if self._paragraphs:
            self.sections.append(
                Section(self._heading, self._page, "\n".join(self._paragraphs), "text")
            )
            self._paragraphs = []

    def finish(self) -> list[Section]:
        self.flush()
        return self.sections
