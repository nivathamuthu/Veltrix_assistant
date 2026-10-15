"""Holiday questions: which calendar applies, what it lists, and how it is cited.

The calendar and its public dates come from ``public_holidays_for`` (requester's region,
reference date, access level; deterministic precedence, never a guess). The names and the
optional rows come from the same calendar document's rows. Public and optional holidays are
kept in separate parts so an optional (restricted) holiday is never described as public.
The text handed to the model is corpus-derived, so it travels as budgeted note parts that cite
the calendar document and a section (locator) of it.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Final, Iterable, Sequence

from app.db.models import KIND_OPTIONAL, KIND_PUBLIC, HolidayRow
from app.llm.budget import KIND_NOTE, CorpusPart
from app.retrieval.holiday_lookup import ApplicableCalendar

_WORD: Final = re.compile(r"[0-9a-z]+")


def split_rows(
    rows: Iterable[HolidayRow], calendar: ApplicableCalendar
) -> tuple[list[HolidayRow], list[HolidayRow]]:
    """``(public, optional)``, by date. The public ones are exactly the calendar's public dates."""
    public: dict[str, HolidayRow] = {}
    optional: dict[str, HolidayRow] = {}
    for row in rows:
        try:
            day = date.fromisoformat(row.date)
        except ValueError:
            continue
        if row.kind == KIND_PUBLIC and day in calendar.dates:
            public.setdefault(row.date, row)
        elif row.kind == KIND_OPTIONAL and day not in calendar.dates:
            optional.setdefault(row.date, row)
    return (
        [public[d] for d in sorted(public)],
        [optional[d] for d in sorted(optional)],
    )


def pick_locator(sections: Sequence[tuple[str, str]], rows: Iterable[HolidayRow]) -> str | None:
    """The section of the calendar document that lists these holidays: the one whose text
    contains the most holiday names (the first section if none matches); None if the document
    has no sections to cite."""
    if not sections:
        return None
    names = [" ".join(_WORD.findall(r.name.casefold())) for r in rows if r.name]
    best, best_hits = sections[0][0], 0
    for locator, text in sections:
        haystack = " ".join(_WORD.findall(text.casefold()))
        hits = sum(1 for n in names if n and n in haystack)
        if hits > best_hits:
            best, best_hits = locator, hits
    return best


def holiday_parts(
    public: Sequence[HolidayRow], optional: Sequence[HolidayRow], calendar: ApplicableCalendar, locator: str
) -> list[CorpusPart]:
    parts: list[CorpusPart] = []
    if public:
        lines = "\n".join(f"{r.date} {r.name}" for r in public)
        parts.append(
            CorpusPart(f"PUBLIC holidays in this calendar:\n{lines}", calendar.doc_id, locator,
                       calendar.effective_date, KIND_NOTE)
        )
    if optional:
        lines = "\n".join(f"{r.date} {r.name}" for r in optional)
        parts.append(
            CorpusPart(f"OPTIONAL holidays in this calendar (not public holidays):\n{lines}",
                       calendar.doc_id, locator, calendar.effective_date, KIND_NOTE)
        )
    return parts


def optional_called_public(answer: str, optional: Sequence[HolidayRow]) -> bool:
    """A sentence that names an optional holiday and says "public" but not "optional"."""
    for sentence in re.split(r"(?<=[.!?\n])\s+", answer):
        lowered = sentence.casefold()
        if "public" in lowered and "optional" not in lowered and any(r.name.casefold() in lowered for r in optional):
            return True
    return False


def render_holidays(
    public: Sequence[HolidayRow], optional: Sequence[HolidayRow], calendar: ApplicableCalendar, locator: str
) -> str:
    """Fixed wording, used when the composed answer is not usable."""
    lines = [f"Holidays for your region (calendar {calendar.doc_id}, {locator}):"]
    if public:
        lines.append("Public holidays:")
        lines.extend(f"- {r.date}: {r.name}" for r in public)
    if optional:
        lines.append("Optional holidays (these are optional, not public holidays):")
        lines.extend(f"- {r.date}: {r.name}" for r in optional)
    return "\n".join(lines)
