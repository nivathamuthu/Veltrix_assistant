"""Tolerant but never-guessing date parsing for table cells.

Accepted (case-insensitive), each optionally with a weekday before, after, or in
parentheses, and with ordinal suffixes / "the" / "of" / commas ignored:

* ISO                      2026-01-15   2026/01/15   2026.01.15
* day month year           15 January 2026   15th of Jan, 2026   15-Jan-2026
* month day year           Jan 15, 2026   January 15th 2026
* numeric day/month/year   15/01/2026   15.01.2026   (only if unambiguous)

Never guessed (the cell is rejected and the caller skips the row):

* a date with no year, a two-digit year, a range ("24-26 December 2026");
* ``03/04/2026``: day-first or month-first cannot be told apart, so it is
  rejected unless the day is above 12 or both numbers are equal;
* impossible dates (30 February), unknown month names, any other trailing text.

The weekday, if the cell names one, is returned so callers can cross-check it;
it is never used to "repair" the date.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2,
    "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4,
    "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
_WEEKDAY_ALTERNATION = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))

_PAREN_WEEKDAY = re.compile(rf"\(\s*({_WEEKDAY_ALTERNATION})\s*\.?\s*\)")
_LEADING_WEEKDAY = re.compile(rf"^({_WEEKDAY_ALTERNATION})\b\.?[\s,:\-]*")
_TRAILING_WEEKDAY = re.compile(rf"[\s,\-]*\b({_WEEKDAY_ALTERNATION})\.?$")
_ORDINAL = re.compile(r"(?<=\d)(?:st|nd|rd|th)\b")
_FILLER_WORDS = re.compile(r"\b(?:the|of)\b")
_PERIOD_AFTER_LETTER = re.compile(r"(?<=[a-z])\.")
_WHITESPACE = re.compile(r"\s+")

_ISO = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
_DAY_MONTH_YEAR = re.compile(r"(\d{1,2})[\s\-/.]*([a-z]+)[\s\-/.]*(\d{4})")
_MONTH_DAY_YEAR = re.compile(r"([a-z]+)[\s\-/.]*(\d{1,2})[\s\-/.]*(\d{4})")
_NUMERIC = re.compile(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})")

_MIN_YEAR, _MAX_YEAR = 1900, 2200


@dataclass(frozen=True)
class ParsedDate:
    value: date
    weekday: int | None  # Monday = 0, if the cell also named a weekday


def _build(year: int, month: int, day: int) -> date | None:
    if not _MIN_YEAR <= year <= _MAX_YEAR:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _normalise(text: str) -> tuple[str, int | None]:
    """Lower-case, strip weekday/ordinals/filler; return (clean text, weekday)."""
    cleaned = unicodedata.normalize("NFKC", text).lower()
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    weekdays: list[int] = []

    def take(match: re.Match) -> str:
        weekdays.append(_WEEKDAYS[match.group(1)])
        return " "

    cleaned = _PAREN_WEEKDAY.sub(take, cleaned).strip()
    cleaned = _LEADING_WEEKDAY.sub(take, cleaned, count=1).strip()
    cleaned = _TRAILING_WEEKDAY.sub(take, cleaned, count=1).strip()
    cleaned = _ORDINAL.sub("", cleaned)
    cleaned = _FILLER_WORDS.sub(" ", cleaned)
    cleaned = _PERIOD_AFTER_LETTER.sub("", cleaned)
    cleaned = cleaned.replace(",", " ")
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if len(set(weekdays)) > 1:  # "Monday ... Tuesday": contradictory, reject
        return "", None
    return cleaned, (weekdays[0] if weekdays else None)


def _numeric(first: int, second: int, year: int) -> date | None:
    """dd/mm/yyyy or mm/dd/yyyy, only when the order cannot matter."""
    if first == second:
        return _build(year, second, first)
    if first > 12 and second <= 12:
        return _build(year, second, first)  # first must be the day
    if second > 12 and first <= 12:
        return _build(year, first, second)  # second must be the day
    return None  # both <= 12 and different: day-first or month-first, unknowable


def parse_date_cell(text: str | None) -> ParsedDate | None:
    """The date in a table cell, or None if it is not exactly one clear date."""
    if not text or not text.strip():
        return None
    cleaned, weekday = _normalise(text)
    if not cleaned:
        return None

    if match := _ISO.fullmatch(cleaned):
        year, month, day = (int(g) for g in match.groups())
        result = _build(year, month, day)
    elif match := _NUMERIC.fullmatch(cleaned):
        first, second, year = (int(g) for g in match.groups())
        result = _numeric(first, second, year)
    elif match := _DAY_MONTH_YEAR.fullmatch(cleaned):
        day, month_name, year = match.groups()
        month = _MONTHS.get(month_name)
        result = _build(int(year), month, int(day)) if month else None
    elif match := _MONTH_DAY_YEAR.fullmatch(cleaned):
        month_name, day, year = match.groups()
        month = _MONTHS.get(month_name)
        result = _build(int(year), month, int(day)) if month else None
    else:
        result = None
    return ParsedDate(result, weekday) if result else None


def parse_date(text: str | None) -> date | None:
    parsed = parse_date_cell(text)
    return parsed.value if parsed else None


def parse_weekday(text: str | None) -> int | None:
    """Monday = 0 if the cell is exactly one weekday name ("Thursday", "Thu."), else None."""
    if not text:
        return None
    cleaned = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).lower()).strip()
    return _WEEKDAYS.get(cleaned.rstrip("."))
