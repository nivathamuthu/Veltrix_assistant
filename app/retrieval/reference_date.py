"""The reference date for a question (README B4).

Pure: regular expressions and ``datetime`` only. No clock (``today`` is passed
in, normally APP_TODAY), no LLM, no database, no I/O.

The reference date is ``today`` unless the message asks about a specific PAST
date, month or year. Rules:

* Full dates use that date: ``2025-08-01``, ``1 August 2025``, ``1st Aug 2025``,
  ``August 1, 2025``, and numeric ``13/8/2025`` or ``8/13/2025`` (a numeric date
  is read only when it is unambiguous: if both numbers could be the month and
  they differ, e.g. ``1/8/2025``, it is NOT guessed).
* A month and year only ("in March 2025") means the last day of that month if
  that month is before today's month; the current or a future month means today.
* A year only ("in 2025") means the LAST DAY OF THAT YEAR if the year is before
  today's year, and today if it is the current year.
* "last year" means the previous calendar year, resolved like a past year (its
  last day). "this year" is the current year, so it means today.
* A day and month with no year ("on 1 August", "1st of March") means DEFAULT_YEAR
  (2026, README B9). It is then treated like any full date: in the past it is
  used, today or later it is not in the past, so the result is today. A day and
  month followed by a stray number ("1 August 25") is unclear, not guessed. The
  word "may" counts as the month only when nothing but punctuat
* If a no-year date sits next to a different explicit year ("in 2025, on 1
  August", "last year on 1 August") it is unclear: 2026 would contradict the
  year the user wrote.
* Future dates, months and years never change the reference date (today).
* Numbers that only look like years (``$2025``, ``2,025``, ``2025.50``,
  ``2025%``, ``FY2025``, ``2025/26``, ``12025``) are not years.
* A more specific date wins: full dates ion follows it
  ("on 5 May."), so "up to 5 may be carried over" is not a date.(including no-year ones), then months,
  then years. Several DIFFERENT past dates (or months, or years) at the same
  level are ambiguous, as is any impossible date (31 February, or 29 February
  in a non-leap default year) or ambiguous numeric date: the value stays
  ``today`` and ``unclear`` is set so a caller can ask the user to clarify.
* Only English month names, only four-digit years (1900-2099). Other relative
  phrases such as "yesterday" or "next year" do not change the date.

The date only chooses which document versions are eligible; it never grants
access, which depends solely on the requester's role and region.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime

# README B9: "A date with no year means 2026." A rule of the task, not a corpus fact;
# callers may pass another ``default_year``.
DEFAULT_YEAR = 2026

SOURCE_TODAY = "today"
SOURCE_DATE = "date"
SOURCE_MONTH = "month"
SOURCE_YEAR = "year"


@dataclass(frozen=True)
class ReferenceDate:
    """``value`` is the date to use; ``source`` says where it came from
    ("today", "date", "month", "year"); ``unclear`` is True when the message
    contained something date-like that could not be read safely (an ambiguous
    or impossible date, or several different past dates), in which case
    ``value`` is today."""

    value: date
    source: str
    unclear: bool = False


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sept": 9, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")(?![a-z])"

# Not glued to a longer word or number on either side, and not a decimal/range
# part. ASCII only, so other scripts' digits are never read as dates.
_BEFORE = r"(?<![\w.\-/])"
_AFTER = r"(?![\w\-/]|[.,]\d)"
_FLAGS = re.IGNORECASE | re.ASCII

_ISO = re.compile(_BEFORE + r"(\d{4})-(\d{2})-(\d{2})" + _AFTER, _FLAGS)
_NUMERIC = re.compile(_BEFORE + r"(\d{1,2})([/.\-])(\d{1,2})\2(\d{4})" + _AFTER, _FLAGS)
_DAY_MONTH_YEAR = re.compile(
    r"(?<!\w)(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + _MONTH + r")\.?,?\s+(\d{4})" + _AFTER, _FLAGS
)
_MONTH_DAY_YEAR = re.compile(
    r"(?<!\w)(" + _MONTH + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})" + _AFTER, _FLAGS
)
_MONTH_YEAR = re.compile(r"(?<!\w)(" + _MONTH + r")\.?,?\s+(\d{4})" + _AFTER, _FLAGS)
_YEAR = re.compile(r"(?<![\w$\u20ac\u00a3\u20b9.,/\-])((?:19|20)\d{2})(?![\w%\-/]|[.,]\d)", _FLAGS)

# Day and month with no year. The trailing lookahead captures whether a word
# character follows (after optional spaces), so a stray number can be flagged
# and "may" the verb can be told from May the month.
_DAY_MONTH = re.compile(
    r"(?<!\w)(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + _MONTH + r")\.?(?=(\s*\w)?)", _FLAGS
)
_MONTH_DAY = re.compile(
    r"(?<!\w)(" + _MONTH + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?!\w)(?=(\s*\w)?)", _FLAGS
)
_LAST_YEAR = re.compile(r"(?<!\w)last\s+year(?!\w)", _FLAGS)
_THIS_YEAR = re.compile(r"(?<!\w)this\s+year(?!\w)", _FLAGS)

_MIN_YEAR, _MAX_YEAR = 1900, 2099


def _make_date(year: int, month: int, day: int) -> date | None:
    if not _MIN_YEAR <= year <= _MAX_YEAR:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _numeric_date(first: int, second: int, year: int) -> date | None:
    """A numeric date, only when exactly one reading is possible."""
    if first > 12 and second <= 12:
        return _make_date(year, second, first)  # day first
    if second > 12 and first <= 12:
        return _make_date(year, first, second)  # month first
    if first == second and first <= 12:
        return _make_date(year, first, second)  # both readings agree
    return None  # ambiguous (or impossible)


class _Scan:
    """What a message names. Each kind of match is masked before the next kind is searched."""

    def __init__(self, message: str, today: date, default_year: int) -> None:
        self.dates: set[date] = set()  # complete dates the user wrote
        self.no_year_dates: set[date] = set()  # day + month, completed with default_year
        self.months: set[tuple[int, int]] = set()
        self.years: set[int] = set()
        self.unclear = False
        self._today = today
        self._default_year = default_year
        text = message
        text = self._sweep(text, _ISO, self._iso)
        text = self._sweep(text, _NUMERIC, self._numeric)
        text = self._sweep(text, _DAY_MONTH_YEAR, self._day_month_year)
        text = self._sweep(text, _MONTH_DAY_YEAR, self._month_day_year)
        text = self._sweep(text, _MONTH_YEAR, self._month_year)
        text = self._sweep(text, _YEAR, self._year)
        text = self._sweep(text, _DAY_MONTH, self._day_month)
        text = self._sweep(text, _MONTH_DAY, self._month_day)
        text = self._sweep(text, _LAST_YEAR, self._last_year)
        self._sweep(text, _THIS_YEAR, self._this_year)

    @staticmethod
    def _sweep(text: str, pattern: re.Pattern[str], handler) -> str:
        matches = list(pattern.finditer(text))
        for match in matches:
            handler(match)
        # blank every match in one pass (spans never overlap within one pattern)
        pieces, last = [], 0
        for match in matches:
            pieces.append(text[last : match.start()])
            pieces.append(" " * (match.end() - match.start()))
            last = match.end()
        pieces.append(text[last:])
        return "".join(pieces)

    def _add_date(self, value: date | None) -> None:
        if value is None:
            self.unclear = True
        else:
            self.dates.add(value)

    def _iso(self, m: re.Match[str]) -> None:
        self._add_date(_make_date(int(m[1]), int(m[2]), int(m[3])))

    def _numeric(self, m: re.Match[str]) -> None:
        self._add_date(_numeric_date(int(m[1]), int(m[3]), int(m[4])))

    def _day_month_year(self, m: re.Match[str]) -> None:
        self._add_date(_make_date(int(m[3]), _MONTHS[m[2].lower()], int(m[1])))

    def _month_day_year(self, m: re.Match[str]) -> None:
        self._add_date(_make_date(int(m[3]), _MONTHS[m[1].lower()], int(m[2])))

    def _month_year(self, m: re.Match[str]) -> None:
        year = int(m[2])
        if _MIN_YEAR <= year <= _MAX_YEAR:
            self.months.add((year, _MONTHS[m[1].lower()]))

    def _year(self, m: re.Match[str]) -> None:
        year = int(m[1])
        if _MIN_YEAR <= year <= _MAX_YEAR:
            self.years.add(year)

    def _no_year(self, month_word: str, day: int, following: str | None) -> None:
        """``following`` is the optional space + first word character after the match."""
        month = _MONTHS[month_word.lower()]
        if following and following[-1].isdigit():
            self.unclear = True  # "1 August 25": a year? a day? not guessed
            return
        if month == 5 and following:
            return  # "5 may be carried over": the verb, not the month
        value = _make_date(self._default_year, month, day)
        if value is None:
            self.unclear = True  # 31 February (or 29 February in a non-leap default year)
        else:
            self.no_year_dates.add(value)

    def _day_month(self, m: re.Match[str]) -> None:
        self._no_year(m[2], int(m[1]), m[3])

    def _month_day(self, m: re.Match[str]) -> None:
        self._no_year(m[1], int(m[2]), m[3])

    def _last_year(self, m: re.Match[str]) -> None:
        self.years.add(self._today.year - 1)

    def _this_year(self, m: re.Match[str]) -> None:
        self.years.add(self._today.year)


def resolve_reference_date(
    message: str | None, today: date, default_year: int = DEFAULT_YEAR
) -> ReferenceDate:
    """Like ``extract_reference_date`` but also says where the date came from and
    whether the message held something unreadable or ambiguous.

    ``default_year`` completes a day and month written without a year.
    """
    if isinstance(today, datetime):
        today = today.date()
    if not isinstance(today, date):
        raise TypeError("today must be a datetime.date")
    if message is None:
        message = ""
    if not isinstance(message, str):
        raise TypeError("message must be text")

    scan = _Scan(message, today, default_year)
    if scan.unclear:
        return ReferenceDate(today, SOURCE_TODAY, unclear=True)
    if scan.no_year_dates:
        written = {d.year for d in scan.dates} | {y for y, _ in scan.months} | scan.years
        if any(year != default_year for year in written):
            # "in 2025, on 1 August": completing it with another year would contradict the user
            return ReferenceDate(today, SOURCE_TODAY, unclear=True)

    levels = (
        (SOURCE_DATE, {d for d in scan.dates | scan.no_year_dates if d < today}),
        (
            SOURCE_MONTH,
            {
                date(y, m, calendar.monthrange(y, m)[1])
                for y, m in scan.months
                if (y, m) < (today.year, today.month)
            },
        ),
        (SOURCE_YEAR, {date(y, 12, 31) for y in scan.years if y < today.year}),
    )
    for source, past in levels:
        if len(past) > 1:
            return ReferenceDate(today, SOURCE_TODAY, unclear=True)
        if past:
            return ReferenceDate(next(iter(past)), source)
    return ReferenceDate(today, SOURCE_TODAY)


def extract_reference_date(
    message: str | None, today: date, default_year: int = DEFAULT_YEAR
) -> date:
    """The date to resolve documents as of: ``today`` unless ``message`` asks
    about a specific past date, month or year (rules in the module docstring)."""
    return resolve_reference_date(message, today, default_year).value
