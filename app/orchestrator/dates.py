"""Leave-date parsing (README B9): pure, deterministic, given ``today``. Never guesses.

``parse_leave_dates(text, today)`` returns a ``DateResult`` with ``start`` and ``end`` (both
set, a single day has ``start == end``) or a ``reason`` the orchestrator can show to the
user to ask again. Anything that has more than one reading is "ambiguous", not a guess.

Rules (README B9)
    * No year means 2026 (the leave year), not the current year; any year other than 2026
      is "outside 2026". Weeks start on Monday; "next week" is Monday to Friday of the
      following week; "next Friday" (and a bare "Friday") is the first Friday AFTER today;
      naming today's own weekday without "next" is ambiguous.
    * Ranges: "from X to Y", "X - Y", "X till Y" (also until / through), "19-21 October",
      "next Mon to Wed". The end weekday of a range is the first such day on or after the
      start.
    * Slash and dot dates ("19/10/2026") are ambiguous (day-first or month-first), as are
      "next month", a bare "the 19th", a month alone, a duration ("for 3 days"), and two
      dates with no connector.
    * Half days are not supported.
    * Reasons are checked in this order: half day, ambiguous, invalid date, no date, then
      for a parsed range: outside 2026, end before start, in the past.

No clock is read here: ``today`` is always a parameter (APP_TODAY from config). Only text
is looked at; the result never says anything about balances, holidays or policy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final

LEAVE_YEAR: Final = 2026

REASON_AMBIGUOUS: Final = "ambiguous"
REASON_PAST: Final = "in the past"
REASON_OUTSIDE_YEAR: Final = f"outside {LEAVE_YEAR}"
REASON_MISSING: Final = "no date"
REASON_END_BEFORE_START: Final = "end before start"
REASON_INVALID: Final = "invalid date"
REASON_HALF_DAY: Final = "half day"


@dataclass(frozen=True)
class DateResult:
    """Dates (``start`` <= ``end``), or the ``reason`` there are none.

    On "in the past" the dates are still reported, so a read-only view (a manager's team
    calendar) may use them; for every other reason both are None."""

    start: date | None = None
    end: date | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None and self.start is not None and self.end is not None


def _fail(reason: str) -> DateResult:
    return DateResult(None, None, reason)


# --- the grammar -------------------------------------------------------------------------------------

_MONTHS: Final = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sept": 9, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}
_WEEKDAYS: Final = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tues": 1, "tue": 1, "weds": 2, "wed": 2, "thurs": 3, "thur": 3, "thu": 3,
    "fri": 4, "sat": 5, "sun": 6,
}
_FULL_WEEKDAYS: Final = frozenset(n for n in _WEEKDAYS if len(n) > 4 or n in ("tues", "weds", "thurs"))


def _alt(words: dict[str, int]) -> str:
    return "|".join(sorted(words, key=len, reverse=True))


_MON = rf"(?:{_alt(_MONTHS)})"
_WD = rf"(?:{_alt(_WEEKDAYS)})"
_ORD = r"(?:st|nd|rd|th)?"
_LINK = r"(?:-|to|till|until|through|thru)"

_EXPR = re.compile(
    rf"""
    (?P<iso>\b(?P<iso_y>\d{{4}})-(?P<iso_m>\d{{2}})-(?P<iso_d>\d{{2}})\b)
  | (?P<dr>(?<![\d/.:])(?P<dr_d1>\d{{1,2}}){_ORD}\s*{_LINK}\s*(?P<dr_d2>\d{{1,2}}){_ORD}\s+(?:of\s+)?
        (?<![a-z])(?P<dr_m>{_MON})\b\.?(?:,?\s+(?P<dr_y>\d{{4}}))?(?!\d))
  | (?P<mr>(?<![a-z])(?P<mr_m>{_MON})\b\.?\s+(?P<mr_d1>\d{{1,2}}){_ORD}\s*{_LINK}\s*(?P<mr_d2>\d{{1,2}}){_ORD}
        (?!\d)(?!\s*(?:of\s+)?{_MON}\b)(?!\s*,?\s*\d{{4}}))
  | (?P<dm>(?<![\d/.:])(?P<dm_d>\d{{1,2}}){_ORD}\s+(?:of\s+)?(?<![a-z])(?P<dm_m>{_MON})\b\.?
        (?:,?\s+(?P<dm_y>\d{{4}})(?!\d))?)
  | (?P<md>(?<![a-z])(?P<md_m>{_MON})\b\.?\s+(?P<md_d>\d{{1,2}}){_ORD}(?!\d)(?![/.:])
        (?:,?\s+(?P<md_y>\d{{4}})(?!\d))?)
  | (?P<wn>(?:\bon\s+)?\b(?P<wn_a>{_WD})\b\s+(?:of\s+)?next\s+week\b
        |\bnext\s+week\b\s+(?:on\s+)?\b(?P<wn_b>{_WD})\b)
  | (?P<nw>\bnext\s+week\b)
  | (?P<rel>\b(?:day\s+after\s+tomorrow|tomorrow|today|yesterday)\b)
  | (?P<wd>\b(?:(?P<wd_mod>next|this|coming|last)\s+)?(?P<wd_name>{_WD})\b)
    """,
    re.VERBOSE,
)

_ABBREV_CONTEXT = re.compile(r"(?:^\s*|\b(?:on|from|to|till|until|through|thru)\s+|-\s*)$")

_HALF_DAY = re.compile(r"\bhalf[\s-]*(?:a[\s-]+)?days?\b")
_VAGUE_STRONG = re.compile(
    r"\b(?:next\s+month|this\s+month|this\s+week|(?:this|next|the|a)\s+weekend|next\s+year|this\s+year"
    r"|the\s+week\s+after|sometime|a\s+few\s+days|a\s+couple\s+of\s+days|couple\s+of\s+weeks|few\s+weeks)\b"
)
_VAGUE_WEAK = re.compile(r"\b(?:soon|later|whenever|asap|eventually|at\s+some\s+point)\b")
_MONTH_ONLY = re.compile(
    rf"\b(?:in|during|for|throughout)\s+(?:the\s+month\s+of\s+)?(?<![a-z]){_MON}\b(?!\s*\d)"
)
_DURATION = re.compile(
    r"\b(?:(?:[2-9]|\d{2,}|two|three|four|five|six|seven|eight|nine|ten)\s+(?:working\s+|business\s+)?days?"
    r"|(?:\d+|a|an|one|two|three|four|five|six)\s+weeks?)\b"
)
_ORDINAL_ALONE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\b")
_NUMERIC = re.compile(r"(?<!\d)\d{1,4}[/.]\d{1,2}(?:[/.]\d{1,4})?(?!\d)|\b\d{1,2}-\d{1,2}-\d{2,4}\b")
_STRAY_LINK = re.compile(r"(?:^|\s)(?:to|till|until|through|thru|-)\s*\d")
_DANGLING_LINK = re.compile(r"\b(?:to|till|until|through|thru)\s*[.!?]*\s*$|-\s*[.!?]*\s*$")
_FILLER = frozenset({"the", "on", "of"})
_LINKS = frozenset({"to", "till", "until", "through", "thru", "-"})

# kinds of expression
_DATE, _RANGE, _WEEKDAY = "date", "range", "weekday"


@dataclass(frozen=True)
class _Expr:
    kind: str
    start: int  # span in the normalised text
    stop: int
    first: date | None = None
    last: date | None = None
    weekday: int | None = None
    modifier: str | None = None
    invalid: bool = False


def _normalise(text: str) -> str:
    text = text.lower()
    for dash in ("\u2013", "\u2014", "\u2212"):
        text = text.replace(dash, "-")
    return re.sub(r"\s+", " ", text).strip()


def _make_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _as_expr(match: re.Match[str], text: str, today: date) -> _Expr | None:
    """One grammar match as an expression, or None when it is not one (e.g. 'sat' in a sentence)."""
    g = match.groupdict()
    span = match.span()

    def single(day: date | None) -> _Expr:
        return _Expr(_DATE, *span, first=day, last=day, invalid=day is None)

    if g["iso"]:
        return single(_make_date(int(g["iso_y"]), int(g["iso_m"]), int(g["iso_d"])))
    if g["dr"]:
        year = int(g["dr_y"]) if g["dr_y"] else LEAVE_YEAR
        month = _MONTHS[g["dr_m"]]
        first = _make_date(year, month, int(g["dr_d1"]))
        last = _make_date(year, month, int(g["dr_d2"]))
        return _Expr(_RANGE, *span, first=first, last=last, invalid=first is None or last is None)
    if g["mr"]:
        month = _MONTHS[g["mr_m"]]
        first = _make_date(LEAVE_YEAR, month, int(g["mr_d1"]))
        last = _make_date(LEAVE_YEAR, month, int(g["mr_d2"]))
        return _Expr(_RANGE, *span, first=first, last=last, invalid=first is None or last is None)
    if g["dm"]:
        year = int(g["dm_y"]) if g["dm_y"] else LEAVE_YEAR
        return single(_make_date(year, _MONTHS[g["dm_m"]], int(g["dm_d"])))
    if g["md"]:
        year = int(g["md_y"]) if g["md_y"] else LEAVE_YEAR
        return single(_make_date(year, _MONTHS[g["md_m"]], int(g["md_d"])))
    if g["wn"]:
        name = g["wn_a"] or g["wn_b"]
        day = _monday(today) + timedelta(days=7 + _WEEKDAYS[name])
        return _Expr(_DATE, *span, first=day, last=day)
    if g["nw"]:
        monday = _monday(today) + timedelta(days=7)
        return _Expr(_RANGE, *span, first=monday, last=monday + timedelta(days=4))
    if g["rel"]:
        word = g["rel"]
        offset = {"today": 0, "tomorrow": 1, "yesterday": -1}.get(word, 2)
        return single(today + timedelta(days=offset))
    # a weekday name
    name, modifier = g["wd_name"], g["wd_mod"]
    if name not in _FULL_WEEKDAYS and modifier is None:
        # 'mon', 'sat', 'sun' ... only count where a date is plausible, so "I sat down" is just text
        if not _ABBREV_CONTEXT.search(text[: match.start()]):
            return None
    return _Expr(_WEEKDAY, *span, weekday=_WEEKDAYS[name], modifier=modifier)


# --- resolving weekdays ----------------------------------------------------------------------------------


def _resolve_start_weekday(expr: _Expr, today: date) -> date | None:
    """The date a weekday name means on its own, or None when it is ambiguous."""
    target = expr.weekday
    assert target is not None
    if expr.modifier == "last":
        back = (today.weekday() - target) % 7
        return today - timedelta(days=back or 7)
    ahead = (target - today.weekday()) % 7
    if ahead == 0:
        return today + timedelta(days=7) if expr.modifier == "next" else None
    return today + timedelta(days=ahead)


def _resolve_end_weekday(expr: _Expr, start: date) -> date | None:
    """The first such weekday on or after ``start``."""
    if expr.modifier == "last":
        return None
    assert expr.weekday is not None
    return start + timedelta(days=(expr.weekday - start.weekday()) % 7)


# --- the parser --------------------------------------------------------------------------------------------


def parse_leave_dates(text: str, today: date) -> DateResult:
    """The leave dates written in ``text``, resolved against ``today`` (see the module docstring)."""
    if isinstance(today, datetime) or not isinstance(today, date):
        raise TypeError("today must be a datetime.date (APP_TODAY), passed in by the caller")
    if not isinstance(text, str) or not text.strip():
        return _fail(REASON_MISSING)

    t = _normalise(text)
    if _HALF_DAY.search(t):
        return _fail(REASON_HALF_DAY)

    exprs = [e for m in _EXPR.finditer(t) if (e := _as_expr(m, t, today)) is not None]

    # Blank out what was understood. The filler is NUL, not a space, so that "to <date>" does not
    # read as a link word dangling at the end of the text.
    masked = list(t)
    for e in exprs:
        masked[e.start : e.stop] = "\x00" * (e.stop - e.start)
    rest = "".join(masked)

    if _unclear(rest, exprs):
        return _fail(REASON_AMBIGUOUS)
    if any(e.invalid for e in exprs):
        return _fail(REASON_INVALID)
    if not exprs:
        return _fail(REASON_MISSING)

    resolved = _combine(t, exprs, today)
    if isinstance(resolved, DateResult):
        return resolved
    start, end = resolved

    if start.year != LEAVE_YEAR or end.year != LEAVE_YEAR:
        return _fail(REASON_OUTSIDE_YEAR)
    if end < start:
        return _fail(REASON_END_BEFORE_START)
    if start < today:
        return DateResult(start, end, REASON_PAST)
    return DateResult(start, end, None)


def _unclear(rest: str, exprs: list[_Expr]) -> bool:
    """True when the text has something with more than one reading, next to or instead of dates."""
    if _NUMERIC.search(rest) or _ORDINAL_ALONE.search(rest):
        return True
    if _VAGUE_STRONG.search(rest) or _MONTH_ONLY.search(rest):
        return True
    if not exprs:
        return _VAGUE_WEAK.search(rest) is not None
    if len(exprs) <= 1 and _DURATION.search(rest):
        return True
    stripped = rest.strip()
    return bool(_STRAY_LINK.search(rest) or _DANGLING_LINK.search(stripped))


def _combine(t: str, exprs: list[_Expr], today: date) -> tuple[date, date] | DateResult:
    """One expression is one day (or one range); two joined by a link word are a range."""
    if len(exprs) == 1:
        e = exprs[0]
        if e.kind == _RANGE:
            return _checked(e.first, e.last)
        if e.kind == _DATE:
            return _checked(e.first, e.first)
        day = _resolve_start_weekday(e, today)
        return _fail(REASON_AMBIGUOUS) if day is None else (day, day)

    if len(exprs) != 2:
        return _fail(REASON_AMBIGUOUS)
    a, b = exprs
    if a.kind == _RANGE or b.kind == _RANGE or not _joined(t[a.stop : b.start]):
        return _fail(REASON_AMBIGUOUS)
    start = _resolve_start_weekday(a, today) if a.kind == _WEEKDAY else a.first
    if start is None:
        return _fail(REASON_AMBIGUOUS)
    end = _resolve_end_weekday(b, start) if b.kind == _WEEKDAY else b.first
    if end is None:
        return _fail(REASON_AMBIGUOUS)
    return start, end


def _checked(first: date | None, last: date | None) -> tuple[date, date] | DateResult:
    if first is None or last is None:
        return _fail(REASON_INVALID)
    return first, last


def _joined(between: str) -> bool:
    """Is the text between two dates just a link word (to, till, -, ...) and filler?"""
    words = [w for w in re.split(r"[\s,.;]+", between.strip()) if w and w not in _FILLER]
    return len(words) == 1 and words[0] in _LINKS
