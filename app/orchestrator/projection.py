"""Hybrid questions: a projected leave balance, calculated in CODE (Phase 7b item 4).

"What will my leave balance be by December?" needs the requester's own balance (a tool result)
and the accrual rule (a document). Where the rule value is a plain number with a rate unit
("<n> days per month" / "per year"), the arithmetic is done here with ``Decimal``, step by step,
and the model is only asked to explain the result. When the rule cannot be turned into numbers
(or the target date cannot be read, or the projection would cross a year end and need a
carry-forward rule), ``project_balance`` returns ``None`` and the model must show its own working
and say the result is an estimate.

Nothing here knows a rule value: the rate comes from the resolved document fact passed in.
The assumptions are stated by the code, so they are always printed with the result.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Final

from app.tools import LEAVE_TYPES

RATE_TOPIC_WORDS: Final = ("accru", "earn")
_RATE: Final = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:working\s+|business\s+)?(?:days?|d)?\s*(?:per|a|each|every|/)\s*"
    r"(month|year|annum)\b",
    re.IGNORECASE,
)
_MONTHS: Final = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sept": 9, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_YEAR: Final = re.compile(
    r"(?<![a-z])(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?(?:\s+(?:of\s+)?(20\d{2}))?(?![a-z])",
    re.IGNORECASE,
)
_YEAR_END: Final = re.compile(r"\b(?:end\s+of\s+(?:the\s+)?year|year[\s-]?end)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Rate:
    per_month: Decimal
    text: str  # how the rule reads, for the working


@dataclass(frozen=True)
class Projection:
    leave_type: str
    current: Decimal
    target: date
    credits: int  # month ends between today and the target (inclusive of the target)
    rate: Rate
    result: Decimal
    steps: tuple[str, ...]
    assumptions: tuple[str, ...]

    def text(self) -> str:
        """The calculation as plain lines (for the composition call and the fallback answer)."""
        return (
            "\n".join(self.steps)
            + "\nAssumptions:\n"
            + "\n".join(f"- {a}" for a in self.assumptions)
        )


def fmt(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text


def parse_rate(value: str) -> Rate | None:
    """A rate such as "1.5 days per month" or "18 days per year" as a per-month amount."""
    match = _RATE.search(value or "")
    if not match:
        return None
    amount = Decimal(match.group(1))
    unit = match.group(2).lower()
    per_month = amount if unit == "month" else amount / Decimal(12)
    return Rate(per_month, " ".join(match.group(0).split()))


def is_rate_topic(topic: str) -> bool:
    lowered = topic.casefold()
    return any(word in lowered for word in RATE_TOPIC_WORDS)


def leave_type_of(topic: str) -> str | None:
    """A leave type named in a topic (matched against the tools' own list), if exactly one."""
    named = [t for t in LEAVE_TYPES if t in topic.casefold()]
    return named[0] if len(named) == 1 else None


def parse_target(text: str | None, today: date) -> date | None:
    """The last day of the month (or year) the employee names, if it is not in the past."""
    if not text:
        return None
    if _YEAR_END.search(text):
        return date(today.year, 12, 31)
    match = _MONTH_YEAR.search(text)
    if not match:
        return None
    month = _MONTHS[match.group(1).lower()]
    year = int(match.group(2)) if match.group(2) else today.year
    target = date(year, month, calendar.monthrange(year, month)[1])
    return target if target >= today else None


def _month_ends_after(today: date, target: date) -> list[date]:
    ends: list[date] = []
    year, month = today.year, today.month
    while True:
        last = date(year, month, calendar.monthrange(year, month)[1])
        if last > target:
            break
        if last > today:
            ends.append(last)
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return ends


def project_balance(
    *,
    leave_type: str,
    current_days: float | int | Decimal,
    rate: Rate,
    target: date,
    today: date,
) -> Projection | None:
    """The balance at ``target``, or None when it cannot be computed safely (the target is not
    after today, or it lies in a later year, which would need a carry-forward rule)."""
    if target <= today or target.year != today.year:
        return None
    current = Decimal(str(current_days))
    ends = _month_ends_after(today, target)
    gained = rate.per_month * len(ends)
    result = current + gained
    steps = (
        f"1. Current {leave_type} balance (from your leave record): {fmt(current)} days.",
        f"2. Accrual rule from the documents: {rate.text}, i.e. {fmt(rate.per_month)} days per month.",
        f"3. Month ends after {today.isoformat()} up to {target.isoformat()}: {len(ends)}.",
        f"4. Accrual until then: {len(ends)} x {fmt(rate.per_month)} = {fmt(gained)} days.",
        f"5. Projected balance on {target.isoformat()}: {fmt(current)} + {fmt(gained)} = {fmt(result)} days.",
    )
    assumptions = (
        "Leave accrues at each month end at the rate above (a yearly rate is spread evenly over 12 months).",
        "No further leave is taken before then (approved and pending leave is already in your balance).",
        f"The target is in {today.year}, so no year-end carry-forward applies.",
    )
    return Projection(leave_type, current, target, len(ends), rate, result, steps, assumptions)
