"""Unambiguous tool intents decided in code, so the analyzer LLM is not called.

Only matches that are already complete enough for the orchestrator to route to a tool
without guessing. Policy, holiday, hybrid, off-topic and anything mixed still go to the
model. Permission checks stay inside the tools.
"""
from __future__ import annotations

import re
from datetime import date

from app.orchestrator.analyzer import (
    INTENT_APPLY,
    INTENT_BALANCE,
    INTENT_CANCEL,
    INTENT_HOLIDAY,
    INTENT_LIST,
    INTENT_POLICY,
    INTENT_TEAM,
    MISSING_DATES,
    MISSING_LEAVE_TYPE,
    Understanding,
)
from app.orchestrator.dates import REASON_MISSING, parse_leave_dates

_ASK = r"(?:what(?:'s| is)|show(?:\s+me)?|check|tell me|give me)"
_OWN_BALANCE = re.compile(
    rf"(?is)^\s*(?:{_ASK}\s+)?"
    r"my\s+(?:(annual|sick)\s+)?leave\s+balance\s*[.?]?\s*$"
)
_OWN_BALANCE_OF = re.compile(
    rf"(?is)^\s*(?:{_ASK}\s+)?"
    r"(?:the\s+)?balance\s+(?:of|for)\s+(?:my\s+)?(?:(annual|sick)\s+)?leave\s*[.?]?\s*$"
)
_OWN_TYPED_BALANCE = re.compile(
    rf"(?is)^\s*(?:{_ASK}\s+)?"
    r"(?:my\s+)?(annual|sick)\s+(?:leave\s+)?balance\s*[.?]?\s*$"
)
_OTHER_BALANCE = re.compile(
    r"(?is)^\s*(?:what(?:'s| is)|show(?:\s+me)?|check|tell me)\s+"
    r"(?!my\b|the\b|our\b|a\b|an\b)([A-Za-z][A-Za-z0-9_-]{1,31})(?:'s)?\s+"
    r"(?:(annual|sick)\s+)?leave\s+(?:balance|details)\s*[.?]?\s*$"
)
_LIST_MINE = re.compile(
    r"(?is)^\s*show(?:\s+me)?\s+my\s+(pending\s+)?leave\s+requests\s*[.?]?\s*$"
)
# Ids used in the sample DB look like E2004 / E-NOBODY. Evaluator databases may differ;
# require a digit or a hyphen so ordinary words are not treated as an id.
_MENTIONED_ID = re.compile(
    r"\b([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+|[A-Za-z]+\d[A-Za-z0-9]*)\b"
)
_TEAM = re.compile(
    r"(?is)^\s*who\s+(?:in|on)\s+my\s+team\s+is\s+on\s+leave(?:\s+(.+?))?\s*[.?]?\s*$"
)
_APPLY = re.compile(
    r"(?is)^\s*(?:i\s+(?:want\s+to|would\s+like\s+to|'d\s+like\s+to)\s+)?"
    r"apply\s+(?:for\s+)?(?:(annual|anual|annnual|sick)\s+)?leave(?:\s+(.+))?\s*$"
)
_TYPE_ONLY = re.compile(r"(?is)^\s*(annual|anual|annnual|sick)(?:\s+leave)?[.!]?\s*$")
_TYPE_AND_DATES = re.compile(r"(?is)^\s*(annual|anual|annnual|sick)(?:\s+leave)?\s+(.+?)\s*$")
_NAMED_LEAVE_TYPE = re.compile(r"(?i)\b(annual|anual|annnual|sick)\b")
_DATE_FILLER = re.compile(r"(?is)^\s*(?:only|just|please|on)\s+")
_LEAVE_TYPE_ALIASES = {"annual": "annual", "anual": "annual", "annnual": "annual", "sick": "sick"}
_CANCEL = re.compile(
    r"(?is)^\s*cancel\s+(?:your\s+|my\s+|the\s+)?(?:leave\s+)?request\s+"
    r"([A-Za-z0-9][A-Za-z0-9-]{0,63})\s*[.!]?\s*$"
)

# "How do I apply" / "policy for cancelling" must not look like a write.
_HOW_TO = re.compile(r"(?i)\bhow\s+(?:do\s+i|to|can\s+i)\b")
_POLICY_NEW_JOINER = re.compile(
    r"(?is)^\s*(?:what (?:should|are|do)|where do)\s+"
    r".{0,40}\b(?:new joiners?|first week|onboarding|laptop)\b.{0,40}\s*[.?]?\s*$"
)
_POLICY_WFH = re.compile(
    r"(?is)^\s*what is the (?:work[-\s]?from[-\s]?home|hybrid work|remote work) policy\s*[.?]?\s*$"
)
_POLICY_LEAVE_DAYS = re.compile(
    r"(?is)^\s*(?:"
    r"how many days of annual leave do employees in my region get|"
    r"how many days of unused annual leave can i carry forward|"
    r"how many sick days do i get|"
    r"is the annual leave entitlement changing next year|"
    r"how do i request leave"
    r")\s*[.?]?\s*$"
)
_POLICY_HOLIDAYS = re.compile(
    r"(?is)^\s*what public holidays does my office observe\s*[.?]?\s*$"
)
_POLICY_HANDBOOK = re.compile(
    r"(?is)^\s*(?:"
    r"what does the (?:people )?manager handbook say(?: about .+?)?|"
    r"what is (?:in )?the (?:people )?manager handbook"
    r")\s*[.?]?\s*$"
)
_POLICY_TRAVEL = re.compile(
    r"(?is)^\s*(?:"
    r"what is the travel(?: and expense)? policy|"
    r"how do i book business travel|"
    r"how long do i have to file a travel expense claim|"
    r"what is the daily allowance for business travel"
    r")\s*[.?]?\s*$"
)


def match_unambiguous_tool(message: object) -> Understanding | None:
    """An ``Understanding`` when the message is a single, complete tool request; else None."""
    if not isinstance(message, str):
        return None
    text = message.strip()
    if not text or _HOW_TO.search(text):
        return None

    match = _OWN_BALANCE.fullmatch(text) or _OWN_BALANCE_OF.fullmatch(text) or _OWN_TYPED_BALANCE.fullmatch(text)
    if match:
        return Understanding(intent=INTENT_BALANCE, leave_type=_canonical_leave_type(match.group(1)))

    match = _OTHER_BALANCE.fullmatch(text)
    if match:
        return Understanding(
            intent=INTENT_BALANCE, target_employee_id=match.group(1), leave_type=match.group(2)
        )

    match = _LIST_MINE.fullmatch(text)
    if match:
        status = "pending" if match.group(1) else None
        return Understanding(intent=INTENT_LIST, status=status)

    match = _TEAM.fullmatch(text)
    if match:
        dates = (match.group(1) or "").strip() or None
        return Understanding(intent=INTENT_TEAM, dates_text=dates)

    match = _APPLY.fullmatch(text)
    if match:
        leave_type = _canonical_leave_type(match.group(1))
        dates_text = (match.group(2) or "").strip() or None
        missing = []
        if leave_type is None:
            missing.append(MISSING_LEAVE_TYPE)
        if dates_text is None:
            missing.append(MISSING_DATES)
        return Understanding(
            intent=INTENT_APPLY,
            leave_type=leave_type,
            dates_text=dates_text,
            missing_fields=tuple(missing),
        )

    match = _CANCEL.fullmatch(text)
    if match:
        return Understanding(intent=INTENT_CANCEL, request_id=match.group(1))

    return None


def match_unambiguous_policy(message: object) -> Understanding | None:
    """Policy or holiday chips that must not wait on the analyzer LLM."""
    if not isinstance(message, str):
        return None
    text = message.strip()
    if not text:
        return None
    if _POLICY_HOLIDAYS.fullmatch(text):
        return Understanding(intent=INTENT_HOLIDAY)
    if (
        _POLICY_NEW_JOINER.fullmatch(text)
        or _POLICY_WFH.fullmatch(text)
        or _POLICY_LEAVE_DAYS.fullmatch(text)
        or _POLICY_HANDBOOK.fullmatch(text)
        or _POLICY_TRAVEL.fullmatch(text)
    ):
        return Understanding(intent=INTENT_POLICY)
    return None


def _canonical_leave_type(word: str | None) -> str | None:
    if not word:
        return None
    return _LEAVE_TYPE_ALIASES.get(word.strip().casefold())


def extract_named_leave_type(message: object) -> str | None:
    """The single annual/sick word in ``message``, or None when it is missing or said twice."""
    if not isinstance(message, str):
        return None
    hits = [_canonical_leave_type(m.group(1)) for m in _NAMED_LEAVE_TYPE.finditer(message)]
    hits = [h for h in hits if h]
    return hits[0] if len(hits) == 1 else None


def match_apply_fragment(message: object, today: date) -> Understanding | None:
    """A leave type and/or dates that can finish an apply already in progress."""
    if not isinstance(message, str):
        return None
    text = message.strip()
    if not text or _HOW_TO.search(text):
        return None
    match = _TYPE_ONLY.fullmatch(text)
    if match:
        return Understanding(
            intent=INTENT_APPLY,
            leave_type=_canonical_leave_type(match.group(1)),
            missing_fields=(MISSING_DATES,),
        )
    match = _TYPE_AND_DATES.fullmatch(text)
    if match:
        dates_text = match.group(2).strip()
        if dates_text.casefold() in {"balance", "balances", "details"}:
            return None
        return Understanding(
            intent=INTENT_APPLY,
            leave_type=_canonical_leave_type(match.group(1)),
            dates_text=dates_text,
        )
    cleaned = _DATE_FILLER.sub("", text).strip() or text
    if parse_leave_dates(cleaned, today).reason != REASON_MISSING:
        return Understanding(intent=INTENT_APPLY, dates_text=cleaned, missing_fields=(MISSING_LEAVE_TYPE,))
    return None


def extract_mentioned_employee_id(message: object) -> str | None:
    """The single id-shaped token in ``message``, or None when there is not exactly one."""
    if not isinstance(message, str):
        return None
    hits = [m.group(1) for m in _MENTIONED_ID.finditer(message)]
    return hits[0] if len(hits) == 1 else None
