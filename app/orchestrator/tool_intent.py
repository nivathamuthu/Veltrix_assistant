"""Unambiguous tool intents decided in code, so the analyzer LLM is not called.

Only matches that are already complete enough for the orchestrator to route to a tool
without guessing. Policy, holiday, hybrid, off-topic and anything mixed still go to the
model. Permission checks stay inside the tools.
"""
from __future__ import annotations

import re
from app.orchestrator.analyzer import (
    INTENT_APPLY,
    INTENT_BALANCE,
    INTENT_CANCEL,
    INTENT_LIST,
    INTENT_TEAM,
    Understanding,
)

_OWN_BALANCE = re.compile(
    r"(?is)^\s*(?:what(?:'s| is)|show(?:\s+me)?|check|tell me)\s+"
    r"my\s+(?:(annual|sick)\s+)?leave\s+balance\s*[.?]?\s*$"
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
    r"(?is)^\s*who\s+in\s+my\s+team\s+is\s+on\s+leave(?:\s+(.+?))?\s*[.?]?\s*$"
)
_APPLY = re.compile(
    r"(?is)^\s*(?:i\s+(?:want\s+to|would\s+like\s+to|'d\s+like\s+to)\s+)?"
    r"apply\s+for\s+(annual|sick)\s+leave\s+(.+?)\s*$"
)
_CANCEL = re.compile(
    r"(?is)^\s*cancel\s+(?:your\s+|my\s+|the\s+)?(?:leave\s+)?request\s+"
    r"([A-Za-z0-9][A-Za-z0-9-]{0,63})\s*[.!]?\s*$"
)

# "How do I apply" / "policy for cancelling" must not look like a write.
_HOW_TO = re.compile(r"(?i)\bhow\s+(?:do\s+i|to|can\s+i)\b")


def match_unambiguous_tool(message: object) -> Understanding | None:
    """An ``Understanding`` when the message is a single, complete tool request; else None."""
    if not isinstance(message, str):
        return None
    text = message.strip()
    if not text or _HOW_TO.search(text):
        return None

    match = _OWN_BALANCE.fullmatch(text)
    if match:
        return Understanding(intent=INTENT_BALANCE, leave_type=match.group(1))

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
        return Understanding(intent=INTENT_APPLY, leave_type=match.group(1), dates_text=match.group(2).strip())

    match = _CANCEL.fullmatch(text)
    if match:
        return Understanding(intent=INTENT_CANCEL, request_id=match.group(1))

    return None


def extract_mentioned_employee_id(message: object) -> str | None:
    """The single id-shaped token in ``message``, or None when there is not exactly one."""
    if not isinstance(message, str):
        return None
    hits = [m.group(1) for m in _MENTIONED_ID.finditer(message)]
    return hits[0] if len(hits) == 1 else None
