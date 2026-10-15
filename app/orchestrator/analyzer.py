"""The analyzer interface: what the (Phase 7b) LLM understands about a message.

``Analyzer.analyze(message, session_context)`` returns a structured ``Understanding``: an
intent plus the details the message gave (leave type, a dates phrase, a target employee id,
a request id) and the fields it found missing. It only UNDERSTANDS. It never decides what
the requester may see or do: the orchestrator calls the tools with the authenticated
requester, and every permission check runs inside the tools.

Nothing the analyzer returns is trusted. ``Understanding`` normalises its fields (an unknown
intent becomes "other", non-text becomes None) and the tools re-validate every argument.

``ScriptedAnalyzer`` (tests) replays prepared answers and records every call, so a test can
prove the analyzer was or was not reached. ``NullAnalyzer`` understands nothing; it is the
default until the real LLM analyzer exists.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Final

INTENT_POLICY: Final = "policy_question"
INTENT_BALANCE: Final = "leave_balance"
INTENT_LIST: Final = "list_requests"
INTENT_TEAM: Final = "team_leave"
INTENT_APPLY: Final = "apply_leave"
INTENT_CANCEL: Final = "cancel_leave"
INTENT_HOLIDAY: Final = "holiday_question"
INTENT_OFF_TOPIC: Final = "off_topic"  # outside HR / leave: the service refuses politely
INTENT_OTHER: Final = "other"  # unclear: the service asks what is needed
INTENTS: Final = (
    INTENT_POLICY, INTENT_BALANCE, INTENT_LIST, INTENT_TEAM,
    INTENT_APPLY, INTENT_CANCEL, INTENT_HOLIDAY, INTENT_OFF_TOPIC, INTENT_OTHER,
)

# Names the analyzer may report as missing; the orchestrator also computes its own.
MISSING_LEAVE_TYPE: Final = "leave_type"
MISSING_DATES: Final = "dates"
MISSING_REQUEST: Final = "request_id"
MISSING_EMPLOYEE: Final = "employee"
MISSING_FIELDS: Final = (MISSING_LEAVE_TYPE, MISSING_DATES, MISSING_REQUEST, MISSING_EMPLOYEE)


def _text(value: object, *, lower: bool = False) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    return value.lower() if lower else value


@dataclass(frozen=True)
class Understanding:
    intent: str = INTENT_OTHER
    leave_type: str | None = None
    dates_text: str | None = None  # the user's own words; parsed by app.orchestrator.dates
    target_employee_id: str | None = None
    request_id: str | None = None
    status: str | None = None  # a leave request status to filter a listing by
    missing_fields: tuple[str, ...] = ()
    # a policy question that ALSO needs the requester's own leave data (e.g. a projected balance)
    needs_leave_data: bool = False

    def __post_init__(self) -> None:
        intent = _text(self.intent, lower=True)
        object.__setattr__(self, "intent", intent if intent in INTENTS else INTENT_OTHER)
        object.__setattr__(self, "leave_type", _text(self.leave_type, lower=True))
        object.__setattr__(self, "dates_text", _text(self.dates_text))
        object.__setattr__(self, "target_employee_id", _text(self.target_employee_id))
        object.__setattr__(self, "request_id", _text(self.request_id))
        object.__setattr__(self, "status", _text(self.status, lower=True))
        raw = self.missing_fields if isinstance(self.missing_fields, (tuple, list, set, frozenset)) else ()
        missing = tuple(m for m in MISSING_FIELDS if m in raw)  # known names, in a fixed order
        object.__setattr__(self, "missing_fields", missing)
        object.__setattr__(self, "needs_leave_data", self.needs_leave_data is True)


@dataclass(frozen=True)
class SessionContext:
    """What the analyzer may know about the session: only whether a proposal is waiting and
    its leave type, dates or request id. No names, no contact details, no corpus text.

    ``role`` and ``region`` are the requester's, from the database record (the analyzer may
    use them to classify; it never decides permissions). ``request_id`` only labels the log
    line of the analyzer's LLM call."""

    pending_action: str | None = None
    pending_details: dict[str, str] = field(default_factory=dict)
    role: str | None = None
    region: str | None = None
    request_id: str | None = None


class Analyzer(ABC):
    @abstractmethod
    def analyze(self, message: str, session_context: SessionContext) -> Understanding:
        """Understand ``message`` (never act on it)."""


class NullAnalyzer(Analyzer):
    """Understands nothing: every message is "other" (the orchestrator then asks what is needed)."""

    def analyze(self, message: str, session_context: SessionContext) -> Understanding:
        return Understanding(INTENT_OTHER)


Scripted = Understanding | BaseException | Callable[[str, SessionContext], Any]


class ScriptedAnalyzer(Analyzer):
    """Replays prepared results in order (for tests). An item is an ``Understanding``, an
    exception to raise, or a callable ``(message, context) -> result``. When the script runs
    out it returns "other". Every call is recorded in ``calls`` as ``(message, context)``."""

    def __init__(self, *items: Scripted) -> None:
        self._items: deque[Scripted] = deque(items)
        self._lock = threading.Lock()
        self.calls: list[tuple[str, SessionContext]] = []

    def queue(self, *items: Scripted) -> None:
        with self._lock:
            self._items.extend(items)

    def analyze(self, message: str, session_context: SessionContext) -> Understanding:
        with self._lock:
            self.calls.append((message, session_context))
            item = self._items.popleft() if self._items else Understanding(INTENT_OTHER)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Understanding):
            return item
        return item(message, session_context)
