"""A Provider that never touches the network. Records what it was sent; returns scripted text.

Used by every Phase 7b unit test. ``complete`` is the same method the real adapters implement,
so the analyzer, the policy responder and ``LLMClient`` run exactly as they do in production.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from app.llm.providers.base import Provider

Handler = Callable[[str, list[dict], dict | None], str]

_LABEL = re.compile(
    r"\[(?P<doc>[^|\]]+?)\s*\|\s*(?P<loc>[^|\]]+?)\s*\|\s*effective\s+(?P<date>\d{4}-\d{2}-\d{2})\]"
)
_DAYS = re.compile(r"(\d+(?:\.\d+)?)\s+days?", re.IGNORECASE)
_RATE = re.compile(
    r"(?P<n>\d+(?:\.\d+)?)\s*(?:days?\s*/\s*(?P<slash>month|year)|days?\s+per\s+(?P<per>month|year))",
    re.IGNORECASE,
)
_BAND_AFTER = re.compile(r"(?:for|if|when)\s+([^.\n;|]+)", re.IGNORECASE)
_WEEK = re.compile(
    r"(two|three|four|five|\d+)\s+days?\s+(?:each|per|a)\s+week", re.IGNORECASE
)
_NOTICE = re.compile(r"(\d+)\s+weeks?", re.IGNORECASE)
_CARRY = re.compile(
    r"carr(?:y|ied)(?:\s+forward|\s+over)\s+(?:up\s+to\s+)?(\d+(?:\.\d+)?)\s+days|"
    r"up\s+to\s+(\d+(?:\.\d+)?)\s+days\s+can\s+be\s+carr(?:y|ied)(?:\s+forward|\s+over)",
    re.IGNORECASE,
)
_SICK = re.compile(
    r"(\d+(?:\.\d+)?)\s+(?:working\s+)?days?\s+of\s+paid\s+sick|"
    r"(\d+(?:\.\d+)?)\s+paid\s+sick\s+days|"
    r"sick leave[^\n.]{0,40}?(\d+(?:\.\d+)?)\s+(?:working\s+)?days",
    re.IGNORECASE,
)
_ANNUAL_FLAT = re.compile(
    r"(\d+(?:\.\d+)?)\s+working days of paid annual leave|"
    r"annual leave entitlement is (\d+(?:\.\d+)?)\s+days|"
    r"receive (\d+(?:\.\d+)?)\s+working days of paid annual leave",
    re.IGNORECASE,
)
_PAYOUT = re.compile(r"encash|paid out|on leaving|leaving the company", re.IGNORECASE)
_WORD_NUM = {"two": "2", "three": "3", "four": "4", "five": "5"}


def analysis(
    intent: str = "policy_question",
    *,
    leave_type: str | None = None,
    dates_text: str | None = None,
    target_employee_id: str | None = None,
    request_id: str | None = None,
    status: str | None = None,
    missing_fields: list[str] | None = None,
    needs_leave_data: bool = False,
) -> str:
    return json.dumps(
        {
            "intent": intent,
            "leave_type": leave_type,
            "dates_text": dates_text,
            "target_employee_id": target_employee_id,
            "request_id": request_id,
            "status": status,
            "missing_fields": missing_fields or [],
            "needs_leave_data": needs_leave_data,
        }
    )


def extraction(claims: list[dict], not_covered: list[str] | None = None) -> str:
    return json.dumps({"claims": claims, "not_covered": not_covered or []})


def claim(topic: str, value: str, doc_id: str, locator: str) -> dict:
    return {"topic": topic, "value": value, "doc_id": doc_id, "locator": locator}


def _blocks(text: str) -> list[tuple[str, str, str]]:
    """``(doc_id, locator, body)`` for each labelled excerpt in ``text``."""
    matches = list(_LABEL.finditer(text))
    out = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        out.append((match.group("doc").strip(), match.group("loc").strip(), body))
    return out


def _last_tag(text: str, name: str) -> str:
    """The last ``<name>...</name>`` body. The first match can be the system prompt
    talking about the tag, so we must not start there."""
    matches = list(re.finditer(rf"<{name}>\n?(.*?)\n?</{name}>", text, re.DOTALL | re.IGNORECASE))
    return matches[-1].group(1) if matches else ""


def _table_rates(body: str) -> list[str]:
    """Accrual or entitlement rates, one value per tenure band (band stays in the value)."""
    values: list[str] = []
    seen: set[str] = set()
    for line in body.splitlines() or [body]:
        for match in _RATE.finditer(line):
            unit = match.group("slash") or match.group("per")
            value = f"{match.group('n')} days per {unit.lower()}"
            band = ""
            stripped = line.strip()
            if stripped.lower().startswith("row:"):
                others = [
                    cell.strip()
                    for cell in stripped.split(":", 1)[-1].split("|")
                    if match.group(0) not in cell
                ]
                band = " ".join(others).strip()
            if not band:
                after = _BAND_AFTER.search(line[match.end() :])
                if after:
                    band = after.group(1).strip()
            if band:
                value = f"{value} for {band}"
            if value not in seen:
                seen.add(value)
                values.append(value)
    return values


def _flat_day_count(body: str) -> str | None:
    """A leave-day figure that is not a per-month / per-year rate."""
    for match in _DAYS.finditer(body):
        tail = body[match.end() : match.end() + 16].lower()
        if tail.lstrip().startswith("per ") or tail.lstrip().startswith("/"):
            continue
        return f"{match.group(1)} days"
    return None


def extract_from_prompt(text: str) -> str:
    """A tiny deterministic extractor: report what the labelled excerpts actually say.

    Used so a modified corpus (a different number in the same document) changes the
    claims, and so an invented doc_id can never appear: only labels that were sent.
    Only topics the question is about are reported (a real model is asked the same).
    A table of tenure-band rates is several claims for one topic, band in the value.
    """
    question = _last_tag(text, "question").casefold()
    wants_remote = any(w in question for w in ("work from home", "work-from-home", "remote", "wfh"))
    wants_notice = "notice" in question
    wants_carry = any(w in question for w in ("carry", "carried forward", "carry over"))
    wants_sick = "sick" in question and not wants_carry
    wants_accrual = any(w in question for w in ("accrue", "accrual", "balance", "by december", "project"))
    wants_leave = (not wants_remote and not wants_carry and not wants_sick) and any(
        w in question for w in ("leave", "annual", "entitlement", "how many days")
    )
    claims: list[dict] = []
    for doc_id, locator, body in _blocks(text):
        rates = _table_rates(body)
        if rates and (wants_accrual or wants_leave):
            for value in rates:
                claims.append(claim("annual leave accrual", value, doc_id, locator))
        if wants_remote and "week" in body.lower() and "home" in body.lower():
            match = _WEEK.search(body)
            if match:
                n = _WORD_NUM.get(match.group(1).lower(), match.group(1))
                claims.append(claim("remote work days", f"{n} days each week", doc_id, locator))
        elif wants_carry:
            match = _CARRY.search(body)
            if match:
                number = next(g for g in match.groups() if g)
                claims.append(claim("annual leave carry forward", f"{number} days", doc_id, locator))
        elif wants_sick:
            match = _SICK.search(body)
            if match:
                number = next(g for g in match.groups() if g)
                claims.append(claim("sick leave", f"{number} days", doc_id, locator))
        elif wants_notice and "notice" in body.lower() and _NOTICE.search(body):
            claims.append(claim("notice period", f"{_NOTICE.search(body).group(1)} weeks", doc_id, locator))
        elif wants_leave and not wants_accrual and "leave" in body.lower():
            annual = _ANNUAL_FLAT.search(body)
            if annual:
                number = next(g for g in annual.groups() if g)
                claims.append(claim("annual leave accrual", f"{number} days", doc_id, locator))
            elif not rates and not _CARRY.search(body) and not _PAYOUT.search(body):
                flat = _flat_day_count(body)
                if flat:
                    claims.append(claim("leave days", flat, doc_id, locator))
    return extraction(claims)


def compose_from_prompt(text: str) -> str:
    """Restate the resolved facts in natural words (no bracketed ids, no settled-by)."""
    current: list[tuple[str, str, str]] = []
    upcoming: list[tuple[str, str, str]] = []
    conflicts: list[tuple[str, str, str]] = []
    extras: list[str] = []
    for _doc_id, locator, body in _blocks(text):
        name = ""
        date = ""
        in_force = ""
        upcoming_value = ""
        conflict_value = ""
        for raw in body.splitlines():
            raw = raw.strip()
            if raw.startswith("Source name:"):
                name = raw.split(":", 1)[1].strip()
            elif raw.startswith("Effective date:"):
                date = raw.split(":", 1)[1].strip()
            elif raw.startswith("In force"):
                in_force = raw.split(":", 1)[1].strip()
            elif raw.startswith("UNRESOLVED") or raw.lower().startswith("unresolved"):
                conflict_value = raw.split(":", 1)[-1].strip()
            elif raw.startswith("UPCOMING"):
                upcoming_value = raw.split(":", 1)[-1].strip()
            elif raw.startswith("Calculation") or raw[:1].isdigit():
                extras.append(raw)
            elif raw.startswith("PUBLIC holidays") or raw.startswith("OPTIONAL"):
                extras.append(raw)
            elif re.match(r"\d{4}-\d{2}-\d{2} ", raw):
                extras.append(raw)
        name = name or locator or "the source document"
        if conflict_value:
            conflicts.append((conflict_value, name, date))
        elif in_force:
            current.append((in_force, name, date))
        if upcoming_value:
            upcoming.append((upcoming_value, name, date))

    lines: list[str] = []
    if conflicts:
        parts = "; ".join(
            f"{name} ({date}) says {value}" if date else f"{name} says {value}"
            for value, name, date in conflicts
        )
        lines.append(f"The documents conflict and I cannot tell which applies. {parts}.")
    if current:
        values = "; ".join(dict.fromkeys(v for v, _, _ in current))
        name, date = current[0][1], current[0][2]
        if date:
            lines.append(f"The current rule is {values}, from {name}, effective {date}.")
        else:
            lines.append(f"The current rule is {values}, from {name}.")
    if upcoming:
        values = "; ".join(dict.fromkeys(v for v, _, _ in upcoming))
        name, date = upcoming[0][1], upcoming[0][2]
        current_values = "; ".join(dict.fromkeys(v for v, _, _ in current))
        if current_values:
            lines.append(
                f"From {date} this changes from {current_values} to {values}, from {name}. "
                "This is not in force yet."
            )
        else:
            lines.append(f"From {date} this changes to {values}, from {name}. This is not in force yet.")
    lines.extend(extras)
    return "\n".join(lines) or "The documents do not cover this. Please contact HR."


def default_handler(system: str, messages: list[dict], json_schema: dict | None) -> str:
    title = (json_schema or {}).get("title") if isinstance(json_schema, dict) else None
    # User messages only. The system prompts name <message> / <question> as delimiters, so
    # joining them in would make those words look like the employee's text.
    blob = "\n".join(m.get("content", "") for m in messages)
    if title == "analysis":
        return _analyse(blob)
    if title == "extraction":
        return extract_from_prompt(blob)
    return compose_from_prompt(blob)


_EMPLOYEE_ID = re.compile(r"\b([A-Z]{1,3}\d+[A-Z0-9]*|[A-Z]-[A-Z0-9]+)\b")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_REQUEST_ID = re.compile(r"\bR[A-Z0-9]+\b")
_DATE_HINT = re.compile(
    r"\b(?:from|to|through|thru|next|this|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|january|february|march|april|june|july|august|september|"
    r"october|november|december|\d{4}-\d{2}-\d{2})\b"
)


def _target_id(raw: str) -> str | None:
    match = _EMPLOYEE_ID.search(raw)
    return match.group(1) if match else None


def _analyse(blob: str) -> str:
    # Last <message> only: the analyzer system prompt names the tag and the intents.
    raw = _last_tag(blob, "message")
    text = raw.casefold()
    target = _target_id(raw)
    if any(w in text for w in ("capital of", "write a python", "tell me a joke", "weather today", "recipe")):
        return analysis("off_topic")
    if "holiday" in text:
        return analysis("holiday_question")
    if "balance" in text and any(w in text for w in ("will", "by december", "by the end", "projected")):
        return analysis("policy_question", leave_type="annual", dates_text="by December", needs_leave_data=True)
    if "balance" in text:
        leave = "annual" if "annual" in text else ("sick" if "sick" in text else None)
        return analysis("leave_balance", leave_type=leave, target_employee_id=target)
    if "leave details" in text:
        return analysis("leave_balance", target_employee_id=target)
    if "pending" in text and "request" in text:
        return analysis("list_requests", status="pending")
    if "apply" in text:
        leave = "annual" if "annual" in text else ("sick" if "sick" in text else None)
        dates = raw.strip() if _DATE_HINT.search(text) else None
        return analysis("apply_leave", leave_type=leave, dates_text=dates)
    if "team" in text:
        return analysis("team_leave", dates_text="next week")
    if "cancel" in text:
        rid = None
        uuid_match = _UUID.search(raw)
        if uuid_match:
            rid = uuid_match.group(0)
        else:
            req_match = _REQUEST_ID.search(raw)
            rid = req_match.group(0) if req_match else None
        return analysis("cancel_leave", request_id=rid)
    return analysis("policy_question")


class MockProvider(Provider):
    """A ``Provider`` with no HTTP. ``replies`` are consumed first; then ``handler``."""

    name = "mock"
    default_model = "mock"

    def __init__(
        self,
        *replies: str,
        handler: Handler | None = default_handler,
        error: BaseException | None = None,
        errors: list[BaseException] | None = None,
        api_key: str = "test-key",
        delay: float = 0.0,
    ) -> None:
        super().__init__(api_key=api_key, model="mock", timeout=5.0)
        self._replies = list(replies)
        self.handler = handler
        self.error = error
        self._errors = list(errors or [])
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    def _complete_once(
        self, system: str, messages: list[dict], json_schema: dict | None, timeout: float
    ) -> str:
        # The mock answers through ``complete`` (no HTTP, no retry). This exists so the
        # abstract Provider can be instantiated.
        raise AssertionError("MockProvider answers in complete(), not _complete_once()")

    def complete(self, system: str, messages: list[dict], json_schema: dict | None = None) -> str:
        if self.delay:
            time.sleep(self.delay)
        self.calls.append({"system": system, "messages": list(messages), "json_schema": json_schema})
        if self._errors:
            raise self._errors.pop(0)
        if self.error is not None:
            raise self.error
        if self._replies:
            return self._replies.pop(0)
        if self.handler is not None:
            return self.handler(system, messages, json_schema)
        raise AssertionError("MockProvider has no scripted reply and no handler")

    def queue(self, *replies: str) -> None:
        self._replies.extend(replies)

    @property
    def received_text(self) -> str:
        pieces: list[str] = []
        for call in self.calls:
            pieces.append(call["system"])
            pieces.extend(m.get("content", "") for m in call["messages"])
        return "\n".join(pieces)

    @property
    def purposes_seen(self) -> list[str]:
        """Heuristic: schema title, or 'compose' when no schema."""
        out = []
        for call in self.calls:
            schema = call.get("json_schema") or {}
            out.append(schema.get("title") or "compose")
        return out
