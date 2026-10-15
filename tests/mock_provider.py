"""A Provider that never touches the network. Records what it was sent; returns scripted text.

Used by every Phase 7b unit test. ``complete`` is the same method the real adapters implement,
so the analyzer, the policy responder and ``LLMClient`` run exactly as they do in production.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from app.llm.providers.base import Provider

Handler = Callable[[str, list[dict], dict | None], str]

_LABEL = re.compile(
    r"\[(?P<doc>[^|\]]+?)\s*\|\s*(?P<loc>[^|\]]+?)\s*\|\s*effective\s+(?P<date>\d{4}-\d{2}-\d{2})\]"
)
_DAYS = re.compile(r"(\d+(?:\.\d+)?)\s+days?", re.IGNORECASE)
_PER = re.compile(r"(\d+(?:\.\d+)?)\s+days?\s+per\s+(month|year)", re.IGNORECASE)
_WEEK = re.compile(r"(two|three|four|five|\d+)\s+days?\s+each\s+week", re.IGNORECASE)
_NOTICE = re.compile(r"(\d+)\s+weeks?", re.IGNORECASE)
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


def extract_from_prompt(text: str) -> str:
    """A tiny deterministic extractor: report what the labelled excerpts actually say.

    Used so a modified corpus (a different number in the same document) changes the
    claims, and so an invented doc_id can never appear: only labels that were sent.
    Only topics the question is about are reported (a real model is asked the same)."""
    question = _last_tag(text, "question").casefold()
    wants_remote = any(w in question for w in ("work from home", "remote", "wfh"))
    wants_notice = "notice" in question
    wants_accrual = any(w in question for w in ("accrue", "accrual", "balance", "by december", "project"))
    wants_leave = (not wants_remote) and any(
        w in question for w in ("leave", "annual", "entitlement", "how many days")
    )
    claims: list[dict] = []
    for doc_id, locator, body in _blocks(text):
        if _PER.search(body):
            if wants_accrual:
                match = _PER.search(body)
                claims.append(claim("annual leave accrual", match.group(0).lower(), doc_id, locator))
            continue
        if wants_remote and "each week" in body.lower() and "home" in body.lower():
            match = _WEEK.search(body)
            if match:
                n = _WORD_NUM.get(match.group(1).lower(), match.group(1))
                claims.append(claim("remote work days", f"{n} days each week", doc_id, locator))
        elif wants_notice and "notice" in body.lower() and _NOTICE.search(body):
            claims.append(claim("notice period", f"{_NOTICE.search(body).group(1)} weeks", doc_id, locator))
        elif wants_leave and "leave" in body.lower() and _DAYS.search(body):
            claims.append(claim("leave days", f"{_DAYS.search(body).group(1)} days", doc_id, locator))
    return extraction(claims)


def compose_from_prompt(text: str) -> str:
    """Restate the resolved facts the composition call was given, so the answer follows them."""
    lines: list[str] = []
    conflict = "unresolved conflict" in text.lower()
    for doc_id, locator, body in _blocks(text):
        for raw in body.splitlines():
            raw = raw.strip()
            if raw.startswith("In force"):
                lines.append(f"{raw} (source {doc_id}, {locator}).")
            elif raw.startswith("UNRESOLVED") or raw.lower().startswith("unresolved"):
                lines.append(f"The documents conflict. {doc_id}, {locator}: {raw}")
            elif raw.startswith("UPCOMING"):
                lines.append(f"{raw} (source {doc_id}, {locator}). This is not in force yet.")
            elif raw.startswith("Calculation") or raw[:1].isdigit():
                lines.append(raw)
            elif raw.startswith("PUBLIC holidays") or raw.startswith("OPTIONAL"):
                lines.append(raw)
            elif re.match(r"\d{4}-\d{2}-\d{2} ", raw):
                lines.append(raw)
    if conflict and "conflict" not in " ".join(lines).lower():
        lines.append("The documents conflict.")
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


def _analyse(blob: str) -> str:
    # Last <message> only: the analyzer system prompt names the tag and the intents.
    text = _last_tag(blob, "message").casefold()
    if any(w in text for w in ("capital of", "write a python", "tell me a joke", "weather today", "recipe")):
        return analysis("off_topic")
    if "holiday" in text:
        return analysis("holiday_question")
    if "balance" in text and any(w in text for w in ("will", "by december", "by the end", "projected")):
        return analysis("policy_question", leave_type="annual", dates_text="by December", needs_leave_data=True)
    if "balance" in text:
        return analysis("leave_balance")
    if "pending" in text and "request" in text:
        return analysis("list_requests", status="pending")
    if "apply" in text:
        return analysis("apply_leave")
    if "team" in text:
        return analysis("team_leave", dates_text="next week")
    if "cancel" in text:
        return analysis("cancel_leave")
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
    ) -> None:
        super().__init__(api_key=api_key, model="mock", timeout=5.0)
        self._replies = list(replies)
        self.handler = handler
        self.error = error
        self._errors = list(errors or [])
        self.calls: list[dict[str, Any]] = []

    def _complete_once(
        self, system: str, messages: list[dict], json_schema: dict | None, timeout: float
    ) -> str:
        # The mock answers through ``complete`` (no HTTP, no retry). This exists so the
        # abstract Provider can be instantiated.
        raise AssertionError("MockProvider answers in complete(), not _complete_once()")

    def complete(self, system: str, messages: list[dict], json_schema: dict | None = None) -> str:
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
