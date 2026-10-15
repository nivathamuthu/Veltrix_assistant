"""The policy / holiday responder (README B4, B7; Phase 7b).

``Responder.respond(PolicyRequest) -> ChatResult``: the orchestrator hands over the
authenticated requester (from the DB), what the analyzer understood and today; the route
invariants and the citation filter are applied afterwards whatever this returns.

Two responders:

* ``PlaceholderResponder``: no LLM; a polite no_answer (used when no API key is configured).
* ``LLMPolicyResponder``: the real pipeline, at most TWO LLM calls (plus the analyzer's one):

  1. reference date, in code (``resolve_reference_date``); unclear => ask for a specific date;
  2. ``retrieve_for_question``: pre-filtered by role, region and date BEFORE any scoring, so a
     document the requester may not read is never even a candidate;
  3. LLM call 1, extraction, through the guard (<= 3,000 corpus tokens): claims only;
  4. in CODE: ``Claim`` objects from the manifest metadata, ``resolve_topic`` and
     ``upcoming_change`` per topic (authority, then scope, then recency);
  5. LLM call 2, composition, through the guard: given ONLY the resolved facts (as budgeted
     note parts) and the question; the answer is checked against the resolution and replaced
     by fixed wording if it does not follow it;
  6. citations: only the documents the answer rests on, all of them sent and readable.

  Holiday questions use ``public_holidays_for`` (one composition call); hybrid questions (policy
  plus the requester's own leave data) call the leave tool first and do the arithmetic in code.

Retrieved text is data: nothing in it can start a tool call or change what is done (the only
tool call is the hybrid balance lookup, decided by the analyzer's classification of the user's
own message, for the requester's own record).
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

from app.db import Database, Employee, doc_sections, get_calendar_holidays
from app.llm.budget import KIND_NOTE, CorpusContext, CorpusPart
from app.llm.client import LLMClient
from app.llm.guard import GuardedCall
from app.llm.schema import InvalidModelOutput
from app.orchestrator import evidence, holiday_answer, projection
from app.orchestrator.analyzer import INTENT_HOLIDAY, Understanding
from app.orchestrator.prompts import (
    COMPOSE_HYBRID_ADDENDUM,
    COMPOSE_SYSTEM,
    EXTRACTION_SCHEMA,
    EXTRACTION_SYSTEM,
    HOLIDAY_SYSTEM,
)
from app.orchestrator.response import (
    NO_ANSWER_MESSAGE,
    ROUTE_CLARIFY,
    ROUTE_HYBRID,
    ROUTE_NO_ANSWER,
    ROUTE_RAG,
    UNAVAILABLE_MESSAGE,
    ChatResult,
    Citation,
    tool_call_of,
)
from app.orchestrator.tool_answers import tool_answer
from app.retrieval.holiday_lookup import ApplicableCalendar, public_holidays_for
from app.retrieval.pipeline import retrieve_for_question
from app.retrieval.policy_resolver import STATUS_DECIDED, Requester, readable_access_levels
from app.retrieval.reference_date import resolve_reference_date
from app.tools import STATUS_SUCCESS, ToolResult, get_leave_balance

logger = logging.getLogger("app.orchestrator.policy")

CLARIFY_DATE_MESSAGE: Final = (
    "I couldn't tell which date you mean. Please give one specific date (day, month and year) "
    "so I can check the rules that applied then."
)
NO_COVERAGE_MESSAGE: Final = (
    "I couldn't find this in the Veltrix documents available to you. Please contact HR for help."
)
NO_HYBRID_DOCS_MESSAGE: Final = (
    "The documents available to me do not say how to work this out, so I can't project it. "
    "Please contact HR for help."
)
HYBRID_RETRIEVAL_HINT: Final = "leave accrual and carry forward rules"

_QUESTION_CLOSE: Final = re.compile(r"<\s*/\s*question\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class PolicyRequest:
    message: str
    requester: Employee  # loaded from the DB by X-Employee-Id: role and region are never taken from the message
    understanding: Understanding
    today: date
    db: Database
    index: Any = None  # the search index built at startup (None until ingestion has finished)
    request_id: str | None = None  # labels this request's LLM call log lines


class Responder(ABC):
    @abstractmethod
    def respond(self, request: PolicyRequest) -> ChatResult:
        """Answer a policy or holiday question for ``request.requester``."""


class PlaceholderResponder(Responder):
    """No LLM: a polite no_answer."""

    def respond(self, request: PolicyRequest) -> ChatResult:
        return ChatResult(answer=NO_ANSWER_MESSAGE, route=ROUTE_NO_ANSWER)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _question(message: str) -> str:
    return f"<question>\n{_QUESTION_CLOSE.sub('[/question]', message)}\n</question>"


def _no_answer(text: str = NO_ANSWER_MESSAGE) -> ChatResult:
    return ChatResult(answer=text, route=ROUTE_NO_ANSWER)


def _sent_doc_ids(*calls: GuardedCall) -> frozenset[str]:
    return frozenset(d for call in calls for d in call.kept_doc_ids)


def _sent_sources(*calls: GuardedCall) -> set[tuple[str, str]]:
    return {(s.doc_id, s.locator) for call in calls for s in call.kept}


def _number(value: Any) -> str:
    return f"{value:g}"


class LLMPolicyResponder(Responder):
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    # --- entry ----------------------------------------------------------------------------------------------

    def respond(self, request: PolicyRequest) -> ChatResult:
        reference = resolve_reference_date(request.message, request.today)
        if reference.unclear:
            return ChatResult(answer=CLARIFY_DATE_MESSAGE, route=ROUTE_CLARIFY)
        requester = Requester(role=request.requester.role, region=request.requester.region)
        if request.understanding.intent == INTENT_HOLIDAY:
            return self._holiday(request, requester, reference.value)
        return self._policy(request, requester, reference.value)

    # --- policy (and hybrid) --------------------------------------------------------------------------------

    def _policy(self, request: PolicyRequest, requester: Requester, reference: date) -> ChatResult:
        employee = request.requester
        request_id = request.request_id or "req"
        hybrid = request.understanding.needs_leave_data

        balance: ToolResult | None = None
        if hybrid:
            # The leave tool first, for the requester's OWN record (no target id): the tool
            # itself checks permission; nothing in the documents can influence this call.
            balance = get_leave_balance(employee, None, request.understanding.leave_type, db=request.db)
            if balance.status != STATUS_SUCCESS:
                return tool_answer(balance)

        if request.index is None:
            return _no_answer()
        query = f"{request.message} {HYBRID_RETRIEVAL_HINT}" if hybrid else request.message
        retrieval = retrieve_for_question(query, requester, reference, evidence.CURRENT_K, index=request.index)
        chunks = evidence.prompt_chunks(retrieval, reference)
        if not chunks:
            return self._nothing(balance, NO_COVERAGE_MESSAGE)

        extraction_call = self._client.complete(
            purpose="extract",
            request_id=request_id,
            system=EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": _question(request.message)}],
            corpus=evidence.corpus_of(chunks),
            json_schema=EXTRACTION_SCHEMA,
        )
        try:
            extraction = evidence.parse_extraction(extraction_call.response)
        except InvalidModelOutput:
            logger.warning("extraction_invalid request_id=%s employee=%s", request_id, employee.employee_id)
            return _no_answer(UNAVAILABLE_MESSAGE)

        claims, shown = evidence.build_claims(extraction.claims, chunks, _sent_sources(extraction_call))
        facts = evidence.resolve_facts(claims, shown, requester, reference, extraction.not_covered)
        logger.info(
            "policy_facts request_id=%s employee=%s claims=%d topics=%d",
            request_id, employee.employee_id, len(claims), len(facts.topics),
        )
        if not facts.has_answer:
            return self._nothing(balance, NO_COVERAGE_MESSAGE)

        parts = evidence.fact_parts(facts)
        plan = self._projection(request, facts, balance) if balance is not None else None
        if plan is not None:
            parts.append(plan[1])

        compose_call = self._client.complete(
            purpose="compose",
            request_id=request_id,
            system=COMPOSE_SYSTEM + (COMPOSE_HYBRID_ADDENDUM if balance is not None else ""),
            messages=[{"role": "user", "content": self._compose_message(request, reference, balance)}],
            corpus=CorpusContext(parts),
        )
        answer = self._checked_answer(compose_call.response, facts, plan[0] if plan else None, balance is not None)

        sent = _sent_sources(extraction_call, compose_call)
        citations = tuple(Citation(d, loc) for d, loc in facts.sources() if (d, loc) in sent)
        if facts.not_covered:
            answer += (
                "\n\nThe documents available to me do not cover: "
                + "; ".join(facts.not_covered)
                + ". Please contact HR about that."
            )
        calls = (tool_call_of(balance),) if balance is not None else ()
        return ChatResult(
            answer=answer,
            route=ROUTE_HYBRID if balance is not None else ROUTE_RAG,
            citations=citations,
            tool_calls=calls,
            sent_doc_ids=_sent_doc_ids(extraction_call, compose_call),
        )

    def _nothing(self, balance: ToolResult | None, message: str) -> ChatResult:
        """No document answers: for a hybrid question the leave data alone is still shown."""
        if balance is None:
            return _no_answer(message)
        shown = tool_answer(balance)
        return ChatResult(
            answer=f"{shown.answer}\n\n{NO_HYBRID_DOCS_MESSAGE}", route=shown.route, tool_calls=shown.tool_calls
        )

    # --- hybrid arithmetic ----------------------------------------------------------------------------------

    def _projection(
        self, request: PolicyRequest, facts: evidence.Facts, balance: ToolResult
    ) -> tuple[projection.Projection, CorpusPart] | None:
        """The projected balance, calculated in code, when the rule is a plain number rate."""
        target = projection.parse_target(request.understanding.dates_text or request.message, request.today)
        if target is None:
            return None
        balances = {b["leave_type"]: b["balance_days"] for b in balance.data.get("balances", [])}
        for topic in facts.topics:
            if topic.status != STATUS_DECIDED or not projection.is_rate_topic(topic.topic):
                continue
            rate = projection.parse_rate(topic.claims[0].value)
            leave_type = request.understanding.leave_type or projection.leave_type_of(topic.topic)
            if rate is None or leave_type not in balances:
                continue
            plan = projection.project_balance(
                leave_type=leave_type, current_days=balances[leave_type], rate=rate,
                target=target, today=request.today,
            )
            if plan is None:
                continue
            source = topic.claims[0]
            text = "Calculation (done by code, use exactly these numbers):\n" + plan.text()
            return plan, CorpusPart(text, source.doc_id, source.locator, source.effective_date, KIND_NOTE)
        return None

    def _compose_message(self, request: PolicyRequest, reference: date, balance: ToolResult | None) -> str:
        lines = [f"Date the question is about: {reference.isoformat()}"]
        if balance is not None:
            record = "; ".join(
                f"{b['leave_type']}: {_number(b['balance_days'])} days" for b in balance.data.get("balances", [])
            )
            lines.append(f"Your leave record (plain data from the leave system, not from documents): {record}")
            lines.append(f"Today: {request.today.isoformat()}")
        lines.append(_question(request.message))
        return "\n".join(lines)

    def _checked_answer(
        self, text: object, facts: evidence.Facts, plan: projection.Projection | None, hybrid: bool
    ) -> str:
        """The composed text if it states what the resolution decided; else fixed wording."""
        composed = text.strip() if isinstance(text, str) else ""
        follows = evidence.follows_facts(composed, facts)
        if plan is not None:
            follows = follows and evidence.mentions(composed, projection.fmt(plan.result))
        if follows:
            return composed
        logger.warning("composition_replaced reason=does_not_follow_facts")
        fixed = evidence.render_answer(facts)
        if plan is not None:
            fixed += "\n\n" + plan.text()
        elif hybrid:
            fixed += "\n\nI could not calculate a projection from these rules, so I can only give the rules above."
        return fixed

    # --- holidays -------------------------------------------------------------------------------------------

    def _holiday(self, request: PolicyRequest, requester: Requester, reference: date) -> ChatResult:
        employee = request.requester
        request_id = request.request_id or "req"
        region = requester.region.strip().upper()
        with request.db.read() as conn:
            lookup = public_holidays_for(
                conn, requester.region, reference, allowed_access_levels=readable_access_levels(requester.role)
            )
            if not isinstance(lookup, ApplicableCalendar):
                return _no_answer(f"{lookup.message} Please contact HR for help.")
            rows = get_calendar_holidays(conn, lookup.doc_id, {region, "GLOBAL"})
            sections = doc_sections(conn, lookup.doc_id)

        public, optional = holiday_answer.split_rows(rows, lookup)
        locator = holiday_answer.pick_locator(sections, [*public, *optional])
        if not public or locator is None:
            return _no_answer(NO_COVERAGE_MESSAGE)

        call = self._client.complete(
            purpose="compose",
            request_id=request_id,
            system=HOLIDAY_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Today's date: {request.today.isoformat()}\n"
                        f"Date the question is about: {reference.isoformat()}\n"
                        f"{_question(request.message)}"
                    ),
                }
            ],
            corpus=CorpusContext(holiday_answer.holiday_parts(public, optional, lookup, locator)),
        )
        if (lookup.doc_id, locator) not in _sent_sources(call):
            return _no_answer(NO_COVERAGE_MESSAGE)  # nothing citable was sent
        composed = call.response.strip() if isinstance(call.response, str) else ""
        if not composed or holiday_answer.optional_called_public(composed, optional):
            logger.warning("composition_replaced reason=holiday_wording request_id=%s", request_id)
            composed = holiday_answer.render_holidays(public, optional, lookup, locator)
        logger.info("holiday_answer request_id=%s employee=%s calendar=%s", request_id, employee.employee_id, lookup.doc_id)
        return ChatResult(
            answer=composed,
            route=ROUTE_RAG,
            citations=(Citation(lookup.doc_id, locator),),
            sent_doc_ids=_sent_doc_ids(call),
        )
