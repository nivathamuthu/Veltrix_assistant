"""Route invariants (README B7), enforced by one function on every response."""
from __future__ import annotations

import dataclasses

import pytest

from app.orchestrator.response import (
    NO_ANSWER_MESSAGE,
    ROUTES,
    ChatResult,
    Citation,
    ToolCall,
    enforce_invariants,
)

READABLE = {"POL-1", "POL-2", "MEMO-1"}


def cite(doc_id="POL-1", locator="2. Accrual"):
    return Citation(doc_id, locator)


def call(name="get_leave_balance", status="success", args=None):
    return ToolCall(name, args or {}, status)


def draft(route, *, citations=(), tool_calls=(), answer="An answer.", pending=False, sent=("POL-1", "POL-2", "MEMO-1")):
    return ChatResult(
        answer=answer, route=route, citations=tuple(citations), tool_calls=tuple(tool_calls),
        pending_confirmation=pending, sent_doc_ids=frozenset(sent),
    )


def run(result, readable=READABLE):
    return enforce_invariants(result, readable_doc_ids=readable)


def check_invariants(result):
    """The B7 table, stated independently of the implementation."""
    c, t = len(result.citations), len(result.tool_calls)
    return {
        "rag": c >= 1 and t == 0,
        "tool": t >= 1 and c == 0,
        "hybrid": c >= 1 and t >= 1,
        "clarify": c == 0 and t == 0,
        "no_answer": c == 0 and t == 0,
        "refuse": c == 0 and t == 0,
    }[result.route]


def test_the_routes_are_the_six_in_the_spec():
    assert set(ROUTES) == {"rag", "tool", "hybrid", "clarify", "no_answer", "refuse"}


# --- results that are already valid pass through untouched ---------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        draft("rag", citations=[cite()]),
        draft("tool", tool_calls=[call()]),
        draft("hybrid", citations=[cite()], tool_calls=[call()]),
        draft("clarify"),
        draft("no_answer"),
        draft("refuse"),
    ],
)
def test_valid_results_are_unchanged(result):
    assert run(result) == result


# --- downgrades ------------------------------------------------------------------------------------------------


def test_rag_without_citations_becomes_no_answer_with_a_safe_message():
    out = run(draft("rag", answer="The policy says 25 days."))
    assert out.route == "no_answer" and out.citations == () and out.tool_calls == ()
    assert out.answer == NO_ANSWER_MESSAGE  # the unsupported claim is not passed on
    assert "25" not in out.answer


def test_rag_with_tool_calls_becomes_hybrid_when_it_has_citations():
    out = run(draft("rag", citations=[cite()], tool_calls=[call()]))
    assert out.route == "hybrid" and len(out.tool_calls) == 1 and len(out.citations) == 1


def test_tool_with_citations_becomes_hybrid():
    out = run(draft("tool", citations=[cite()], tool_calls=[call()]))
    assert out.route == "hybrid" and check_invariants(out)


def test_tool_without_tool_calls_becomes_no_answer():
    out = run(draft("tool"))
    assert out.route == "no_answer" and out.answer == NO_ANSWER_MESSAGE


def test_hybrid_with_only_citations_becomes_rag_and_with_only_tool_calls_becomes_tool():
    assert run(draft("hybrid", citations=[cite()])).route == "rag"
    assert run(draft("hybrid", tool_calls=[call()])).route == "tool"
    assert run(draft("hybrid")).route == "no_answer"


@pytest.mark.parametrize("route", ["clarify", "no_answer", "refuse"])
def test_routes_that_must_be_empty_have_citations_and_tool_calls_removed(route):
    out = run(draft(route, citations=[cite()], tool_calls=[call()], answer="Please tell me more."))
    assert out.route == route and out.citations == () and out.tool_calls == ()
    assert out.answer == "Please tell me more."  # the text is kept; only the extras go


def test_an_unknown_route_becomes_no_answer():
    for bad in ("", "RAG", "chat", None, 5):
        out = run(dataclasses.replace(draft("rag", citations=[cite()]), route=bad))  # type: ignore[arg-type]
        assert out.route == "no_answer"


def test_an_empty_answer_is_replaced_by_a_safe_message():
    for bad in ("", "   ", None):
        out = run(dataclasses.replace(draft("refuse"), answer=bad))  # type: ignore[arg-type]
        assert out.answer.strip()


# --- citations -------------------------------------------------------------------------------------------------


def test_citations_are_deduplicated_keeping_the_first_and_the_order():
    out = run(draft("rag", citations=[cite("POL-1", "A"), cite("POL-2", "B"), cite("POL-1", "A"), cite("POL-1", "C")]))
    assert [(c.doc_id, c.locator) for c in out.citations] == [("POL-1", "A"), ("POL-2", "B"), ("POL-1", "C")]


def test_a_citation_for_a_document_that_was_not_sent_is_dropped():
    out = run(draft("rag", citations=[cite("POL-1"), cite("POL-2")], sent=("POL-1",)))
    assert [c.doc_id for c in out.citations] == ["POL-1"]


def test_a_citation_for_a_document_the_requester_cannot_read_is_dropped_even_if_it_was_sent():
    out = run(draft("rag", citations=[cite("POL-1"), cite("HR-ONLY")], sent=("POL-1", "HR-ONLY")), readable={"POL-1"})
    assert [c.doc_id for c in out.citations] == ["POL-1"]


def test_when_every_citation_is_dropped_rag_falls_back_to_no_answer():
    out = run(draft("rag", citations=[cite("HR-ONLY")], sent=("HR-ONLY",), answer="secret detail"), readable={"POL-1"})
    assert out.route == "no_answer" and out.citations == ()
    assert "secret" not in out.answer and "HR-ONLY" not in out.answer


def test_when_every_citation_is_dropped_hybrid_falls_back_to_tool():
    out = run(draft("hybrid", citations=[cite("X")], tool_calls=[call()], sent=()))
    assert out.route == "tool" and out.citations == () and len(out.tool_calls) == 1


def test_nothing_is_cited_when_no_document_was_sent_to_the_llm():
    out = run(draft("rag", citations=[cite()], sent=()))
    assert out.route == "no_answer"


def test_citations_with_blank_fields_are_dropped():
    out = run(draft("rag", citations=[Citation("", "x"), Citation("POL-1", ""), cite("POL-2", "ok")]))
    assert [(c.doc_id, c.locator) for c in out.citations] == [("POL-2", "ok")]


def test_the_unreadable_document_is_never_named_in_the_replacement_answer():
    out = run(draft("rag", citations=[cite("HR-ONLY")], sent=("HR-ONLY",)), readable=set())
    assert "HR-ONLY" not in out.answer
    assert all("HR-ONLY" not in c.doc_id for c in out.citations)


# --- pending confirmation ---------------------------------------------------------------------------------------


def test_the_pending_flag_follows_the_tool_call_status():
    pending = call("apply_leave", "pending_confirmation")
    assert run(draft("tool", tool_calls=[pending], pending=False)).pending_confirmation is True
    assert run(draft("tool", tool_calls=[call()], pending=True)).pending_confirmation is False
    assert run(draft("clarify", pending=True)).pending_confirmation is False
    assert run(draft("refuse", pending=True)).pending_confirmation is False


def test_a_pending_call_is_removed_with_its_flag_when_the_route_cannot_carry_it():
    out = run(draft("no_answer", tool_calls=[call("apply_leave", "pending_confirmation")], pending=True))
    assert out.pending_confirmation is False and out.tool_calls == ()


# --- safety ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("n_cites", [0, 1, 3])
@pytest.mark.parametrize("n_calls", [0, 1, 2])
def test_every_combination_ends_in_a_valid_response(route, n_cites, n_calls):
    result = draft(route, citations=[cite(f"POL-{i % 2 + 1}", f"S{i}") for i in range(n_cites)],
                   tool_calls=[call() for _ in range(n_calls)])
    out = run(result)
    assert out.route in ROUTES and check_invariants(out), (route, n_cites, n_calls, out)
    assert out.answer.strip()
    # whatever was kept was allowed
    assert all(c.doc_id in READABLE and c.doc_id in result.sent_doc_ids for c in out.citations)


def test_it_never_raises_whatever_it_is_given():
    class Bomb:
        def __iter__(self):
            raise RuntimeError("boom")

    weird = [
        draft("rag", citations=[None]),  # type: ignore[list-item]
        draft("rag", citations=["POL-1"]),  # type: ignore[list-item]
        draft("tool", tool_calls=[None]),  # type: ignore[list-item]
        dataclasses.replace(draft("rag"), citations=Bomb()),  # type: ignore[arg-type]
        dataclasses.replace(draft("rag"), sent_doc_ids=None),  # type: ignore[arg-type]
    ]
    for result in weird:
        out = run(result)
        assert isinstance(out, ChatResult) and out.route in ROUTES and check_invariants(out)
    assert isinstance(enforce_invariants(None, readable_doc_ids=READABLE), ChatResult)  # type: ignore[arg-type]
    assert enforce_invariants(None, readable_doc_ids=READABLE).route == "no_answer"  # type: ignore[arg-type]
    assert enforce_invariants(draft("rag", citations=[cite()]), readable_doc_ids=None).route == "no_answer"  # type: ignore[arg-type]


def test_the_result_is_a_new_object_and_the_input_is_not_modified():
    original = draft("rag", citations=[cite("POL-1"), cite("POL-1")])
    snapshot = dataclasses.replace(original)
    out = run(original)
    assert out is not original and original == snapshot and len(original.citations) == 2


def test_enforcement_is_idempotent():
    for route in sorted(ROUTES):
        once = run(draft(route, citations=[cite(), cite()], tool_calls=[call()]))
        assert run(once) == once
