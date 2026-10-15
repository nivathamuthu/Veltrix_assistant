"""From retrieved chunks to resolved facts, in code (no LLM judgement)."""
from __future__ import annotations

from datetime import date

import pytest

from app.llm.schema import InvalidModelOutput
from app.orchestrator.evidence import (
    MAX_UPCOMING_CHUNKS,
    build_claims,
    follows_facts,
    parse_extraction,
    prompt_chunks,
    render_answer,
    resolve_facts,
)
from app.retrieval.pipeline import Retrieval, RetrievedChunk
from app.retrieval.policy_resolver import Requester
from tests.llm_helpers import TODAY, all_chunks


EMPLOYEE_IN = Requester("employee", "IN")
EMPLOYEE_DE = Requester("employee", "DE")


def rc(chunk, score=1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, doc_type=chunk.doc_type,
        regions=tuple(chunk.regions), effective_date=date.fromisoformat(chunk.effective_date),
        locator=chunk.locator, text=chunk.text, token_count=chunk.token_count,
        score=score, access_level=chunk.access_level,
    )


def by_id(*doc_ids) -> list[RetrievedChunk]:
    wanted = set(doc_ids)
    return [rc(c) for c in all_chunks() if c.doc_id in wanted]


def test_upcoming_chunks_are_capped_and_only_newer_than_the_reference():
    current = by_id("POL-V1", "POL-V2")
    upcoming = [
        rc(next(c for c in all_chunks() if c.doc_id == "POL-V3")),
        rc(next(c for c in all_chunks() if c.doc_id == "POL-V2")),  # not newer than 2026-09-15
    ]
    # four extra "upcoming" chunks dated after the reference, to hit the cap
    extras = [
        RetrievedChunk(
            f"FUT-{i}", f"FUT-{i}", "policy", ("GLOBAL",), date(2027, 2, i + 1),
            "x", "future text", 3, 0.1, "all",
        )
        for i in range(5)
    ]
    kept = prompt_chunks(Retrieval(tuple(current), tuple(upcoming + extras)), TODAY)
    upcoming_kept = [c for c in kept if c.effective_date > TODAY]
    assert all(c.effective_date > TODAY for c in upcoming_kept)
    assert len(upcoming_kept) <= MAX_UPCOMING_CHUNKS
    assert "POL-V2" not in {c.doc_id for c in upcoming_kept}  # same-or-earlier date is not upcoming


def test_an_invented_source_or_a_value_not_in_the_chunk_is_dropped():
    chunks = by_id("POL-V2")
    sent = {(c.doc_id, c.locator) for c in chunks}
    raw = [
        {"topic": "leave days", "value": "3 days", "doc_id": "POL-V2", "locator": chunks[0].locator},
        {"topic": "leave days", "value": "3 days", "doc_id": "INVENTED", "locator": chunks[0].locator},
        {"topic": "leave days", "value": "99 days", "doc_id": "POL-V2", "locator": chunks[0].locator},
        {"topic": "leave days", "value": "3 days", "doc_id": "POL-V2", "locator": "not-a-real-locator"},
    ]
    claims, shown = build_claims(raw, chunks, sent)
    assert len(claims) == 1
    assert shown[claims[0]] == "3 days"
    assert claims[0].doc.doc_id == "POL-V2"
    assert claims[0].doc.doc_type == "policy"  # from the chunk / manifest, not the model


def test_document_metadata_comes_from_the_chunk_not_the_model():
    chunks = by_id("FAQ-1")
    sent = {(c.doc_id, c.locator) for c in chunks}
    claims, _ = build_claims(
        [{"topic": "leave days", "value": "2 days", "doc_id": "FAQ-1", "locator": chunks[0].locator}],
        chunks, sent,
    )
    assert claims[0].doc.doc_type == "faq"
    assert claims[0].doc.effective_date == date(2026, 5, 1)


def test_worked_example_facts_match_readme_b4():
    chunks = by_id("POL-V1", "POL-V2", "FAQ-1", "ADD-DE", "POL-V3")
    sent = {(c.doc_id, c.locator) for c in chunks}
    raw = [
        {"topic": "leave days", "value": "2 days", "doc_id": "POL-V1", "locator": "Leave policy v1"},
        {"topic": "leave days", "value": "3 days", "doc_id": "POL-V2", "locator": "Leave policy v2"},
        {"topic": "leave days", "value": "2 days", "doc_id": "FAQ-1", "locator": "How many leave days"},
        {"topic": "leave days", "value": "5 days", "doc_id": "ADD-DE", "locator": "Germany leave addendum"},
        {"topic": "leave days", "value": "4 days", "doc_id": "POL-V3", "locator": "Leave policy v3"},
    ]
    claims, shown = build_claims(raw, chunks, sent)

    india = resolve_facts(claims, shown, EMPLOYEE_IN, TODAY)
    assert india.has_answer
    topic = india.topics[0]
    assert topic.status == "decided" and topic.claims[0].value == "3 days"
    assert topic.claims[0].doc_id == "POL-V2"
    assert topic.upcoming is not None
    assert topic.upcoming.doc_id == "POL-V3" and topic.upcoming.value == "4 days"
    assert topic.upcoming.effective_date == date(2027, 1, 1)

    germany = resolve_facts(claims, shown, EMPLOYEE_DE, TODAY)
    topic = germany.topics[0]
    assert topic.status == "decided" and topic.claims[0].value == "5 days"
    assert topic.claims[0].doc_id == "ADD-DE"
    assert topic.upcoming is None

    past = resolve_facts(claims, shown, EMPLOYEE_IN, date(2025, 8, 1))
    topic = past.topics[0]
    assert topic.status == "decided" and topic.claims[0].value == "2 days"
    assert topic.claims[0].doc_id == "POL-V1"


def test_a_conflict_is_both_values_and_follows_facts_requires_the_word():
    chunks = by_id("POL-NOTICE-A", "POL-NOTICE-B")
    sent = {(c.doc_id, c.locator) for c in chunks}
    raw = [
        {"topic": "notice period", "value": "2 weeks", "doc_id": "POL-NOTICE-A", "locator": "Notice A"},
        {"topic": "notice period", "value": "4 weeks", "doc_id": "POL-NOTICE-B", "locator": "Notice B"},
    ]
    claims, shown = build_claims(raw, chunks, sent)
    facts = resolve_facts(claims, shown, EMPLOYEE_IN, TODAY)
    assert facts.topics[0].status == "conflict"
    both = {r.doc_id for r in facts.topics[0].claims}
    assert both == {"POL-NOTICE-A", "POL-NOTICE-B"}
    answer = render_answer(facts)
    assert "conflict" in answer.lower() and "2 weeks" in answer and "4 weeks" in answer
    assert follows_facts(answer, facts)
    assert not follows_facts("The notice period is 2 weeks.", facts)


def test_parse_extraction_rejects_invalid_json():
    with pytest.raises(InvalidModelOutput):
        parse_extraction("not json")
    parsed = parse_extraction('{"claims": [], "not_covered": ["x"]}')
    assert parsed.claims == () and parsed.not_covered == ("x",)
