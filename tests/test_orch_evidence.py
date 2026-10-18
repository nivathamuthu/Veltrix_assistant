"""From retrieved chunks to resolved facts, in code (no LLM judgement)."""
from __future__ import annotations

from datetime import date

import pytest

from app.llm.schema import InvalidModelOutput
from app.orchestrator.evidence import (
    DROP_NOT_KEPT,
    DROP_VALUE,
    MAX_UPCOMING_CHUNKS,
    ClaimRef,
    Facts,
    TopicFact,
    build_claims,
    build_claims_report,
    corpus_of,
    facts_for_question,
    fold_topic,
    follows_facts,
    group_chunks_by_document,
    normalise_support_text,
    parse_extraction,
    prompt_chunks,
    render_answer,
    resolve_facts,
    skipped_numeric_documents,
    _supported,
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


def test_a_wrong_locator_is_repaired_when_the_value_is_in_exactly_one_sent_chunk():
    chunks = by_id("POL-V2")
    sent = {(c.doc_id, c.locator) for c in chunks}
    raw = [{"topic": "leave days", "value": "3 days", "doc_id": "POL-V2", "locator": "guessed heading"}]
    claims, shown = build_claims(raw, chunks, sent)
    assert len(claims) == 1
    assert claims[0].locator == chunks[0].locator
    assert shown[claims[0]] == "3 days"


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
    assert topic.upcoming
    assert topic.upcoming[0].doc_id == "POL-V3" and topic.upcoming[0].value == "4 days"
    assert topic.upcoming[0].effective_date == date(2027, 1, 1)

    germany = resolve_facts(claims, shown, EMPLOYEE_DE, TODAY)
    topic = germany.topics[0]
    assert topic.status == "decided" and topic.claims[0].value == "5 days"
    assert topic.claims[0].doc_id == "ADD-DE"
    assert topic.upcoming == ()

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


def test_support_check_accepts_days_per_month_variants_and_numbers_alone():
    chunk = "Row: Tenure Under 2 years | Accrual 1.5 days/month"
    assert "day per month" in normalise_support_text(chunk)
    assert _supported("1.5 days per month for under 2 years", chunk)
    assert _supported("1.5 DAYS/MONTH", chunk)
    assert _supported("1.5 days per month for under 2 years", "accrual is 1.5 for under two years")
    assert not _supported("99 days per month", chunk)


def test_dropped_claims_report_kept_set_and_value_reasons():
    chunks = by_id("POL-V2")
    sent = {(c.doc_id, c.locator) for c in chunks}
    raw = [
        {"topic": "leave days", "value": "3 days", "doc_id": "POL-V2", "locator": chunks[0].locator},
        {"topic": "leave days", "value": "3 days", "doc_id": "INVENTED", "locator": chunks[0].locator},
        {"topic": "leave days", "value": "99 days", "doc_id": "POL-V2", "locator": chunks[0].locator},
    ]
    claims, shown, dropped, names, extracted, model_kept = build_claims_report(raw, chunks, sent)
    assert len(claims) == 1 and shown[claims[0]] == "3 days"
    assert extracted == 3 and model_kept == 1 and extracted >= model_kept
    assert dropped.count(DROP_NOT_KEPT) == 1
    assert dropped.count(DROP_VALUE) == 1
    assert names[claims[0]]


def test_banded_values_from_one_document_are_decided():
    text = (
        "Annual Leave Accrual Schedule\n"
        "Row: Tenure under 2 years | Accrual 1.5 days/month\n"
        "Row: Tenure 2 to 5 years | Accrual 1.75 days/month"
    )
    chunk = RetrievedChunk(
        "TBL-1", "TBL-LEAVE", "policy", ("GLOBAL",), date(2025, 1, 1),
        "Annual Leave Accrual Schedule", text, 40, 1.0, "all",
    )
    sent = {(chunk.doc_id, chunk.locator)}
    raw = [
        {
            "topic": "annual leave accrual",
            "value": "1.5 days per month for under 2 years",
            "doc_id": "TBL-LEAVE",
            "locator": chunk.locator,
        },
        {
            "topic": "annual leave accrual",
            "value": "1.75 days per month for 2 to 5 years",
            "doc_id": "TBL-LEAVE",
            "locator": chunk.locator,
        },
    ]
    claims, shown = build_claims(raw, [chunk], sent)
    facts = resolve_facts(claims, shown, EMPLOYEE_IN, TODAY, names={c: "Annual Leave Accrual Schedule" for c in claims})
    topic = facts.topics[0]
    assert topic.status == "decided"
    assert {r.value for r in topic.claims} == {
        "1.5 days per month for under 2 years",
        "1.75 days per month for 2 to 5 years",
    }
    answer = render_answer(facts)
    assert "settled by" not in answer.lower()
    assert "TBL-LEAVE" not in answer
    assert "[" not in answer
    assert "Annual Leave Accrual Schedule" in answer
    assert "1.5" in answer and "1.75" in answer
    assert follows_facts(answer, facts)
    assert not follows_facts("Annual leave is 1.5 days per month only.", facts)


def test_render_answer_states_current_next_to_upcoming_without_bracketed_ids():
    chunks = by_id("POL-V2", "POL-V3")
    sent = {(c.doc_id, c.locator) for c in chunks}
    raw = [
        {"topic": "leave days", "value": "3 days", "doc_id": "POL-V2", "locator": "Leave policy v2"},
        {"topic": "leave days", "value": "4 days", "doc_id": "POL-V3", "locator": "Leave policy v3"},
    ]
    claims, shown = build_claims(raw, chunks, sent)
    names = {c: ("Leave policy v2" if c.doc.doc_id == "POL-V2" else "Leave policy v3") for c in claims}
    facts = resolve_facts(claims, shown, EMPLOYEE_IN, TODAY, names=names)
    answer = render_answer(facts)
    assert "3 days" in answer and "4 days" in answer
    assert "2027-01-01" in answer
    assert "not in force yet" in answer.lower()
    assert "settled by" not in answer.lower()
    assert "[POL-V2" not in answer and "[POL-V3" not in answer
    assert "Leave policy v2" in answer
    assert follows_facts(answer, facts)
    assert not follows_facts("Settled by regional scope: 3 days [POL-V2 | x].", facts)


def test_band_specific_topics_fold_into_one_rule():
    assert fold_topic("annual leave accrual, 2 to less than 5 years of service") == "annual leave accrual"
    assert fold_topic("Annual leave entitlement") == "annual leave accrual"
    assert fold_topic("leave days") == "annual leave accrual"
    assert fold_topic("paid annual leave") == "annual leave accrual"
    assert fold_topic("annual leave carry forward") == "annual leave carry forward"
    assert fold_topic("annual leave encashment") == "annual leave encashment"


def test_a_carry_forward_excerpt_is_not_treated_as_the_accrual_rule():
    from app.orchestrator.evidence import _topic_for_chunk

    carry = "Employees may carry forward up to 15 days of unused annual leave."
    assert _topic_for_chunk("annual leave accrual", carry, "15 days") == "annual leave carry forward"
    mixed = (
        "Annual leave is earned monthly; see the Accrual Schedule. "
        "Up to 10 days can be carried forward; use them by 31 March."
    )
    assert _topic_for_chunk("annual leave accrual", mixed, "10 days") == "annual leave carry forward"
    payout = "On leaving the company, employees are paid for up to 30 days of unused annual leave."
    assert _topic_for_chunk("annual leave accrual", payout, "30 days") == "annual leave encashment"
    rates = "Length of service Less than 2 years. Days earned per month 1.5."
    assert _topic_for_chunk("annual leave accrual", rates, "1.5 days per month") == "annual leave accrual"


def test_encashment_is_dropped_from_an_entitlement_question():
    accrual = TopicFact(
        topic="annual leave accrual",
        status="decided",
        claims=(
            ClaimRef(
                "TBL-LEAVE-ACCRUAL",
                "1. Accrual rates",
                date(2025, 1, 1),
                "policy",
                "1.5 days per month",
                "Annual Leave Accrual Schedule",
            ),
        ),
    )
    payout = TopicFact(
        topic="annual leave encashment",
        status="decided",
        claims=(
            ClaimRef(
                "ADD-LEAVE-IN",
                "4. Leave encashment",
                date(2025, 1, 1),
                "addendum",
                "30 days",
                "Leave Policy Addendum: India",
            ),
        ),
    )
    facts = Facts(TODAY, (accrual, payout))
    kept = facts_for_question(facts, "How many days of annual leave do employees in my region get?")
    assert [t.topic for t in kept.topics] == ["annual leave accrual"]
    carry = facts_for_question(facts, "How many days of unused annual leave can I carry forward?")
    assert carry.topics == facts.topics  # no carry topic, so nothing is dropped


TBL_RATES = (
    "Annual Leave Accrual Schedule\n\n"
    "Row: Length of service Less than 2 years | Days earned per month 1.5 | Days earned per full year 18\n"
    "Row: Length of service 2 years to less than 5 years | Days earned per month 1.75 | Days earned per full year 21\n"
    "Row: Length of service 5 years or more | Days earned per month 2.0 | Days earned per full year 24\n"
)
V3_CHANGES = (
    "1. What changes\n\n"
    "Row: Topic Annual leave accrual, less than 2 years of service | Version 2 1.5 days per month | "
    "Version 3 (from 1 January 2027) 1.5 days per month (no change)\n"
    "Row: Topic Annual leave accrual, 2 to less than 5 years of service | Version 2 1.75 days per month | "
    "Version 3 (from 1 January 2027) 2.0 days per month\n"
    "Row: Topic Annual leave accrual, 5 or more years of service | Version 2 2.0 days per month | "
    "Version 3 (from 1 January 2027) 2.25 days per month\n"
)


def test_table_rows_answer_when_the_model_extracts_nothing_or_splits_bands():
    current = RetrievedChunk(
        "TBL-1", "TBL-LEAVE-ACCRUAL", "policy", ("GLOBAL",), date(2025, 1, 1),
        "1. Accrual rates", TBL_RATES, 80, 1.0, "all",
    )
    title = RetrievedChunk(
        "TBL-0", "TBL-LEAVE-ACCRUAL", "policy", ("GLOBAL",), date(2025, 1, 1),
        "Annual Leave Accrual Schedule", "Annual Leave Accrual Schedule\n", 10, 0.5, "all",
    )
    upcoming = RetrievedChunk(
        "V3-1", "POL-LEAVE-GLOBAL-V3", "policy", ("GLOBAL",), date(2027, 1, 1),
        "1. What changes", V3_CHANGES, 80, 1.0, "all",
    )
    chunks = [title, current, upcoming]
    sent = {(c.doc_id, c.locator) for c in chunks}
    raw = [
        {
            "topic": "annual leave accrual, less than 2 years of service",
            "value": "1.5 days per month",
            "doc_id": "POL-LEAVE-GLOBAL-V3",
            "locator": "1. What changes",
        }
    ]
    claims, shown, _dropped, names, extracted, model_kept = build_claims_report(raw, chunks, sent)
    assert extracted >= model_kept
    facts = resolve_facts(
        claims, shown, EMPLOYEE_IN, date(2026, 10, 15),
        not_covered=["my region", "annual leave accrual 2 to less than 5 years of service"],
        names=names,
    )
    assert facts.has_answer
    topic = facts.topics[0]
    assert topic.status == "decided"
    current_vals = {r.value for r in topic.claims}
    assert any("1.5" in v and "Less than 2 years" in v for v in current_vals)
    assert any("1.75" in v for v in current_vals)
    assert any("2.0" in v and "5 years or more" in v for v in current_vals)
    assert {r.doc_id for r in topic.claims} == {"TBL-LEAVE-ACCRUAL"}
    upcoming_vals = {r.value for r in topic.upcoming}
    assert any("2.0" in v and "2 to less than 5" in v for v in upcoming_vals)
    assert any("2.25" in v for v in upcoming_vals)
    assert not any("1.75" in v for v in upcoming_vals)  # Version 2 column is not the V3 rate
    assert facts.not_covered == ()
    answer = render_answer(facts)
    assert "1.5" in answer and "1.75" in answer and "2.25" in answer
    assert "2027-01-01" in answer
    assert "Annual Leave Accrual Schedule" in answer
    assert "settled by" not in answer.lower()
    assert "[POL-LEAVE" not in answer
    assert follows_facts(answer, facts)


def test_chunks_are_grouped_by_document_in_the_extract_corpus():
    chunks = by_id("POL-V2", "POL-V3", "POL-V1")
    grouped = group_chunks_by_document(chunks)
    assert {g[0].doc_id for g in grouped} == {c.doc_id for c in chunks}
    parts = corpus_of(chunks).parts
    headers = [p for p in parts if p.locator == "document"]
    assert {p.doc_id for p in headers} == {c.doc_id for c in chunks}
    assert all("effective" in p.text and p.doc_id in p.text for p in headers)


def test_skipped_numeric_documents_are_those_with_numbers_and_question_terms():
    chunks = by_id("POL-V2", "POL-REMOTE")
    sent = {(c.doc_id, c.locator) for c in chunks}
    skipped = skipped_numeric_documents(
        "How many days of annual leave do employees get?", chunks, sent, claimed_doc_ids=()
    )
    assert "POL-V2" in {c.doc_id for c in skipped}
    assert "POL-REMOTE" not in {c.doc_id for c in skipped}
    none = skipped_numeric_documents(
        "How many days of annual leave do employees get?", chunks, sent, claimed_doc_ids=["POL-V2"]
    )
    assert "POL-V2" not in {c.doc_id for c in none}
