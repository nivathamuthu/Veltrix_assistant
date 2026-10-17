"""POST /chat with a mock provider: wiring, off-topic, failures, budget, restricted docs."""
from __future__ import annotations

import logging
import re

from app.llm.budget import MAX_CORPUS_TOKENS
from app.llm.providers.base import LLMUnavailable
from app.orchestrator.guards import OFF_TOPIC_MESSAGE
from app.orchestrator.response import UNAVAILABLE_MESSAGE
from tests.chat_helpers import assert_valid_chat_body
from tests.llm_helpers import SECRET, policy_docs
from tests.mock_provider import MockProvider, analysis, extraction
from tests.tool_helpers import pii_values


def body_of(response, status=200):
    assert response.status_code == status, (response.status_code, response.text)
    body = response.json()
    if status == 200:
        assert_valid_chat_body(body)
    return body


def _token_counts(caplog) -> list[int]:
    counts = []
    for record in caplog.records:
        msg = record.getMessage()
        if "corpus_tokens=" in msg:
            counts.append(int(msg.split("corpus_tokens=")[1].split()[0]))
    return counts


def test_off_topic_is_refused_with_what_the_assistant_can_help_with(make_chat):
    provider = MockProvider(analysis("off_topic"))
    chat = make_chat(provider=provider, extra_docs=policy_docs())
    body = body_of(chat.post("Write a python function to sort a list and tell me a joke."))
    assert body["route"] == "refuse" and body["answer"] == OFF_TOPIC_MESSAGE
    low = body["answer"].lower()
    for topic in ("leave balance", "apply", "cancel", "team leave", "holiday"):
        assert topic in low
    assert body["citations"] == [] and body["tool_calls"] == []


def test_a_provider_timeout_is_200_no_answer_and_does_not_write(make_chat):
    chat = make_chat(provider=MockProvider(error=LLMUnavailable("timeout")), extra_docs=policy_docs())
    before = chat.leave_tables()
    response = chat.post("How many days of annual leave do employees get?")
    body = body_of(response)
    assert body["route"] == "no_answer"
    assert body["answer"] == UNAVAILABLE_MESSAGE
    assert body["citations"] == [] and body["tool_calls"] == []
    assert "timeout" not in response.text
    assert "Traceback" not in response.text
    assert chat.leave_tables() == before and chat.proposals() == []


def test_invalid_analyzer_json_is_200_no_answer_and_does_not_write(make_chat):
    chat = make_chat(provider=MockProvider("not json at all"), extra_docs=policy_docs())
    before = chat.leave_tables()
    response = chat.post("How many days of annual leave do employees get?")
    body = body_of(response)
    assert body["route"] == "no_answer"
    assert body["answer"] == UNAVAILABLE_MESSAGE
    assert body["citations"] == [] and body["tool_calls"] == []
    assert "Traceback" not in response.text
    assert chat.leave_tables() == before and chat.proposals() == []


def test_invalid_extraction_json_is_200_no_answer(make_chat):
    provider = MockProvider(analysis("policy_question"), "<<<not json>>>")
    chat = make_chat(provider=provider, extra_docs=policy_docs())
    body = body_of(chat.post("How many days of annual leave do employees get?"))
    assert body["route"] == "no_answer"
    assert body["answer"] == UNAVAILABLE_MESSAGE
    assert body["citations"] == [] and body["tool_calls"] == []


def test_worked_example_through_the_api(make_chat):
    chat = make_chat(provider=MockProvider(), extra_docs=policy_docs())
    india = body_of(chat.post("How many days of annual leave do employees get?", employee="E1", session="in"))
    assert india["route"] == "rag"
    assert "3" in india["answer"]
    assert "4" in india["answer"] and "2027-01-01" in india["answer"]
    assert "POL-V2" in {c["doc_id"] for c in india["citations"]}
    assert "ADD-DE" not in {c["doc_id"] for c in india["citations"]}

    germany = body_of(chat.post("How many days of annual leave do employees get?", employee="E4", session="de"))
    assert "5" in germany["answer"]
    assert "ADD-DE" in {c["doc_id"] for c in germany["citations"]}
    assert "2027-01-01" not in germany["answer"]

    past = body_of(chat.post(
        "How many days of annual leave did employees get on 1 August 2025?", employee="E1", session="past"
    ))
    assert "2" in past["answer"]
    cited_past = {c["doc_id"] for c in past["citations"]}
    assert "POL-V1" in cited_past
    assert "ADD-DE" not in cited_past and "FAQ-1" not in cited_past


def test_the_memo_beats_remote_work_for_india_not_germany(make_chat):
    chat = make_chat(provider=MockProvider(), extra_docs=policy_docs())
    india = body_of(chat.post("How many days can I work from home each week?", employee="E1", session="a"))
    assert "3" in india["answer"] and "MEMO-RW" in {c["doc_id"] for c in india["citations"]}
    germany = body_of(chat.post("How many days can I work from home each week?", employee="E4", session="b"))
    assert "MEMO-RW" not in {c["doc_id"] for c in germany["citations"]}


def test_a_restricted_document_never_reaches_the_mock_or_the_response(make_chat):
    provider = MockProvider()
    chat = make_chat(provider=provider, extra_docs=policy_docs())
    response = chat.post("What are the salary bands?", employee="E1")
    body = body_of(response)
    assert SECRET not in provider.received_text
    assert "HR-SECRET" not in provider.received_text
    assert SECRET not in response.text
    assert "HR-SECRET" not in response.text
    assert "you can't access" not in body["answer"].lower()


def test_every_llm_call_logs_corpus_tokens_at_or_below_3000(make_chat, caplog):
    caplog.set_level(logging.INFO)
    chat = make_chat(provider=MockProvider(), extra_docs=policy_docs())
    body_of(chat.post("How many days of annual leave do employees get?"))
    counts = _token_counts(caplog)
    assert counts  # analyze (0) + extract + compose
    assert all(n <= MAX_CORPUS_TOKENS for n in counts)
    assert 0 in counts  # the analyzer call
    assert any(n > 0 for n in counts)  # extraction has corpus


def test_hybrid_through_the_api(make_chat):
    provider = MockProvider(
        analysis("policy_question", leave_type="annual", dates_text="by December", needs_leave_data=True)
    )
    chat = make_chat(provider=provider, extra_docs=policy_docs())
    body = body_of(chat.post("What will my annual leave balance be by December?"))
    assert body["route"] == "hybrid"
    assert body["tool_calls"] and body["tool_calls"][0]["name"] == "get_leave_balance"
    assert body["citations"]
    assert "10" in body["answer"]


def test_leave_writes_still_only_go_through_confirmation(make_chat):
    provider = MockProvider(
        analysis("apply_leave", leave_type="annual", dates_text="from 19 October to 21 October")
    )
    chat = make_chat(provider=provider, extra_docs=policy_docs())
    before = chat.leave_tables()
    proposal = body_of(chat.post("Apply for annual leave from 19 October to 21 October"))
    assert proposal["pending_confirmation"] is True
    assert proposal["tool_calls"][0]["status"] == "pending_confirmation"
    assert chat.leave_tables() == before
    # a second analyze is not needed: confirmation is deterministic
    provider.queue("should not be consumed")
    done = body_of(chat.post("yes"))
    assert done["tool_calls"][0]["status"] == "success"
    assert chat.balance("E1", "annual") == 7.0


def test_create_app_keeps_the_offline_analyzer_without_a_key(make_chat):
    # no provider, no key: policy questions stay the Phase 7a placeholder
    from app.orchestrator.analyzer import ScriptedAnalyzer
    from app.orchestrator.response import NO_ANSWER_MESSAGE

    chat = make_chat(analyzer=ScriptedAnalyzer())
    from app.orchestrator.analyzer import Understanding
    chat.analyzer.queue(Understanding(intent="policy_question"))
    body = body_of(chat.post("How many sick days do I get?"))
    assert body["route"] == "no_answer" and body["answer"] == NO_ANSWER_MESSAGE


def test_logs_from_an_llm_turn_contain_no_personal_data(make_chat, caplog):
    caplog.set_level(logging.DEBUG)
    chat = make_chat(provider=MockProvider(), extra_docs=policy_docs())
    chat.post("How many days of annual leave do Eli Fenn eli.fenn@veltrix.example +91 90000 00004 employees get?")
    text = "\n".join(r.getMessage() for r in caplog.records)
    names, emails, phones = pii_values(chat.db)
    for value in (*names, *emails, *phones, "eli.fenn", "90000"):
        assert value not in text
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)


def test_the_llm_is_not_sent_emails_or_phones(make_chat):
    provider = MockProvider()
    chat = make_chat(provider=provider, extra_docs=policy_docs())
    body_of(chat.post("How many days of annual leave do employees get?", employee="E1"))
    sent = provider.received_text
    names, emails, phones = pii_values(chat.db)
    for value in (*names, *emails, *phones):
        assert value not in sent
    assert "Eli Fenn" not in sent
    assert "+91" not in sent
    assert "requester_role: employee" in sent
    assert "requester_region: IN" in sent
