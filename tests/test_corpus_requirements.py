"""TASK.pdf rules against the real assessment corpus (not the synthetic B4 numbers)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.corpus import LoadedDocument, load_manifest
from app.ingestion.holidays import extract_holidays
from app.ingestion.index import HybridIndex
from app.ingestion.parsers import parse_document
from app.orchestrator.evidence import fold_topic
from app.orchestrator.response import ROUTE_RAG
from app.retrieval.pipeline import retrieve_for_question
from app.retrieval.policy_resolver import Requester
from tests.fake_embedder import FakeEmbedder
from tests.mock_provider import MockProvider
from tests.test_llm_policy import ask

TODAY = date(2026, 10, 15)
PACK_ROOTS = (
    Path(__file__).resolve().parent.parent / "data",
    Path(__file__).resolve().parent.parent / "veltrix_assessment",
)


def _pack_root() -> Path | None:
    for root in PACK_ROOTS:
        if (root / "manifest.json").is_file() and (root / "corpus").is_dir():
            return root
    return None


def _chunks_for(doc_ids: set[str]):
    root = _pack_root()
    if root is None:
        pytest.skip("assessment corpus pack is not present")
    entries = {e.doc_id: e for e in load_manifest(root / "manifest.json")}
    chunks = []
    for doc_id in doc_ids:
        entry = entries.get(doc_id)
        if entry is None:
            continue
        path = root / "corpus" / entry.file
        if not path.is_file():
            continue
        parsed = parse_document(path)
        if not parsed.ok:
            continue
        chunks.extend(chunk_document(entry, parsed))
    if not chunks:
        pytest.skip("none of the requested corpus documents parsed")
    return chunks


def _index(doc_ids: set[str]) -> HybridIndex:
    return HybridIndex.build(_chunks_for(doc_ids), FakeEmbedder())


def _ask(tools_db, index, message: str, employee: str):
    return ask(
        tools_db,
        index,
        MockProvider(),
        message,
        employee=employee,
        today=TODAY,
    )


LEAVE_DOCS = {
    "POL-LEAVE-GLOBAL-V1",
    "POL-LEAVE-GLOBAL-V2",
    "POL-LEAVE-GLOBAL-V2-SUMMARY",
    "POL-LEAVE-GLOBAL-V3",
    "TBL-LEAVE-ACCRUAL",
    "ADD-LEAVE-IN",
    "ADD-LEAVE-US",
    "ADD-LEAVE-DE",
    "FAQ-LEAVE",
    "WIKI-LEAVE-HOWTO",
}


def test_india_annual_leave_uses_the_accrual_schedule_and_mentions_v3(tools_db):
    result = _ask(
        tools_db,
        _index(LEAVE_DOCS),
        "How many days of annual leave do employees in my region get?",
        "E1",
    )
    assert result.route == ROUTE_RAG
    assert "1.5" in result.answer and "1.75" in result.answer and "2.0" in result.answer
    cited = {c.doc_id for c in result.citations}
    assert "TBL-LEAVE-ACCRUAL" in cited
    assert "ADD-LEAVE-DE" not in cited
    assert "30" not in result.answer
    if "POL-LEAVE-GLOBAL-V3" in cited:
        assert "2.25" in result.answer and "2027-01-01" in result.answer


def test_germany_annual_leave_uses_the_regional_addendum_not_the_schedule(tools_db):
    result = _ask(
        tools_db,
        _index(LEAVE_DOCS),
        "How many days of annual leave do employees in my region get?",
        "E4",
    )
    assert result.route == ROUTE_RAG
    assert "30" in result.answer or "2.5" in result.answer
    cited = {c.doc_id for c in result.citations}
    assert "ADD-LEAVE-DE" in cited
    assert "TBL-LEAVE-ACCRUAL" not in cited
    assert "POL-LEAVE-GLOBAL-V3" not in cited
    assert "2027-01-01" not in result.answer


def test_india_carry_forward_uses_the_addendum_not_the_global_limit(tools_db):
    result = _ask(
        tools_db,
        _index(LEAVE_DOCS),
        "How many days of unused annual leave can I carry forward?",
        "E1",
    )
    assert result.route == ROUTE_RAG
    assert "15" in result.answer
    cited = {c.doc_id for c in result.citations}
    assert "ADD-LEAVE-IN" in cited
    assert "12" not in result.answer


def test_us_carry_forward_uses_the_addendum(tools_db):
    result = _ask(
        tools_db,
        _index(LEAVE_DOCS),
        "How many days of unused annual leave can I carry forward?",
        "E3",
    )
    assert result.route == ROUTE_RAG
    assert "5" in result.answer
    assert "ADD-LEAVE-US" in {c.doc_id for c in result.citations}


def test_sick_leave_for_india_follows_policy_not_the_faq(tools_db):
    result = _ask(tools_db, _index(LEAVE_DOCS), "How many sick days do I get?", "E1")
    assert result.route == ROUTE_RAG
    assert "12" in result.answer
    cited = {c.doc_id for c in result.citations}
    assert "POL-LEAVE-GLOBAL-V2" in cited or "POL-LEAVE-GLOBAL-V2-SUMMARY" in cited
    assert "FAQ-LEAVE" not in cited
    assert "WIKI-LEAVE-HOWTO" not in cited


def test_india_leave_question_retrieves_the_addendum_and_the_schedule():
    idx = _index(LEAVE_DOCS)
    result = retrieve_for_question(
        "How many days of annual leave do employees in my region get?",
        Requester("employee", "IN"),
        TODAY,
        12,
        index=idx,
    )
    seen = {c.doc_id for c in (*result.current, *result.upcoming)}
    assert "TBL-LEAVE-ACCRUAL" in seen
    assert "ADD-LEAVE-IN" in seen


def test_employee_cannot_retrieve_hr_or_manager_documents():
    docs = {
        "HR-COMP-BANDS-2026",
        "HB-MANAGER",
        "HR-PIP-GUIDE",
        "POL-LEAVE-GLOBAL-V2",
        "POL-CONDUCT",
    }
    chunks = _chunks_for(docs)
    if not any(c.access_level != "all" for c in chunks):
        pytest.skip("restricted corpus documents did not parse")
    idx = HybridIndex.build(chunks, FakeEmbedder())
    result = retrieve_for_question(
        "What are the salary bands and the manager handbook rules?",
        Requester("employee", "IN"),
        TODAY,
        12,
        index=idx,
    )
    seen = {c.doc_id for c in (*result.current, *result.upcoming)}
    assert "HR-COMP-BANDS-2026" not in seen
    assert "HB-MANAGER" not in seen
    assert "HR-PIP-GUIDE" not in seen


def test_india_2026_public_holidays_are_extracted_from_the_calendar():
    root = _pack_root()
    if root is None:
        pytest.skip("assessment corpus pack is not present")
    entries = [e for e in load_manifest(root / "manifest.json") if e.doc_id == "CAL-HOL-IN-2026"]
    if not entries:
        pytest.skip("India 2026 calendar is not in the pack")
    entry = entries[0]
    parsed = parse_document(root / "corpus" / entry.file)
    assert parsed.ok
    result = extract_holidays([entry], [LoadedDocument(entry, parsed)])
    public = {h.date for h in result.holidays if h.kind == "public"}
    optional = {h.date for h in result.holidays if h.kind == "optional"}
    assert "2026-01-26" in public
    assert "2026-10-20" in public
    assert "2026-03-04" in optional
    assert "2026-03-04" not in public


def test_fold_topic_keeps_carry_forward_apart_from_accrual():
    assert fold_topic("leave days") == "annual leave accrual"
    assert fold_topic("30 working days of paid annual leave") == "annual leave accrual"
    assert fold_topic("annual leave carry forward") == "annual leave carry forward"
    assert fold_topic("annual leave encashment") == "annual leave encashment"
    assert fold_topic("sick leave") == "sick leave"
