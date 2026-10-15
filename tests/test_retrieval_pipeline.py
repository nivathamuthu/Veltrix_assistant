"""Pre-filtered retrieval for policy questions (Phase 5): CURRENT and UPCOMING sets."""
from __future__ import annotations

import os
import re
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import app.ingestion.index as index_module
from app.ingestion.corpus import load_corpus
from app.ingestion.embeddings import EmbeddingModelUnavailable, FastEmbedEmbedder
from app.ingestion.index import HybridIndex, SearchFilters
from app.ingestion.manifest import load_manifest
from app.llm.budget import CorpusContext
from app.llm.fake_provider import FakeProvider
from app.llm.guard import call_with_corpus
from app.retrieval.policy_resolver import DocMeta, Requester, is_eligible, readable_access_levels
from app.retrieval.pipeline import Retrieval, RetrievedChunk, retrieve_for_question, select_diverse
from tests.fake_embedder import FakeEmbedder
from tests.test_corpus_loading import write_corpus
from tests.test_index import chunk

REF = date(2026, 9, 15)
SECRET = "Zebraquartz salary midpoint for senior engineers is confidential to the compensation committee."

EMPLOYEE_IN = Requester("employee", "IN")
EMPLOYEE_US = Requester("employee", "US")
EMPLOYEE_DE = Requester("employee", "DE")
MANAGER_IN = Requester("manager", "IN")
HR_IN = Requester("hr", "IN")

CHUNKS = [
    chunk("POL-REMOTE::1", "Remote work policy: employees may work from home two days each week.",
          doc_id="POL-REMOTE", doc_type="policy", date="2025-01-01"),
    chunk("MEMO-RW::1", "Memo remote work update: employees in India and the United States may work from "
          "home three days each week.", doc_id="MEMO-RW", doc_type="memo", regions=("IN", "US"), date="2026-07-01"),
    chunk("FAQ-RW::1", "FAQ remote work: how many days can I work from home? Two days each week.",
          doc_id="FAQ-RW", doc_type="faq", date="2026-05-01"),
    chunk("DE-RW::1", "Germany remote work addendum: work from home days each week follow works council rules.",
          doc_id="DE-RW", doc_type="addendum", regions=("DE",), date="2025-01-01"),
    chunk("FUT-RW::1", "Future remote work policy version: work from home four days each week from next year.",
          doc_id="FUT-RW", doc_type="policy", date="2027-01-01"),
    chunk("FUT-DE::1", "Future Germany remote work rule: work from home five days each week.",
          doc_id="FUT-DE", doc_type="addendum", regions=("DE",), date="2027-03-01"),
    chunk("HR-SECRET::1", SECRET, doc_id="HR-SECRET", access="hr", date="2026-01-01"),
    chunk("MGR-RW::1", "Manager guide remote work: approve work from home days for the team each week.",
          doc_id="MGR-RW", access="managers", date="2026-01-01"),
    chunk("HR-FUT::1", "Future hr only remote work calibration: work from home exceptions for hr cases.",
          doc_id="HR-FUT", access="hr", date="2027-02-01"),
    chunk("TRAVEL::1", "Travel expense claims must be filed within thirty days with receipts attached.",
          doc_id="TRAVEL", date="2026-01-01"),
    chunk("BENEFITS-US::1", "United States health insurance enrolment opens each November.",
          doc_id="BENEFITS-US", regions=("US",), date="2026-01-01"),
]


@pytest.fixture()
def index() -> HybridIndex:
    return HybridIndex.build(CHUNKS, FakeEmbedder())


def docs(chunks) -> set[str]:
    return {c.doc_id for c in chunks}


def get(index, query="work from home days each week", who=EMPLOYEE_IN, ref=REF, k=50) -> Retrieval:
    return retrieve_for_question(query, who, ref, k, index=index)


# --- the shape of the result ---------------------------------------------------------------------


def test_the_result_has_two_sets_of_retrieved_chunks_with_all_the_fields(index):
    result = get(index)
    assert isinstance(result, Retrieval)
    assert result.current and result.upcoming
    for c in (*result.current, *result.upcoming):
        assert isinstance(c, RetrievedChunk)
        assert c.doc_id and c.doc_type and c.regions and c.locator and c.text
        assert isinstance(c.effective_date, date) and isinstance(c.token_count, int) and c.token_count > 0
        assert isinstance(c.score, float) and c.score > 0
    first = result.current[0]
    assert first.token_count == next(x.token_count for x in CHUNKS if x.chunk_id == first.chunk_id)


def test_each_set_is_best_first_and_has_no_duplicates(index):
    result = get(index)
    for chunks in (result.current, result.upcoming):
        scores = [c.score for c in chunks]
        assert scores == sorted(scores, reverse=True)
        assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_the_two_sets_never_share_a_chunk(index):
    result = get(index)
    assert not ({c.chunk_id for c in result.current} & {c.chunk_id for c in result.upcoming})


def test_k_limits_each_set(index):
    result = get(index, k=3)
    assert len(result.current) <= 3 and len(result.upcoming) <= 3
    assert len(get(index, k=1).current) == 1


@pytest.mark.parametrize("k", [0, -1, 2.5, "3", None, True])
def test_k_must_be_a_positive_integer(index, k):
    with pytest.raises((ValueError, TypeError)):
        retrieve_for_question("remote work", EMPLOYEE_IN, REF, k, index=index)


def test_an_empty_or_blank_query_gives_two_empty_sets(index):
    for query in ("", "   ", "\n"):
        assert get(index, query=query) == Retrieval(current=(), upcoming=())


def test_retrieved_chunks_convert_to_doc_meta_for_the_resolver(index):
    c = next(c for c in get(index).current if c.doc_id == "MEMO-RW")
    meta = c.doc_meta()
    assert meta == DocMeta("MEMO-RW", "memo", ("IN", "US"), date(2026, 7, 1), "all")
    assert is_eligible(meta, "employee", "IN", REF)


# --- restricted chunks never appear ------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        SECRET,  # verbatim
        "zebraquartz",
        "Zebraquartz salary midpoint for senior engineers",
        "Manager guide remote work: approve work from home days for the team each week.",
        "Future hr only remote work calibration: work from home exceptions for hr cases.",
    ],
)
def test_a_restricted_chunk_never_appears_in_either_set_for_an_employee_even_for_a_verbatim_query(index, query):
    result = get(index, query=query, who=EMPLOYEE_IN)
    seen = docs(result.current) | docs(result.upcoming)
    assert not seen & {"HR-SECRET", "MGR-RW", "HR-FUT"}
    text = " ".join(c.text for c in (*result.current, *result.upcoming))
    assert "zebraquartz" not in text.lower()
    assert all(c.access_level == "all" for c in (*result.current, *result.upcoming))


def test_control_the_same_queries_do_find_those_chunks_when_access_is_granted(index):
    """Makes the test above meaningful: the verbatim query really does match."""
    assert get(index, query=SECRET, who=HR_IN).current[0].doc_id == "HR-SECRET"
    manager_hit = get(index, query="Manager guide remote work: approve work from home days for the team",
                      who=MANAGER_IN)
    assert manager_hit.current[0].doc_id == "MGR-RW"
    assert "HR-FUT" in docs(get(index, query="hr only remote work calibration exceptions", who=HR_IN).upcoming)


def test_a_manager_gets_manager_chunks_but_not_hr_chunks(index):
    result = get(index, who=MANAGER_IN)
    seen = docs(result.current) | docs(result.upcoming)
    assert "MGR-RW" in seen
    assert not seen & {"HR-SECRET", "HR-FUT"}
    assert "HR-SECRET" not in docs(get(index, query=SECRET, who=MANAGER_IN).current)


def test_an_unknown_role_gets_nothing_at_all(index):
    for role in ("admin", "", "contractor", None):
        assert get(index, who=Requester(role, "IN")) == Retrieval(current=(), upcoming=())


def test_the_requesters_region_is_required(index):
    for region in ("", "  ", None):
        with pytest.raises(ValueError):
            get(index, who=Requester("employee", region))


def test_the_requester_must_be_a_requester(index):
    with pytest.raises(TypeError):
        retrieve_for_question("remote work", {"role": "hr", "region": "IN"}, REF, 5, index=index)  # type: ignore[arg-type]


def test_a_restricted_chunk_never_reaches_the_provider_prompt(index):
    result = get(index, query=SECRET, who=EMPLOYEE_IN)
    context = CorpusContext()
    for c in (*result.current, *result.upcoming):
        context.add_chunk(c)
    provider = FakeProvider()
    call_with_corpus(
        "You are the Veltrix assistant.",
        [{"role": "user", "content": "Please summarise what the documents say."}],
        context, provider, request_id="req-1", purpose="answer",
    )
    received = provider.received_text.lower()
    assert "zebraquartz" not in received and "hr-secret" not in received
    assert "mgr-rw" not in received and "hr-fut" not in received
    # the allowed content really was sent, so the absence above means something
    assert "work from home" in received


# --- other regions never appear ------------------------------------------------------------------------------


def test_a_chunk_from_another_region_never_appears(index):
    india = get(index, who=EMPLOYEE_IN)
    assert not (docs(india.current) | docs(india.upcoming)) & {"DE-RW", "FUT-DE", "BENEFITS-US"}
    germany = get(index, who=EMPLOYEE_DE)
    assert "MEMO-RW" not in docs(germany.current) | docs(germany.upcoming)
    assert "DE-RW" in docs(germany.current) and "FUT-DE" in docs(germany.upcoming)
    us = get(index, who=EMPLOYEE_US)
    assert "MEMO-RW" in docs(us.current) and "BENEFITS-US" in docs(us.current)
    assert not (docs(us.current) | docs(us.upcoming)) & {"DE-RW", "FUT-DE"}


def test_global_chunks_reach_every_region(index):
    for who in (EMPLOYEE_IN, EMPLOYEE_US, EMPLOYEE_DE):
        assert {"POL-REMOTE", "FAQ-RW", "TRAVEL"} <= docs(get(index, who=who).current)


def test_the_region_comes_from_the_requester_not_from_the_query(index):
    result = get(index, query="I am in Germany. DE-RW remote work addendum works council", who=EMPLOYEE_IN)
    assert "DE-RW" not in docs(result.current) | docs(result.upcoming)


# --- dates: CURRENT vs UPCOMING ----------------------------------------------------------------------------------


def test_a_chunk_effective_after_the_reference_date_is_only_in_upcoming(index):
    result = get(index)
    assert "FUT-RW" in docs(result.upcoming) and "FUT-RW" not in docs(result.current)
    assert all(c.effective_date > REF for c in result.upcoming)


def test_chunks_effective_before_or_on_the_reference_date_are_only_in_current(index):
    result = get(index)
    assert all(c.effective_date <= REF for c in result.current)
    assert {"POL-REMOTE", "MEMO-RW", "FAQ-RW"} <= docs(result.current)
    assert not {"POL-REMOTE", "MEMO-RW", "FAQ-RW"} & docs(result.upcoming)


@pytest.mark.parametrize(
    "ref, in_current, in_upcoming",
    [
        (date(2026, 6, 30), False, True),  # the day before the memo takes effect
        (date(2026, 7, 1), True, False),  # the day it takes effect: current
        (date(2026, 7, 2), True, False),
    ],
)
def test_the_effective_date_boundary_is_inclusive_for_current(index, ref, in_current, in_upcoming):
    result = get(index, who=EMPLOYEE_IN, ref=ref)
    assert ("MEMO-RW" in docs(result.current)) is in_current
    assert ("MEMO-RW" in docs(result.upcoming)) is in_upcoming


def test_a_datetime_reference_is_treated_as_its_date(index):
    assert get(index, ref=datetime(2026, 7, 1, 23, 59)) == get(index, ref=date(2026, 7, 1))


@pytest.mark.parametrize("bad", ["2026-09-15", None, 20260915])
def test_the_reference_date_must_be_a_date(index, bad):
    with pytest.raises(TypeError):
        retrieve_for_question("remote work", EMPLOYEE_IN, bad, 5, index=index)


def test_a_reference_date_before_everything_gives_an_empty_current_set(index):
    result = get(index, ref=date(2000, 1, 1))
    assert result.current == () and result.upcoming


def test_a_reference_date_after_everything_gives_an_empty_upcoming_set(index):
    result = get(index, ref=date(2035, 1, 1))
    assert result.upcoming == () and result.current


# --- the pre-filter: scoring only ever sees eligible chunks --------------------------------------------------------


def _eligible(who: Requester, low: date | None, high: date | None) -> int:
    levels = readable_access_levels(who.role)
    count = 0
    for c in CHUNKS:
        d = date.fromisoformat(c.effective_date)
        if c.access_level not in levels or not ({"GLOBAL", who.region} & set(c.regions)):
            continue
        if (low and d < low) or (high and d > high):
            continue
        count += 1
    return count


def test_only_eligible_rows_are_scored_for_each_of_the_two_searches(index, monkeypatch):
    seen_rows: list[int] = []
    real = index_module._cosine_scores

    def spy(matrix, query_vector):
        seen_rows.append(matrix.shape[0])
        return real(matrix, query_vector)

    monkeypatch.setattr(index_module, "_cosine_scores", spy)
    spy_bm25_sizes: list[int] = []
    real_bm25 = index_module.BM25Okapi

    class SpyBM25(real_bm25):
        def __init__(self, corpus, *a, **kw):
            spy_bm25_sizes.append(len(corpus))
            super().__init__(corpus, *a, **kw)

    monkeypatch.setattr(index_module, "BM25Okapi", SpyBM25)

    get(index, who=EMPLOYEE_IN, k=5)
    expected = [_eligible(EMPLOYEE_IN, None, REF), _eligible(EMPLOYEE_IN, date(2026, 9, 16), None)]
    assert seen_rows == expected
    assert sorted(spy_bm25_sizes) == sorted(expected)
    assert max(expected) < len(CHUNKS)  # restricted, other-region and out-of-range chunks were never rows


# --- one document must not crowd out the others -------------------------------------------------------------------------

QUERY = "carry over of unused annual leave days"


def versions_index() -> HybridIndex:
    chunks = [
        chunk(f"POL-A::{i}", f"{QUERY}. {QUERY}. Rule {i}: {QUERY} to the next year.", doc_id="POL-A",
              date="2020-01-01")
        for i in range(6)
    ]
    for v, year in enumerate(range(2021, 2026), start=1):
        chunks.append(
            chunk(f"POL-V{v}::1", f"Policy version {v}: unused annual leave may be carried over, up to {v + 2} days.",
                  doc_id=f"POL-V{v}", date=f"{year}-01-01")
        )
    chunks += [chunk(f"OTHER-{i}::1", f"Unrelated topic number {i}: office parking and printers.", doc_id=f"OTHER-{i}")
               for i in range(5)]
    return HybridIndex.build(chunks, FakeEmbedder())


def test_control_a_plain_search_lets_one_documents_chunks_crowd_out_the_versions():
    idx = versions_index()
    plain = idx.search(QUERY, SearchFilters(allowed_access_levels=frozenset({"all"}), region="IN"), 6)
    assert {r.doc_id for r in plain} == {"POL-A"}  # all six slots taken by one document


def test_with_five_versions_of_one_policy_all_five_are_represented():
    idx = versions_index()
    result = retrieve_for_question(QUERY, EMPLOYEE_IN, REF, 6, index=idx)
    represented = docs(result.current)
    assert {f"POL-V{v}" for v in range(1, 6)} <= represented
    assert "POL-A" in represented  # and the strongest document is still there
    assert len(result.current) == 6


def test_the_top_chunks_of_each_document_come_first_then_remaining_slots_go_by_score():
    idx = versions_index()
    result = retrieve_for_question(QUERY, EMPLOYEE_IN, REF, 9, index=idx)
    per_doc = {}
    for c in result.current:
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
    assert all(per_doc.get(f"POL-V{v}") == 1 for v in range(1, 6))
    assert per_doc["POL-A"] >= 1 and len(result.current) == 9


def test_versions_and_types_of_one_topic_all_appear_when_k_allows(index):
    result = get(index, k=10)
    assert {"POL-REMOTE", "MEMO-RW", "FAQ-RW"} <= docs(result.current)  # policy, memo and faq of one topic


# --- select_diverse on its own --------------------------------------------------------------------------------------------


def ranked(*spec):
    """spec: (doc_id, chunk_id) in rank order (best first); scores fall with rank."""
    return [SimpleNamespace(doc_id=d, chunk_id=c, score=1.0 - i / 100) for i, (d, c) in enumerate(spec)]


def test_select_diverse_takes_each_documents_best_chunk_before_second_chunks():
    results = ranked(("A", "A1"), ("A", "A2"), ("A", "A3"), ("B", "B1"), ("C", "C1"), ("B", "B2"))
    picked = select_diverse(results, 3)
    assert [r.chunk_id for r in picked] == ["A1", "B1", "C1"]


def test_select_diverse_fills_the_rest_by_score():
    results = ranked(("A", "A1"), ("A", "A2"), ("B", "B1"), ("A", "A3"), ("C", "C1"), ("B", "B2"))
    picked = select_diverse(results, 5)
    assert {r.chunk_id for r in picked} == {"A1", "B1", "C1", "A2", "A3"}
    assert [r.chunk_id for r in picked] == ["A1", "A2", "B1", "A3", "C1"]  # back in rank order


def test_select_diverse_with_fewer_slots_than_documents_keeps_the_best_documents():
    results = ranked(*[(f"D{i}", f"D{i}::1") for i in range(6)])
    assert [r.doc_id for r in select_diverse(results, 3)] == ["D0", "D1", "D2"]


def test_select_diverse_never_exceeds_k_or_repeats_and_handles_small_inputs():
    results = ranked(("A", "A1"), ("B", "B1"))
    assert select_diverse(results, 10) == results
    assert select_diverse([], 5) == []
    assert select_diverse(results, 1) == results[:1]
    many = ranked(*[("A", f"A{i}") for i in range(20)])
    assert [r.chunk_id for r in select_diverse(many, 4)] == ["A0", "A1", "A2", "A3"]


# --- a modified copy of the corpus changes what is retrieved ---------------------------------------------------------------------------


def test_a_modified_copy_of_the_corpus_changes_the_retrieved_text(tmp_path):
    root = write_corpus(
        tmp_path,
        {
            "leave.md": (
                "# Leave Rules\n\nStaff may carry over 28 days of leave into the next year.\n",
                {"doc_id": "D-LEAVE"},
            ),
            "other.md": ("# Parking\n\nParking permits are issued by facilities.\n", {"doc_id": "D-OTHER"}),
        },
    )

    def retrieve() -> str:
        corpus = load_corpus(root / "corpus", root / "manifest.json")
        idx = HybridIndex.build(corpus.chunks, FakeEmbedder())
        result = retrieve_for_question("carry over days of leave", EMPLOYEE_IN, REF, 5, index=idx)
        return " ".join(c.text for c in result.current)

    before = retrieve()
    assert "28 days" in before and "35 days" not in before
    path = root / "corpus" / "leave.md"
    path.write_text(path.read_text(encoding="utf-8").replace("28 days", "35 days"), encoding="utf-8")
    after = retrieve()
    assert "35 days" in after and "28 days" not in after


# --- the real corpus (skipped when data/ is absent) --------------------------------------------------------------------------------------

DATA = Path(__file__).resolve().parent.parent / "data"
# Read at import: the autouse fixture in conftest clears the environment per test.
CACHE_DIR = Path(os.environ.get("EMBED_CACHE_DIR", "./models"))
needs_data = pytest.mark.skipif(not (DATA / "manifest.json").is_file(), reason="data/ not present")


@pytest.fixture(scope="module")
def real_corpus():
    return load_corpus(DATA / "corpus", DATA / "manifest.json")


@pytest.fixture(scope="module")
def real_embedder():
    try:
        return FastEmbedEmbedder("BAAI/bge-small-en-v1.5", CACHE_DIR)
    except EmbeddingModelUnavailable:
        pytest.skip(f"embedding model not available in {CACHE_DIR}")


@pytest.fixture(scope="module")
def real_index(real_corpus, real_embedder):
    return HybridIndex.build(real_corpus.chunks, real_embedder)


@needs_data
def test_real_restricted_documents_never_appear_for_an_employee_even_for_their_own_text(real_corpus):
    restricted = [c for c in real_corpus.chunks if c.access_level != "all"]
    if not restricted:
        pytest.skip("no restricted chunks in the corpus")
    idx = HybridIndex.build(real_corpus.chunks, FakeEmbedder())
    forbidden = {c.chunk_id for c in restricted}
    for c in restricted[:12]:
        for region in ("IN", "US", "DE"):
            result = retrieve_for_question(c.text, Requester("employee", region), date(2026, 9, 15), 20, index=idx)
            assert not forbidden & {x.chunk_id for x in (*result.current, *result.upcoming)}
    # control: the same verbatim query finds it for a role that may read it
    hr = retrieve_for_question(restricted[0].text, HR_IN, date(2030, 1, 1), 5, index=idx)
    assert restricted[0].chunk_id in {x.chunk_id for x in hr.current}


def _find_memo_and_policy():
    entries = load_manifest(DATA / "manifest.json")
    memo = [e for e in entries if e.doc_type == "memo" and {"IN", "US"} <= set(e.regions)]
    policy = [e for e in entries if e.doc_type == "policy" and "remote" in e.title.lower() and e.regions == ("GLOBAL",)]
    return (memo[0] if memo else None), (policy[0] if policy else None)


@needs_data
def test_real_corpus_smoke_work_from_home_for_india_and_the_us(real_corpus, real_index):
    memo, policy = _find_memo_and_policy()
    if memo is None or policy is None:
        pytest.skip("remote work memo/policy not in this corpus")
    if memo.doc_id not in real_corpus.chunks_per_doc:
        pytest.skip("the memo is a scan and needs OCR (tesseract) to be read")

    india = retrieve_for_question("work from home days", EMPLOYEE_IN, date(2026, 9, 15), 8, index=real_index)
    assert {memo.doc_id, policy.doc_id} <= docs(india.current)
    assert memo.doc_id not in docs(india.upcoming)

    us_before = retrieve_for_question("work from home days", EMPLOYEE_US, date(2026, 6, 1), 8, index=real_index)
    assert memo.doc_id in docs(us_before.upcoming) and memo.doc_id not in docs(us_before.current)
    assert policy.doc_id in docs(us_before.current)

    germany = retrieve_for_question("work from home days", EMPLOYEE_DE, date(2026, 9, 15), 8, index=real_index)
    assert memo.doc_id not in docs(germany.current) | docs(germany.upcoming)  # not a DE document
