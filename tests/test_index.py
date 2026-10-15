"""Hybrid index: pre-filtering, ranking, persistence (Phase 3b). Synthetic corpus."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

import app.ingestion.index as index_module
from app.db.models import Chunk, Segment
from app.ingestion.index import RRF_K, HybridIndex, SearchFilters, tokenize
from app.tokens import count_tokens
from tests.fake_embedder import FakeEmbedder

RESTRICTED_TEXT = (
    "Compensation band for senior engineers: midpoint zebraquartz salary figure "
    "is confidential to the people team."
)


def chunk(chunk_id, text, *, access="all", regions=("GLOBAL",), date="2026-01-01", doc_id=None,
          doc_type="policy", locator="Heading", segments=()):
    return Chunk(
        chunk_id=chunk_id, doc_id=doc_id or chunk_id.split("::")[0], locator=locator,
        regions=tuple(regions), effective_date=date, doc_type=doc_type, access_level=access,
        text=text, token_count=count_tokens(text), segments=tuple(segments),
    )


CHUNKS = [
    chunk("LEAVE::001", "Annual leave accrues monthly for every employee. Unused leave carries over."),
    chunk("HR-ONLY::001", RESTRICTED_TEXT, access="hr"),
    chunk("MGR::001", "Team approval workflow: managers approve leave requests of direct reports.",
          access="managers"),
    chunk("SICK-DE::001", "Statutory sick leave in Germany needs a doctor certificate after three days.",
          regions=("DE",)),
    chunk("HOL-IN::001", "Public holidays India: Diwali, Holi and Republic Day are office closed days.",
          regions=("IN",)),
    chunk("FUTURE::001", "Future leave policy version effective next year replaces the current one.",
          date="2027-01-01"),
    chunk("MULTI::001", "Benefits enrolment applies to employees in India and the United States only.",
          regions=("IN", "US")),
]
RESTRICTED_ID = "HR-ONLY::001"

EMPLOYEE = SearchFilters(allowed_access_levels=frozenset({"all"}), region="IN")
MANAGER = SearchFilters(allowed_access_levels=frozenset({"all", "managers"}), region="IN")
HR = SearchFilters(allowed_access_levels=frozenset({"all", "managers", "hr"}), region="IN")


@pytest.fixture()
def index() -> HybridIndex:
    return HybridIndex.build(CHUNKS, FakeEmbedder())


def ids(results):
    return [r.chunk_id for r in results]


# --- the filter is required -------------------------------------------------


def test_search_cannot_be_called_without_filters(index):
    with pytest.raises(TypeError):
        index.search("annual leave", k=3)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        index.search("annual leave", None, 3)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        index.search("annual leave", {"allowed_access_levels": ["all"]}, 3)  # type: ignore[arg-type]


def test_filters_cannot_be_built_without_levels_or_region():
    with pytest.raises(TypeError):
        SearchFilters()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        SearchFilters(region="IN")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        SearchFilters(allowed_access_levels=frozenset({"all"}))  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        SearchFilters(allowed_access_levels=frozenset({"all"}), region="  ")
    with pytest.raises(TypeError):
        SearchFilters(allowed_access_levels="all", region="IN")  # type: ignore[arg-type]


def test_no_allowed_access_level_means_no_results(index):
    nothing = SearchFilters(allowed_access_levels=frozenset(), region="IN")
    assert index.search("annual leave", nothing, 5) == []


# --- restricted chunks never come back, even for a verbatim query ------------


@pytest.mark.parametrize("query", [RESTRICTED_TEXT, "zebraquartz", "midpoint zebraquartz salary figure"])
def test_restricted_chunk_is_never_returned_even_for_its_own_text(index, query):
    for filters in (EMPLOYEE, MANAGER):
        results = index.search(query, filters, 10)
        assert RESTRICTED_ID not in ids(results)
        assert all(r.access_level in filters.allowed_access_levels for r in results)
        assert "zebraquartz" not in " ".join(r.text for r in results)


def test_control_the_same_query_does_find_it_when_access_is_granted(index):
    """Makes the tests above meaningful: the query really does match that chunk."""
    assert ids(index.search(RESTRICTED_TEXT, HR, 3))[0] == RESTRICTED_ID
    assert ids(index.search("zebraquartz", HR, 3))[0] == RESTRICTED_ID


def test_manager_content_is_hidden_from_employees_only(index):
    query = "managers approve leave requests of direct reports"
    assert "MGR::001" not in ids(index.search(query, EMPLOYEE, 10))
    assert ids(index.search(query, MANAGER, 10))[0] == "MGR::001"


def test_restricted_chunk_is_never_scored_bm25_path(index, monkeypatch):
    """BM25 statistics are built from the eligible chunks only."""
    seen_corpora: list[list[list[str]]] = []
    real = index_module.BM25Okapi

    class SpyBM25(real):
        def __init__(self, corpus, *args, **kwargs):
            seen_corpora.append([list(doc) for doc in corpus])
            super().__init__(corpus, *args, **kwargs)

    monkeypatch.setattr(index_module, "BM25Okapi", SpyBM25)
    results = index.search(RESTRICTED_TEXT, EMPLOYEE, 10)

    assert seen_corpora, "BM25 was expected to run"
    for corpus in seen_corpora:
        assert not any("zebraquartz" in doc for doc in corpus)  # restricted tokens absent
        # an employee in IN reads: LEAVE, HOL-IN, FUTURE, MULTI (not HR-ONLY, MGR, SICK-DE)
        assert len(corpus) == 4
    assert all(r.bm25_score is None or math.isfinite(r.bm25_score) for r in results)


def test_restricted_chunk_is_never_scored_vector_path(index, monkeypatch):
    """Only eligible rows of the matrix ever reach the dot product."""
    seen_matrices: list[np.ndarray] = []
    real = index_module._cosine_scores

    def spy(matrix, query_vector):
        seen_matrices.append(matrix.copy())
        return real(matrix, query_vector)

    monkeypatch.setattr(index_module, "_cosine_scores", spy)
    embedder = FakeEmbedder()
    restricted_vector = embedder.embed_passages([RESTRICTED_TEXT])[0]
    index.search(RESTRICTED_TEXT, EMPLOYEE, 10)

    (matrix,) = seen_matrices
    eligible = index._eligible_positions(EMPLOYEE)
    assert matrix.shape[0] == len(eligible) < len(CHUNKS)
    assert not any(np.allclose(row, restricted_vector) for row in matrix)


def test_restricted_chunk_data_is_never_even_read(index):
    """Poison the restricted chunk's vector and tokens: if anything touched them
    during an employee search the result would be NaN or an exception."""

    class Poison(list):
        def __iter__(self):
            raise AssertionError("restricted tokens were read")

        def __contains__(self, item):
            raise AssertionError("restricted tokens were read")

    pos = [c.chunk_id for c in CHUNKS].index(RESTRICTED_ID)
    index._vectors[pos, :] = np.nan
    index._tokens[pos] = Poison()
    for query in (RESTRICTED_TEXT, "zebraquartz", "annual leave", "public holidays India"):
        results = index.search(query, EMPLOYEE, 10)
        assert results
        assert all(math.isfinite(r.score) for r in results)
        assert all(r.vector_score is None or math.isfinite(r.vector_score) for r in results)


def test_restricted_chunks_do_not_influence_scores_of_visible_ones(index):
    """Scores equal those of an index that never contained the restricted chunk
    (the BM25 IDF would differ if the restricted chunk were in the statistics)."""
    visible_only = HybridIndex.build([c for c in CHUNKS if c.access_level == "all"], FakeEmbedder())
    for query in ("annual leave carries over", "sick leave certificate", RESTRICTED_TEXT):
        a = index.search(query, EMPLOYEE, 10)
        b = visible_only.search(query, EMPLOYEE, 10)
        assert ids(a) == ids(b)
        assert [r.score for r in a] == pytest.approx([r.score for r in b])
        def bm25(results):
            return [-1.0 if r.bm25_score is None else r.bm25_score for r in results]

        assert bm25(a) == pytest.approx(bm25(b))


# --- region and date filters -------------------------------------------------


def test_region_filter_keeps_global_and_own_region_only(index):
    de = SearchFilters(allowed_access_levels=frozenset({"all"}), region="DE")
    in_ = EMPLOYEE
    de_ids = set(ids(index.search("leave holidays benefits sick", de, 10)))
    in_ids = set(ids(index.search("leave holidays benefits sick", in_, 10)))
    assert "SICK-DE::001" in de_ids and "HOL-IN::001" not in de_ids and "MULTI::001" not in de_ids
    assert "HOL-IN::001" in in_ids and "MULTI::001" in in_ids and "SICK-DE::001" not in in_ids
    assert "LEAVE::001" in de_ids and "LEAVE::001" in in_ids  # GLOBAL reaches everyone


def test_region_matching_ignores_case(index):
    lower = SearchFilters(allowed_access_levels=frozenset({"all"}), region="in")
    assert "HOL-IN::001" in ids(index.search("public holidays India", lower, 5))


def test_unknown_region_only_sees_global_chunks(index):
    other = SearchFilters(allowed_access_levels=frozenset({"all"}), region="BR")
    results = index.search("leave holidays benefits sick", other, 10)
    assert all("GLOBAL" in r.regions for r in results)


def test_max_effective_date_excludes_later_chunks(index):
    query = "future leave policy version effective next year"
    assert "FUTURE::001" in ids(index.search(query, EMPLOYEE, 10))
    capped = SearchFilters(
        allowed_access_levels=frozenset({"all"}), region="IN", max_effective_date="2026-10-15"
    )
    assert "FUTURE::001" not in ids(index.search(query, capped, 10))


def test_max_effective_date_must_be_a_date():
    with pytest.raises(ValueError):
        SearchFilters(allowed_access_levels=frozenset({"all"}), region="IN", max_effective_date="soon")


# --- ranking ------------------------------------------------------------------


def test_results_carry_the_requested_fields_and_are_sorted_best_first(index):
    results = index.search("annual leave carries over", EMPLOYEE, 4)
    assert 1 <= len(results) <= 4
    for r in results:
        assert r.doc_id and r.locator and r.text and r.token_count == count_tokens(r.text)
        assert r.chunk_id.startswith(r.doc_id)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert results[0].chunk_id == "LEAVE::001"


def test_a_chunk_ranked_first_by_both_methods_gets_the_full_rrf_score(index):
    top = index.search("Annual leave accrues monthly for every employee", EMPLOYEE, 3)[0]
    assert top.chunk_id == "LEAVE::001"
    assert top.score == pytest.approx(2.0 / (RRF_K + 1))


def test_hybrid_finds_what_only_one_method_can(index):
    # words absent from the corpus: BM25 has nothing, vectors still rank something
    results = index.search("qwertyuiop asdfghjkl", EMPLOYEE, 3)
    assert results and all(r.bm25_score is None for r in results)
    # an exact rare word: BM25 puts the matching chunk first
    assert index.search("Diwali", EMPLOYEE, 3)[0].chunk_id == "HOL-IN::001"


def test_search_is_deterministic(index):
    first = [(r.chunk_id, r.score) for r in index.search("leave policy", EMPLOYEE, 5)]
    for _ in range(3):
        assert [(r.chunk_id, r.score) for r in index.search("leave policy", EMPLOYEE, 5)] == first


def test_k_limits_results_and_blank_queries_return_nothing(index):
    assert len(index.search("leave", EMPLOYEE, 2)) == 2
    assert index.search("leave", EMPLOYEE, 0) == []
    assert index.search("   ", EMPLOYEE, 3) == []


def test_empty_index_returns_nothing():
    empty = HybridIndex.build([], FakeEmbedder())
    assert empty.search("anything", EMPLOYEE, 3) == []
    assert len(empty) == 0


def test_locator_is_the_heading_of_the_best_matching_section_in_a_merged_chunk():
    text = "Q1. Parking\n\nBring a permit for the car park.\n\nQ2. Sick leave\n\nSick leave needs a certificate."
    first_end = text.index("\n\nQ2.")
    merged = chunk(
        "FAQ::001", text, locator="Q1. Parking",
        segments=[Segment("Q1. Parking", 0, first_end), Segment("Q2. Sick leave", first_end + 2, len(text))],
    )
    idx = HybridIndex.build([merged, chunk("OTHER::001", "Unrelated expense text.")], FakeEmbedder())
    assert idx.search("sick leave certificate", EMPLOYEE, 1)[0].locator == "Q2. Sick leave"
    assert idx.search("car park permit", EMPLOYEE, 1)[0].locator == "Q1. Parking"


def test_tokenizer_folds_plurals_and_drops_stop_words():
    assert tokenize("The holidays and the days of leave") == ["holiday", "day", "leave"]
    assert tokenize("Access status bonus") == ["access", "status", "bonus"]


def test_concurrent_searches_agree(index):
    from concurrent.futures import ThreadPoolExecutor

    def run(_):
        return [r.chunk_id for r in index.search("public holidays India", EMPLOYEE, 3)]

    with ThreadPoolExecutor(8) as pool:
        outcomes = list(pool.map(run, range(32)))
    assert all(o == outcomes[0] for o in outcomes)


# --- persistence --------------------------------------------------------------


def test_save_writes_vectors_ids_and_meta_and_overwrites_older_files(index, tmp_path):
    target = tmp_path / "idx"
    target.mkdir()
    (target / "vectors.npy").write_bytes(b"stale")
    index.save(target)
    index.save(target)  # twice: still a single consistent set
    vectors = np.load(target / "vectors.npy")
    assert vectors.shape == (len(CHUNKS), 64)
    assert json.loads((target / "chunk_ids.json").read_text()) == [c.chunk_id for c in CHUNKS]
    meta = json.loads((target / "index_meta.json").read_text())
    assert meta["chunk_count"] == len(CHUNKS) and meta["model"] == "fake-hashing-embedder"
    assert not list(target.glob(".*.tmp"))


def test_vectors_are_normalised(index):
    norms = np.linalg.norm(index._vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# --- effective-date range (Phase 5: CURRENT vs UPCOMING) -----------------------


def test_the_date_bounds_default_to_none_and_never_replace_access_or_region():
    filters = SearchFilters(allowed_access_levels=frozenset({"all"}), region="IN")
    assert filters.min_effective_date is None and filters.max_effective_date is None
    with pytest.raises(TypeError):
        SearchFilters(min_effective_date="2026-01-01")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        SearchFilters(region="IN", min_effective_date="2026-01-01")  # type: ignore[call-arg]


def test_min_effective_date_is_inclusive_and_excludes_earlier_chunks(index):
    later = SearchFilters(frozenset({"all"}), "IN", min_effective_date="2027-01-01")
    assert ids(index.search("leave policy version", later, 10)) == ["FUTURE::001"]
    day_after = SearchFilters(frozenset({"all"}), "IN", min_effective_date="2027-01-02")
    assert index.search("leave policy version", day_after, 10) == []


def test_min_and_max_together_form_an_inclusive_range(index):
    both = SearchFilters(frozenset({"all"}), "IN", max_effective_date="2026-12-31", min_effective_date="2026-01-01")
    results = index.search("leave", both, 10)
    assert results and all("2026-01-01" <= r.effective_date <= "2026-12-31" for r in results)
    assert "FUTURE::001" not in ids(results)


def test_an_inverted_range_matches_nothing(index):
    inverted = SearchFilters(frozenset({"all"}), "IN", max_effective_date="2025-01-01", min_effective_date="2026-01-01")
    assert index.search("leave", inverted, 10) == []


def test_date_bounds_accept_dates_datetimes_and_iso_strings_and_reject_junk():
    from datetime import date, datetime

    for value in (date(2026, 1, 1), datetime(2026, 1, 1, 9, 30), "2026-01-01"):
        f = SearchFilters(frozenset({"all"}), "IN", min_effective_date=value, max_effective_date=value)
        assert f.min_effective_date == f.max_effective_date == "2026-01-01"
    for junk in ("soon", "01/01/2026", "2026-13-01"):
        with pytest.raises(ValueError):
            SearchFilters(frozenset({"all"}), "IN", min_effective_date=junk)


def test_the_restricted_chunk_stays_hidden_whatever_the_date_range(index):
    wide = SearchFilters(frozenset({"all"}), "IN", min_effective_date="1990-01-01", max_effective_date="2100-01-01")
    assert RESTRICTED_ID not in ids(index.search(RESTRICTED_TEXT, wide, 10))
