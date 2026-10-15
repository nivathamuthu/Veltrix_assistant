"""Pre-filtered retrieval for policy questions (README B11, Phase 5).

``retrieve_for_question`` runs the hybrid search twice for one requester:

* CURRENT:  documents the requester's role may read, for their region plus GLOBAL,
            already in force on the reference date (``effective_date <= reference``);
* UPCOMING: the same role and region filters, but taking effect AFTER the reference
            date (``effective_date > reference``).

Both searches are pre-filtered: the set of candidate chunks is computed from
metadata alone, before anything is scored (see ``HybridIndex.search``). A chunk the
requester may not read is never a candidate, so it cannot be ranked, returned,
or reach an LLM. Role and region are taken from the ``Requester`` (which the caller
loads from the DB record), never from the query text.

The search is asked for a larger pool than ``k``, and the pool is then reduced by
``select_diverse``: the best chunk of each document first, then the remaining
slots by score. That keeps several versions and types of one topic (policy, memo,
FAQ, regional addendum) from being crowded out by many chunks of a single document.

Nothing here calls an LLM or touches the database. Retrieved text is information
only; this module never interprets it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol, TypeVar

from app.ingestion.index import HybridIndex, SearchFilters, SearchResult
from app.retrieval.policy_resolver import DocMeta, Requester, readable_access_levels

POOL_FACTOR = 3  # the search is asked for k * POOL_FACTOR chunks, then reduced to k


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: date
    locator: str
    text: str
    token_count: int
    score: float
    access_level: str

    def doc_meta(self) -> DocMeta:
        """The manifest facts the policy resolver works with."""
        return DocMeta(self.doc_id, self.doc_type, self.regions, self.effective_date, self.access_level)


@dataclass(frozen=True)
class Retrieval:
    current: tuple[RetrievedChunk, ...] = ()
    upcoming: tuple[RetrievedChunk, ...] = ()


class _Ranked(Protocol):
    doc_id: str


T = TypeVar("T", bound=_Ranked)


def select_diverse(ranked: Sequence[T], k: int) -> list[T]:
    """Pick at most ``k`` of ``ranked`` (best first): each document's best chunk first
    (best documents first), then the remaining slots by rank. Returned in rank order."""
    if k <= 0:
        return []
    chosen: set[int] = set()
    seen_docs: set[str] = set()
    for position, item in enumerate(ranked):
        if len(chosen) == k:
            break
        if item.doc_id not in seen_docs:
            seen_docs.add(item.doc_id)
            chosen.add(position)
    for position in range(len(ranked)):
        if len(chosen) == k:
            break
        chosen.add(position)
    return [ranked[position] for position in sorted(chosen)]


def _validated_k(k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k < 1:
        raise ValueError("k must be at least 1")
    return k


def _as_day(reference_date: object) -> date:
    if isinstance(reference_date, datetime):
        return reference_date.date()
    if isinstance(reference_date, date):
        return reference_date
    raise TypeError("reference_date must be a datetime.date")


def _convert(result: SearchResult) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=result.chunk_id,
        doc_id=result.doc_id,
        doc_type=result.doc_type,
        regions=tuple(result.regions),
        effective_date=date.fromisoformat(result.effective_date),
        locator=result.locator,
        text=result.text,
        token_count=result.token_count,
        score=float(result.score),
        access_level=result.access_level,
    )


def retrieve_for_question(
    query: str,
    requester: Requester,
    reference_date: date,
    k: int,
    *,
    index: HybridIndex,
) -> Retrieval:
    """CURRENT and UPCOMING candidates (at most ``k`` each) for ``query``.

    ``requester`` carries the role and region from the DB record. An unknown role
    reads nothing, so both sets are empty. ``reference_date`` is the date the
    question is about (see ``reference_date.py``), not the clock.
    """
    if not isinstance(requester, Requester):
        raise TypeError("requester must be a Requester built from the DB record")
    k = _validated_k(k)
    reference = _as_day(reference_date)
    if not isinstance(requester.region, str) or not requester.region.strip():
        raise ValueError("the requester's region is required")
    region = requester.region.strip().upper()
    if not isinstance(query, str) or not query.strip():
        return Retrieval()

    levels = readable_access_levels(requester.role) if isinstance(requester.role, str) else frozenset()
    if not levels:
        return Retrieval()

    pool = k * POOL_FACTOR
    in_force = SearchFilters(allowed_access_levels=levels, region=region, max_effective_date=reference)
    later = SearchFilters(
        allowed_access_levels=levels, region=region, min_effective_date=reference + timedelta(days=1)
    )
    current = select_diverse(index.search(query, in_force, pool), k)
    upcoming = select_diverse(index.search(query, later, pool), k)
    return Retrieval(
        current=tuple(_convert(r) for r in current),
        upcoming=tuple(_convert(r) for r in upcoming),
    )
