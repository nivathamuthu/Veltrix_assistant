"""Hybrid search index: BM25 + dense vectors fused with reciprocal rank fusion.

Access control lives HERE, in code, before any scoring (project rule 2):

1. ``SearchFilters`` (allowed access levels + requester region, optional
   effective-date range) is REQUIRED. There is no default and no "search
   everything" mode: access levels and region are mandatory, and the dates
   only narrow the result further.
2. ``_eligible_positions`` decides which chunks may be considered using
   metadata only. It never reads chunk text, tokens or vectors.
3. BM25 statistics are built from the eligible chunks only (so a restricted
   chunk cannot influence IDF or any score), and the vector matrix is sliced
   to the eligible rows before the dot product. Restricted chunks are never
   scored and then dropped; they are never scored at all.

The index is rebuilt from scratch at every startup; ``save`` writes the
artifacts to INDEX_DIR.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from app.db.models import Chunk
from app.ingestion.embeddings import Embedder
from app.logging_setup import get_logger

log = get_logger(__name__)

GLOBAL_REGION = "GLOBAL"  # scope value meaning "applies everywhere" (README B3)
RRF_K = 60
_BM25_CACHE_SIZE = 64
_MIN_CANDIDATES = 25

ARTIFACT_FILES = ("vectors.npy", "chunk_ids.json", "index_meta.json")

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i if in into is it its "
    "me my of on or our so than that the their them then there these they this to us was "
    "we what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case words, stop words dropped, plain plurals folded ("days" -> "day")."""
    tokens: list[str] = []
    for word in _WORD.findall(text.lower()):
        if word in _STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        tokens.append(word)
    return tokens


@dataclass(frozen=True)
class SearchFilters:
    """Who may read what. Required for every search; has no permissive default.

    ``allowed_access_levels`` and ``region`` must come from the requester's DB
    record, never from the request text (rules 4 and 5).
    """

    allowed_access_levels: frozenset[str]
    region: str
    max_effective_date: str | None = None  # ISO date; chunks effective later are excluded
    min_effective_date: str | None = None  # ISO date; chunks effective EARLIER are excluded (inclusive)

    def __post_init__(self) -> None:
        levels = self.allowed_access_levels
        if isinstance(levels, str) or not isinstance(levels, Iterable):
            raise TypeError("allowed_access_levels must be a collection of access levels")
        object.__setattr__(self, "allowed_access_levels", frozenset(levels))
        if not isinstance(self.region, str) or not self.region.strip():
            raise ValueError("region is required")
        object.__setattr__(self, "region", self.region.strip().upper())
        for name in ("max_effective_date", "min_effective_date"):
            bound = getattr(self, name)
            if isinstance(bound, datetime):
                bound = bound.date()
            if isinstance(bound, date):
                object.__setattr__(self, name, bound.isoformat())
            elif bound is not None:
                object.__setattr__(self, name, date.fromisoformat(bound).isoformat())


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    doc_id: str
    locator: str  # the heading in the chunk that best matches the query
    text: str
    token_count: int
    score: float  # fused (RRF) score; higher is better
    bm25_score: float | None  # None if the chunk had no lexical match
    vector_score: float | None  # cosine similarity; None if outside the vector top-N
    regions: tuple[str, ...]
    effective_date: str
    doc_type: str
    access_level: str


@dataclass
class BuildStats:
    chunk_count: int = 0
    embed_seconds: float = 0.0
    dim: int = 0
    model: str = ""


def _cosine_scores(matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """Vectors are L2-normalised, so the dot product is the cosine similarity.

    ``matrix`` holds ONLY eligible rows (module-level so tests can observe it).
    """
    return matrix @ query_vector


class HybridIndex:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder, vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("one vector per chunk is required")
        self._chunks = tuple(chunks)
        self._embedder = embedder
        self._vectors = np.asarray(vectors, dtype=np.float32)
        # Metadata used for eligibility, kept apart from text and vectors.
        self._access = tuple(c.access_level for c in self._chunks)
        self._regions = tuple(frozenset(r.upper() for r in c.regions) for c in self._chunks)
        self._effective = tuple(c.effective_date for c in self._chunks)
        # Tokenising at build time is index construction, not scoring.
        self._tokens = [tokenize(c.text) for c in self._chunks]
        self._bm25_cache: dict[tuple[int, ...], BM25Okapi] = {}
        self._cache_lock = threading.Lock()
        self.stats = BuildStats(
            chunk_count=len(self._chunks), dim=int(self._vectors.shape[1]) if len(chunks) else 0,
            model=getattr(embedder, "name", ""),
        )

    # ------------------------------------------------------------------ build

    @classmethod
    def build(cls, chunks: Sequence[Chunk], embedder: Embedder) -> "HybridIndex":
        started = time.monotonic()
        vectors = embedder.embed_passages([c.text for c in chunks])
        index = cls(chunks, embedder, vectors)
        index.stats.embed_seconds = time.monotonic() - started
        log.info(
            "index built: chunks=%d dim=%d embed_seconds=%.2f",
            len(chunks), index.stats.dim, index.stats.embed_seconds,
        )
        return index

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, index_dir: Path) -> None:
        """Write vectors and metadata to ``index_dir``, replacing any older files."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACT_FILES:  # rebuilt from scratch: drop the previous index
            (index_dir / name).unlink(missing_ok=True)

        def write_atomic(name: str, writer) -> None:
            tmp = index_dir / f".{name}.tmp"
            with open(tmp, "wb") as fh:
                writer(fh)
            os.replace(tmp, index_dir / name)

        write_atomic("vectors.npy", lambda fh: np.save(fh, self._vectors))
        write_atomic(
            "chunk_ids.json",
            lambda fh: fh.write(json.dumps([c.chunk_id for c in self._chunks]).encode("utf-8")),
        )
        meta = {
            "model": self.stats.model,
            "dim": self.stats.dim,
            "chunk_count": self.stats.chunk_count,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rrf_k": RRF_K,
        }
        write_atomic("index_meta.json", lambda fh: fh.write(json.dumps(meta).encode("utf-8")))
        log.info("index saved: dir=%s chunks=%d", index_dir, len(self._chunks))

    # ----------------------------------------------------------------- search

    def _eligible_positions(self, filters: SearchFilters) -> tuple[int, ...]:
        """Positions of chunks the requester may read. Metadata only."""
        allowed = filters.allowed_access_levels
        limit = filters.max_effective_date
        floor = filters.min_effective_date
        eligible = []
        for pos in range(len(self._chunks)):
            if self._access[pos] not in allowed:
                continue
            regions = self._regions[pos]
            if GLOBAL_REGION not in regions and filters.region not in regions:
                continue
            if limit is not None and self._effective[pos] > limit:
                continue
            if floor is not None and self._effective[pos] < floor:
                continue
            eligible.append(pos)
        return tuple(eligible)

    def _bm25_for(self, positions: tuple[int, ...]) -> BM25Okapi:
        """BM25 over exactly these chunks (cached per distinct eligible set)."""
        with self._cache_lock:
            cached = self._bm25_cache.get(positions)
        if cached is not None:
            return cached
        model = BM25Okapi([self._tokens[p] for p in positions])
        with self._cache_lock:
            if len(self._bm25_cache) >= _BM25_CACHE_SIZE:
                self._bm25_cache.pop(next(iter(self._bm25_cache)))
            self._bm25_cache[positions] = model
        return model

    def _bm25_ranking(
        self, positions: tuple[int, ...], query_tokens: list[str], depth: int
    ) -> list[tuple[int, float]]:
        if not query_tokens:
            return []
        wanted = set(query_tokens)
        scores = self._bm25_for(positions).get_scores(query_tokens)
        matched = [
            (positions[i], float(scores[i]))
            for i in range(len(positions))
            if wanted.intersection(self._tokens[positions[i]])
        ]
        matched.sort(key=lambda item: (-item[1], item[0]))
        return matched[:depth]

    def _vector_ranking(
        self, positions: tuple[int, ...], query: str, depth: int
    ) -> list[tuple[int, float]]:
        query_vector = self._embedder.embed_query(query)
        eligible_rows = self._vectors[np.asarray(positions, dtype=np.int64)]
        sims = _cosine_scores(eligible_rows, query_vector)
        order = sorted(range(len(positions)), key=lambda i: (-float(sims[i]), positions[i]))
        return [(positions[i], float(sims[i])) for i in order[:depth]]

    def search(self, query: str, filters: SearchFilters, k: int) -> list[SearchResult]:
        """Top-``k`` chunks the requester may read, best first.

        ``filters`` is mandatory. Anything outside it is not scored.
        """
        if not isinstance(filters, SearchFilters):
            raise TypeError("search() requires SearchFilters; there is no unfiltered search")
        if k <= 0 or not query or not query.strip():
            return []
        positions = self._eligible_positions(filters)
        if not positions:
            return []

        depth = min(len(positions), max(k * 5, _MIN_CANDIDATES))
        query_tokens = tokenize(query)
        bm25 = self._bm25_ranking(positions, query_tokens, depth)
        vector = self._vector_ranking(positions, query, depth)

        fused: dict[int, float] = {}
        for ranking in (bm25, vector):
            for rank, (pos, _) in enumerate(ranking, start=1):
                fused[pos] = fused.get(pos, 0.0) + 1.0 / (RRF_K + rank)
        bm25_scores = dict(bm25)
        vector_scores = dict(vector)

        ordered = sorted(fused, key=lambda p: (-fused[p], p))[:k]
        query_terms = set(query_tokens)
        return [
            SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                locator=_best_locator(chunk, query_terms),
                text=chunk.text,
                token_count=chunk.token_count,
                score=fused[pos],
                bm25_score=bm25_scores.get(pos),
                vector_score=vector_scores.get(pos),
                regions=chunk.regions,
                effective_date=chunk.effective_date,
                doc_type=chunk.doc_type,
                access_level=chunk.access_level,
            )
            for pos in ordered
            for chunk in (self._chunks[pos],)
        ]


def _best_locator(chunk: Chunk, query_terms: set[str]) -> str:
    """For a merged chunk, the heading of the section that best matches the query."""
    if len(chunk.segments) <= 1 or not query_terms:
        return chunk.locator
    best, best_hits = chunk.segments[0].locator, -1
    for segment in chunk.segments:
        hits = len(query_terms.intersection(tokenize(chunk.text[segment.start : segment.end])))
        if hits > best_hits:
            best, best_hits = segment.locator, hits
    return best
