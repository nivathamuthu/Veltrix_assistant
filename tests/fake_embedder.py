"""A tiny deterministic embedder for tests (no model download, no network).

Hashes words into a fixed number of buckets, so texts that share words get a
high cosine similarity and a text compared with itself gets exactly 1.0. That
is enough to exercise the vector path of the index, including "a query copied
verbatim from a restricted chunk".
"""
from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from app.ingestion.embeddings import normalise_rows

_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    name = "fake-hashing-embedder"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.passage_calls = 0
        self.query_calls = 0

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in _WORD.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        return vec

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        self.passage_calls += 1
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalise_rows(np.vstack([self._vector(t) for t in texts]))

    def embed_query(self, query: str) -> np.ndarray:
        self.query_calls += 1
        return normalise_rows(self._vector(query))
