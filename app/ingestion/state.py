"""Thread-safe readiness flag shared between ingestion and /health."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing numpy/rank_bm25 just to read the flag
    from app.ingestion.index import HybridIndex


class IngestionState:
    """/health returns 200 only after ``mark_ready()``; until then it is 503.

    The search index is attached before the service is marked ready, so any
    request that is served can rely on ``index`` being present.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False
        self._error: str | None = None
        self._index: "HybridIndex | None" = None
        self._details: dict = {}

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def index(self) -> "HybridIndex | None":
        with self._lock:
            return self._index

    @property
    def details(self) -> dict:
        """Phase timings and skip lists from the last successful ingestion."""
        with self._lock:
            return dict(self._details)

    def mark_ready(self, index: "HybridIndex | None" = None, details: dict | None = None) -> None:
        with self._lock:
            if index is not None:
                self._index = index
            if details is not None:
                self._details = dict(details)
            self._ready = True
            self._error = None

    def mark_failed(self, reason: str) -> None:
        # Stays not-ready: the service reports 503 rather than serving
        # answers from a half-built index.
        with self._lock:
            self._ready = False
            self._error = reason
            self._index = None
            self._details = {}
