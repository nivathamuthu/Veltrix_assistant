"""Thread-safe readiness flag shared between ingestion and /health."""
from __future__ import annotations

import threading


class IngestionState:
    """/health returns 200 only after ``mark_ready()``; until then it is 503."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False
        self._error: str | None = None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def mark_ready(self) -> None:
        with self._lock:
            self._ready = True
            self._error = None

    def mark_failed(self, reason: str) -> None:
        # Stays not-ready: the service reports 503 rather than serving
        # answers from a half-built index.
        with self._lock:
            self._ready = False
            self._error = reason
