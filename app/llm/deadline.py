"""The request deadline, visible to the provider adapters (project rule 10).

``/chat`` finishes within one deadline (``REQUEST_TIMEOUT_SECONDS``, under 30 s). The
orchestrator runs in a worker thread that cannot be cancelled, so a provider call that is
still waiting when the deadline passes would keep running. The request sets the deadline in
a ``ContextVar`` (``asyncio.to_thread`` copies the context into the thread) and the adapters
shorten their per-call timeout to what is left, and skip the retry when too little remains.

No deadline set (tests, scripts) means "no limit beyond the per-call timeout".
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_DEADLINE: ContextVar[float | None] = ContextVar("llm_request_deadline", default=None)


def set_deadline(seconds_from_now: float) -> None:
    """Start the clock for the current request (monotonic)."""
    _DEADLINE.set(time.monotonic() + max(0.0, seconds_from_now))


def clear_deadline() -> None:
    _DEADLINE.set(None)


def remaining() -> float | None:
    """Seconds left, or None when no deadline is set."""
    deadline = _DEADLINE.get()
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


@contextmanager
def deadline_of(seconds_from_now: float) -> Iterator[None]:
    token = _DEADLINE.set(time.monotonic() + max(0.0, seconds_from_now))
    try:
        yield
    finally:
        _DEADLINE.reset(token)
