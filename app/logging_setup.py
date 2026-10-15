"""PII-safe logging.

Personal data in this project means ``full_name``, ``email`` and ``phone``.
Log lines must use employee IDs instead. This module is the safety net: every
formatted log line (message, args and traceback) passes through
``redact()`` before it is written.

Redaction has two layers:
  1. Registered values: exact names / emails / phones loaded from the employee
     table (``register_pii``). Catches names, which no regex can recognise.
  2. Patterns: any email address, and phone-number shapes.
"""
from __future__ import annotations

import logging
import re
import sys
import threading
from typing import Iterable, TextIO

REDACTED_NAME = "[REDACTED_NAME]"
REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Phone shapes. Deliberately avoids ISO dates (2026-10-15) and UUIDs.
_PHONE_RES = (
    # International: +91 98221 51875, +1 512-555-0182, +49 30 5550 2681
    re.compile(r"(?<![\w+])\+\d[\d\s().\-]{6,18}\d"),
    # North-American: (512) 555-0182, 512-555-0182, 512.555.0182
    re.compile(r"(?<![\w-])\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?![\w-])"),
    # Long bare digit runs: 9822151875
    re.compile(r"(?<![\w-])\d{10,15}(?![\w-])"),
)


class _Registry:
    """Thread-safe store of exact PII values to scrub."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._names: list[str] = []
        self._emails: list[str] = []
        self._phones: list[str] = []
        self._name_re: re.Pattern[str] | None = None
        self._literal_re: re.Pattern[str] | None = None

    def clear(self) -> None:
        with self._lock:
            self._names, self._emails, self._phones = [], [], []
            self._name_re = self._literal_re = None

    def register(
        self,
        names: Iterable[str] = (),
        emails: Iterable[str] = (),
        phones: Iterable[str] = (),
    ) -> None:
        with self._lock:
            self._names = _merge(self._names, names)
            self._emails = _merge(self._emails, emails)
            self._phones = _merge(self._phones, phones)
            self._name_re = _compile(self._names)
            self._literal_re = _compile(self._emails + self._phones)

    def scrub(self, text: str) -> str:
        with self._lock:
            name_re, literal_re = self._name_re, self._literal_re
            emails, phones = tuple(self._emails), tuple(self._phones)
        if literal_re is not None:

            def _literal(m: re.Match[str]) -> str:
                hit = m.group(0).lower()
                return REDACTED_EMAIL if hit in {e.lower() for e in emails} else REDACTED_PHONE

            text = literal_re.sub(_literal, text)
        if name_re is not None:
            text = name_re.sub(REDACTED_NAME, text)
        return text


def _merge(existing: list[str], new: Iterable[str]) -> list[str]:
    seen = {v.lower(): v for v in existing}
    for value in new:
        value = (value or "").strip()
        if len(value) >= 3:  # ignore empties / trivially short tokens
            seen.setdefault(value.lower(), value)
    # longest first so "Anna Maria Lee" wins over "Anna Maria"
    return sorted(seen.values(), key=len, reverse=True)


def _compile(values: list[str]) -> re.Pattern[str] | None:
    if not values:
        return None
    return re.compile("|".join(re.escape(v) for v in values), re.IGNORECASE)


_registry = _Registry()


def register_pii(
    names: Iterable[str] = (),
    emails: Iterable[str] = (),
    phones: Iterable[str] = (),
) -> None:
    """Teach the logger exact values to scrub (called after reading the DB)."""
    _registry.register(names, emails, phones)


def clear_pii() -> None:
    _registry.clear()


def redact(text: str) -> str:
    """Remove names, emails and phone numbers from ``text``."""
    text = _registry.scrub(text)
    text = _EMAIL_RE.sub(REDACTED_EMAIL, text)
    for pattern in _PHONE_RES:
        text = pattern.sub(REDACTED_PHONE, text)
    return text


class RedactingFormatter(logging.Formatter):
    """Formatter that scrubs the *final* line, including exception tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# Loggers that uvicorn configures with its own handlers.
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def _harden(logger: logging.Logger, formatter: logging.Formatter) -> None:
    for handler in logger.handlers:
        handler.setFormatter(formatter)


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    """Install the redacting formatter on the root and uvicorn loggers.

    ``stream`` is where a root handler is created if none exists (default
    stdout). Safe to call more than once.
    """
    formatter = RedactingFormatter(LOG_FORMAT)
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler(stream or sys.stdout))
    root.setLevel(level)
    _harden(root, formatter)
    for name in _UVICORN_LOGGERS:
        _harden(logging.getLogger(name), formatter)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
