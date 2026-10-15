"""Harness for the POST /chat tests: a real app (lifespan, ingestion, DB) on synthetic data.

The corpus is three tiny invented documents (all / managers / hr access). The database is
the Phase 6 synthetic org (``tests/tool_helpers.py``). Startup ingestion rebuilds the
holiday tables from that corpus (which has no calendar), so the synthetic holidays are
loaded AFTER the app reports ready.

TODAY (APP_TODAY) is Thursday 2026-10-15.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from app.db import Database
from app.orchestrator.analyzer import ScriptedAnalyzer, Understanding

CORPUS = {
    "SYN-SAMPLE": ("all", "# Sample Policy\n\nSynthetic text for everyone.\n"),
    "SYN-MGR": ("managers", "# Manager Guide\n\nSynthetic text for managers.\n"),
    "SYN-HR": ("hr", "# HR Handbook\n\nSynthetic text for hr.\n"),
}

RESPONSE_KEYS = {"request_id", "answer", "route", "citations", "tool_calls", "pending_confirmation"}
ROUTES = {"rag", "tool", "hybrid", "clarify", "no_answer", "refuse"}
B6_ARGUMENTS = {"employee_id", "leave_type", "status", "start_date", "end_date", "request_id"}
TOOL_STATUSES = {"success", "error", "pending_confirmation"}


def write_corpus(root: Path, extra: dict | None = None) -> tuple[Path, Path]:
    """Write the default tiny corpus, plus any extra docs.

    An extra entry is ``doc_id -> text`` or
    ``doc_id -> (access, text)`` or
    ``doc_id -> {access, text, doc_type, regions, effective_date}``.
    """
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    manifest = []
    docs = dict(CORPUS)
    for doc_id, spec in (extra or {}).items():
        if isinstance(spec, str):
            docs[doc_id] = ("all", spec)
        elif isinstance(spec, tuple) and len(spec) == 2:
            docs[doc_id] = spec
        else:
            docs[doc_id] = spec
    for doc_id, spec in docs.items():
        if isinstance(spec, dict):
            access = spec.get("access", spec.get("access_level", "all"))
            text = spec["text"]
            doc_type = spec.get("doc_type", "policy")
            regions = spec.get("regions", ["GLOBAL"])
            effective = spec.get("effective_date", "2026-01-01")
        else:
            access, text = spec
            doc_type, regions, effective = "policy", ["GLOBAL"], "2026-01-01"
        name = f"{doc_id.lower()}.md"
        (corpus / name).write_text(text, encoding="utf-8")
        manifest.append(
            {
                "doc_id": doc_id, "file": name, "title": doc_id, "doc_type": doc_type,
                "regions": list(regions), "effective_date": effective, "access_level": access,
            }
        )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return corpus, manifest_path


def Q(intent: str, **fields) -> Understanding:
    return Understanding(intent=intent, **fields)


@dataclass
class Chat:
    client: TestClient
    app: object
    db: Database
    analyzer: ScriptedAnalyzer
    settings: object

    URL = "/chat"

    def post(self, message="hello", *, employee: str | None = "E1", session: str | None = "s1", **extra):
        body = {"session_id": session, "message": message}
        body = {k: v for k, v in body.items() if v is not None}
        body.update(extra)
        headers = {} if employee is None else {"X-Employee-Id": employee}
        return self.client.post(self.URL, json=body, headers=headers)

    def raw(self, *, content=None, json_body=None, headers=None, employee: str | None = "E1"):
        merged = {} if employee is None else {"X-Employee-Id": employee}
        merged.update(headers or {})
        if json_body is not None:
            return self.client.post(self.URL, json=json_body, headers=merged)
        return self.client.post(self.URL, content=content, headers=merged)

    def say(self, message="hello", *, understanding: Understanding | None = None, **kwargs):
        """Queue an understanding for the analyzer, then send the message."""
        if understanding is not None:
            self.analyzer.queue(understanding)
        return self.post(message, **kwargs)

    # direct DB views (read-only connections; never through the app)
    def _rows(self, sql: str, args=()) -> list[tuple]:
        conn = sqlite3.connect(f"file:{self.db.path.resolve().as_posix()}?mode=ro", uri=True)
        try:
            return conn.execute(sql, args).fetchall()
        finally:
            conn.close()

    def requests(self, employee_id: str = "E1") -> list[tuple]:
        return self._rows("SELECT * FROM leave_requests WHERE employee_id = ? ORDER BY request_id", (employee_id,))

    def balance(self, employee_id: str, leave_type: str) -> float:
        return self._rows(
            "SELECT balance_days FROM leave_balances WHERE employee_id = ? AND leave_type = ?",
            (employee_id, leave_type),
        )[0][0]

    def proposals(self) -> list[tuple]:
        return self._rows("SELECT session_id, action, payload_json FROM pending_proposals ORDER BY session_id")

    def sessions(self) -> dict[str, str]:
        return dict(self._rows("SELECT session_id, employee_id FROM sessions"))

    def leave_tables(self) -> dict[str, list[tuple]]:
        return {
            t: sorted(self._rows(f"SELECT * FROM {t}"), key=repr)  # test-only
            for t in ("employees", "leave_balances", "leave_requests")
        }


def wait_until_ready(client: TestClient, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get("/health").status_code == 200:
            return
        time.sleep(0.05)
    raise AssertionError("the app never became ready")


def assert_valid_chat_body(body: dict) -> None:
    """The response contract (README B7 + the invariants), checked on every response."""
    assert set(body) == RESPONSE_KEYS, set(body) ^ RESPONSE_KEYS
    assert uuid.UUID(body["request_id"]).version == 4
    assert isinstance(body["answer"], str) and body["answer"].strip()
    assert body["route"] in ROUTES
    assert isinstance(body["pending_confirmation"], bool)
    for citation in body["citations"]:
        assert set(citation) == {"doc_id", "locator"} and citation["doc_id"] and citation["locator"]
    for call in body["tool_calls"]:
        assert set(call) == {"name", "args", "status"}
        assert call["status"] in TOOL_STATUSES
        assert set(call["args"]) <= B6_ARGUMENTS
    c, t, route = len(body["citations"]), len(body["tool_calls"]), body["route"]
    if route == "rag":
        assert c >= 1 and t == 0
    elif route == "tool":
        assert t >= 1 and c == 0
    elif route == "hybrid":
        assert c >= 1 and t >= 1
    else:
        assert c == 0 and t == 0
    pending = any(call["status"] == "pending_confirmation" for call in body["tool_calls"])
    assert body["pending_confirmation"] is pending


LEAK_PATTERNS = (
    re.compile(r"Traceback", re.I),
    re.compile(r"\bFile \""),
    re.compile(r"\.py\b"),
    re.compile(r"sqlite", re.I),
    re.compile(r"Exception|Error:"),
    re.compile(r"\bat 0x[0-9a-f]+"),
)


def assert_no_internal_detail(text: str) -> None:
    for pattern in LEAK_PATTERNS:
        assert not pattern.search(text), (pattern.pattern, text[:300])
