"""Shared fixtures. All data here is synthetic: no real corpus or employees."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from app.config import load_settings
from app.db import Database
from app.logging_setup import clear_pii
from tests.db_helpers import build_synthetic_db

ENV_VARS = (
    "CORPUS_DIR", "MANIFEST_PATH", "DB_PATH", "INDEX_DIR", "EMBED_CACHE_DIR", "APP_TODAY",
    "LLM_PROVIDER", "LLM_MODEL", "EMBED_MODEL", "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "LOG_LEVEL",
    "INGEST_STARTUP_DELAY_SECONDS", "LLM_TIMEOUT_SECONDS", "REQUEST_TIMEOUT_SECONDS",
)


@pytest.fixture(autouse=True)
def _clean_env_and_pii(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    clear_pii()
    yield
    clear_pii()


REAL_DB = Path(__file__).resolve().parent.parent / "data" / "veltrix.db"


@pytest.fixture()
def synthetic_db(tmp_path: Path) -> Database:
    """Hand-built database with known rows (see tests/db_helpers.py)."""
    return Database(build_synthetic_db(tmp_path / "synthetic.db"))


@pytest.fixture()
def real_db_copy(tmp_path: Path) -> Database:
    """A temporary COPY of data/veltrix.db; the original is never touched."""
    if not REAL_DB.is_file():
        pytest.skip("data/veltrix.db not present")
    target = tmp_path / "veltrix_copy.db"
    shutil.copyfile(REAL_DB, target)
    return Database(target)


@pytest.fixture(params=["synthetic", "real_copy"])
def core_db(request, tmp_path: Path) -> Database:
    """Schema-level tests run against both the synthetic DB and a copy of the real one."""
    if request.param == "synthetic":
        return request.getfixturevalue("synthetic_db")
    return request.getfixturevalue("real_db_copy")


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """A tiny synthetic data folder shaped like the evaluator's (one small document)."""
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "sample.md").write_text(
        "# Sample Policy\n\nSynthetic text used only by tests.\n", encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "doc_id": "SYN-SAMPLE", "file": "sample.md", "title": "Sample",
                    "doc_type": "policy", "regions": ["GLOBAL"],
                    "effective_date": "2026-01-01", "access_level": "all",
                }
            ]
        ),
        encoding="utf-8",
    )
    conn = sqlite3.connect(tmp_path / "veltrix.db")
    conn.execute(
        "CREATE TABLE employees (employee_id TEXT PRIMARY KEY, full_name TEXT, email TEXT,"
        " phone TEXT, region TEXT, join_date TEXT, level TEXT, role TEXT, manager_id TEXT)"
    )
    conn.execute(
        "INSERT INTO employees VALUES ('E9001','Zed Testperson','zed.testperson@veltrix.example',"
        "'+91 90000 12345','IN','2020-01-01','L3','employee',NULL)"
    )
    conn.commit()
    conn.close()
    return tmp_path


@pytest.fixture()
def fake_embedder():
    from tests.fake_embedder import FakeEmbedder

    return FakeEmbedder()


@pytest.fixture()
def settings(data_dir: Path, monkeypatch):
    monkeypatch.setenv("CORPUS_DIR", str(data_dir / "corpus"))
    monkeypatch.setenv("MANIFEST_PATH", str(data_dir / "manifest.json"))
    monkeypatch.setenv("DB_PATH", str(data_dir / "veltrix.db"))
    monkeypatch.setenv("INDEX_DIR", str(data_dir / "index"))
    return load_settings(env_file=None)
