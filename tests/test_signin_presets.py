"""Sign-in preset queries (TASK §2.3 fresh evaluator database)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.db import Database, get_signin_presets
from tests.db_helpers import build_synthetic_db


def test_signin_presets_pick_one_per_region_and_role(tmp_path: Path) -> None:
    db_path = build_synthetic_db(tmp_path / "synthetic.db")
    with Database(db_path).read() as conn:
        body = get_signin_presets(conn)
    ids = {p["employee_id"] for p in body["presets"]}
    assert "E1" in ids or "E3" in ids  # synthetic employees
    assert body["hr_lookup_example_id"] == "E1"
    assert all("full_name" not in p for p in body["presets"])
