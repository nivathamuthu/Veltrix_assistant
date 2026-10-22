"""Sign-in shortcuts for the optional chat UI (TASK §2.3: DB rows may differ per evaluator)."""
from __future__ import annotations

import sqlite3
from typing import Any

_REGIONS = ("IN", "US", "DE")
_UI_ROLES = ("employee", "manager")


def get_signin_presets(conn: sqlite3.Connection) -> dict[str, Any]:
    """One employee and one manager per region, plus one HR id; no personal data."""
    presets: list[dict[str, str]] = []
    for region in _REGIONS:
        for role in _UI_ROLES:
            row = conn.execute(
                "SELECT employee_id, region, role FROM employees "
                "WHERE role = ? AND region = ? ORDER BY employee_id LIMIT 1",
                (role, region),
            ).fetchone()
            if row is None:
                continue
            presets.append(_preset_row(row))

    hr = conn.execute(
        "SELECT employee_id, region, role FROM employees WHERE role = 'hr' "
        "ORDER BY employee_id LIMIT 1"
    ).fetchone()
    if hr is not None:
        presets.append(_preset_row(hr))

    lookup = conn.execute(
        "SELECT employee_id FROM employees WHERE role = 'employee' ORDER BY employee_id LIMIT 1"
    ).fetchone()
    return {
        "presets": presets,
        "hr_lookup_example_id": lookup["employee_id"] if lookup else "",
    }


def _preset_row(row: sqlite3.Row) -> dict[str, str]:
    employee_id, region, role = row["employee_id"], row["region"], row["role"]
    return {
        "employee_id": employee_id,
        "role": role,
        "region": region,
        "label": f"{employee_id} {role} {region}",
    }
