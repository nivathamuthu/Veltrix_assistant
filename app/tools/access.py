"""Who may see whose leave data (README B3), decided in code from the DB record.

* employee: themselves.
* manager:  themselves and their direct reports (``manager_id`` equals their id).
* hr:       any ONE employee at a time (no tool takes more than one id).
* any other role: themselves only.

The decision uses the requester's id and role and the target id, and nothing read from
the target's data: for a manager it is a single relationship query that answers yes or
no. A forbidden id and a nonexistent id take the same path and get the same answer, so
a requester cannot probe which ids exist.
"""
from __future__ import annotations

import sqlite3

from app.db import Employee, is_direct_report

ROLE_EMPLOYEE = "employee"
ROLE_MANAGER = "manager"
ROLE_HR = "hr"


def may_view_employee(conn: sqlite3.Connection, requester: Employee, target_id: str) -> bool:
    """True if ``requester`` may see leave data of ``target_id``. Run BEFORE any read of
    that employee's balances or requests."""
    if target_id == requester.employee_id:
        return True
    if requester.role == ROLE_HR:
        return True
    if requester.role == ROLE_MANAGER:
        return is_direct_report(conn, requester.employee_id, target_id)
    return False


def is_manager(requester: Employee) -> bool:
    """Managers only: HR is not a manager for the team view."""
    return requester.role == ROLE_MANAGER
