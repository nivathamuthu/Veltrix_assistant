"""The ``holidays`` and ``holiday_calendars`` tables (both rebuilt at startup).

Plain reads and writes only. Deciding WHICH calendar applies to a requester is
the job of app/retrieval/holiday_lookup.py.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Iterable

from app.db.models import KIND_PUBLIC, CalendarRow, HolidayRow

_SQL_CLEAR_HOLIDAYS = "DELETE FROM holidays"
_SQL_CLEAR_CALENDARS = "DELETE FROM holiday_calendars"
_SQL_INSERT_HOLIDAY = (
    "INSERT INTO holidays (region, date, name, doc_id, kind, effective_date, regions) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)
_SQL_INSERT_CALENDAR = (
    "INSERT INTO holiday_calendars (doc_id, effective_date, regions, access_level, status, "
    "public_count, optional_count, unknown_count, skipped_rows) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_SQL_ALL_CALENDARS = (
    "SELECT doc_id, effective_date, regions, access_level, status, public_count, "
    "optional_count, unknown_count, skipped_rows FROM holiday_calendars ORDER BY doc_id"
)
_SQL_ALL_HOLIDAYS = (
    "SELECT region, date, name, doc_id, kind, effective_date, regions FROM holidays "
    "ORDER BY doc_id, date, region"
)
# The kind is a bound parameter: nothing but a row explicitly marked public counts.
_SQL_PUBLIC_DATES = (
    "SELECT DISTINCT date FROM holidays WHERE doc_id = ? AND kind = ? ORDER BY date"
)


def replace_holiday_data(
    conn: sqlite3.Connection,
    calendars: Iterable[CalendarRow],
    holidays: Iterable[HolidayRow],
) -> tuple[int, int]:
    """Clear both tables and refill them. Call inside ``Database.transaction()``
    so a failure keeps the previous contents and never leaves a half-built set.
    Returns ``(calendar rows, holiday rows)`` written."""
    calendar_rows = [
        (
            c.doc_id, c.effective_date, json.dumps(list(c.regions)), c.access_level, c.status,
            c.public_count, c.optional_count, c.unknown_count, c.skipped_rows,
        )
        for c in calendars
    ]
    holiday_rows = [
        (
            h.region, h.date, h.name, h.doc_id, h.kind, h.effective_date,
            json.dumps(list(h.regions)),
        )
        for h in holidays
    ]
    conn.execute(_SQL_CLEAR_HOLIDAYS)
    conn.execute(_SQL_CLEAR_CALENDARS)
    conn.executemany(_SQL_INSERT_CALENDAR, calendar_rows)
    conn.executemany(_SQL_INSERT_HOLIDAY, holiday_rows)
    return len(calendar_rows), len(holiday_rows)


def list_calendars(conn: sqlite3.Connection) -> list[CalendarRow]:
    return [
        CalendarRow(
            doc_id=r["doc_id"], effective_date=r["effective_date"],
            regions=tuple(json.loads(r["regions"])), access_level=r["access_level"],
            status=r["status"], public_count=r["public_count"],
            optional_count=r["optional_count"], unknown_count=r["unknown_count"],
            skipped_rows=r["skipped_rows"],
        )
        for r in conn.execute(_SQL_ALL_CALENDARS)
    ]


def load_holidays(conn: sqlite3.Connection) -> list[HolidayRow]:
    return [
        HolidayRow(
            region=r["region"], date=r["date"], name=r["name"], doc_id=r["doc_id"],
            kind=r["kind"], effective_date=r["effective_date"],
            regions=tuple(json.loads(r["regions"])),
        )
        for r in conn.execute(_SQL_ALL_HOLIDAYS)
    ]


def get_public_dates(conn: sqlite3.Connection, doc_id: str) -> list[str]:
    """ISO dates of the rows of ONE calendar document that are marked public."""
    return [r["date"] for r in conn.execute(_SQL_PUBLIC_DATES, (doc_id, KIND_PUBLIC))]
