"""Holidays at startup: rebuild from scratch, modified corpus, report (Phase 3c)."""
from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from app.config import load_settings
from app.db import (
    CALENDAR_OK,
    CALENDAR_UNAVAILABLE,
    Database,
    list_calendars,
    load_holidays,
    run_migrations,
)
from app.ingestion import holiday_report
from app.ingestion.corpus import load_corpus
from app.ingestion.manifest import load_manifest
from app.ingestion.pipeline import build_holidays, run_ingestion
from app.ingestion.state import IngestionState
from app.retrieval.holiday_lookup import (
    REASON_UNUSABLE,
    ApplicableCalendar,
    NoApplicableCalendar,
    public_holidays_for,
)
from tests.test_corpus_loading import write_corpus

DATA = Path(__file__).resolve().parent.parent / "data"


def calendar_md(public: list[tuple[str, str, str]], restricted: list[tuple[str, str, str]] = ()) -> str:
    def block(heading, rows):
        lines = [f"## {heading}", "", "| Date | Day | Holiday |", "|---|---|---|"]
        lines += [f"| {d} | {w} | {n} |" for d, w, n in rows]
        return "\n".join(lines) + "\n"

    parts = ["# Synthetic calendar\n", block("1. Public holidays 2026", public)]
    if restricted:
        parts.append(block("2. Restricted holidays 2026", restricted))
    return "\n".join(parts)


PUBLIC = [
    ("15 January 2026", "Thursday", "Alpha Day"),
    ("26 January 2026", "Monday", "Beta Day"),
]
RESTRICTED = [("4 March 2026", "Wednesday", "Gamma Day")]

# A policy that ALSO has a dated table under a "public holidays" heading: it is not
# a calendar document, so none of its rows may reach the holidays table.
POLICY_MD = (
    "# Leave policy\n\n## Public holidays\n\n| Date | Day | Holiday |\n|---|---|---|\n"
    "| 9 February 2026 | Monday | Policy Day |\n"
)


def meta(doc_id, doc_type="calendar", regions=("IN",), effective="2026-01-01", access="all"):
    return {
        "doc_id": doc_id, "doc_type": doc_type, "regions": list(regions),
        "effective_date": effective, "access_level": access,
    }


@pytest.fixture()
def corpus(tmp_path) -> Path:
    return write_corpus(
        tmp_path / "c",
        {
            "in_2026.md": (calendar_md(PUBLIC, RESTRICTED), meta("CAL-IN-2026")),
            "in_2025.md": (
                calendar_md([("10 January 2025", "Friday", "Old Day")]),
                meta("CAL-IN-2025", effective="2025-01-01"),
            ),
            "policy.md": (POLICY_MD, meta("POL-LEAVE", doc_type="policy", regions=("GLOBAL",))),
        },
    )


@pytest.fixture()
def pipeline_settings(settings, corpus, monkeypatch):
    monkeypatch.setenv("CORPUS_DIR", str(corpus / "corpus"))
    monkeypatch.setenv("MANIFEST_PATH", str(corpus / "manifest.json"))
    return load_settings(env_file=None)


def ingest(settings, embedder) -> IngestionState:
    state = IngestionState()
    run_ingestion(settings, state, embedder)
    return state


def snapshot(settings):
    db = Database(settings.db_path)
    with db.read() as conn:
        return list_calendars(conn), load_holidays(conn)


def lookup(settings, region, reference, levels=("all",)):
    db = Database(settings.db_path)
    with db.read() as conn:
        return public_holidays_for(conn, region, reference, allowed_access_levels=levels)


# --- startup -------------------------------------------------------------------


def test_startup_extracts_holidays_and_marks_ready(pipeline_settings, fake_embedder):
    state = ingest(pipeline_settings, fake_embedder)
    assert state.ready and state.error is None
    assert state.details["holiday_rows"] == 4  # 2 + 1 restricted + 1 from the 2025 calendar
    calendars, holidays = snapshot(pipeline_settings)
    by_id = {c.doc_id: c for c in calendars}
    assert set(by_id) == {"CAL-IN-2026", "CAL-IN-2025"}  # the policy is not a calendar
    assert (by_id["CAL-IN-2026"].public_count, by_id["CAL-IN-2026"].optional_count) == (2, 1)
    assert all(h.name != "Policy Day" for h in holidays)
    assert {h.kind for h in holidays if h.doc_id == "CAL-IN-2026"} == {"public", "optional"}
    # manifest facts are stored with every row
    for h in holidays:
        assert h.regions == ("IN",) and h.region == "IN"
    assert {h.effective_date for h in holidays if h.doc_id == "CAL-IN-2025"} == {"2025-01-01"}


def test_restricted_dates_never_reach_the_lookup(pipeline_settings, fake_embedder):
    ingest(pipeline_settings, fake_embedder)
    result = lookup(pipeline_settings, "IN", date(2026, 6, 1))
    assert isinstance(result, ApplicableCalendar) and result.doc_id == "CAL-IN-2026"
    assert result.dates == {date(2026, 1, 15), date(2026, 1, 26)}
    assert date(2026, 3, 4) not in result.dates and date(2026, 2, 9) not in result.dates


def test_an_earlier_reference_date_reaches_the_older_calendar(pipeline_settings, fake_embedder):
    ingest(pipeline_settings, fake_embedder)
    result = lookup(pipeline_settings, "IN", date(2025, 6, 1))
    assert result.doc_id == "CAL-IN-2025" and result.dates == {date(2025, 1, 10)}


def test_the_holiday_tables_are_never_left_with_stale_rows_from_another_run(
    pipeline_settings, fake_embedder
):
    db = Database(pipeline_settings.db_path)
    run_migrations(db)
    with db.transaction() as conn:  # leftovers from some earlier run
        conn.execute(
            "INSERT INTO holidays (region, date, name, doc_id, kind, effective_date, regions) "
            "VALUES ('IN', '2030-01-01', 'Stale', 'GONE', 'public', '2030-01-01', '[\"IN\"]')"
        )
    ingest(pipeline_settings, fake_embedder)
    _, holidays = snapshot(pipeline_settings)
    assert all(h.doc_id != "GONE" for h in holidays)


# --- rebuild ---------------------------------------------------------------------


def test_rebuilding_twice_is_idempotent(pipeline_settings, fake_embedder):
    ingest(pipeline_settings, fake_embedder)
    first = snapshot(pipeline_settings)
    ingest(pipeline_settings, fake_embedder)
    second = snapshot(pipeline_settings)
    assert first == second and first[1]
    conn = sqlite3.connect(pipeline_settings.db_path)
    (total,) = conn.execute("SELECT COUNT(*) FROM holidays").fetchone()
    (distinct,) = conn.execute("SELECT COUNT(DISTINCT region || date || doc_id) FROM holidays").fetchone()
    conn.close()
    assert total == distinct == len(second[1])


def test_a_removed_calendar_disappears_on_rebuild(pipeline_settings, fake_embedder, corpus):
    ingest(pipeline_settings, fake_embedder)
    manifest = [e for e in json.loads((corpus / "manifest.json").read_text()) if e["doc_id"] != "CAL-IN-2025"]
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ingest(pipeline_settings, fake_embedder)
    calendars, holidays = snapshot(pipeline_settings)
    assert {c.doc_id for c in calendars} == {"CAL-IN-2026"}
    assert {h.doc_id for h in holidays} == {"CAL-IN-2026"}


def test_a_modified_corpus_changes_the_lookup_result(pipeline_settings, fake_embedder, corpus):
    ingest(pipeline_settings, fake_embedder)
    before = lookup(pipeline_settings, "IN", date(2026, 6, 1))
    assert date(2026, 1, 15) in before.dates

    path = corpus / "corpus" / "in_2026.md"
    path.write_text(path.read_text(encoding="utf-8").replace("15 January 2026", "16 January 2026"),
                    encoding="utf-8")
    ingest(pipeline_settings, fake_embedder)
    after = lookup(pipeline_settings, "IN", date(2026, 6, 1))
    assert date(2026, 1, 15) not in after.dates and date(2026, 1, 16) in after.dates
    assert after.dates != before.dates
    assert after.dates - {date(2026, 1, 16)} == before.dates - {date(2026, 1, 15)}  # only that date moved


def test_changing_a_heading_from_public_to_restricted_removes_the_dates(
    pipeline_settings, fake_embedder, corpus
):
    ingest(pipeline_settings, fake_embedder)
    path = corpus / "corpus" / "in_2026.md"
    path.write_text(path.read_text(encoding="utf-8").replace("1. Public holidays 2026", "1. Floating holidays 2026"),
                    encoding="utf-8")
    ingest(pipeline_settings, fake_embedder)
    result = lookup(pipeline_settings, "IN", date(2026, 6, 1))
    assert isinstance(result, NoApplicableCalendar) and result.reason == REASON_UNUSABLE


# --- bad inputs never stop startup ----------------------------------------------------


def test_an_unreadable_calendar_file_keeps_startup_alive_and_blocks_its_region(
    tmp_path, settings, monkeypatch, fake_embedder, caplog
):
    root = write_corpus(
        tmp_path / "bad",
        {
            "ok.md": (calendar_md(PUBLIC), meta("CAL-OK", regions=("DE",))),
            "newest.pdf": ("%PDF-1.4 not really a pdf", meta("CAL-BROKEN", effective="2026-02-01")),
            "older.md": (calendar_md(PUBLIC), meta("CAL-OLDER", effective="2025-01-01")),
        },
    )
    monkeypatch.setenv("CORPUS_DIR", str(root / "corpus"))
    monkeypatch.setenv("MANIFEST_PATH", str(root / "manifest.json"))
    s = load_settings(env_file=None)
    with caplog.at_level(logging.WARNING):
        state = ingest(s, fake_embedder)
    assert state.ready
    by_id = {c.doc_id: c for c in snapshot(s)[0]}
    assert by_id["CAL-BROKEN"].status == CALENDAR_UNAVAILABLE and by_id["CAL-OK"].status == CALENDAR_OK
    # the newest IN calendar is unreadable: do NOT quietly use the older one
    result = lookup(s, "IN", date(2026, 6, 1))
    assert isinstance(result, NoApplicableCalendar) and result.reason == REASON_UNUSABLE
    assert lookup(s, "DE", date(2026, 6, 1)).doc_id == "CAL-OK"


def test_a_manifest_without_calendars_gives_no_applicable_calendar(
    tmp_path, settings, monkeypatch, fake_embedder
):
    root = write_corpus(tmp_path / "nocal", {"p.md": (POLICY_MD, meta("POL", doc_type="policy"))})
    monkeypatch.setenv("CORPUS_DIR", str(root / "corpus"))
    monkeypatch.setenv("MANIFEST_PATH", str(root / "manifest.json"))
    s = load_settings(env_file=None)
    assert ingest(s, fake_embedder).ready
    assert snapshot(s) == ([], [])
    assert isinstance(lookup(s, "IN", date(2026, 6, 1)), NoApplicableCalendar)


def test_holiday_logs_contain_ids_and_row_numbers_but_no_holiday_text(
    tmp_path, settings, monkeypatch, fake_embedder, caplog
):
    rows = PUBLIC + [("sometime in spring", "-", "SecretHolidayName")]
    root = write_corpus(tmp_path / "log", {"c.md": (calendar_md(rows), meta("CAL-LOG"))})
    monkeypatch.setenv("CORPUS_DIR", str(root / "corpus"))
    monkeypatch.setenv("MANIFEST_PATH", str(root / "manifest.json"))
    s = load_settings(env_file=None)
    with caplog.at_level(logging.INFO):
        assert ingest(s, fake_embedder).ready
    assert "doc_id=CAL-LOG table=1 row=3" in caplog.text
    assert "SecretHolidayName" not in caplog.text and "Alpha Day" not in caplog.text


# --- the report ------------------------------------------------------------------------


def test_the_report_lists_counts_per_calendar_and_public_dates_per_region(
    pipeline_settings, monkeypatch, capsys
):
    monkeypatch.setenv("APP_TODAY", "2026-06-01")
    assert holiday_report.main(["--year", "2026"]) == 0
    out = capsys.readouterr().out
    assert "holidays per calendar" in out
    in_2026 = next(line for line in out.splitlines() if line.startswith("CAL-IN-2026"))
    assert re.search(r"\bok\s+2\s+1\s+0\s+0\b", in_2026)  # public 2, optional 1
    assert "IN: CAL-IN-2026" in out and "2026-01-15 Thu" in out and "2026-01-26 Mon" in out
    assert "2026-03-04" not in out.split("public holidays in 2026")[1]  # restricted: not listed


# --- real corpus (skipped when data/ is absent) -----------------------------------------


@pytest.mark.skipif(not (DATA / "manifest.json").is_file(), reason="data/ not present")
def test_real_calendars_each_yield_public_holidays_and_no_unknown_tables():
    """No expected dates are written here (they would be corpus facts); this only checks
    the extraction is usable and strict."""
    entries = [e for e in load_manifest(DATA / "manifest.json") if e.doc_type == "calendar"]
    if not entries:
        pytest.skip("no calendar documents in the manifest")
    kept = [e for e in entries if e.file.endswith((".html", ".md"))]  # no OCR needed
    if not kept:
        pytest.skip("no text-format calendar")
    from app.ingestion.holidays import extract_holidays

    corpus = load_corpus(DATA / "corpus", DATA / "manifest.json")
    result = extract_holidays(corpus.entries, corpus.documents)
    for calendar in result.calendars:
        if calendar.doc_id in {e.doc_id for e in kept}:
            assert calendar.status == CALENDAR_OK and calendar.public_count > 0
    assert all(h.kind in ("public", "optional", "unknown") for h in result.holidays)


@pytest.mark.skipif(not (DATA / "manifest.json").is_file(), reason="data/ not present")
def test_changing_one_date_in_a_copy_of_a_real_html_calendar_changes_the_lookup(tmp_path, monkeypatch, fake_embedder):
    from bs4 import BeautifulSoup

    from app.ingestion.dates import parse_date_cell

    entries = [e for e in load_manifest(DATA / "manifest.json")
               if e.doc_type == "calendar" and e.file.endswith(".html")]
    if not entries:
        pytest.skip("no HTML calendar in the corpus")
    entry = entries[0]
    raw = next(m for m in json.loads((DATA / "manifest.json").read_text()) if m["doc_id"] == entry.doc_id)

    (tmp_path / "corpus").mkdir()
    html = (DATA / "corpus" / entry.file).read_text(encoding="utf-8")
    (tmp_path / "corpus" / entry.file).write_text(html, encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps([raw]), encoding="utf-8")
    shutil.copyfile(DATA / "veltrix.db", tmp_path / "veltrix.db")
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("MANIFEST_PATH", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "veltrix.db"))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    s = load_settings(env_file=None)

    region = entry.regions[0]
    assert ingest(s, fake_embedder).ready
    before = lookup(s, region, date(2026, 6, 1), levels=(entry.access_level,))
    assert isinstance(before, ApplicableCalendar)

    soup = BeautifulSoup(html, "html.parser")
    holiday_tables = [  # tables with a column headed exactly "Date" (not the document-info box)
        t for t in soup.find_all("table")
        if any(c.get_text(strip=True).lower() == "date" for c in t.find_all(["th", "td"]))
    ]
    cell = next(
        td for t in holiday_tables for td in t.find_all("td")
        if (p := parse_date_cell(td.get_text(strip=True))) and p.value in before.dates
    )
    old = parse_date_cell(cell.get_text(strip=True)).value
    new = date(old.year, 12, 30) if (old.month, old.day) != (12, 30) else date(old.year, 12, 29)
    assert new not in before.dates
    cell.string = new.isoformat()
    (tmp_path / "corpus" / entry.file).write_text(str(soup), encoding="utf-8")

    assert ingest(s, fake_embedder).ready
    after = lookup(s, region, date(2026, 6, 1), levels=(entry.access_level,))
    assert old not in after.dates and new in after.dates
    assert len(after.dates) == len(before.dates)
