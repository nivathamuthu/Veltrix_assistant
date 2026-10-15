"""Developer report: extract holidays from the corpus and print what was found.

    python -m app.ingestion.holiday_report [--year 2026] [--as-of YYYY-MM-DD]

Runs the real parse + extraction + table rebuild (the same functions startup
uses; no embedding model is needed), then reads the result back through
``public_holidays_for``, i.e. exactly what apply_leave will see. Compare the
printed dates with the calendar PDFs / DOCX / HTML.

The lookup is made with every access level present among the calendars (an
all-seeing view for this dev tool); the service itself passes the requester's
levels.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

from app.config import load_settings
from app.db import Database, list_calendars, load_holidays, run_migrations
from app.ingestion.corpus import load_corpus
from app.ingestion.pipeline import build_holidays
from app.logging_setup import configure_logging
from app.retrieval.holiday_lookup import ApplicableCalendar, public_holidays_for


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2026, help="year whose public dates are listed")
    parser.add_argument(
        "--as-of", type=date.fromisoformat, default=None,
        help="reference date for choosing the calendar (default APP_TODAY, else today)",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.log_level, stream=sys.stderr)  # report goes to stdout
    reference = args.as_of or settings.app_today or date.today()

    db = Database(settings.db_path)
    run_migrations(db)
    corpus = load_corpus(settings.corpus_dir, settings.manifest_path)
    extraction = build_holidays(db, corpus)

    with db.read() as conn:
        calendars = list_calendars(conn)
        rows = load_holidays(conn)

        print("\n== holidays per calendar ==")
        print(
            f"{'doc_id':20} {'regions':8} {'effective':11} {'status':20} "
            f"{'public':>6} {'optional':>8} {'unknown':>7} {'skipped':>7}"
        )
        for c in calendars:
            print(
                f"{c.doc_id:20} {','.join(c.regions):8} {c.effective_date:11} {c.status:20} "
                f"{c.public_count:>6} {c.optional_count:>8} {c.unknown_count:>7} {c.skipped_rows:>7}"
            )
        if not calendars:
            print("(no calendar documents in the manifest)")

        print("\n== how each table was read ==")
        for t in extraction.tables:
            if not t.holiday_table:
                print(f"{t.doc_id:20} table {t.table_no}: not a holiday table ({t.heading!r})")
                continue
            print(
                f"{t.doc_id:20} table {t.table_no}: kind={t.kind:8} (from {t.kind_source}) "
                f"rows={t.rows} extracted={t.extracted} skipped={t.skipped}  heading={t.heading!r}"
            )

        print(f"\n== public holidays in {args.year}, per region (calendar chosen as of {reference}) ==")
        levels = frozenset(c.access_level for c in calendars)
        regions = sorted({r for c in calendars for r in c.regions})
        names = {(h.doc_id, h.date): h.name for h in rows}
        for region in regions:
            result = public_holidays_for(conn, region, reference, allowed_access_levels=levels)
            if not isinstance(result, ApplicableCalendar):
                print(f"\n{region}: NO APPLICABLE CALENDAR ({result.reason}): {result.message}")
                continue
            in_year = sorted(d for d in result.dates if d.year == args.year)
            weekend = sum(1 for d in in_year if d.weekday() >= 5)
            print(
                f"\n{region}: {result.doc_id} (effective {result.effective_date}) -> "
                f"{len(in_year)} public dates in {args.year} ({weekend} on a weekend)"
            )
            for d in in_year:
                flag = "  (weekend: no effect on working days)" if d.weekday() >= 5 else ""
                print(f"  {d.isoformat()} {d.strftime('%a')}  {names.get((result.doc_id, d.isoformat()), '?')}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
