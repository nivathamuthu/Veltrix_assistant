"""Developer report: run the real startup ingestion, then print chunk statistics,
timings and a few sample searches.

    python -m app.ingestion.chunk_report

Uses the same settings and the same ``run_ingestion`` as the service, so the
numbers are what a real startup produces. It rewrites the ``chunks`` table in
DB_PATH and the files in INDEX_DIR, exactly like startup does.

The sample searches are made AS a given requester (access levels + region).
These are dev-only inputs; the service derives them from the DB (rules 4, 5).
"""
from __future__ import annotations

import argparse
import sys

from app.config import load_settings
from app.db import Database, load_chunks
from app.ingestion.index import SearchFilters
from app.ingestion.pipeline import run_ingestion
from app.ingestion.state import IngestionState
from app.logging_setup import configure_logging

# (query, requester region, access levels the requester may read)
SAMPLE_SEARCHES = [
    ("work from home days", "IN", ("all",)),
    ("public holidays India", "IN", ("all",)),
    ("sick leave Germany", "DE", ("all",)),
    ("how much annual leave do I earn per month", "US", ("all",)),
    ("can I carry over unused leave", "DE", ("all",)),
]
# The same query asked with and without access to restricted content.
ACCESS_DEMO = ("salary compensation bands", "IN")


def _fmt_seconds(value: float) -> str:
    return f"{value:6.2f}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top", type=int, default=3, help="results per sample search")
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.log_level, stream=sys.stderr)  # table goes to stdout
    state = IngestionState()
    run_ingestion(settings, state)
    if not state.ready:
        print(f"ingestion FAILED: {state.error}")
        return 1
    index, details = state.index, state.details

    with Database(settings.db_path).read() as conn:
        chunks = load_chunks(conn)

    # ---- per document --------------------------------------------------
    print("\n== chunks per document ==")
    print(f"{'doc_id':32} {'chunks':>6}   tokens (min / avg / max)")
    by_doc: dict[str, list[int]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c.token_count)
    for doc_id, counts in sorted(by_doc.items()):
        avg = sum(counts) / len(counts)
        print(f"{doc_id:32} {len(counts):>6}   {min(counts):>4} / {avg:>5.0f} / {max(counts):>4}")

    counts = [c.token_count for c in chunks]
    bands = {
        "<50": sum(n < 50 for n in counts),
        "50-99": sum(50 <= n < 100 for n in counts),
        "100-199": sum(100 <= n < 200 for n in counts),
        "200-350": sum(200 <= n <= 350 for n in counts),
        "351-450": sum(350 < n <= 450 for n in counts),
        ">450": sum(n > 450 for n in counts),
    }
    print("\n== chunk token_count (cl100k_base) ==")
    print(f"documents indexed : {len(by_doc)}")
    print(f"chunks            : {len(chunks)}   (index holds {len(index)})")
    print(f"min / avg / max   : {min(counts)} / {sum(counts) / len(counts):.1f} / {max(counts)}")
    print("distribution      : " + ", ".join(f"{k}: {v}" for k, v in bands.items()))

    # ---- skips ------------------------------------------------------------
    print("\n== skipped ==")
    print(f"manifest entry, file missing   : {details['skipped_missing'] or 'none'}")
    print(f"manifest entry, failed to parse: {details['skipped_unparsed'] or 'none'}")
    print(f"manifest entry, no text        : {details['skipped_empty'] or 'none'}")
    print(f"corpus file, no manifest entry : {details['unmanifested_files'] or 'none'}")

    # ---- timings ----------------------------------------------------------
    print("\n== startup / indexing time ==")
    for name, value in details["seconds"].items():
        print(f"{name:24}: {_fmt_seconds(value)}")
    print(f"{'TOTAL run_ingestion':24}: {_fmt_seconds(details['total_seconds'])}")

    # ---- sample searches -----------------------------------------------------
    print(f"\n== sample searches (top {args.top}) ==")
    for query, region, levels in SAMPLE_SEARCHES:
        filters = SearchFilters(allowed_access_levels=frozenset(levels), region=region)
        print(f'\n"{query}"   [region={region}, access={"+".join(levels)}]')
        for rank, r in enumerate(index.search(query, filters, args.top), start=1):
            print(f"  {rank}. {r.doc_id:30} {r.locator}   (score {r.score:.4f})")

    query, region = ACCESS_DEMO
    print(f'\n== pre-filter demo: "{query}" (region={region}) ==')
    for label, levels in (("employee", ("all",)), ("hr", ("all", "managers", "hr"))):
        filters = SearchFilters(allowed_access_levels=frozenset(levels), region=region)
        top = index.search(query, filters, args.top)
        shown = ", ".join(f"{r.doc_id} [{r.access_level}]" for r in top)
        print(f"  as {label:8}: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
