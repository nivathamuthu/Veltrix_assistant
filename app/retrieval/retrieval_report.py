"""Developer report: run the real startup ingestion, then retrieve for six sample
questions across roles and regions and print, per question, the doc_ids in the
CURRENT and UPCOMING sets and the corpus tokens left after the 3,000-token budget.

    python -m app.retrieval.retrieval_report

Uses the same settings and the same ``run_ingestion`` as the service. The role and
region of each sample requester are dev-only inputs; the service takes them from the
DB record (rules 4 and 5). Only doc_ids and token counts are printed, no chunk text.
"""
from __future__ import annotations

import argparse
import sys

from app.config import load_settings
from app.ingestion.pipeline import run_ingestion
from app.ingestion.state import IngestionState
from app.llm.budget import CorpusContext, count_tokens, enforce_budget, render_block
from app.logging_setup import configure_logging
from app.retrieval.pipeline import Retrieval, retrieve_for_question
from app.retrieval.policy_resolver import Requester
from app.retrieval.reference_date import resolve_reference_date

# (question, role, region)
SAMPLE_QUESTIONS = [
    ("How many work from home days do I get?", "employee", "IN"),
    ("How many work from home days do I get? I mean on 1 June.", "employee", "US"),
    ("How many work from home days do I get?", "employee", "DE"),
    ("How do I approve leave requests from my team?", "manager", "IN"),
    ("What are the salary compensation bands?", "hr", "US"),
    ("Can I carry over unused annual leave?", "employee", "DE"),
]


def _context_for(result: Retrieval) -> CorpusContext:
    """CURRENT chunks first (they decide the answer), then UPCOMING ones."""
    context = CorpusContext()
    for chunk in (*result.current, *result.upcoming):
        context.add_chunk(chunk)
    return context


def _docs(chunks) -> str:
    return ", ".join(f"{c.doc_id}[{c.effective_date}]" for c in chunks) or "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=6, help="chunks per set (CURRENT and UPCOMING)")
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.log_level, stream=sys.stderr)
    state = IngestionState()
    run_ingestion(settings, state)
    if not state.ready:
        print(f"ingestion FAILED: {state.error}")
        return 1
    today = settings.today()

    print(f"today (APP_TODAY or system date): {today}   k={args.k} per set\n")
    for number, (question, role, region) in enumerate(SAMPLE_QUESTIONS, start=1):
        reference = resolve_reference_date(question, today)
        result = retrieve_for_question(
            question, Requester(role, region), reference.value, args.k, index=state.index
        )
        context = _context_for(result)
        before = count_tokens(render_block(context.parts))
        kept = enforce_budget(context)
        print(f'{number}. "{question}"   [role={role}, region={region}]')
        print(f"   reference date : {reference.value} ({reference.source}{', unclear' if reference.unclear else ''})")
        print(f"   CURRENT  ({len(result.current)}): {_docs(result.current)}")
        print(f"   UPCOMING ({len(result.upcoming)}): {_docs(result.upcoming)}")
        print(
            f"   corpus tokens  : {before} before budget -> {kept.tokens} after "
            f"(limit {kept.limit}); sent docs: {', '.join(kept.kept_doc_ids) or '-'}"
        )
        if kept.dropped or kept.truncated:
            print(f"   dropped {len(kept.dropped)} part(s), truncated {len(kept.truncated)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
