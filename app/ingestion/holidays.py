"""Holiday extraction from calendar documents (Phase 3c).

Works from the parsed TABLE sections of every manifest document whose doc_type
is "calendar" (``Section.rows``: structured ``(column header, value)`` pairs),
not from chunk text. Nothing about the corpus is hard-coded: no dates, no
holiday names, no per-document rules.

* A table is a holiday table only if it has a date column: a column headed
  "date", or (headerless tables) one where at least half the cells parse as
  dates. A document-info box that happens to contain one date is therefore not
  mistaken for a holiday list, while a table whose dates are mostly unreadable
  is not silently dropped: its bad rows are skipped and logged one by one.
* Each row yields a date and a holiday name. A row whose date cannot be parsed
  unambiguously is skipped and logged by doc_id, table number and row number
  only; the date is never guessed (see app/ingestion/dates.py).
* ``kind`` is decided from the table's own heading, falling back to the
  caption (the text directly above it under the same heading):
    - "restricted" / "optional" ...        -> optional
    - "public" / "paid holidays" ...       -> public
    - nothing, or both, or a negation      -> unknown: logged, never public
  Only ``public`` rows can ever count as non-working days.
* If the "Day" column names a weekday that does not match the date, the date
  cell still wins (a corrected date with a stale weekday must not lose a
  holiday) and a warning is logged.

Logs carry identifiers and numbers only, never holiday names or other content.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from app.db.models import (
    CALENDAR_NO_PUBLIC,
    CALENDAR_OK,
    CALENDAR_UNAVAILABLE,
    KIND_OPTIONAL,
    KIND_PUBLIC,
    KIND_UNKNOWN,
    CalendarRow,
    HolidayRow,
)
from app.ingestion.corpus import LoadedDocument
from app.ingestion.dates import parse_date_cell, parse_weekday
from app.ingestion.manifest import ManifestEntry
from app.ingestion.parsers.models import Section
from app.logging_setup import get_logger

log = get_logger(__name__)

CALENDAR_DOC_TYPE = "calendar"  # a doc_type from the spec's authority list (B4)

# --- how a table's heading says what kind of holidays it lists ---------------
_OPTIONAL_WORDS = re.compile(r"\b(?:restricted|optional|floating|elective)\b")
_PUBLIC_WORDS = re.compile(r"\bpublic\b|\bpaid\s+holidays?\b|\bstatutory\b|\bbank\s+holidays?\b")
_NEGATED_PUBLIC = re.compile(
    r"\b(?:non|not)[\s\-]+(?:public|statutory|paid)\b|\bnonpublic\b|\bunpaid\b"
)
_CONFLICT = "conflict"

# A column header that says "this column holds the holiday's name".
_NAME_HEADER = re.compile(r"holiday|name|occasion|festival|event|description|title")

# A column header that says "this column holds dates".
_DATE_HEADER = re.compile(r"\bdates?\b")

_MIN_DATE_SHARE = 0.5  # headerless tables: share of a column's cells that must be dates
_MIN_WEEKDAY_SHARE = 0.6  # share of a column's cells that must be weekday names ("Day" column)

_PRIORITY = {KIND_PUBLIC: 2, KIND_OPTIONAL: 1, KIND_UNKNOWN: 0}


@dataclass(frozen=True)
class KindDecision:
    kind: str  # KIND_PUBLIC | KIND_OPTIONAL | KIND_UNKNOWN
    source: str  # "heading" | "caption" | "conflict" | "none"


@dataclass
class TableReport:
    """What happened to one table (for the report, never for logs)."""

    doc_id: str
    table_no: int
    heading: str | None
    holiday_table: bool
    kind: str = KIND_UNKNOWN
    kind_source: str = "none"
    rows: int = 0
    extracted: int = 0
    skipped: int = 0
    note: str = ""


@dataclass
class Extraction:
    calendars: list[CalendarRow] = field(default_factory=list)
    holidays: list[HolidayRow] = field(default_factory=list)
    tables: list[TableReport] = field(default_factory=list)


@dataclass(frozen=True)
class _Pick:
    kind: str
    name: str


# --------------------------------------------------------------------------- #
# Kind of a table
# --------------------------------------------------------------------------- #


def _signal(text: str) -> str | None:
    """KIND_PUBLIC / KIND_OPTIONAL / _CONFLICT, or None if the text says nothing."""
    lowered = text.lower()
    negated = bool(_NEGATED_PUBLIC.search(lowered))
    lowered = _NEGATED_PUBLIC.sub(" ", lowered)
    public = bool(_PUBLIC_WORDS.search(lowered))
    optional = bool(_OPTIONAL_WORDS.search(lowered))
    if optional and public:
        return _CONFLICT
    if optional:
        return KIND_OPTIONAL
    if negated:
        return _CONFLICT  # explicitly "not public", but not said to be optional
    if public:
        return KIND_PUBLIC
    return None


def classify_kind(heading: str | None, caption: str | None) -> KindDecision:
    """Decide public / optional from the table's own heading, then its caption.

    The heading wins when it is decisive. Contradictory wording, or no wording
    at all, gives ``unknown`` (treated as not public; never guessed).
    """
    for source, text in (("heading", heading), ("caption", caption)):
        if not text:
            continue
        signal = _signal(text)
        if signal == _CONFLICT:
            return KindDecision(KIND_UNKNOWN, "conflict")
        if signal is not None:
            return KindDecision(signal, source)
    return KindDecision(KIND_UNKNOWN, "none")


# --------------------------------------------------------------------------- #
# Reading one table
# --------------------------------------------------------------------------- #


def _grid(rows: tuple) -> tuple[list[str], list[dict[str, str] | None]]:
    """Column names and, per row, ``{column: value}`` (None if the row's shape is odd).

    With a header row every value is keyed by its column header. Without one,
    columns are positional, and a row with a different number of cells (an
    empty cell was dropped, so positions are unreliable) is marked None.
    """
    if all(header for row in rows for header, _ in row):
        columns: list[str] = []
        cells: list[dict[str, str] | None] = []
        for row in rows:
            mapping: dict[str, str] = {}
            for header, value in row:
                if header not in columns:
                    columns.append(header)
                mapping.setdefault(header, value)
            cells.append(mapping)
        return columns, cells

    width = Counter(len(row) for row in rows).most_common(1)[0][0]
    columns = [f"#{i + 1}" for i in range(width)]
    return columns, [
        {columns[i]: value for i, (_, value) in enumerate(row)} if len(row) == width else None
        for row in rows
    ]


def _date_column(
    columns: list[str],
    cells: list[dict[str, str] | None],
    doc_id: str,
    table_no: int,
    declared_holiday_list: bool,
) -> str | None:
    """The column that holds the dates, or None if this is not a holiday table.

    A column whose HEADER says "date" is the date column however many of its
    cells parse: unreadable cells must surface as skipped, logged rows, never
    make the whole table disappear. With no such header (a headerless table)
    the column in which at least half of the cells parse as dates is used.
    """
    hits = {
        column: sum(
            1 for row in cells if row and column in row and parse_date_cell(row[column]) is not None
        )
        for column in columns
    }

    def strongest(candidates: list[str]) -> str:
        return max(candidates, key=lambda c: (hits[c], -columns.index(c)))

    labelled = [c for c in columns if _DATE_HEADER.search(c.lower())]
    if labelled:
        return strongest(labelled)
    if not columns:
        return None
    best = strongest(columns)
    if hits[best] == 0:
        return None
    if hits[best] < _MIN_DATE_SHARE * len(cells):
        # Some dates, but too few to call it a date column. Quiet for an info box
        # that merely contains one date; loud for a table that says it lists holidays.
        level = logging.WARNING if declared_holiday_list else logging.DEBUG
        log.log(
            level,
            "doc_id=%s table=%d: dates in only %d of %d rows; table not read as a holiday list",
            doc_id, table_no, hits[best], len(cells),
        )
        return None
    return best


def _weekday_column(
    columns: list[str], cells: list[dict[str, str] | None], skip: str
) -> str | None:
    for column in columns:
        if column == skip:
            continue
        values = [row[column] for row in cells if row and column in row]
        weekdays = sum(1 for v in values if parse_weekday(v) is not None)
        if weekdays and weekdays >= _MIN_WEEKDAY_SHARE * len(values):
            return column
    return None


def _name_column(candidates: list[str]) -> str | None:
    """The one column that holds holiday names, or None if that is not clear."""
    if len(candidates) == 1:
        return candidates[0]
    labelled = [c for c in candidates if _NAME_HEADER.search(c.lower())]
    return labelled[0] if len(labelled) == 1 else None


def _read_table(
    doc_id: str, table_no: int, section: Section, caption: str | None
) -> tuple[TableReport, list[tuple[str, _Pick]], int]:
    """(report, [(iso date, pick)], skipped row count) for one table."""
    report = TableReport(doc_id, table_no, section.heading, holiday_table=False, rows=len(section.rows))
    columns, cells = _grid(section.rows)
    decision = classify_kind(section.heading, caption)
    date_col = _date_column(
        columns, cells, doc_id, table_no, declared_holiday_list=decision.kind != KIND_UNKNOWN
    )
    if date_col is None:
        report.note = "no date column: not a holiday table"
        log.debug("doc_id=%s table=%d: not a holiday table", doc_id, table_no)
        return report, [], 0
    report.holiday_table = True
    report.kind, report.kind_source = decision.kind, decision.source
    if decision.kind == KIND_UNKNOWN:
        log.warning(
            "doc_id=%s table=%d: kind cannot be determined from its heading (%s); "
            "rows are kept as 'unknown' and never count as public holidays",
            doc_id, table_no, decision.source,
        )

    weekday_col = _weekday_column(columns, cells, skip=date_col)
    name_col = _name_column([c for c in columns if c not in (date_col, weekday_col)])
    if name_col is None:
        report.skipped = len(cells)
        report.note = "no single holiday-name column: table skipped"
        log.warning(
            "doc_id=%s table=%d: cannot tell which column holds the holiday name; "
            "all %d rows skipped", doc_id, table_no, len(cells),
        )
        return report, [], len(cells)

    picks: list[tuple[str, _Pick]] = []
    skipped = 0
    for row_no, row in enumerate(cells, start=1):
        reason = None
        parsed = None
        name = ""
        if row is None:
            reason = "row has a different shape"
        elif date_col not in row:
            reason = "no date cell"
        else:
            parsed = parse_date_cell(row[date_col])
            name = " ".join(row.get(name_col, "").split())
            if parsed is None:
                reason = "date not understood"
            elif not name:
                reason = "no holiday name"
        if reason or parsed is None:
            skipped += 1
            log.warning("skipping holiday row doc_id=%s table=%d row=%d: %s", doc_id, table_no, row_no, reason)
            continue

        named = parsed.weekday
        if named is None and weekday_col and weekday_col in row:
            named = parse_weekday(row[weekday_col])
        if named is not None and named != parsed.value.weekday():
            log.warning(
                "doc_id=%s table=%d row=%d: the weekday does not match the date; the date is kept",
                doc_id, table_no, row_no,
            )
        picks.append((parsed.value.isoformat(), _Pick(decision.kind, name)))

    report.skipped = skipped
    report.extracted = len(picks)
    return report, picks, skipped


# --------------------------------------------------------------------------- #
# One calendar document, then all of them
# --------------------------------------------------------------------------- #


def _caption_for(sections: list[Section], index: int) -> str | None:
    """The text directly above a table under the same heading, if any."""
    if index == 0:
        return None
    previous = sections[index - 1]
    if previous.kind == "text" and previous.heading == sections[index].heading:
        return previous.text
    return None


def _extract_calendar(
    entry: ManifestEntry, sections: list[Section]
) -> tuple[CalendarRow, list[HolidayRow], list[TableReport]]:
    chosen: dict[str, _Pick] = {}  # iso date -> best pick (public > optional > unknown)
    reports: list[TableReport] = []
    skipped_total = 0
    table_no = 0
    for index, section in enumerate(sections):
        if section.kind != "table":
            continue
        table_no += 1
        if not section.rows:
            continue
        report, picks, skipped = _read_table(
            entry.doc_id, table_no, section, _caption_for(sections, index)
        )
        reports.append(report)
        skipped_total += skipped
        for iso, pick in picks:
            existing = chosen.get(iso)
            if existing is None:
                chosen[iso] = pick
                continue
            log.warning(
                "doc_id=%s table=%d: a date is listed more than once (%s / %s); "
                "the stronger kind is kept", entry.doc_id, table_no, existing.kind, pick.kind,
            )
            if _PRIORITY[pick.kind] > _PRIORITY[existing.kind]:
                chosen[iso] = pick

    holidays = [
        HolidayRow(region, iso, pick.name, entry.doc_id, pick.kind, entry.effective_date, entry.regions)
        for iso, pick in sorted(chosen.items())
        for region in entry.regions
    ]
    counts = Counter(pick.kind for pick in chosen.values())
    calendar = CalendarRow(
        doc_id=entry.doc_id,
        effective_date=entry.effective_date,
        regions=entry.regions,
        access_level=entry.access_level,
        status=CALENDAR_OK if counts[KIND_PUBLIC] else CALENDAR_NO_PUBLIC,
        public_count=counts[KIND_PUBLIC],
        optional_count=counts[KIND_OPTIONAL],
        unknown_count=counts[KIND_UNKNOWN],
        skipped_rows=skipped_total,
    )
    return calendar, holidays, reports


def _unavailable(entry: ManifestEntry) -> CalendarRow:
    return CalendarRow(
        doc_id=entry.doc_id, effective_date=entry.effective_date, regions=entry.regions,
        access_level=entry.access_level, status=CALENDAR_UNAVAILABLE,
    )


def extract_holidays(
    entries: list[ManifestEntry], documents: list[LoadedDocument]
) -> Extraction:
    """Holidays of every calendar document in the manifest.

    Every calendar entry yields a ``CalendarRow``, even when its file is
    missing or unreadable (status ``unavailable``), so a later lookup can tell
    "the newest calendar is unusable" from "there is no calendar". One broken
    document never stops the others.
    """
    parsed = {d.entry.doc_id: d.parsed for d in documents}
    result = Extraction()
    for entry in entries:
        if entry.doc_type != CALENDAR_DOC_TYPE:
            continue
        document = parsed.get(entry.doc_id)
        if document is None:
            log.warning("calendar doc_id=%s was not parsed: no holidays from it", entry.doc_id)
            result.calendars.append(_unavailable(entry))
            continue
        try:
            calendar, holidays, reports = _extract_calendar(entry, document.sections)
        except Exception as exc:  # noqa: BLE001 - one calendar must not stop startup
            log.error("holiday extraction failed doc_id=%s error=%s", entry.doc_id, type(exc).__name__)
            result.calendars.append(_unavailable(entry))
            continue
        result.calendars.append(calendar)
        result.holidays.extend(holidays)
        result.tables.extend(reports)
        log.info(
            "calendar doc_id=%s status=%s public=%d optional=%d unknown=%d skipped_rows=%d",
            entry.doc_id, calendar.status, calendar.public_count, calendar.optional_count,
            calendar.unknown_count, calendar.skipped_rows,
        )
    if not result.calendars:
        log.warning("no calendar documents in the manifest: no holidays extracted")
    return result
