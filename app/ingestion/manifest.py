"""manifest.json loading.

The manifest is the source of every chunk's metadata (regions, effective date,
doc type, access level). Values are validated for shape only: no corpus facts
(allowed access levels, region codes, ...) are hard-coded here.

A bad entry is skipped and logged by ``doc_id`` only (never titles or content).
A manifest that cannot be read at all raises ``ManifestError``; the pipeline
turns that into a clean "not ready" state rather than a crash.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.logging_setup import get_logger

log = get_logger(__name__)

_REQUIRED_TEXT_FIELDS = ("doc_id", "file", "title", "doc_type", "effective_date", "access_level")


class ManifestError(RuntimeError):
    """The manifest file itself is unusable."""


@dataclass(frozen=True)
class ManifestEntry:
    doc_id: str
    file: str
    title: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: str  # ISO yyyy-mm-dd
    access_level: str


def _parse_entry(raw: object) -> tuple[ManifestEntry | None, str, str]:
    """Return ``(entry, doc_id_for_logging, problem)``; entry is None on a problem.

    ``problem`` is a fixed short code, never text taken from the entry.
    """
    if not isinstance(raw, dict):
        return None, "?", "entry is not an object"
    doc_id = raw.get("doc_id")
    safe_id = doc_id if isinstance(doc_id, str) and doc_id.strip() else "?"
    for field in _REQUIRED_TEXT_FIELDS:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            return None, safe_id, f"missing or empty field '{field}'"
    regions = raw.get("regions")
    if (
        not isinstance(regions, list)
        or not regions
        or not all(isinstance(r, str) and r.strip() for r in regions)
    ):
        return None, safe_id, "regions must be a non-empty list of strings"
    try:
        date.fromisoformat(raw["effective_date"])
    except ValueError:
        return None, safe_id, "effective_date is not an ISO date"
    file_name = raw["file"].strip()
    if Path(file_name).name != file_name or file_name in (".", ".."):
        return None, safe_id, "file must be a bare file name inside the corpus folder"
    return (
        ManifestEntry(
            doc_id=raw["doc_id"].strip(),
            file=file_name,
            title=raw["title"].strip(),
            doc_type=raw["doc_type"].strip(),
            regions=tuple(r.strip() for r in regions),
            effective_date=raw["effective_date"],
            access_level=raw["access_level"].strip(),
        ),
        safe_id,
        "",
    )


def load_manifest(path: Path) -> list[ManifestEntry]:
    """Valid entries in manifest order. Invalid or duplicate entries are skipped."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"manifest unreadable ({type(exc).__name__})") from exc
    if not isinstance(data, list):
        raise ManifestError("manifest must be a JSON array")

    entries: list[ManifestEntry] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    for raw in data:
        entry, safe_id, problem = _parse_entry(raw)
        if entry is None:
            log.warning("skipping manifest entry doc_id=%s: %s", safe_id, problem)
            continue
        if entry.doc_id in seen_ids:
            log.warning("skipping manifest entry doc_id=%s: duplicate doc_id", entry.doc_id)
            continue
        if entry.file in seen_files:
            log.warning("skipping manifest entry doc_id=%s: file already listed", entry.doc_id)
            continue
        seen_ids.add(entry.doc_id)
        seen_files.add(entry.file)
        entries.append(entry)
    return entries
