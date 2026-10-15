"""Document eligibility, precedence and upcoming changes (README B4).

Pure, deterministic Python: no database, no LLM, no clock, no I/O. Everything
it needs (role, region, reference date, the claims) is passed in. The role and
region must come from the requester's DB record, never from a message.

Claims are one document's value for one rule/topic. Later phases produce them
from retrieved chunks; this module only decides between them.

Order of the rules (B4):

1. Eligibility, applied first and never skipped: the role can read the
   document's access_level, its regions include the requester's region or
   ``GLOBAL``, and its effective_date is on or before the reference date. A
   claim that fails any of these takes no part in anything below, so it can
   neither win, nor tie, nor be reported.
2. Precedence among eligible claims, each test applied only to what the
   previous one left:
   authority (policy, addendum, memo, handbook, calendar outrank faq, wiki; an
   unknown document type has no authority), then scope (a document whose
   regions do not include GLOBAL outranks a GLOBAL one), then recency (the
   later effective_date wins).
3. Whatever is still level after all three tests is a tie. If the tied claims
   all state the same value it is not a conflict: the status is "decided" and
   every agreeing claim is returned for citation. Different values give status
   "conflict" and all the tied claims. Nothing is ever picked arbitrarily.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

# --------------------------------------------------------------------------- #
# Rules from the spec (README B3 / B4). These are policy about HOW to decide,
# not corpus facts.
# --------------------------------------------------------------------------- #

GLOBAL_REGION = "GLOBAL"

# B3: which access_level values each role may read.
ROLE_ACCESS_LEVELS: dict[str, frozenset[str]] = {
    "employee": frozenset({"all"}),
    "manager": frozenset({"all", "managers"}),
    "hr": frozenset({"all", "managers", "hr"}),
}

# B4: these document types outrank the rest; any other type (faq, wiki, or
# something unknown) has no authority.
AUTHORITATIVE_TYPES = frozenset({"policy", "addendum", "memo", "handbook", "calendar"})

STATUS_DECIDED = "decided"
STATUS_CONFLICT = "conflict"
STATUS_NONE = "none"

# What settled a decided result (see Resolution.decided_by).
DECIDED_BY_ONLY_CLAIM = "only_claim"
DECIDED_BY_AUTHORITY = "authority"
DECIDED_BY_SCOPE = "scope"
DECIDED_BY_RECENCY = "recency"
DECIDED_BY_AGREEMENT = "agreement"


# --------------------------------------------------------------------------- #
# Small normalisers
# --------------------------------------------------------------------------- #


def _norm(value: object) -> str:
    """Lower-cased, trimmed text; anything that is not text becomes ''."""
    return value.strip().casefold() if isinstance(value, str) else ""


def _topic_key(topic: str) -> str:
    return " ".join(topic.casefold().split())


def _as_date(value: object, what: str = "date") -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"{what} must be a datetime.date")


_PLAIN_NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")


def _value_key(value: str | int | float) -> tuple[str, object]:
    """What "the same value" means: numbers compare numerically (3 == 3.0 == "3"),
    text compares ignoring case and spacing. "Three" is not 3: no guessing."""
    if isinstance(value, str):
        text = " ".join(value.casefold().split())
        if _PLAIN_NUMBER.fullmatch(text):
            return ("number", Decimal(text).normalize())
        return ("text", text)
    return ("number", Decimal(str(value)).normalize())


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DocMeta:
    """The manifest facts about one document that eligibility and precedence use."""

    doc_id: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: date
    access_level: str

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str):
            raise TypeError("doc_id must be text")
        if not isinstance(self.doc_type, str) or not isinstance(self.access_level, str):
            raise TypeError("doc_type and access_level must be text")
        if isinstance(self.regions, (str, bytes)) or not isinstance(self.regions, Iterable):
            raise TypeError("regions must be a list of region codes, not a single string")
        if isinstance(self.effective_date, datetime) or not isinstance(self.effective_date, date):
            raise TypeError("effective_date must be a datetime.date")
        regions = tuple(r.strip().upper() for r in self.regions if isinstance(r, str) and r.strip())
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "doc_type", _norm(self.doc_type))
        object.__setattr__(self, "access_level", _norm(self.access_level))


@dataclass(frozen=True)
class Claim:
    """One document's value for one rule/topic, with where it was found (locator)."""

    topic: str
    value: str | int | float
    doc: DocMeta
    locator: str

    def __post_init__(self) -> None:
        if not isinstance(self.topic, str) or not isinstance(self.locator, str):
            raise TypeError("topic and locator must be text")
        if not isinstance(self.doc, DocMeta):
            raise TypeError("doc must be a DocMeta")
        value = self.value
        ok = isinstance(value, (str, int, float)) and not isinstance(value, bool)
        if not ok or (isinstance(value, float) and not math.isfinite(value)):
            raise TypeError("a claim value must be text or a finite number")


@dataclass(frozen=True)
class Requester:
    """Who is asking. Both fields come from the DB record, never from a message."""

    role: str
    region: str


@dataclass(frozen=True)
class Resolution:
    """Outcome of ``resolve_topic``.

    * ``status`` "decided": ``winner`` is the claim to state; ``claims`` are all
      the claims that carry that answer (one, or several that agree), to cite.
    * ``status`` "conflict": ``winner`` is None; ``claims`` are the tied claims
      with differing values; the answer must give both and say they conflict.
    * ``status`` "none": no eligible claim; ``claims`` is empty.

    ``decided_by`` says what settled a decided result: "only_claim", "authority",
    "scope", "recency" (the test that left a single claim) or "agreement" (tied
    claims with equal values). None for "conflict" and "none".
    Claims are in a fixed order (doc_id, locator) whatever order they came in.
    """

    status: str
    winner: Claim | None = None
    claims: tuple[Claim, ...] = ()
    decided_by: str | None = None


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #


def readable_access_levels(role: str) -> frozenset[str]:
    """The access_level values ``role`` may read (B3). An unknown role reads nothing."""
    return ROLE_ACCESS_LEVELS.get(_norm(role), frozenset())


def can_read(doc: DocMeta, role: str) -> bool:
    return doc.access_level in readable_access_levels(role)


def covers_region(doc: DocMeta, region: str | None) -> bool:
    """True if the document's regions include ``region`` or GLOBAL."""
    wanted = region.strip().upper() if isinstance(region, str) else ""
    return GLOBAL_REGION in doc.regions or (wanted != "" and wanted in doc.regions)


def is_eligible(doc: DocMeta, role: str, region: str | None, reference_date: date) -> bool:
    """B4 rule 1: readable by the role, covers the region, and already in force."""
    return (
        can_read(doc, role)
        and covers_region(doc, region)
        and doc.effective_date <= _as_date(reference_date, "reference_date")
    )


# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #


def _authority(claim: Claim) -> int:
    return 1 if claim.doc.doc_type in AUTHORITATIVE_TYPES else 0


def _scope(claim: Claim) -> int:
    return 0 if GLOBAL_REGION in claim.doc.regions else 1  # regional outranks GLOBAL


def _recency(claim: Claim) -> int:
    return claim.doc.effective_date.toordinal()


_TESTS = (
    (DECIDED_BY_AUTHORITY, _authority),
    (DECIDED_BY_SCOPE, _scope),
    (DECIDED_BY_RECENCY, _recency),
)


def _order(claim: Claim) -> tuple[str, str, str]:
    return (claim.doc.doc_id, claim.locator, repr(_value_key(claim.value)))


def _checked(claims: Iterable[Claim]) -> list[Claim]:
    items = list(claims)
    if len({_topic_key(c.topic) for c in items}) > 1:
        # Mixing topics would compare unrelated values. The message names no topic or document.
        raise ValueError("claims for different topics cannot be resolved together")
    return items


def resolve_topic(
    claims: Iterable[Claim], reference_date: date, requester: Requester
) -> Resolution:
    """Decide which value of one topic applies to ``requester`` on ``reference_date``.

    Only eligible claims are considered (filtered first); then authority, scope
    and recency are applied in that order, each only to the claims the previous
    test left level. See the module docstring for ties and conflicts.
    """
    items = _checked(claims)
    ref = _as_date(reference_date, "reference_date")
    remaining = sorted(
        (c for c in items if is_eligible(c.doc, requester.role, requester.region, ref)),
        key=_order,
    )
    if not remaining:
        return Resolution(STATUS_NONE)
    if len(remaining) == 1:
        return Resolution(STATUS_DECIDED, remaining[0], (remaining[0],), DECIDED_BY_ONLY_CLAIM)

    decided_by = None
    for name, rank in _TESTS:
        best = max(rank(c) for c in remaining)
        narrowed = [c for c in remaining if rank(c) == best]
        if len(narrowed) < len(remaining):
            decided_by = name
            remaining = narrowed
        if len(remaining) == 1:
            return Resolution(STATUS_DECIDED, remaining[0], (remaining[0],), decided_by)

    # Still level after all three tests.
    if len({_value_key(c.value) for c in remaining}) == 1:
        return Resolution(STATUS_DECIDED, remaining[0], tuple(remaining), DECIDED_BY_AGREEMENT)
    return Resolution(STATUS_CONFLICT, None, tuple(remaining))


# --------------------------------------------------------------------------- #
# Upcoming changes
# --------------------------------------------------------------------------- #


def upcoming_change(
    claims: Iterable[Claim], requester: Requester, reference_date: date
) -> Claim | None:
    """The next change the requester should be told about, or None.

    A claim qualifies only if all of these hold:

    * its document is effective AFTER the reference date (one effective on the
      reference date is current, not upcoming);
    * the requester's role may read it and it covers the requester's region
      (so an unreadable or other-region document is never reported);
    * it would win under the precedence rules ON ITS OWN effective date, with
      eligibility and precedence recomputed as of that date over all the
      topic's claims the requester may read. So a regional addendum that beats
      a future global policy makes the policy a non-event, and a future FAQ
      never qualifies; a claim that would only tie there does not qualify;
    * its value differs from the value in force today (a future document that
      says the same thing changes nothing). If nothing is decided today, any
      qualifying claim counts.

    Of several qualifying claims the nearest effective date is returned (ties
    by doc_id), so the result does not depend on the order of ``claims``.
    """
    items = _checked(claims)
    ref = _as_date(reference_date, "reference_date")
    current = resolve_topic(items, ref, requester)
    current_key = _value_key(current.winner.value) if current.status == STATUS_DECIDED else None

    candidates = sorted(
        (
            c
            for c in items
            if c.doc.effective_date > ref
            and can_read(c.doc, requester.role)
            and covers_region(c.doc, requester.region)
        ),
        key=lambda c: (c.doc.effective_date, *_order(c)),
    )
    on_date: dict[date, Resolution] = {}
    for candidate in candidates:
        when = candidate.doc.effective_date
        if when not in on_date:
            on_date[when] = resolve_topic(items, when, requester)
        outcome = on_date[when]
        if outcome.status != STATUS_DECIDED or candidate not in outcome.claims:
            continue
        if current_key is not None and _value_key(candidate.value) == current_key:
            continue
        return candidate
    return None
