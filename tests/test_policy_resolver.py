"""README B4: eligibility, precedence and upcoming changes. Pure functions only.

The documents and numbers below are the worked example from README B4 and other
synthetic cases; nothing here is read from the corpus.
"""
from __future__ import annotations

import ast
import itertools
import random
from datetime import date, datetime
from pathlib import Path

import pytest

from app.retrieval.policy_resolver import (
    Claim,
    DocMeta,
    Requester,
    is_eligible,
    readable_access_levels,
    resolve_topic,
    upcoming_change,
)

TODAY = date(2026, 9, 15)
TOPIC = "leave_days"


def doc(doc_id, doc_type="policy", regions=("GLOBAL",), effective=date(2025, 1, 1), access="all"):
    return DocMeta(doc_id, doc_type, tuple(regions), effective, access)


def claim(value, d, topic=TOPIC, locator="s1"):
    return Claim(topic, value, d, locator)


def ids(claims):
    return sorted(c.doc.doc_id for c in claims)


EMPLOYEE_IN = Requester("employee", "IN")
EMPLOYEE_DE = Requester("employee", "DE")
MANAGER_IN = Requester("manager", "IN")
HR_IN = Requester("hr", "IN")

# --- README B4 worked example (today 2026-09-15) -------------------------------

V1 = doc("POL-V1", "policy", ("GLOBAL",), date(2025, 1, 1))
V2 = doc("POL-V2", "policy", ("GLOBAL",), date(2026, 4, 1))
FAQ = doc("FAQ-1", "faq", ("GLOBAL",), date(2026, 5, 1))
ADDENDUM_DE = doc("ADD-DE", "addendum", ("DE",), date(2025, 6, 1))
V3 = doc("POL-V3", "policy", ("GLOBAL",), date(2027, 1, 1))

EXAMPLE = [
    claim(2, V1), claim(3, V2), claim(2, FAQ), claim(5, ADDENDUM_DE), claim(4, V3),
]


def test_worked_example_india_employee_gets_v2_and_is_told_about_v3():
    resolution = resolve_topic(EXAMPLE, TODAY, EMPLOYEE_IN)
    assert resolution.status == "decided"
    assert resolution.winner.value == 3 and resolution.winner.doc.doc_id == "POL-V2"

    upcoming = upcoming_change(EXAMPLE, EMPLOYEE_IN, TODAY)
    assert upcoming is not None
    assert (upcoming.doc.doc_id, upcoming.value, upcoming.doc.effective_date) == (
        "POL-V3", 4, date(2027, 1, 1),
    )


def test_worked_example_germany_employee_gets_the_addendum_and_no_upcoming_change():
    resolution = resolve_topic(EXAMPLE, TODAY, EMPLOYEE_DE)
    assert resolution.status == "decided"
    assert resolution.winner.value == 5 and resolution.winner.doc.doc_id == "ADD-DE"
    assert upcoming_change(EXAMPLE, EMPLOYEE_DE, TODAY) is None  # the addendum beats v3 on scope


def test_worked_example_india_employee_asking_about_2025_08_01_gets_v1():
    resolution = resolve_topic(EXAMPLE, date(2025, 8, 1), EMPLOYEE_IN)
    assert resolution.status == "decided"
    assert resolution.winner.value == 2 and resolution.winner.doc.doc_id == "POL-V1"


def test_worked_example_germany_in_2025_08_01_gets_the_addendum_already():
    resolution = resolve_topic(EXAMPLE, date(2025, 8, 1), EMPLOYEE_DE)
    assert resolution.winner.doc.doc_id == "ADD-DE"


def test_worked_example_before_the_addendum_germany_falls_back_to_v1():
    resolution = resolve_topic(EXAMPLE, date(2025, 5, 31), EMPLOYEE_DE)
    assert resolution.winner.doc.doc_id == "POL-V1"


def test_the_upcoming_change_looking_from_2025_08_01_is_v2_the_earliest_that_wins():
    upcoming = upcoming_change(EXAMPLE, EMPLOYEE_IN, date(2025, 8, 1))
    assert upcoming.doc.doc_id == "POL-V2" and upcoming.value == 3


# --- eligibility ---------------------------------------------------------------

ROLE_LEVELS = {
    "employee": {"all"},
    "manager": {"all", "managers"},
    "hr": {"all", "managers", "hr"},
}


@pytest.mark.parametrize("role", sorted(ROLE_LEVELS))
@pytest.mark.parametrize("level", ["all", "managers", "hr"])
def test_role_reads_exactly_its_access_levels(role, level):
    d = doc("D", access=level)
    assert is_eligible(d, role, "IN", TODAY) is (level in ROLE_LEVELS[role])


@pytest.mark.parametrize("role", sorted(ROLE_LEVELS))
def test_readable_access_levels_match_b3(role):
    assert readable_access_levels(role) == frozenset(ROLE_LEVELS[role])


@pytest.mark.parametrize("role", ["", None, "admin", "contractor", "hr manager", "root", 7])
def test_an_unknown_role_reads_nothing(role):
    assert readable_access_levels(role) == frozenset()
    assert is_eligible(doc("D", access="all"), role, "IN", TODAY) is False


def test_role_names_are_matched_ignoring_case_and_padding():
    assert readable_access_levels("HR") == readable_access_levels("hr")
    assert readable_access_levels(" hr ") == readable_access_levels("hr")
    assert is_eligible(doc("D", access="managers"), "Manager", "IN", TODAY)


@pytest.mark.parametrize("level", ["secret", "", "everyone", "hr,managers", "all managers"])
def test_an_unknown_access_level_is_never_readable(level):
    for role in ROLE_LEVELS:
        assert is_eligible(doc("D", access=level), role, "IN", TODAY) is False


def test_access_levels_are_matched_ignoring_case_and_padding():
    assert is_eligible(doc("D", access=" ALL "), "employee", "IN", TODAY)
    assert not is_eligible(doc("D", access="HR"), "manager", "IN", TODAY)
    assert is_eligible(doc("D", access="HR"), "hr", "IN", TODAY)


def test_regions_must_include_the_requesters_region_or_global():
    assert is_eligible(doc("D", regions=("IN",)), "employee", "IN", TODAY)
    assert is_eligible(doc("D", regions=("IN", "US")), "employee", "US", TODAY)
    assert is_eligible(doc("D", regions=("GLOBAL",)), "employee", "DE", TODAY)
    assert not is_eligible(doc("D", regions=("IN", "US")), "employee", "DE", TODAY)


def test_region_matching_ignores_case_and_padding():
    assert is_eligible(doc("D", regions=("in",)), "employee", " IN ", TODAY)
    assert is_eligible(doc("D", regions=("Global",)), "employee", "de", TODAY)


@pytest.mark.parametrize("region", ["", None, "  "])
def test_a_requester_without_a_region_matches_only_global_documents(region):
    assert is_eligible(doc("D", regions=("GLOBAL",)), "employee", region, TODAY)
    assert not is_eligible(doc("D", regions=("IN",)), "employee", region, TODAY)


def test_a_document_without_regions_matches_nobody():
    assert not is_eligible(doc("D", regions=()), "employee", "IN", TODAY)


def test_effective_date_is_inclusive_and_the_future_is_not_eligible():
    d = doc("D", effective=date(2026, 9, 15))
    assert is_eligible(d, "employee", "IN", date(2026, 9, 15))
    assert is_eligible(d, "employee", "IN", date(2026, 9, 16))
    assert not is_eligible(d, "employee", "IN", date(2026, 9, 14))


def test_all_three_conditions_are_required_together():
    d = doc("D", regions=("DE",), effective=date(2027, 1, 1), access="hr")
    assert not is_eligible(d, "employee", "IN", TODAY)
    for role, region, ref in itertools.product(("employee", "hr"), ("IN", "DE"), (TODAY, date(2027, 1, 1))):
        expected = role == "hr" and region == "DE" and ref >= d.effective_date
        assert is_eligible(d, role, region, ref) is expected


def test_doc_meta_normalises_regions_and_requires_real_dates():
    d = DocMeta("D", "Policy", ["in", " Global "], date(2025, 1, 1), "all")
    assert d.regions == ("IN", "GLOBAL") and d.doc_type == "policy"
    with pytest.raises(TypeError):
        DocMeta("D", "policy", ("IN",), "2025-01-01", "all")
    with pytest.raises(TypeError):
        DocMeta("D", "policy", ("IN",), datetime(2025, 1, 1), "all")
    with pytest.raises(TypeError):
        DocMeta("D", "policy", "IN", date(2025, 1, 1), "all")  # a bare string is not a region list


def test_claim_values_are_text_or_numbers_only():
    d = doc("D")
    assert Claim("t", 3, d, "s").value == 3 and Claim("t", "3 days", d, "s").value == "3 days"
    for bad in (None, True, [3], float("nan")):
        with pytest.raises(TypeError):
            Claim("t", bad, d, "s")


# --- precedence ------------------------------------------------------------------


def test_faq_loses_to_a_policy_on_authority_even_when_newer():
    old_policy = claim(3, doc("POL", "policy", effective=date(2024, 1, 1)))
    new_faq = claim(9, doc("FAQ", "faq", effective=date(2026, 8, 1)))
    resolution = resolve_topic([new_faq, old_policy], TODAY, EMPLOYEE_IN)
    assert resolution.winner == old_policy and resolution.decided_by == "authority"


@pytest.mark.parametrize("high", ["policy", "addendum", "memo", "handbook", "calendar"])
@pytest.mark.parametrize("low", ["faq", "wiki"])
def test_every_authoritative_type_outranks_faq_and_wiki(high, low):
    a = claim(1, doc("A", high, effective=date(2024, 1, 1)))
    b = claim(2, doc("B", low, effective=date(2026, 1, 1)))
    assert resolve_topic([b, a], TODAY, EMPLOYEE_IN).winner == a


@pytest.mark.parametrize("a_type", ["policy", "addendum", "memo", "handbook", "calendar"])
@pytest.mark.parametrize("b_type", ["policy", "addendum", "memo", "handbook", "calendar"])
def test_authoritative_types_are_equals_so_recency_decides_between_them(a_type, b_type):
    older = claim(1, doc("A", a_type, effective=date(2025, 1, 1)))
    newer = claim(2, doc("B", b_type, effective=date(2026, 1, 1)))
    resolution = resolve_topic([older, newer], TODAY, EMPLOYEE_IN)
    assert resolution.winner == newer and resolution.decided_by == "recency"


def test_faq_and_wiki_are_equals_so_recency_decides_between_them():
    faq = claim(1, doc("F", "faq", effective=date(2025, 1, 1)))
    wiki = claim(2, doc("W", "wiki", effective=date(2026, 1, 1)))
    assert resolve_topic([faq, wiki], TODAY, EMPLOYEE_IN).winner == wiki


def test_an_unknown_document_type_has_no_authority():
    policy = claim(3, doc("POL", "policy", effective=date(2024, 1, 1)))
    other = claim(9, doc("BLOG", "blog", effective=date(2026, 1, 1)))
    assert resolve_topic([other, policy], TODAY, EMPLOYEE_IN).winner == policy


def test_document_types_are_matched_case_insensitively():
    upper = claim(3, doc("POL", "POLICY", effective=date(2024, 1, 1)))
    faq = claim(9, doc("FAQ", "faq", effective=date(2026, 1, 1)))
    assert resolve_topic([faq, upper], TODAY, EMPLOYEE_IN).winner == upper


@pytest.mark.parametrize("type_", ["memo", "addendum", "policy"])
def test_a_regional_document_beats_an_older_or_newer_global_policy_on_scope(type_):
    regional = claim(5, doc("REG", type_, ("IN",), effective=date(2025, 6, 1)))
    global_new = claim(3, doc("GLB", "policy", ("GLOBAL",), effective=date(2026, 4, 1)))
    resolution = resolve_topic([global_new, regional], TODAY, EMPLOYEE_IN)
    assert resolution.winner == regional and resolution.decided_by == "scope"


def test_authority_is_checked_before_scope():
    regional_faq = claim(9, doc("RF", "faq", ("IN",), effective=date(2026, 1, 1)))
    global_policy = claim(3, doc("GP", "policy", ("GLOBAL",), effective=date(2025, 1, 1)))
    resolution = resolve_topic([regional_faq, global_policy], TODAY, EMPLOYEE_IN)
    assert resolution.winner == global_policy and resolution.decided_by == "authority"


def test_scope_is_checked_before_recency():
    older_regional = claim(5, doc("R", "policy", ("IN",), effective=date(2020, 1, 1)))
    newer_global = claim(3, doc("G", "policy", ("GLOBAL",), effective=date(2026, 9, 1)))
    assert resolve_topic([newer_global, older_regional], TODAY, EMPLOYEE_IN).winner == older_regional


def test_a_document_that_lists_global_among_its_regions_counts_as_global():
    mixed = claim(1, doc("MIX", "policy", ("IN", "GLOBAL"), effective=date(2025, 1, 1)))
    regional = claim(2, doc("REG", "policy", ("IN",), effective=date(2024, 1, 1)))
    assert resolve_topic([mixed, regional], TODAY, EMPLOYEE_IN).winner == regional


def test_a_multi_region_document_is_regional_and_beats_global():
    memo = claim(5, doc("M", "memo", ("IN", "US"), effective=date(2025, 6, 1)))
    global_policy = claim(3, doc("G", "policy", ("GLOBAL",), effective=date(2026, 1, 1)))
    assert resolve_topic([global_policy, memo], TODAY, EMPLOYEE_IN).winner == memo


def test_same_authority_and_scope_goes_to_the_later_effective_date():
    old, new = claim(2, doc("OLD", effective=date(2025, 1, 1))), claim(3, doc("NEW", effective=date(2026, 1, 1)))
    resolution = resolve_topic([old, new], TODAY, EMPLOYEE_IN)
    assert resolution.winner == new and resolution.decided_by == "recency"
    assert resolution.claims == (new,)


def test_an_exact_tie_with_different_values_is_a_conflict_listing_both_claims():
    a = claim(2, doc("A", effective=date(2026, 1, 1)), locator="p1")
    b = claim(3, doc("B", "handbook", effective=date(2026, 1, 1)), locator="p2")
    resolution = resolve_topic([a, b], TODAY, EMPLOYEE_IN)
    assert resolution.status == "conflict"
    assert resolution.winner is None
    assert set(resolution.claims) == {a, b} and len(resolution.claims) == 2


def test_an_exact_tie_with_the_same_value_is_decided_and_cites_every_agreeing_document():
    a = claim(3, doc("A", effective=date(2026, 1, 1)), locator="p1")
    b = claim(3, doc("B", "handbook", effective=date(2026, 1, 1)), locator="p2")
    resolution = resolve_topic([b, a], TODAY, EMPLOYEE_IN)
    assert resolution.status == "decided" and resolution.decided_by == "agreement"
    assert resolution.winner.value == 3
    assert ids(resolution.claims) == ["A", "B"]
    assert resolution.winner in resolution.claims


@pytest.mark.parametrize("a, b", [(3, 3.0), (3, "3"), ("3 Days", " 3  days"), ("Three", "three"), (0.5, "0.50")])
def test_values_that_are_the_same_apart_from_formatting_are_not_a_conflict(a, b):
    x = claim(a, doc("A", effective=date(2026, 1, 1)))
    y = claim(b, doc("B", effective=date(2026, 1, 1)))
    assert resolve_topic([x, y], TODAY, EMPLOYEE_IN).status == "decided"


@pytest.mark.parametrize("a, b", [(3, 4), ("3 days", "4 days"), (3, "three"), (0.5, 0.6)])
def test_values_that_really_differ_conflict(a, b):
    x = claim(a, doc("A", effective=date(2026, 1, 1)))
    y = claim(b, doc("B", effective=date(2026, 1, 1)))
    assert resolve_topic([x, y], TODAY, EMPLOYEE_IN).status == "conflict"


def test_three_way_tie_with_one_odd_value_out_is_a_conflict_with_all_three():
    docs = [doc(n, effective=date(2026, 1, 1)) for n in "ABC"]
    claims = [claim(2, docs[0]), claim(2, docs[1]), claim(3, docs[2])]
    resolution = resolve_topic(claims, TODAY, EMPLOYEE_IN)
    assert resolution.status == "conflict" and ids(resolution.claims) == ["A", "B", "C"]


def test_a_tie_only_among_the_top_group_ignores_lower_ranked_claims():
    top1 = claim(2, doc("A", effective=date(2026, 1, 1)))
    top2 = claim(3, doc("B", effective=date(2026, 1, 1)))
    lower = claim(2, doc("C", effective=date(2025, 1, 1)))
    resolution = resolve_topic([top1, top2, lower], TODAY, EMPLOYEE_IN)
    assert resolution.status == "conflict" and ids(resolution.claims) == ["A", "B"]


def test_one_document_stating_two_values_conflicts_with_itself():
    d = doc("A", effective=date(2026, 1, 1))
    resolution = resolve_topic([claim(2, d, locator="s1"), claim(3, d, locator="s2")], TODAY, EMPLOYEE_IN)
    assert resolution.status == "conflict" and len(resolution.claims) == 2


def test_a_single_eligible_claim_wins_and_says_so():
    only = claim(2, doc("A"))
    resolution = resolve_topic([only], TODAY, EMPLOYEE_IN)
    assert (resolution.status, resolution.winner, resolution.decided_by) == ("decided", only, "only_claim")


def test_no_claims_means_none():
    resolution = resolve_topic([], TODAY, EMPLOYEE_IN)
    assert (resolution.status, resolution.winner, resolution.claims) == ("none", None, ())


def test_when_nothing_is_eligible_the_status_is_none_not_a_guess():
    future = claim(9, doc("F", effective=date(2027, 1, 1)))
    other_region = claim(9, doc("R", regions=("DE",)))
    hidden = claim(9, doc("H", access="hr"))
    resolution = resolve_topic([future, other_region, hidden], TODAY, EMPLOYEE_IN)
    assert (resolution.status, resolution.winner, resolution.claims) == ("none", None, ())


def test_ineligible_claims_are_filtered_before_precedence_is_applied():
    # the future policy and the hr-only memo would both outrank the FAQ, but are not eligible
    faq = claim(2, doc("FAQ", "faq", effective=date(2025, 1, 1)))
    future = claim(9, doc("FUT", "policy", effective=date(2027, 1, 1)))
    hr_memo = claim(8, doc("HRM", "memo", effective=date(2026, 1, 1), access="hr"))
    resolution = resolve_topic([future, hr_memo, faq], TODAY, EMPLOYEE_IN)
    assert resolution.winner == faq


def test_the_result_does_not_depend_on_the_order_of_the_claims():
    rng = random.Random(0)
    claims = list(EXAMPLE) + [
        claim(5, doc("DUP-A", "memo", ("DE",), date(2025, 6, 1))),
        claim(6, doc("DUP-B", "handbook", ("DE",), date(2025, 6, 1))),
    ]
    baseline = {r.region: resolve_topic(claims, TODAY, r) for r in (EMPLOYEE_IN, EMPLOYEE_DE)}
    for _ in range(25):
        rng.shuffle(claims)
        for r in (EMPLOYEE_IN, EMPLOYEE_DE):
            assert resolve_topic(claims, TODAY, r) == baseline[r.region]


def test_inputs_are_not_modified_and_any_iterable_is_accepted():
    claims = list(EXAMPLE)
    resolve_topic(claims, TODAY, EMPLOYEE_IN)
    upcoming_change(claims, EMPLOYEE_IN, TODAY)
    assert claims == EXAMPLE
    assert resolve_topic(iter(EXAMPLE), TODAY, EMPLOYEE_IN).winner.doc.doc_id == "POL-V2"
    assert upcoming_change((c for c in EXAMPLE), EMPLOYEE_IN, TODAY).doc.doc_id == "POL-V3"


def test_claims_for_different_topics_are_never_mixed():
    with pytest.raises(ValueError):
        resolve_topic([claim(2, V1, topic="a"), claim(3, V2, topic="b")], TODAY, EMPLOYEE_IN)
    with pytest.raises(ValueError):
        upcoming_change([claim(2, V1, topic="a"), claim(3, V3, topic="b")], EMPLOYEE_IN, TODAY)


def test_topic_names_are_compared_ignoring_case_and_padding():
    resolution = resolve_topic([claim(2, V1, topic="Leave Days"), claim(3, V2, topic=" leave  days ")], TODAY, EMPLOYEE_IN)
    assert resolution.winner.value == 3


def test_a_datetime_reference_date_is_treated_as_its_date():
    assert resolve_topic(EXAMPLE, datetime(2026, 9, 15, 23, 59), EMPLOYEE_IN).winner.doc.doc_id == "POL-V2"
    with pytest.raises(TypeError):
        resolve_topic(EXAMPLE, "2026-09-15", EMPLOYEE_IN)


# --- roles -------------------------------------------------------------------------

POLICY_ALL = claim(2, doc("ALL", effective=date(2025, 1, 1)))
MANAGERS_ONLY = claim(7, doc("MGR", "policy", effective=date(2026, 1, 1), access="managers"))
HR_ONLY = claim(9, doc("HR", "policy", effective=date(2026, 6, 1), access="hr"))
ROLE_CLAIMS = [POLICY_ALL, MANAGERS_ONLY, HR_ONLY]


def test_an_employee_never_gets_a_manager_only_or_hr_only_claim_as_winner():
    resolution = resolve_topic(ROLE_CLAIMS, TODAY, EMPLOYEE_IN)
    assert resolution.winner == POLICY_ALL
    assert ids(resolution.claims) == ["ALL"]


def test_an_employee_with_only_restricted_claims_gets_none():
    resolution = resolve_topic([MANAGERS_ONLY, HR_ONLY], TODAY, EMPLOYEE_IN)
    assert resolution.status == "none" and resolution.claims == ()


def test_a_manager_gets_manager_claims_but_not_hr_claims():
    resolution = resolve_topic(ROLE_CLAIMS, TODAY, MANAGER_IN)
    assert resolution.winner == MANAGERS_ONLY
    assert "HR" not in ids(resolution.claims)


def test_hr_sees_everything():
    assert resolve_topic(ROLE_CLAIMS, TODAY, HR_IN).winner == HR_ONLY


def test_a_restricted_claim_never_takes_part_in_a_conflict_or_agreement():
    visible = claim(3, doc("V", effective=date(2026, 1, 1)))
    hidden_same_rank = claim(4, doc("H", effective=date(2026, 1, 1), access="hr"))
    resolution = resolve_topic([visible, hidden_same_rank], TODAY, EMPLOYEE_IN)
    assert resolution.status == "decided" and resolution.claims == (visible,)


def test_an_employee_never_gets_a_restricted_upcoming_change():
    future_mgr = claim(8, doc("FM", effective=date(2027, 1, 1), access="managers"))
    future_hr = claim(9, doc("FH", effective=date(2027, 2, 1), access="hr"))
    claims = [POLICY_ALL, future_mgr, future_hr]
    assert upcoming_change(claims, EMPLOYEE_IN, TODAY) is None
    assert upcoming_change(claims, MANAGER_IN, TODAY) == future_mgr
    assert upcoming_change(claims, HR_IN, TODAY) == future_mgr  # earliest that wins on its own date


def test_a_manager_does_not_get_an_hr_only_upcoming_change():
    future_hr = claim(9, doc("FH", effective=date(2027, 2, 1), access="hr"))
    assert upcoming_change([POLICY_ALL, future_hr], MANAGER_IN, TODAY) is None
    assert upcoming_change([POLICY_ALL, future_hr], HR_IN, TODAY) == future_hr


def test_a_future_restricted_claim_does_not_block_a_readable_one_from_winning():
    # on 2027-02-01 the hr-only claim would win for HR; for an employee it does not exist
    future_ok = claim(4, doc("FOK", effective=date(2027, 1, 1)))
    future_hr = claim(9, doc("FH", effective=date(2027, 1, 1), access="hr"))
    assert upcoming_change([POLICY_ALL, future_ok, future_hr], EMPLOYEE_IN, TODAY) == future_ok


# --- regions -----------------------------------------------------------------------


def test_a_claim_from_another_region_is_never_used_or_reported():
    de_only = claim(5, doc("DE-ADD", "addendum", ("DE",), date(2026, 1, 1)))
    us_only = claim(6, doc("US-MEMO", "memo", ("US",), date(2026, 2, 1)))
    resolution = resolve_topic([POLICY_ALL, de_only, us_only], TODAY, EMPLOYEE_IN)
    assert resolution.winner == POLICY_ALL and resolution.claims == (POLICY_ALL,)


def test_with_only_other_region_claims_the_answer_is_none():
    de_only = claim(5, doc("DE-ADD", "addendum", ("DE",), date(2026, 1, 1)))
    assert resolve_topic([de_only], TODAY, EMPLOYEE_IN).status == "none"


def test_another_regions_future_claim_is_never_an_upcoming_change():
    future_de = claim(5, doc("FDE", "addendum", ("DE",), date(2027, 1, 1)))
    future_us_memo = claim(6, doc("FUS", "memo", ("US",), date(2026, 12, 1)))
    assert upcoming_change([POLICY_ALL, future_de, future_us_memo], EMPLOYEE_IN, TODAY) is None
    assert upcoming_change([POLICY_ALL, future_de, future_us_memo], EMPLOYEE_DE, TODAY) == future_de


def test_a_conflict_in_another_region_does_not_affect_this_region():
    de_a = claim(5, doc("A", "addendum", ("DE",), date(2026, 1, 1)))
    de_b = claim(6, doc("B", "addendum", ("DE",), date(2026, 1, 1)))
    assert resolve_topic([POLICY_ALL, de_a, de_b], TODAY, EMPLOYEE_IN).winner == POLICY_ALL
    assert resolve_topic([POLICY_ALL, de_a, de_b], TODAY, EMPLOYEE_DE).status == "conflict"


# --- the memo and policy case ----------------------------------------------------------

REMOTE_POLICY = doc("POL-REMOTE", "policy", ("GLOBAL",), date(2025, 1, 1))
REMOTE_MEMO = doc("MEMO-REMOTE", "memo", ("IN", "US"), date(2026, 7, 1))
REMOTE = [claim(2, REMOTE_POLICY, topic="remote_days"), claim(3, REMOTE_MEMO, topic="remote_days")]


@pytest.mark.parametrize("ref", [date(2026, 7, 1), date(2026, 7, 2), date(2026, 9, 15), date(2030, 1, 1)])
def test_the_regional_memo_beats_the_global_policy_for_india_from_its_effective_date(ref):
    resolution = resolve_topic(REMOTE, ref, EMPLOYEE_IN)
    assert resolution.winner.doc.doc_id == "MEMO-REMOTE" and resolution.winner.value == 3


@pytest.mark.parametrize("ref", [date(2026, 6, 30), date(2026, 1, 1), date(2025, 1, 1)])
def test_before_its_effective_date_the_memo_is_not_eligible_and_the_policy_stands(ref):
    resolution = resolve_topic(REMOTE, ref, EMPLOYEE_IN)
    assert resolution.winner.doc.doc_id == "POL-REMOTE"
    assert "MEMO-REMOTE" not in ids(resolution.claims)


def test_the_memo_applies_to_the_us_too_but_never_to_germany():
    assert resolve_topic(REMOTE, TODAY, Requester("employee", "US")).winner.doc.doc_id == "MEMO-REMOTE"
    assert resolve_topic(REMOTE, TODAY, EMPLOYEE_DE).winner.doc.doc_id == "POL-REMOTE"


def test_before_the_memo_starts_it_is_reported_as_an_upcoming_change_for_india_only():
    ref = date(2026, 6, 30)
    upcoming = upcoming_change(REMOTE, EMPLOYEE_IN, ref)
    assert upcoming is not None and upcoming.doc.doc_id == "MEMO-REMOTE"
    assert upcoming_change(REMOTE, EMPLOYEE_DE, ref) is None


def test_once_the_memo_is_in_force_it_is_current_not_upcoming():
    assert upcoming_change(REMOTE, EMPLOYEE_IN, date(2026, 7, 1)) is None


# --- upcoming change: the "would it win on its own date" rule ---------------------------


def test_a_future_global_policy_does_not_beat_a_current_regional_addendum():
    addendum = claim(5, doc("ADD", "addendum", ("DE",), date(2025, 6, 1)))
    future = claim(4, doc("FUT", "policy", ("GLOBAL",), date(2027, 1, 1)))
    assert upcoming_change([addendum, future], EMPLOYEE_DE, TODAY) is None


def test_a_future_faq_never_qualifies_because_it_loses_on_authority():
    policy = claim(3, doc("POL", "policy", effective=date(2026, 1, 1)))
    future_faq = claim(9, doc("FAQ", "faq", effective=date(2027, 1, 1)))
    assert upcoming_change([policy, future_faq], EMPLOYEE_IN, TODAY) is None


def test_a_future_regional_document_that_will_win_by_scope_qualifies():
    policy = claim(3, doc("POL", "policy", ("GLOBAL",), date(2026, 1, 1)))
    future = claim(5, doc("MEMO", "memo", ("IN",), date(2027, 1, 1)))
    assert upcoming_change([policy, future], EMPLOYEE_IN, TODAY) == future


def test_with_several_qualifying_future_documents_the_nearest_change_is_returned():
    policy = claim(3, doc("POL", effective=date(2026, 1, 1)))
    near = claim(4, doc("NEAR", effective=date(2027, 1, 1)))
    far = claim(5, doc("FAR", effective=date(2028, 1, 1)))
    # both would win on their own dates; the next change is the nearest one
    assert upcoming_change([policy, near, far], EMPLOYEE_IN, TODAY) == near
    assert upcoming_change([far, policy, near], EMPLOYEE_IN, TODAY) == near


def test_a_future_document_with_the_same_value_as_today_is_not_a_change():
    policy = claim(3, doc("POL", effective=date(2026, 1, 1)))
    future_same = claim("3", doc("FUT", effective=date(2027, 1, 1)))
    assert upcoming_change([policy, future_same], EMPLOYEE_IN, TODAY) is None


def test_a_same_value_future_document_does_not_hide_a_later_real_change():
    policy = claim(3, doc("POL", effective=date(2026, 1, 1)))
    same = claim(3, doc("SAME", effective=date(2027, 1, 1)))
    different = claim(4, doc("DIFF", effective=date(2028, 1, 1)))
    assert upcoming_change([policy, same, different], EMPLOYEE_IN, TODAY) == different


def test_a_future_document_that_would_only_tie_on_its_date_is_not_reported():
    policy = claim(3, doc("POL", effective=date(2026, 1, 1)))
    a = claim(4, doc("A", effective=date(2027, 1, 1)))
    b = claim(5, doc("B", effective=date(2027, 1, 1)))
    assert upcoming_change([policy, a, b], EMPLOYEE_IN, TODAY) is None


def test_two_future_documents_that_agree_on_a_tie_are_both_winners_and_one_is_returned():
    policy = claim(3, doc("POL", effective=date(2026, 1, 1)))
    a = claim(4, doc("A", effective=date(2027, 1, 1)))
    b = claim(4, doc("B", effective=date(2027, 1, 1)))
    assert upcoming_change([policy, a, b], EMPLOYEE_IN, TODAY) in (a, b)


def test_the_returned_upcoming_change_is_deterministic_among_agreeing_documents():
    policy = claim(3, doc("POL", effective=date(2026, 1, 1)))
    a = claim(4, doc("A", effective=date(2027, 1, 1)))
    b = claim(4, doc("B", effective=date(2027, 1, 1)))
    assert upcoming_change([policy, a, b], EMPLOYEE_IN, TODAY) == upcoming_change([b, policy, a], EMPLOYEE_IN, TODAY)


def test_an_upcoming_change_is_reported_even_when_there_is_no_current_answer():
    future = claim(4, doc("FUT", effective=date(2027, 1, 1)))
    assert resolve_topic([future], TODAY, EMPLOYEE_IN).status == "none"
    assert upcoming_change([future], EMPLOYEE_IN, TODAY) == future


def test_an_upcoming_change_is_reported_when_the_present_is_a_conflict():
    a = claim(4, doc("A", effective=date(2026, 1, 1)))
    b = claim(5, doc("B", effective=date(2026, 1, 1)))
    future = claim(6, doc("FUT", effective=date(2027, 1, 1)))
    assert upcoming_change([a, b, future], EMPLOYEE_IN, TODAY) == future


def test_a_document_effective_on_the_reference_date_is_current_not_upcoming():
    current = claim(4, doc("CUR", effective=TODAY))
    assert upcoming_change([current], EMPLOYEE_IN, TODAY) is None


def test_no_upcoming_change_without_future_documents():
    assert upcoming_change([POLICY_ALL], EMPLOYEE_IN, TODAY) is None
    assert upcoming_change([], EMPLOYEE_IN, TODAY) is None


# --- purity -----------------------------------------------------------------------------

FORBIDDEN_IMPORT_PREFIXES = (
    "sqlite3", "app.db", "app.llm", "app.ingestion", "openai", "anthropic", "google",
    "httpx", "requests", "urllib", "socket", "subprocess", "asyncio", "fastapi",
)


@pytest.mark.parametrize("module", ["policy_resolver.py", "reference_date.py"])
def test_the_modules_import_no_database_llm_or_network_code(module):
    source = (Path(__file__).resolve().parent.parent / "app" / "retrieval" / module).read_text(encoding="utf-8")
    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [m for m in imported if m.startswith(FORBIDDEN_IMPORT_PREFIXES)]
    assert offenders == []
    assert "datetime.now" not in source and "date.today" not in source  # no hidden clock
