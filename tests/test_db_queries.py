"""Read functions. Exact expectations on the synthetic DB; on a copy of the real
DB every function is checked against an independent plain-Python oracle."""
import pytest

from app.db import (
    Employee,
    get_balance,
    get_direct_reports,
    get_employee,
    get_pii_values,
    get_team_requests,
    list_requests,
)


@pytest.fixture()
def conn(synthetic_db):
    with synthetic_db.read() as c:
        yield c


def _ids(items, attr="request_id"):
    return [getattr(i, attr) for i in items]


# --- get_employee ----------------------------------------------------------


def test_get_employee_returns_the_record(conn):
    e = get_employee(conn, "E1")
    assert isinstance(e, Employee)
    assert (e.employee_id, e.region, e.level, e.role, e.manager_id) == ("E1", "IN", "L2", "employee", "M1")
    assert e.full_name == "Eli Fenn" and e.email == "eli.fenn@veltrix.example"


def test_get_employee_top_level_has_no_manager(conn):
    assert get_employee(conn, "H1").manager_id is None


@pytest.mark.parametrize("bad", ["NOPE", "", "e1", " E1", "E1 ", "E1' OR '1'='1", "E1; DROP TABLE employees", None])
def test_get_employee_unknown_or_malformed_id_is_none(conn, bad):
    assert get_employee(conn, bad) is None


def test_employee_repr_contains_no_personal_data(conn):
    e = get_employee(conn, "E1")
    text = repr(e) + str(e)
    assert "E1" in text
    for secret in (e.full_name, e.email, e.phone, "Fenn", "fenn@"):
        assert secret not in text


# --- get_direct_reports ----------------------------------------------------


def test_direct_reports_are_exactly_those_with_matching_manager_id(conn):
    assert _ids(get_direct_reports(conn, "M1"), "employee_id") == ["E1", "E2", "M3"]
    assert _ids(get_direct_reports(conn, "M2"), "employee_id") == ["E3"]
    assert _ids(get_direct_reports(conn, "H1"), "employee_id") == ["M1", "M2"]


def test_direct_reports_are_not_transitive(conn):
    # M3 reports to M1, so M3's own (empty) team is not merged into M1's.
    assert "E3" not in _ids(get_direct_reports(conn, "M1"), "employee_id")


# --- get_balance -----------------------------------------------------------


def test_get_balance_all_types_sorted(conn):
    balances = get_balance(conn, "E1")
    assert [(b.leave_type, b.balance_days) for b in balances] == [("annual", 10.0), ("sick", 5.0)]
    assert all(b.employee_id == "E1" for b in balances)


def test_get_balance_one_type_and_fractional_days(conn):
    assert [(b.leave_type, b.balance_days) for b in get_balance(conn, "E2", "annual")] == [("annual", 3.5)]


def test_get_balance_type_the_employee_does_not_have_is_none(conn):
    assert get_balance(conn, "E2", "sick") is None


# --- list_requests ---------------------------------------------------------


def test_list_requests_all_statuses_ordered_by_start_date(conn):
    assert _ids(list_requests(conn, "E1")) == ["R1", "R3", "R2"]


@pytest.mark.parametrize(
    "status, expected",
    [("pending", ["R1"]), ("approved", ["R2"]), ("cancelled", ["R3"]), ("rejected", [])],
)
def test_list_requests_status_filter(conn, status, expected):
    assert _ids(list_requests(conn, "E1", status)) == expected


def test_list_requests_only_returns_that_employees_rows(conn):
    assert {r.employee_id for r in list_requests(conn, "E2")} == {"E2"}


def test_list_requests_for_employee_with_none_is_empty_list(conn):
    assert list_requests(conn, "E4") == []


def test_list_request_fields(conn):
    r = list_requests(conn, "E1", "approved")[0]
    assert (r.request_id, r.leave_type, r.start_date, r.end_date, r.working_days, r.status) == (
        "R2", "annual", "2026-12-21", "2026-12-24", 4, "approved",
    )


# --- get_team_requests -----------------------------------------------------


def test_team_requests_overlap_and_status_rules(conn):
    # 2026-11-04 touches R1's last day and R4's first day (inclusive both ways).
    assert _ids(get_team_requests(conn, "M1", "2026-11-04", "2026-11-04")) == ["R1", "R4"]


def test_team_requests_exclude_cancelled_own_leave_and_other_teams(conn):
    got = _ids(get_team_requests(conn, "M1", "2026-01-01", "2026-12-31"))
    assert got == ["R1", "R4", "R7", "R2"]  # not R3 (cancelled), R5 (other team), R6 (M1's own)
    assert "R6" not in got and "R5" not in got and "R3" not in got


def test_team_requests_include_a_reports_own_pending_leave(conn):
    # M3 is M1's direct report, so M3's pending request counts for M1.
    assert _ids(get_team_requests(conn, "M1", "2026-11-05", "2026-11-05")) == ["R4", "R7"]


@pytest.mark.parametrize(
    "start, end, expected",
    [
        ("2026-12-01", "2026-12-20", []),            # ends the day before R2 starts
        ("2026-12-01", "2026-12-21", ["R2"]),        # touches R2's first day
        ("2026-12-24", "2026-12-31", ["R2"]),        # touches R2's last day
        ("2026-12-25", "2026-12-31", []),            # starts the day after R2 ends
        ("2026-11-01", "2026-11-01", []),            # day before R1
        ("2026-11-02", "2026-11-02", ["R1"]),        # R1's first day
    ],
)
def test_team_requests_range_boundaries(conn, start, end, expected):
    assert _ids(get_team_requests(conn, "M1", start, end)) == expected


def test_team_requests_other_manager_sees_only_their_team(conn):
    assert _ids(get_team_requests(conn, "M2", "2026-01-01", "2026-12-31")) == ["R5"]


@pytest.mark.parametrize(
    "bad_start, bad_end",
    [("2026-11-05", "2026-11-04"), ("11/04/2026", "2026-11-05"), ("2026-13-01", "2026-12-01"),
     ("20261104", "20261105"), ("2026-11-04", "2026-11-05; DROP TABLE x"), ("", "")],
)
def test_team_requests_reject_bad_dates(conn, bad_start, bad_end):
    with pytest.raises(ValueError):
        get_team_requests(conn, "M1", bad_start, bad_end)


# --- nonexistent id vs not-allowed id -------------------------------------
# These functions are not an authorisation layer, but a missing id must never
# look different from one the caller may not ask about.


def test_nonexistent_target_returns_none_for_every_target_read(conn):
    assert get_employee(conn, "GHOST") is None
    assert get_balance(conn, "GHOST") is None
    assert get_balance(conn, "GHOST", "annual") is None
    assert list_requests(conn, "GHOST") is None
    assert list_requests(conn, "GHOST", "pending") is None


def test_existing_employee_with_no_balance_rows_looks_like_a_nonexistent_one(conn):
    assert get_balance(conn, "E4") is None
    assert get_balance(conn, "E4") == get_balance(conn, "GHOST")
    assert get_balance(conn, "E4", "annual") == get_balance(conn, "GHOST", "annual")


def test_team_reads_are_identical_for_ghost_non_manager_and_empty_team(conn):
    ghost = get_direct_reports(conn, "GHOST"), get_team_requests(conn, "GHOST", "2026-01-01", "2026-12-31")
    employee = get_direct_reports(conn, "E1"), get_team_requests(conn, "E1", "2026-01-01", "2026-12-31")
    no_reports = get_direct_reports(conn, "E4"), get_team_requests(conn, "E4", "2026-01-01", "2026-12-31")
    assert ghost == employee == no_reports == ([], [])


def test_manager_with_reports_but_no_leave_in_range_is_also_empty_list(conn):
    # M3 is a manager with no reports; M1 has a team but nobody off in June.
    assert get_team_requests(conn, "M3", "2026-01-01", "2026-12-31") == []
    assert get_team_requests(conn, "M1", "2026-06-01", "2026-06-02") == []


# --- injection: values are bound, never executed ---------------------------


@pytest.mark.parametrize("payload", ["' OR '1'='1", "x'; DROP TABLE leave_requests; --", '" OR ""="', "E1' --"])
def test_injection_payloads_are_just_unmatched_values(synthetic_db, payload):
    with synthetic_db.read() as c:
        assert get_employee(c, payload) is None
        assert get_balance(c, payload) is None
        assert get_balance(c, "E1", payload) is None
        assert list_requests(c, payload) is None
        assert list_requests(c, "E1", payload) == []
        assert get_direct_reports(c, payload) == []
        assert get_team_requests(c, payload, "2026-01-01", "2026-12-31") == []
        # the table is still there and complete
        assert c.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0] == 7


def test_get_pii_values_returns_all_three_kinds(conn):
    names, emails, phones = get_pii_values(conn)
    assert len(names) == len(emails) == len(phones) == 8
    assert "Eli Fenn" in names and "eli.fenn@veltrix.example" in emails


# --- the real database copy, against an independent oracle -----------------


def _raw(db, sql, params=()):
    with db.read() as c:
        return [tuple(r) for r in c.execute(sql, params).fetchall()]


def test_every_read_function_matches_an_oracle_on_the_real_data(real_db_copy):
    employees = _raw(real_db_copy, "SELECT * FROM employees")
    balances = _raw(real_db_copy, "SELECT * FROM leave_balances")
    requests = _raw(real_db_copy, "SELECT * FROM leave_requests")
    assert employees and balances and requests

    by_id = {e[0]: e for e in employees}
    ranges = [("2026-01-01", "2026-12-31"), ("2026-02-16", "2026-02-20"), ("2026-11-09", "2026-11-09")]

    with real_db_copy.read() as c:
        for e in employees:
            eid = e[0]
            got = get_employee(c, eid)
            assert tuple(got.__dict__.values()) == e

            # balances
            want_bal = sorted((b[1], b[2]) for b in balances if b[0] == eid)
            have = get_balance(c, eid)
            assert (have is None and not want_bal) or [(b.leave_type, b.balance_days) for b in have] == want_bal
            for lt in ("annual", "sick"):
                one = get_balance(c, eid, lt)
                assert [(b.leave_type, b.balance_days) for b in one or []] == [x for x in want_bal if x[0] == lt]

            # requests
            mine = [r for r in requests if r[1] == eid]
            for status in (None, "pending", "approved", "cancelled"):
                want = sorted(
                    (r for r in mine if status in (None, r[6])), key=lambda r: (r[3], r[4], r[0])
                )
                got_reqs = list_requests(c, eid, status)
                assert [tuple(r.__dict__.values()) for r in got_reqs] == want

            # direct reports + team leave
            reports = sorted(x[0] for x in employees if x[8] == eid)
            assert _ids(get_direct_reports(c, eid), "employee_id") == reports
            for start, end in ranges:
                want_team = sorted(
                    (
                        r for r in requests
                        if r[1] in reports and r[6] in ("pending", "approved")
                        and r[3] <= end and r[4] >= start
                    ),
                    key=lambda r: (r[3], r[1], r[0]),
                )
                got_team = get_team_requests(c, eid, start, end)
                assert [tuple(r.__dict__.values()) for r in got_team] == want_team

        # a nonexistent id is None / [] exactly like the synthetic case
        assert get_employee(c, "E0000-NOPE") is None
        assert get_balance(c, "E0000-NOPE") is None
        assert list_requests(c, "E0000-NOPE") is None
        assert get_team_requests(c, "E0000-NOPE", *ranges[0]) == []
    assert by_id  # sanity
