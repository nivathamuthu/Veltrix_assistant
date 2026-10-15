"""Hybrid arithmetic: projected leave balance, in code, from a numeric rate."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.orchestrator.projection import (
    fmt,
    is_rate_topic,
    parse_rate,
    parse_target,
    project_balance,
)


TODAY = date(2026, 10, 15)


def test_a_monthly_and_a_yearly_rate_parse():
    monthly = parse_rate("1.5 days per month")
    assert monthly is not None and monthly.per_month == Decimal("1.5")
    yearly = parse_rate("18 days per year")
    assert yearly is not None and yearly.per_month == Decimal("18") / Decimal(12)


def test_a_rate_that_is_not_a_plain_number_is_not_parsed():
    assert parse_rate("accrues according to tenure") is None
    assert parse_rate("a few days per month") is None


def test_rate_topics_are_the_ones_about_accrual():
    assert is_rate_topic("annual leave accrual")
    assert is_rate_topic("leave earning rate")
    assert not is_rate_topic("leave days")


def test_the_target_is_the_last_day_of_the_named_month():
    assert parse_target("by December", TODAY) == date(2026, 12, 31)
    assert parse_target("by the end of the year", TODAY) == date(2026, 12, 31)
    assert parse_target("by March 2026", TODAY) is None  # already past
    assert parse_target("tomorrow", TODAY) is None


def test_the_projection_is_current_plus_month_ends_times_the_rate():
    rate = parse_rate("1.5 days per month")
    plan = project_balance(
        leave_type="annual", current_days=10, rate=rate, target=date(2026, 12, 31), today=TODAY
    )
    assert plan is not None
    # month ends after 2026-10-15 up to 2026-12-31: Oct 31, Nov 30, Dec 31 -> 3
    assert plan.credits == 3
    assert plan.result == Decimal("10") + Decimal("1.5") * 3
    text = plan.text()
    assert fmt(plan.result) in text
    assert "Assumptions" in text
    assert "2026" in text  # year-end carry-forward is out of scope


def test_a_target_in_another_year_is_not_computed():
    rate = parse_rate("1.5 days per month")
    assert project_balance(
        leave_type="annual", current_days=10, rate=rate, target=date(2027, 1, 31), today=TODAY
    ) is None
