"""Tests of the time handling in days.py (pure functions, no network).

Run with:  uv run pytest tests/test_days.py
"""

from datetime import date

import pandas as pd
import pytest

from meteo_cache import days as D


def test_as_utc_converts_aware_and_keeps_naive():
    assert D.as_utc(pd.Timestamp("2020-01-01 06:00", tz="America/Chicago")) == pd.Timestamp("2020-01-01 12:00")
    assert D.as_utc(pd.Timestamp("2020-01-01 06:00")) == pd.Timestamp("2020-01-01 06:00")
    assert D.as_utc("2020-01-01 06:00").tz is None


def test_local_to_utc_shifts_by_the_offset():
    assert D.local_to_utc("2020-01-01 06:00", -6) == pd.Timestamp("2020-01-01 12:00")


def test_local_to_utc_rejects_aware():
    with pytest.raises(ValueError, match="naive"):
        D.local_to_utc(pd.Timestamp("2020-01-01", tz="UTC"), -6)


def test_to_local_shifts_a_series():
    t = pd.Series(pd.to_datetime(["2020-01-01 12:00"]))
    assert D.to_local(t, -6).tolist() == [pd.Timestamp("2020-01-01 06:00")]


def test_utc_days_are_calendar_dates():
    assert D.utc_days("2020-01-01 10:00", "2020-01-31 09:00") == (date(2020, 1, 1), date(2020, 1, 31))


def test_months_with_padding():
    d = D.wanted_days(pd.Timestamp("2019-06-01"), pd.Timestamp("2020-06-30"), months=(11, 12, 1, 2, 3), pad_days=3)
    assert (min(d), max(d)) == (date(2019, 10, 29), date(2020, 4, 3))


def test_newest_excludes_recent_days():
    d = D.wanted_days(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31"), newest=date(2020, 1, 5))
    assert max(d) == date(2020, 1, 5) and len(d) == 5


def test_newest_defaults_to_the_archive_delay(monkeypatch):
    now = pd.Timestamp("2020-06-15 23:59", tz="UTC")
    monkeypatch.setattr(D.pd.Timestamp, "now", classmethod(lambda cls, tz=None: now))
    d = D.wanted_days(pd.Timestamp("2020-05-01"), pd.Timestamp("2020-06-15"))
    assert max(d) == date(2020, 6, 15 - D.ARCHIVE_DELAY_DAYS)


def test_day_ranges_split():
    days = pd.date_range("2020-01-01", periods=365, freq="D").date
    r = D.day_ranges(days, 140)
    assert len(r) == 3 and r[0] == (date(2020, 1, 1), date(2020, 5, 19)) and isinstance(r[0][0], date)


def test_day_ranges_break_at_gaps():
    assert D.day_ranges([date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 5)], 10) == [
        (date(2020, 1, 1), date(2020, 1, 2)), (date(2020, 1, 5), date(2020, 1, 5))]
