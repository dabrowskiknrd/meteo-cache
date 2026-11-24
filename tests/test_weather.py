"""Tests of the incremental weather cache; no network (a fake source replaces the API).

Run with:  uv run pytest tests/test_weather.py
"""

import warnings
from datetime import date

import pandas as pd
import pytest

from meteo_cache import weather as W
from meteo_cache.cache import WeatherCache

TEMP, HUM = "temperature_2m", "relative_humidity_2m"
CELL = "52.2,21.0"


@pytest.fixture(autouse=True)
def no_pacing(monkeypatch):
    """No pacing and no sleeping: nothing is sent."""
    monkeypatch.setattr(W._Pacer, "wait", lambda self, calls: None)


def fetch(tmp_path, source, cells, start, end, variables=(TEMP,), utc_offset_hours=-6, **kw):
    return W.fetch_weather(cells, start, end, variables, cache_path=tmp_path / "wx", utc_offset_hours=utc_offset_hours,
                           verbose=False, source=source, **kw)


def test_second_call_makes_no_request(tmp_path, fake):
    a = fetch(tmp_path, fake, ["52.2,21.0"], "2020-01-01 10:00", "2020-01-31 09:00")
    assert len(fake.calls) == 1
    b = fetch(tmp_path, fake, ["52.2,21.0"], "2020-01-01 10:00", "2020-01-31 09:00")
    assert len(fake.calls) == 1
    pd.testing.assert_frame_equal(a, b)
    assert a["t"].min() == pd.Timestamp("2020-01-01 10:00") and a["t"].max() == pd.Timestamp("2020-01-31 09:00")
    assert a["t"].is_unique and a["air_temperature"].notna().all()


def test_longer_period_requests_only_the_added_days(tmp_path, fake):
    fetch(tmp_path, fake, ["52.2,21.0"], "2020-02-01", "2020-02-29 17:00")
    fake.calls.clear()
    out = fetch(tmp_path, fake, ["52.2,21.0"], "2020-01-01", "2020-03-31 17:00")
    assert sorted(c[2:4] for c in fake.calls) == [("2020-01-01", "2020-01-31"), ("2020-03-01", "2020-03-31")]
    assert len(out) == len(pd.date_range("2020-01-01", "2020-03-31 17:00", freq="h"))


def test_new_variable_requests_that_variable_only(tmp_path, fake):
    fetch(tmp_path, fake, ["52.2,21.0"], "2020-01-01", "2020-01-10 17:00")
    fake.calls.clear()
    out = fetch(tmp_path, fake, ["52.2,21.0"], "2020-01-01", "2020-01-10 17:00", variables=(TEMP, HUM))
    assert [c[4] for c in fake.calls] == [(HUM,)]
    assert out[["air_temperature", "relative_humidity"]].notna().all().all()


def test_new_cell_requests_that_cell_only(tmp_path, fake):
    fetch(tmp_path, fake, ["52.2,21.0"], "2020-01-01", "2020-01-10 17:00")
    fake.calls.clear()
    out = fetch(tmp_path, fake, ["52.2,21.0", "41.9,12.5"], "2020-01-01", "2020-01-10 17:00")
    assert [c[:2] for c in fake.calls] == [(41.9, 12.5)]
    assert out["cell"].nunique() == 2


def test_months_restrict_the_requested_days(tmp_path, fake):
    fetch(tmp_path, fake, ["52.2,21.0"], "2019-06-01", "2020-06-30 17:00", months=(11, 12, 1, 2, 3), pad_days=3)
    assert [c[2:4] for c in fake.calls] == [("2019-10-29", "2020-04-03")]


def test_empty_archive_values_are_not_requested_again(tmp_path, fake):
    fake.empty = True
    fetch(tmp_path, fake, ["52.2,21.0"], "2020-01-01", "2020-01-10 17:00")
    fetch(tmp_path, fake, ["52.2,21.0"], "2020-01-01", "2020-01-10 17:00")
    assert len(fake.calls) == 1


def test_budget_partial_then_resume(tmp_path, fake):
    cells = ["52.2,21.0", "41.9,12.5", "41.8,12.4"]
    args = ("2020-01-01", "2020-01-28 17:00")  # 29 UTC days: 29 / 14 calls per cell
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fetch(tmp_path, fake, cells, *args, on_limit="partial", max_calls=5)
    assert len(fake.calls) == 2 and out["cell"].nunique() == 2 and "incomplete" in str(w[-1].message)
    with pytest.raises(W.RateLimitError):
        fetch(tmp_path, fake, cells, *args, max_calls=1)
    out = fetch(tmp_path, fake, cells, *args)
    assert len(fake.calls) == 3 and out["cell"].nunique() == 3


def test_rate_limit_keeps_completed_requests(tmp_path, fake):
    fake.fail = lambda lat: W.RateLimitError("Open-Meteo HTTP 429: Hourly API request limit exceeded.") if lat < 42 else None
    with pytest.raises(W.RateLimitError):
        fetch(tmp_path, fake, ["52.2,21.0", "41.9,12.5"], "2020-01-01", "2020-01-10 17:00")
    fake.fail = lambda lat: None
    fake.calls.clear()
    fetch(tmp_path, fake, ["52.2,21.0", "41.9,12.5"], "2020-01-01", "2020-01-10 17:00")
    assert [c[:2] for c in fake.calls] == [(41.9, 12.5)]


def test_long_period_is_split_below_the_per_minute_limit(tmp_path, fake):
    fetch(tmp_path, fake, ["52.2,21.0"], "2014-01-01", "2014-12-31 17:00", calls_per_minute=10)  # 140 days per request
    assert len(fake.calls) == 3
    assert W.cache_status(tmp_path / "wx")["days"].tolist() == [365]


def test_cells_are_requested_in_the_given_order(tmp_path, fake):
    fetch(tmp_path, fake, ["52.2,21.0", "41.8,12.4", "41.9,12.5"], "2020-01-01", "2020-01-10 17:00")
    assert [c[0] for c in fake.calls] == [52.2, 41.8, 41.9]


def test_per_cell_periods_request_only_their_own_days(tmp_path, fake):
    periods = {"52.2,21.0": ("2020-01-01", "2020-01-10 17:00"), "41.9,12.5": ("2020-02-01", "2020-02-05 17:00")}
    out = W.fetch_weather(list(periods), variables=(TEMP,), periods=periods, cache_path=tmp_path / "wx",
                    utc_offset_hours=-6, verbose=False, source=fake)
    assert [c[2:4] for c in fake.calls] == [("2020-01-01", "2020-01-10"), ("2020-02-01", "2020-02-05")]
    assert out.groupby("cell")["t"].min().to_dict() == {"41.9,12.5": pd.Timestamp("2020-02-01"),
                                                         "52.2,21.0": pd.Timestamp("2020-01-01")}
    fake.calls.clear()  # widening one cell's period requests only the added days of that cell
    periods["41.9,12.5"] = ("2020-02-01", "2020-02-08 17:00")
    W.fetch_weather(list(periods), variables=(TEMP,), periods=periods, cache_path=tmp_path / "wx",
                    utc_offset_hours=-6, verbose=False, source=fake)
    assert [c[:4] for c in fake.calls] == [(41.9, 12.5, "2020-02-06", "2020-02-08")]


def test_time_budget_stops_partially_and_status_is_returned(tmp_path, fake):
    cells = ["52.2,21.0", "41.9,12.5"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        status = fetch(tmp_path, fake, cells, "2020-01-01", "2020-01-10 17:00", on_limit="partial",
                       max_minutes=0, return_data=False)
    assert fake.calls == [] and status.empty and "time budget" in str(w[-1].message)
    status = fetch(tmp_path, fake, cells, "2020-01-01", "2020-01-10 17:00", max_minutes=5, return_data=False)
    assert sorted(status["cell"]) == sorted(cells) and (status["days"] == 10).all()  # UTC days 1-10 January


def test_date_only_end_is_midnight(tmp_path, fake):
    out = fetch(tmp_path, fake, ["52.2,21.0"], "2020-01-01", "2020-01-31", utc_offset_hours=-6)
    assert out["t"].max() == pd.Timestamp("2020-01-31 00:00")  # a date-only end means midnight, not the whole day


def test_aware_times_are_rejected_by_the_wrappers(tmp_path, fake):
    with pytest.raises(ValueError, match="naive"):
        fetch(tmp_path, fake, ["52.2,21.0"], pd.Timestamp("2020-01-01", tz="UTC"), "2020-01-31")


def test_same_call_same_rows(tmp_path, fake):
    january = dict(months=(1,), pad_days=0)
    a = fetch(tmp_path, fake, [CELL], "2020-01-01 06:00", "2020-07-31 05:00", **january)
    fetch(tmp_path, fake, [CELL], "2020-01-01 06:00", "2020-07-31 05:00", months=(6,), pad_days=0)
    b = fetch(tmp_path, fake, [CELL], "2020-01-01 06:00", "2020-07-31 05:00", **january)
    pd.testing.assert_frame_equal(a, b)
    assert set(b["t"].dt.month) == {1}


def test_plan_is_a_dry_run(tmp_path, fake):
    periods = {CELL: ("2020-01-01", "2020-01-10 23:00"), "41.9,12.5": ("2020-02-01", "2020-02-05 23:00")}
    plan = W.plan_requests(WeatherCache(tmp_path / "wx"), periods, (TEMP,), source=fake)
    assert plan == [W.Request(CELL, (TEMP,), date(2020, 1, 1), date(2020, 1, 10), 1.0),
                    W.Request("41.9,12.5", (TEMP,), date(2020, 2, 1), date(2020, 2, 5), 1.0)]
    assert fake.calls == [] and not (tmp_path / "wx").exists()


def test_missing_days_counts(tmp_path, fake):
    cache = WeatherCache(tmp_path / "wx")
    W.run_requests(W.plan_requests(cache, {CELL: ("2020-01-01", "2020-01-10 23:00")}, (TEMP,), source=fake),
                   cache, source=fake, verbose=False)
    missing = W.missing_days(cache, {CELL: ("2020-01-01", "2020-01-31 23:00")}, (TEMP,))
    assert missing.to_dict("records") == [dict(cell=CELL, wanted_days=31, missing_days=21)]


def test_unknown_variable_lists_known_ones(tmp_path, fake):
    with pytest.raises(ValueError, match=r"nonsense.*temperature_2m"):
        W.plan_requests(WeatherCache(tmp_path / "wx"), {CELL: ("2020-01-01", "2020-01-10")}, ("nonsense",), source=fake)


def test_run_requests_reports_stop(tmp_path, fake):
    cache = WeatherCache(tmp_path / "wx")
    periods = {CELL: ("2020-01-01", "2020-01-10 23:00"), "41.9,12.5": ("2020-01-01", "2020-01-10 23:00")}
    plan = W.plan_requests(cache, periods, (TEMP,), source=fake)
    run = W.run_requests(plan, cache, source=fake, max_calls=1, verbose=False)
    assert run.done == 1 and run.spent == 1.0 and "max_calls" in run.stopped and len(fake.calls) == 1


def test_steps_are_utc(tmp_path, fake):
    cache = WeatherCache(tmp_path / "wx")
    naive = {CELL: ("2020-01-01 06:00", "2020-01-10 05:00")}
    aware = {CELL: (pd.Timestamp("2020-01-01", tz="Etc/GMT+6"), pd.Timestamp("2020-01-09 23:00", tz="Etc/GMT+6"))}
    W.run_requests(W.plan_requests(cache, naive, (TEMP,), source=fake), cache, source=fake, verbose=False)
    out = W.read_weather(cache, naive, (TEMP,))
    assert list(out.columns) == ["cell", "t_utc", "lat", "lon", "air_temperature"]
    assert out["t_utc"].min() == pd.Timestamp("2020-01-01 06:00") and out["t_utc"].max() == pd.Timestamp("2020-01-10 05:00")
    pd.testing.assert_frame_equal(out, W.read_weather(cache, aware, (TEMP,)))


def test_fetch_weather_requires_offset(tmp_path, fake):
    with pytest.raises(ValueError, match="utc_offset_hours"):
        W.fetch_weather([CELL], "2020-01-01", "2020-01-10", (TEMP,), cache_path=tmp_path / "wx", source=fake)
    with pytest.raises(ValueError, match="utc_offset_hours"):
        W.fetch_cell(52.2, 21.0, "2020-01-01", "2020-01-10", (TEMP,), source=fake)
    assert fake.calls == [] and not (tmp_path / "wx").exists()


def test_removed_variable_is_still_readable(tmp_path, fake):
    cache = WeatherCache(tmp_path / "wx")
    periods = {CELL: ("2020-01-01", "2020-01-10 23:00")}
    W.run_requests(W.plan_requests(cache, periods, (TEMP,), source=fake), cache, source=fake, verbose=False)
    fake.variables = {k: v for k, v in fake.variables.items() if k != TEMP}  # the source dropped it, the cache holds it
    assert list(W.read_weather(cache, periods, (TEMP,)).columns) == ["cell", "t_utc", "lat", "lon", "air_temperature"]


def test_reading_a_never_stored_variable_names_it(tmp_path, fake):
    cache = WeatherCache(tmp_path / "wx")
    periods = {CELL: ("2020-01-01", "2020-01-10")}
    W.run_requests(W.plan_requests(cache, periods, (TEMP,), source=fake), cache, source=fake, verbose=False)
    with pytest.raises(ValueError, match="relative_humidity_2m"):
        W.read_weather(cache, periods, (HUM,))


def test_missing_days_validates_variables(tmp_path, fake):
    cache = WeatherCache(tmp_path / "wx")
    periods = {CELL: ("2020-01-01", "2020-01-10")}
    with pytest.raises(ValueError, match="nonsense"):
        W.missing_days(cache, periods, ("nonsense",), source=fake)
    with pytest.raises(ValueError, match="no variables"):
        W.missing_days(cache, periods, (), source=fake)


def test_status_only_has_requested_cells(tmp_path, fake):
    fetch(tmp_path, fake, [CELL, "41.9,12.5"], "2020-01-01", "2020-01-10 17:00")
    status = fetch(tmp_path, fake, [CELL], "2020-01-01", "2020-01-10 17:00", return_data=False)
    assert set(status["cell"]) == {CELL}


def test_hourly_limit_warning_only_for_sources_with_a_limit(tmp_path, fake):
    plan = W.plan_requests(WeatherCache(tmp_path / "wx"), {CELL: ("2020-01-01", "2020-12-31")}, (TEMP,), source=fake)
    fake.limit_per_hour = 5
    with pytest.warns(UserWarning, match="hourly limit of 5"):
        W._announce(plan, 1, 500, None, fake, False)
    del fake.limit_per_hour
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        W._announce(plan, 1, 500, None, fake, False)


def test_partial_stop_before_a_new_variable_returns_it_as_nan(tmp_path, fake):
    fetch(tmp_path, fake, [CELL], "2020-01-01", "2020-01-10 17:00")
    with pytest.warns(UserWarning, match="incomplete"):
        out = fetch(tmp_path, fake, [CELL], "2020-01-01", "2020-01-10 17:00", variables=(TEMP, HUM),
                    on_limit="partial", max_calls=0)
    assert out["air_temperature"].notna().all() and out["relative_humidity"].isna().all() and len(out) > 0


def test_partial_stop_on_an_empty_cache_returns_empty_frame(tmp_path, fake):
    with pytest.warns(UserWarning, match="incomplete"):
        out = fetch(tmp_path, fake, [CELL], "2020-01-01", "2020-01-10 17:00", on_limit="partial", max_calls=0)
    assert out.empty and list(out.columns) == ["cell", "t", "lat", "lon", "air_temperature"]
