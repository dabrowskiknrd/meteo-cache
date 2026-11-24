"""Tests of the directory cache and its _meta.json; no network (a fake source replaces the API).

Run with:  uv run pytest tests/test_cache.py
"""

import json
from datetime import date

import pandas as pd
import pytest

from meteo_cache import openmeteo
from meteo_cache import weather as W
from meteo_cache.cache import CacheMismatchError, WeatherCache
from meteo_cache.openmeteo import VARIABLES

CELL, TEMP = "52.2,21.0", "temperature_2m"


@pytest.fixture(autouse=True)
def no_pacing(monkeypatch):
    monkeypatch.setattr(W._Pacer, "wait", lambda self, calls: None)


def _listing(d):
    return sorted(p.name for p in d.iterdir())


def _add_ten_days(cache, source):
    fresh = source.fetch(52.2, 21.0, date(2020, 1, 1), date(2020, 1, 10), [TEMP])
    cache.add(CELL, source, [TEMP], date(2020, 1, 1), date(2020, 1, 10), fresh)


@pytest.fixture
def stored(tmp_path, fake):
    """A cache holding ten days of temperature_2m from the source "open-meteo-archive"."""
    fake.name = "open-meteo-archive"
    _add_ten_days(WeatherCache(tmp_path / "wx"), fake)
    fake.calls.clear()
    return tmp_path / "wx"


def test_format_contract(tmp_path, fake):
    _add_ten_days(WeatherCache(tmp_path / "wx"), fake)
    cell = pd.read_parquet(tmp_path / "wx" / f"{CELL}.parquet")
    assert list(cell.columns) == ["t_utc", "air_temperature"]
    assert cell["t_utc"].dt.tz is None and len(cell) == 240
    coverage = (tmp_path / "wx" / "_coverage.json").read_text()
    assert json.loads(coverage) == {CELL: {TEMP: [["2020-01-01", "2020-01-10"]]}}
    assert coverage == json.dumps(json.loads(coverage), indent=1)


def test_meta_written_on_first_add(tmp_path, fake):
    fake.name = "open-meteo-archive"
    _add_ten_days(WeatherCache(tmp_path / "wx"), fake)
    assert json.loads((tmp_path / "wx" / "_meta.json").read_text()) == {
        "time": "UTC", "source": "open-meteo-archive",
        "variables": {"temperature_2m": {"column": "air_temperature", "unit": "degF"}}}


def test_new_variable_is_appended_to_meta(stored, fake):
    cache = WeatherCache(stored)
    fresh = fake.fetch(52.2, 21.0, date(2020, 1, 1), date(2020, 1, 10), ["precipitation"])
    cache.add(CELL, fake, ["precipitation"], date(2020, 1, 1), date(2020, 1, 10), fresh)
    meta = json.loads((stored / "_meta.json").read_text())
    assert meta["variables"] == {"temperature_2m": {"column": "air_temperature", "unit": "degF"},
                                 "precipitation": {"column": "precipitation", "unit": "inch"}}


def test_read_only_open_writes_nothing(stored, fake):
    before = {p.name: p.read_bytes() for p in stored.iterdir()}
    cache = WeatherCache.open(stored)
    cache.check(fake, [TEMP])
    assert cache.cells() == [CELL] and len(cache.read(CELL)) == 240
    assert len(cache.covered(CELL, TEMP)) == 10 and cache.status()["days"].tolist() == [10]
    assert {p.name: p.read_bytes() for p in stored.iterdir()} == before


def test_renamed_column_is_refused(tmp_path, fake):
    fetch = lambda: W.fetch_weather([CELL], "2020-01-01", "2020-01-10 17:00", (TEMP,), cache_path=tmp_path / "wx",
                                    utc_offset_hours=-6, verbose=False, source=fake)
    fetch()
    fake.variables = {**VARIABLES, TEMP: ("temp_air", "degF")}
    fake.calls.clear()
    with pytest.raises(CacheMismatchError, match="air_temperature.*temp_air"):
        fetch()
    assert fake.calls == []


def test_changed_unit_is_refused(stored, fake):
    fake.variables = {**VARIABLES, TEMP: ("air_temperature", "degC")}
    with pytest.raises(CacheMismatchError, match="degC"):
        WeatherCache.open(stored).check(fake, [TEMP])


def test_edited_variable_table_is_refused(stored, fake, monkeypatch):
    # an edit of openmeteo.VARIABLES must not redefine what the cache holds
    before = _bytes(stored)
    monkeypatch.setitem(openmeteo.VARIABLES, TEMP, ("temp_air", "degF"))
    assert fake.variables is openmeteo.VARIABLES
    with pytest.raises(CacheMismatchError, match="air_temperature.*temp_air"):
        WeatherCache.open(stored).check(fake, [TEMP])
    assert _bytes(stored) == before


def test_other_source_is_refused(stored, fake):
    fake.name = "other-api"
    before = _bytes(stored)
    with pytest.raises(CacheMismatchError) as err:
        W.fetch_weather([CELL], "2020-01-01", "2020-01-20", (TEMP,), cache_path=stored,
                        utc_offset_hours=-6, verbose=False, source=fake)
    assert str(stored) in str(err.value) and "open-meteo-archive" in str(err.value) and "other-api" in str(err.value)
    assert fake.calls == [] and _bytes(stored) == before


def _bytes(d):
    return {p.name: p.read_bytes() for p in d.iterdir()}


@pytest.mark.parametrize("change", ["renamed column", "other source"])
def test_run_requests_refuses_a_mismatch(stored, fake, change):
    # the step API (plan_requests -> run_requests) must not write into a cache it does not match
    if change == "renamed column":
        fake.variables = {**VARIABLES, TEMP: ("temp_air", "degC")}
    else:
        fake.name = "other-api"
    before = _bytes(stored)
    request = W.Request(CELL, (TEMP,), date(2020, 1, 11), date(2020, 1, 20), 1.0)
    with pytest.raises(CacheMismatchError):
        W.run_requests([request], WeatherCache(stored), source=fake, verbose=False)
    assert _bytes(stored) == before


def test_new_variable_may_not_reuse_a_stored_column(stored, fake):
    fake.variables = {**VARIABLES, "relative_humidity_2m": ("air_temperature", "%")}
    before = _bytes(stored)
    with pytest.raises(CacheMismatchError, match="air_temperature.*temperature_2m"):
        WeatherCache(stored).check(fake, ["relative_humidity_2m"])
    request = W.Request(CELL, ("relative_humidity_2m",), date(2020, 1, 1), date(2020, 1, 10), 1.0)
    with pytest.raises(CacheMismatchError):
        W.run_requests([request], WeatherCache(stored), source=fake, verbose=False)
    assert _bytes(stored) == before


def test_meta_is_written_before_the_cell_file(tmp_path, fake, monkeypatch):
    # interrupted after _meta.json: the cache is labelled with its real source, and nothing counts as retrieved
    fake.name = "other-api"

    def fail(*a, **k):
        raise OSError("interrupted")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError):
        _add_ten_days(WeatherCache(tmp_path / "wx"), fake)
    assert _listing(tmp_path / "wx") == ["_meta.json"]
    assert json.loads((tmp_path / "wx" / "_meta.json").read_text())["source"] == "other-api"


def test_empty_cache_accepts_any_source(tmp_path, fake):
    WeatherCache(tmp_path / "none").check(fake, [TEMP])
    assert not (tmp_path / "none").exists()


def test_interrupted_add_refetches_without_duplicates(tmp_path, fake, monkeypatch):
    real, failed = WeatherCache._write, []

    def write_once_failing(self, name, obj):
        if name == "_coverage.json" and not failed:
            failed.append(name)
            raise OSError("interrupted")
        real(self, name, obj)

    monkeypatch.setattr(WeatherCache, "_write", write_once_failing)
    fetch = lambda: W.fetch_weather([CELL], "2020-01-01", "2020-01-10 17:00", (TEMP,), cache_path=tmp_path / "wx",
                                    utc_offset_hours=-6, verbose=False, source=fake)
    with pytest.raises(OSError):
        fetch()
    assert (tmp_path / "wx" / f"{CELL}.parquet").exists() and not (tmp_path / "wx" / "_coverage.json").exists()
    out = fetch()
    assert [c[2:4] for c in fake.calls] == [("2020-01-01", "2020-01-10")] * 2
    assert pd.read_parquet(tmp_path / "wx" / f"{CELL}.parquet")["t_utc"].is_unique and out["t"].is_unique


def test_unknown_variable_says_where_to_add_it(tmp_path, fake):
    with pytest.raises(ValueError, match=r"add them to source\.variables"):
        W.fetch_weather([CELL], "2020-01-01", "2020-01-10", ("nonsense",), cache_path=tmp_path / "wx",
                        utc_offset_hours=-6, verbose=False, source=fake)


def test_in_memory_cache(fake):
    cache = WeatherCache(None)
    _add_ten_days(cache, fake)
    assert cache.cells() == [CELL] and len(cache.read(CELL)) == 240 and cache.status()["days"].tolist() == [10]
