"""Tests of the Open-Meteo source; urllib is replaced, nothing is sent and nothing sleeps.

Run with:  uv run pytest tests/test_openmeteo.py
"""

import http.client
import io
import json
import urllib.error
from datetime import date

import pandas as pd
import pytest

from meteo_cache import openmeteo
from meteo_cache.openmeteo import OpenMeteoArchive
from meteo_cache.source import RateLimitError

HOURLY = {"time": ["2020-01-01T00:00", "2020-01-01T01:00"],
          "temperature_2m": [10, None], "relative_humidity_2m": [80.5, 70]}


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def urls(monkeypatch):
    """Answer every request with ``HOURLY``; the list collects the requested URLs."""
    seen = []

    def urlopen(url, timeout):
        seen.append(url)
        return Response(json.dumps({"hourly": HOURLY}).encode())

    monkeypatch.setattr(openmeteo.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(openmeteo.time, "sleep", lambda s: pytest.fail("sleep"))
    return seen


def test_parse_response(urls):
    out = OpenMeteoArchive().fetch(52.2, 21.0, date(2020, 1, 1), date(2020, 1, 1),
                                   ["temperature_2m", "relative_humidity_2m"])
    assert list(out.columns) == ["t_utc", "air_temperature", "relative_humidity"]
    assert out["t_utc"].tolist() == [pd.Timestamp("2020-01-01 00:00"), pd.Timestamp("2020-01-01 01:00")]
    assert (out.dtypes[1:] == "float64").all() and out["air_temperature"].isna().tolist() == [False, True]
    assert "hourly=temperature_2m%2Crelative_humidity_2m" in urls[0] and "timezone=GMT" in urls[0]
    for unit in ("temperature_unit=fahrenheit", "wind_speed_unit=mph", "precipitation_unit=inch"):
        assert unit in urls[0]


def test_hourly_limit_raises_rate_limit(monkeypatch):
    def urlopen(url, timeout):
        raise urllib.error.HTTPError(url, 429, "Too Many", {}, io.BytesIO(b'{"reason":"Hourly API request limit exceeded"}'))

    monkeypatch.setattr(openmeteo.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(openmeteo.time, "sleep", lambda s: pytest.fail("sleep"))
    with pytest.raises(RateLimitError, match="Hourly API request limit exceeded"):
        OpenMeteoArchive().fetch(1, 2, date(2020, 1, 1), date(2020, 1, 1), ["temperature_2m"])


def test_minutely_limit_waits_and_retries(monkeypatch, urls):
    answers = [urllib.error.HTTPError("u", 429, "x", {}, io.BytesIO(b'{"reason":"Minutely API request limit exceeded"}'))]
    real = openmeteo.urllib.request.urlopen
    sleeps = []

    def urlopen(url, timeout):
        if answers:
            raise answers.pop()
        return real(url, timeout)

    monkeypatch.setattr(openmeteo.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(openmeteo.time, "sleep", sleeps.append)
    OpenMeteoArchive().fetch(1, 2, date(2020, 1, 1), date(2020, 1, 1), ["temperature_2m"])
    assert sleeps == [65]


def test_cost_formula():
    s = OpenMeteoArchive()
    assert s.cost(4, 29) == pytest.approx(29 / 14)
    assert s.cost(15, 2190) == pytest.approx(1.5 * 2190 / 14)
    assert s.cost(4, 3) == 1


def test_max_days_formula():
    s = OpenMeteoArchive()
    assert s.max_days(4, 500) == 7000 and s.max_days(15, 10) == 93 and s.max_days(4, 0) == 1


def test_url_is_configurable(urls):
    OpenMeteoArchive(url="http://x").fetch(1, 2, date(2020, 1, 1), date(2020, 1, 1), ["temperature_2m"])
    assert urls[0].startswith("http://x?")


def test_default_variables_are_the_active_ones():
    assert tuple(openmeteo.VARIABLES) == openmeteo.DEFAULT_VARIABLES == (
        "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m")


@pytest.mark.parametrize("error", [http.client.IncompleteRead(b"{", 10), TimeoutError("read timed out"),
                                   ConnectionResetError("reset")])
def test_broken_response_is_retried(monkeypatch, urls, error):
    answers = [error]
    real = openmeteo.urllib.request.urlopen
    sleeps = []

    def urlopen(url, timeout):
        if answers:
            raise answers.pop()
        return real(url, timeout)

    monkeypatch.setattr(openmeteo.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(openmeteo.time, "sleep", sleeps.append)
    OpenMeteoArchive().fetch(1, 2, date(2020, 1, 1), date(2020, 1, 1), ["temperature_2m"])
    assert sleeps == [4]
