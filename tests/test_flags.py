"""Tests of the freeze indicators."""

import pandas as pd
import pytest

from meteo_cache import weather as W
from meteo_cache.flags import add_weather_flags


def _frame(**columns):
    return pd.DataFrame(columns)


def test_flags_from_weather_code():
    out = add_weather_flags(_frame(weather_code=[0, 56, 66, 71, 85], precipitation=[0, 1, 1, 1, 0],
                                   air_temperature=[40, 30, 40, 20, 20]))
    assert out["freezing_precip"].tolist() == [False, True, True, False, False]
    assert out["snowing"].tolist() == [False, False, False, True, True]
    assert out["wet_below_freezing"].tolist() == [False, True, False, True, False]


def test_flags_warn_without_weather_code():
    df = _frame(precipitation=[1, 0], air_temperature=[20, 20])
    with pytest.warns(UserWarning, match="weather_code"):
        out = add_weather_flags(df)
    assert out["wet_below_freezing"].tolist() == [True, False]
    assert not out["freezing_precip"].any() and not out["snowing"].any()


def test_flags_are_still_exported_by_weather():
    assert W.add_weather_flags is add_weather_flags
