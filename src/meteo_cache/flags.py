"""Boolean indicators of freezing precipitation and snowfall, derived from the weather columns."""

from __future__ import annotations

import warnings

import pandas as pd

# WMO weather codes (column weather_code) that denote freezing precipitation.
FREEZING_DRIZZLE = (56, 57)
FREEZING_RAIN = (66, 67)
SNOW = (71, 73, 75, 77, 85, 86)

NEEDED = ("weather_code", "precipitation", "air_temperature")


def add_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with three boolean columns added.

    freezing_precip     weather_code denotes freezing drizzle or freezing rain (WMO 56, 57, 66, 67)
    snowing             weather_code denotes snow (WMO 71-77, 85, 86)
    wet_below_freezing  precipitation > 0 and air_temperature <= 32 degF

    A flag whose input column is absent is False throughout, with a warning. weather_code is not among the
    default variables, so freezing_precip and snowing need it added to ``openmeteo.VARIABLES``.
    """
    out = df.copy()
    if absent := [c for c in NEEDED if c not in out]:
        warnings.warn(f"no column {absent} in the frame: the flags that need them are all False", stacklevel=2)
    code = out["weather_code"] if "weather_code" in out else pd.Series(float("nan"), index=out.index)
    out["freezing_precip"] = code.isin(FREEZING_DRIZZLE + FREEZING_RAIN)
    out["snowing"] = code.isin(SNOW)
    if {"precipitation", "air_temperature"} <= set(out):
        out["wet_below_freezing"] = (out["precipitation"] > 0) & (out["air_temperature"] <= 32)
    else:
        out["wet_below_freezing"] = False
    return out
