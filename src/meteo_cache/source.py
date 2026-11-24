"""The interface a weather API must provide; the rest of the package knows the API only through it.

To use another API, write a class with these attributes and methods and pass an instance as ``source=``;
use a new cache directory for it. ``openmeteo.OpenMeteoArchive`` is the implementation in use.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class RateLimitError(RuntimeError):
    """Raised when the API's hourly or daily request limit is reached; a later call can resume."""


class Source(Protocol):
    """A provider of hourly weather in UTC.

    name       identifies the source; it is stored in the cache's _meta.json
    variables  {API variable: (column name, unit)}, e.g. {"temperature_2m": ("air_temperature", "degF")}
    Optional: limit_per_hour and limit_per_day (API calls); fetch_weather warns when a plan exceeds them.
    """

    name: str
    variables: dict[str, tuple[str, str]]

    def cost(self, n_variables: int, n_days: int) -> float:
        """Return the API calls that one request for ``n_variables`` over ``n_days`` days counts as."""

    def max_days(self, n_variables: int, calls: float) -> int:
        """Return the longest period, in days, of one request that counts as at most ``calls`` API calls."""

    def fetch(self, lat: float, lon: float, first: date, last: date, variables: list[str]) -> pd.DataFrame:
        """Fetch the UTC days ``first`` to ``last`` (both included) with one request.

        Returns hourly rows with the column t_utc (naive UTC) and one column per variable, named by the
        column in ``variables`` (e.g. air_temperature). Raises RateLimitError at the hourly or daily limit.
        """
