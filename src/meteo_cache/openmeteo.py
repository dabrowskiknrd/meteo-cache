"""Open-Meteo historical weather API (free, no key, non-commercial): everything specific to it.

Endpoint https://archive-api.open-meteo.com/v1/archive. Values are reanalysis model output (ERA5
family, cells of about 10-25 km), not site measurements. Accumulated variables (precipitation, rain,
snowfall) are totals over the hour that ENDS at the timestamp. Limits of the free API
(https://open-meteo.com/en/pricing): 600 calls/minute, 5,000/hour, 10,000/day; a request for more
than 10 variables or 14 days counts as (variables / 10) x (days / 14) calls, each factor at least 1.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from meteo_cache.source import RateLimitError

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Limits of the free API, in weighted calls (https://open-meteo.com/en/pricing).
LIMIT_PER_MINUTE = 600
LIMIT_PER_HOUR = 5_000
LIMIT_PER_DAY = 10_000

# Open-Meteo variable -> (column name in the returned table, unit with the units requested below)
VARIABLES: dict[str, tuple[str, str]] = {
    "temperature_2m": ("air_temperature", "degF"),
    #"apparent_temperature": ("apparent_temperature", "degF"),
    #"dew_point_2m": ("dew_point", "degF"),
    "relative_humidity_2m": ("relative_humidity", "%"),
    "precipitation": ("precipitation", "inch"),
    #"rain": ("rain", "inch"),
    #"snowfall": ("snowfall", "inch"),
    #"snow_depth": ("snow_depth", "ft"),
    #"weather_code": ("weather_code", "WMO code"),
    #"cloud_cover": ("cloud_cover", "%"),
    #"surface_pressure": ("surface_pressure", "hPa"),
    "wind_speed_10m": ("wind_speed", "mph"),
    #"wind_gusts_10m": ("wind_gust", "mph"),
    #"wind_direction_10m": ("wind_direction", "deg"),
    #"soil_temperature_0_to_7cm": ("soil_temperature", "degF"),
}

# The default variable set: every active variable above.
DEFAULT_VARIABLES: tuple[str, ...] = tuple(VARIABLES)

UNITS = {"temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch"}


@dataclass
class OpenMeteoArchive:
    """The Open-Meteo historical archive as a ``Source``; ``OPEN_METEO`` is the default instance.

    Change a field to change the request, e.g. ``OpenMeteoArchive(url=..., timeout=300)``.
    url        endpoint of the archive API
    variables  {API variable: (column, unit)}; the units follow from ``units``
    units      unit parameters sent with every request (Fahrenheit, mph, inch)
    timeout    seconds to wait for one response
    retries    repeats after a per-minute rate limit, a server error or a network error
    """

    name: str = "open-meteo-archive"
    url: str = ARCHIVE_URL
    variables: dict[str, tuple[str, str]] = field(default_factory=lambda: VARIABLES)
    units: dict[str, str] = field(default_factory=lambda: UNITS)
    timeout: float = 120
    retries: int = 3
    limit_per_hour: int = LIMIT_PER_HOUR  # optional source attributes: weather.py warns when a plan exceeds them
    limit_per_day: int = LIMIT_PER_DAY

    def cost(self, n_variables: int, n_days: int) -> float:
        """Return (variables / 10) x (days / 14), each factor at least 1: Open-Meteo's count for one request."""
        return max(1, n_variables / 10) * max(1, n_days / 14)

    def max_days(self, n_variables: int, calls: float) -> int:
        """Return the longest period, in days, of one request that counts as at most ``calls`` API calls."""
        return max(1, int(14 * calls / max(1, n_variables / 10)))

    def fetch(self, lat: float, lon: float, first: date, last: date, variables: list[str]) -> pd.DataFrame:
        """Fetch the UTC days ``first`` to ``last`` (both included) with one request.

        Returns hourly rows with the column t_utc and one column per variable (e.g. air_temperature).
        """
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{first:%Y-%m-%d}", "end_date": f"{last:%Y-%m-%d}",
            "hourly": ",".join(variables),
            "timezone": "GMT",
            **self.units,
        }
        hourly = _get_json(self.url + "?" + urllib.parse.urlencode(params), self.timeout, self.retries)["hourly"]
        out = pd.DataFrame({"t_utc": pd.to_datetime(hourly["time"])})
        for v in variables:
            out[self.variables[v][0]] = pd.to_numeric(pd.Series(hourly[v]), errors="coerce").astype(float)
        return out


OPEN_METEO = OpenMeteoArchive()


def _get_json(url: str, timeout: float, retries: int) -> dict:
    """Download a JSON document, waiting and retrying where the error is temporary.

    HTTP 429 with the per-minute limit: wait 65 s (or the Retry-After header) and retry.
    HTTP 429 with the hourly or daily limit: raise ``RateLimitError`` at once.
    HTTP 5xx, network errors, timeouts and responses cut off: retry after 4, 8, 16 s.
    """
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as err:
            reason = _reason(err)
            if err.code == 429:
                if "minutely" not in reason.lower():  # hourly or daily limit: waiting is not practical
                    raise RateLimitError(f"Open-Meteo HTTP 429: {reason} Completed cells are kept in the "
                                         f"cache; repeat the call later to fetch the remaining cells.") from err
                if attempt == retries:
                    raise RateLimitError(f"Open-Meteo HTTP 429: {reason} (after {retries} retries)") from err
                delay = _retry_after(err) or 65
            elif err.code >= 500 and attempt < retries:
                delay = 2 ** (attempt + 2)
            else:
                raise RuntimeError(f"Open-Meteo HTTP {err.code}: {reason}") from err
        except (http.client.HTTPException, OSError):  # network error, timeout, response cut off
            if attempt == retries:
                raise
            delay = 2 ** (attempt + 2)
        time.sleep(delay)
    raise AssertionError("unreachable")


def _reason(err: urllib.error.HTTPError) -> str:
    body = err.read().decode("utf-8", "replace")
    try:
        return str(json.loads(body).get("reason", body))
    except (json.JSONDecodeError, AttributeError):
        return body


def _retry_after(err: urllib.error.HTTPError) -> float | None:
    value = err.headers.get("Retry-After") if err.headers else None
    try:
        return float(value) + 1 if value else None
    except ValueError:
        return None
