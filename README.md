# meteo-cache

Hourly weather per weather cell, from the Open-Meteo historical archive, kept in an incremental on-disk cache: only the days and variables the cache lacks are requested. Values are reanalysis model output (ERA5 family, cells of about 10-25 km), not site measurements; temperature is good, precipitation and gusts are local and rougher.

A weather cell is `"lat,lon"` rounded to 0.1 degree, for example `"52.2,21.1"` (`cell_key(lat, lon)`).

## Files

Modules of the package `meteo_cache` (in `src/meteo_cache/`):

| File | Holds |
|---|---|
| `days.py` | UTC time: local <-> UTC, which days are wanted, consecutive-day runs |
| `source.py` | `Source` protocol (the contract of an API), `RateLimitError` |
| `openmeteo.py` | everything Open-Meteo: `OpenMeteoArchive`, `OPEN_METEO`, `VARIABLES`, `DEFAULT_VARIABLES` |
| `cache.py` | `WeatherCache` (directory cache, `_meta.json`), `CacheMismatchError` |
| `flags.py` | `add_weather_flags`: indicators of freezing precipitation and snowfall, derived from the weather columns |
| `weather.py` | the public steps and the wrappers below; `meteo_cache` re-exports its public names |

## Installation and tests

The package requires Python 3.13 and is managed with uv. `uv sync` creates the environment and installs the package in editable mode; another project installs it with `uv add <path to this repository>` or `pip install <path to this repository>`.

The tests are run with `uv run pytest -q`; they use a fake source and send no requests. `tests/test_real_cache.py` compares the output against a copy of a real cache: it reads the cache directory from the environment variable `METEO_CACHE_REAL` (default `data/cache/weather_cells`) and the golden file from `METEO_CACHE_GOLDEN` (default `local/weather_golden.parquet`), and it is skipped when the cache directory is absent.

The public names are imported from the package: `from meteo_cache import fetch_weather, WeatherCache`.

## Time

- Disk, cache and all step functions are **UTC** (`t_utc`, naive). Naive input is taken as UTC; tz-aware input is converted.
- Local time exists only in `fetch_weather` and `fetch_cell`, which require `utc_offset_hours`, the fixed offset of local time from UTC in hours (no default; `-6` denotes UTC-6 and `0` denotes UTC). They return `t` = `t_utc` + offset.
- `end` is **inclusive**: `"2014-01-10"` means 2014-01-10 00:00, so that day contributes one hour. Use `"2014-01-10 23:00"` for the whole day. The start is floored to the hour in UTC (the same as local for whole-hour offsets).
- With `months=[12, 1, 2]`, only the wanted UTC days (those months plus `pad_days=3` around them) are requested **and** returned, so a call returns the same rows whatever the cache held before.
- Days newer than 7 days ago are never requested (the archive lags about 5 days).
- Accumulated variables (`precipitation`, `rain`, `snowfall`) are totals over the hour that ends at the timestamp; the others are values at the timestamp.

## Cache layout

```
<cache>/52.2,21.0.parquet   t_utc + one column per variable (one file per cell)
<cache>/_coverage.json       {cell: {api_variable: [[first_day, last_day], ...]}}
<cache>/_meta.json           {"time": "UTC", "source": "open-meteo-archive", "variables": {"temperature_2m": {"column": "air_temperature", "unit": "degF"}}}
```

- `_coverage.json` is the only record of what was retrieved (an empty value in a cell file can be a real archive gap).
- `_meta.json` is created by the first fetch that stores data; later variables are appended, existing definitions never change; reading never writes.
- A different source, column name or unit raises `CacheMismatchError` before any request: use a new directory.
- Writes go through a temp file; the cell file is written before `_coverage.json`, so an interrupted run only causes re-fetching, never loss.
- External consumers may read the parquet files and `_coverage.json` directly; the layout is therefore part of the public interface and must remain unchanged.

## Walk-through (offline)

Read-only on the real cache: nothing below writes.

```python
from meteo_cache import *   # WeatherCache, plan_requests, missing_days, run_requests, read_weather, ...

cache = WeatherCache.open("data/cache/weather_cells")
cache.status().head()                                      # per cell and variable: days, first, last (UTC)
periods = {"52.3, 21.1": ("2014-01-01", "2014-01-31 23:00")}   # {cell: (start, end)} in UTC, end inclusive
plan_requests(cache, periods, DEFAULT_VARIABLES)           # [] : already cached; else Request(cell, variables, first, last, cost)
missing_days(cache, periods, DEFAULT_VARIABLES)            # cell, wanted_days, missing_days (0 here)
read_weather(cache, periods, DEFAULT_VARIABLES).head()     # cell, t_utc, lat, lon, air_temperature, ...
read_weather(cache, periods, DEFAULT_VARIABLES, months=[1], pad_days=0).shape   # (744, 8)
```

Fetching writes, so try it on a throwaway directory with a source that needs no network. `ConstantSource` is a minimal `Source` (it is also the "replace the API" example):

```python
import pandas as pd

class ConstantSource:
    name = "constant"
    variables = {"temp": ("air_temperature", "degC")}      # api variable -> (column, unit)
    def cost(self, n_variables, n_days): return 1.0        # weighted calls of one request
    def max_days(self, n_variables, calls): return 30      # longest period of one request
    def fetch(self, lat, lon, first, last, variables):     # UTC days first..last, both included
        t = pd.date_range(first, pd.Timestamp(last) + pd.Timedelta(hours=23), freq="h")
        return pd.DataFrame({"t_utc": t, **{self.variables[v][0]: 20.0 for v in variables}})
```

```python
import tempfile
cache_dir = tempfile.mkdtemp() + "/cache"
src, cache = ConstantSource(), WeatherCache.open(cache_dir)
periods = {"1.0,2.0": ("2014-01-01", "2014-01-10")}
plan = plan_requests(cache, periods, ["temp"], source=src)    # one Request: 10 days, cost 1.0
run_requests(plan, cache, source=src)                         # RunResult(spent=1.0, done=1, stopped=None)
plan_requests(cache, periods, ["temp"], source=src)           # []
fetch_weather(["1.0,2.0"], "2014-01-01", "2014-01-10", ["temp"], cache_path=cache_dir,
              utc_offset_hours=-6, source=src, verbose=False).head(2)   # cell, t (local), lat, lon, air_temperature
fetch_weather(["1.0,2.0"], "2014-01-01", "2014-01-10", ["temp"], cache_path=cache_dir,
              utc_offset_hours=-6, source=src, return_data=False)       # only the cache status
fetch_weather(["1.0,2.0"], "2014-01-01", "2014-01-10", cache_path=cache_dir, utc_offset_hours=-6)
# CacheMismatchError: ... written by source 'constant', not 'open-meteo-archive'
```

## Steps and wrappers (all in `weather.py`)

| Function | Does |
|---|---|
| `WeatherCache.open(path)` | opens a cache directory (`None`: in memory); never writes |
| `cache.status()` / `cache_status(path)` | `cell, variable, days, first, last` |
| `cache.columns(variables)` | column names as the cache stores them |
| `plan_requests(cache, periods, variables, *, source, months, pad_days, calls_per_minute)` | the `Request`s that fill the gaps; sends nothing |
| `missing_days(cache, periods, variables, *, source, months, pad_days)` | per cell: wanted and missing days |
| `run_requests(requests, cache, *, source, calls_per_minute, max_calls, max_minutes, verbose)` | sends in order, stores every answer; a budget or limit stops the run (`RunResult.stopped`), never raises |
| `read_weather(cache, periods, variables, *, months, pad_days)` | cached hours of the periods: `cell, t_utc, lat, lon, <columns>`; never writes |
| `fetch_weather(cells, start, end, variables, cache_path, utc_offset_hours, ..., on_limit, max_calls, periods, return_data, source)` | local-time wrapper: plan, run, read. `cache_path=None` caches in memory; `return_data=False` returns the status instead of the data. `on_limit="raise"` raises `RateLimitError`, `"partial"` warns and returns what the cache holds |
| `fetch_cell(lat, lon, start, end, variables, utc_offset_hours)` | one location, no cache |
| `cell_key`, `column_names(variables)` | cell key; column names of the variables |
| `add_weather_flags(df)` | adds `freezing_precip`, `snowing`, `wet_below_freezing` (needs `weather_code`, `precipitation`, `air_temperature`; warns when absent) |

`periods` is `{cell: (start, end)}`; the order of the cells is the order of the requests, so put the important cells first. `fetch_weather` takes `start`/`end` for all cells, or `periods` for per-cell ones.

## Variables and limits (Open-Meteo)

`VARIABLES` in `openmeteo.py` maps an Open-Meteo variable to `(column, unit)`; the active ones are `temperature_2m` (`air_temperature`, degF), `relative_humidity_2m` (%), `precipitation` (inch), `wind_speed_10m` (`wind_speed`, mph). Others sit commented out beside them; units are set by `UNITS` (degF, mph, inch).

The free API allows 600 calls/minute, 5,000/hour, 10,000/day (https://open-meteo.com/en/pricing). A request counts as `max(1, variables/10) x max(1, days/14)` calls. `fetch_weather` paces to `calls_per_minute` (default 500), prints the plan, and warns when it exceeds the hourly limit. A per-minute 429 is retried; an hourly or daily 429 stops the run, the retrieved days stay in the cache, and a repeated call continues. Server and network errors are retried after 4, 8 and 16 s.

## Replace the API

- Another URL: `OpenMeteoArchive(url="https://...")`, passed as `source=`.
- Another variable: add `"api_name": ("column", "unit")` to `VARIABLES`.
- Another provider: a class like `ConstantSource` from the walk-through, with `name`, `variables`, `cost`, `max_days`, `fetch` (and optionally `limit_per_hour` / `limit_per_day`, which only drive the warning), and a **new cache directory** (an existing one raises `CacheMismatchError`).


`fetch` returns `t_utc` (naive UTC, hourly) and one column per requested variable, named by the column in `variables`.

## Data license and attribution

Weather data by [Open-Meteo.com](https://open-meteo.com/).

The data returned by the Open-Meteo API is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([Open-Meteo licence](https://open-meteo.com/en/licence)). Wherever data fetched with this package is displayed or redistributed, Open-Meteo must be credited with a link to https://open-meteo.com/ (for example, "Weather data by Open-Meteo.com"), and any changes made to the data must be indicated. The historical archive is derived from the reanalysis data of the Copernicus Climate Change Service (C3S).

The free API may be used for non-commercial purposes only ([terms](https://open-meteo.com/en/terms)); commercial use requires a paid plan ([pricing](https://open-meteo.com/en/pricing)). The MIT license of this package applies to the code only and does not apply to the weather data.

Citation recommended by Open-Meteo: Zippenfenig, P. (2023). *Open-Meteo.com Weather API* [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649
