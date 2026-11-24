"""Hourly weather per weather cell, fetched from a weather API and stored in an incremental on-disk cache.

Terms used throughout:
    cell        a weather location: "lat,lon" with both coordinates rounded to 0.1 degree, e.g. "52.2,21.0"
    period      (start, end) of one cell, end included; ``periods`` is {cell: (start, end)}
    variable    an API variable name, e.g. "temperature_2m"; its column in a table is e.g. "air_temperature"
    wanted days the UTC calendar days a period covers, optionally restricted to ``months``
    API calls   the weighted count the API charges for a request (see ``Source.cost``)

The four steps work in UTC and can be run one at a time:
    plan_requests -> list the requests the cache needs (sends nothing)
    missing_days  -> count the wanted days the cache lacks, per cell
    run_requests  -> send requests and store the answers in the cache
    read_weather  -> read cached hours as one table
``fetch_weather`` runs all four and converts to local time; it requires ``utc_offset_hours`` (e.g. -6 for UTC-6).
Accumulated variables (precipitation, rain, snowfall) are totals over the hour that ends at the timestamp.

    weather = fetch_weather(cells, start, end, cache_path=CACHE / "weather_cells", utc_offset_hours=-6)
"""

from __future__ import annotations

import collections
import time
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from meteo_cache.cache import CacheMismatchError, WeatherCache
from meteo_cache.days import as_utc, day_ranges, local_to_utc, to_local, utc_days, wanted_days
from meteo_cache.flags import add_weather_flags
from meteo_cache.openmeteo import DEFAULT_VARIABLES, OPEN_METEO, VARIABLES, OpenMeteoArchive
from meteo_cache.source import RateLimitError, Source

__all__ = ["DEFAULT_VARIABLES", "VARIABLES", "OpenMeteoArchive", "RateLimitError", "WeatherCache", "CacheMismatchError",
           "Request", "RunResult", "cell_key", "column_names", "wanted_days", "plan_requests", "missing_days",
           "run_requests", "read_weather", "fetch_cell", "fetch_weather", "cache_status", "add_weather_flags"]


@dataclass(frozen=True)
class Request:
    """One API request, as planned by ``plan_requests`` and sent by ``run_requests``.

    cell        the weather cell, e.g. "41.9,12.5"
    variables   API variable names fetched together, e.g. ("temperature_2m", "precipitation")
    first, last first and last UTC day, both included
    cost        API calls the request counts as
    """

    cell: str
    variables: tuple[str, ...]
    first: date
    last: date
    cost: float


@dataclass
class RunResult:
    """Outcome of ``run_requests``.

    spent    API calls used by the completed requests
    done     number of completed requests; each one is stored in the cache
    stopped  why the run ended early (budget or rate limit), or None when every request completed
    """

    spent: float
    done: int
    stopped: str | None


def cell_key(lat, lon) -> pd.Series | str:
    """Return the cell "lat,lon" with both coordinates rounded to 0.1 degree.

    Accepts two scalars (returns a string) or two pandas Series (returns a Series):
        sites["cell"] = cell_key(sites["latitude"], sites["longitude"])
    """
    if isinstance(lat, pd.Series):
        return lat.round(1).astype(str) + "," + pd.Series(lon, index=lat.index).round(1).astype(str)
    return f"{round(float(lat), 1)},{round(float(lon), 1)}"


def column_names(variables: Iterable[str] = DEFAULT_VARIABLES, source: Source = OPEN_METEO) -> list[str]:
    """Return the table column of each API variable, e.g. ["temperature_2m"] -> ["air_temperature"]."""
    return [source.variables[v][0] for v in variables]


# ---------------------------------------------------------------- the four steps (UTC only)


def plan_requests(cache: WeatherCache, periods: dict, variables: Iterable[str], *, source: Source = OPEN_METEO,
                  months=None, pad_days: int = 3, calls_per_minute: float = 500) -> list[Request]:
    """List the API requests needed to fill what the cache lacks; nothing is sent.

    periods           {cell: (start, end)} in UTC; the dict order is the order in which cells are fetched
    variables         API variable names, e.g. DEFAULT_VARIABLES
    months, pad_days  restrict to these calendar months plus pad_days days on each side; None means all months
    calls_per_minute  a long run of missing days is split so that one request stays at or below this cost

    Returns one Request per run of consecutive missing days; an empty list when the cache is complete.

        plan_requests(cache, {"41.9,12.5": ("2020-01-01", "2020-01-31 23:00")}, DEFAULT_VARIABLES)
    """
    variables = _check_variables(variables, source)
    max_days = source.max_days(len(variables), calls_per_minute)
    plan = []
    for cell, wanted in _wanted_by_cell(periods, months, pad_days).items():
        for days, lacking in _lacking_variables(cache, cell, wanted, variables).items():
            plan += [Request(cell, lacking, first, last, source.cost(len(lacking), (last - first).days + 1))
                     for first, last in day_ranges(days, max_days)]
    return plan


def missing_days(cache: WeatherCache, periods: dict, variables: Iterable[str], *, source: Source = OPEN_METEO,
                 months=None, pad_days: int = 3) -> pd.DataFrame:
    """Count, per cell, the wanted days and those that lack at least one of the variables.

    Arguments as in ``plan_requests``. Returns a table with the columns cell, wanted_days, missing_days;
    missing_days = 0 means the cell is complete for its period.
    """
    variables = _check_variables(variables, source)
    rows = [dict(cell=cell, wanted_days=len(wanted), missing_days=len(wanted - _complete_days(cache, cell, variables)))
            for cell, wanted in _wanted_by_cell(periods, months, pad_days).items()]
    return pd.DataFrame(rows, columns=["cell", "wanted_days", "missing_days"])


def run_requests(requests: list[Request], cache: WeatherCache, *, source: Source = OPEN_METEO,
                 calls_per_minute: float = 500, max_calls: float | None = None, max_minutes: float | None = None,
                 verbose: bool = True) -> RunResult:
    """Send the requests in order and store each answer in the cache as soon as it arrives.

    calls_per_minute  requests are delayed so that the API calls of any 60 s window stay at or below this value
    max_calls         stop before a request that would bring the API calls of this run above this value
    max_minutes       stop when this much time has passed

    A budget or a rate limit ends the run without an exception; ``RunResult.stopped`` gives the reason, and a
    later call continues with what is still missing. Raises ``CacheMismatchError`` when the cache was written
    by another source or with other column names or units.
    """
    pacer, began, run = _Pacer(calls_per_minute), time.monotonic(), RunResult(spent=0.0, done=0, stopped=None)
    for request in requests:
        minutes = (time.monotonic() - began) / 60
        run.stopped = _budget_used(run.spent, request.cost, minutes, max_calls, max_minutes)
        if run.stopped:
            break
        try:
            fresh = _send(request, source, pacer)
        except RateLimitError as err:
            run.stopped = str(err)
            break
        cache.add(request.cell, source, list(request.variables), request.first, request.last, fresh)
        run.spent, run.done = run.spent + request.cost, run.done + 1
        if verbose:
            print(f"  {run.done}/{len(requests)} {request.cell} {request.first:%Y-%m-%d} to "
                  f"{request.last:%Y-%m-%d}: {len(fresh):,} hours")
    if verbose and requests:
        print(f"weather run: {run.spent:,.0f} API calls in {(time.monotonic() - began) / 60:.1f} min")
    return run


def read_weather(cache: WeatherCache, periods: dict, variables: Iterable[str], *, months=None,
                 pad_days: int = 3, fallback_columns: dict[str, str] | None = None) -> pd.DataFrame:
    """Read the cached hours of the periods as one table; nothing is fetched.

    periods, variables, months, pad_days  as in ``plan_requests``
    fallback_columns                       {variable: column} for variables the cache has never stored (their
                                           column is then all NaN); without it such a variable raises ValueError

    Returns the columns cell, t_utc, lat, lon and one column per variable, sorted by cell and t_utc. Only hours
    on wanted days are included; hours the cache lacks are absent rows, not NaN rows.
    """
    values = cache.columns(list(variables), fallback_columns)
    wanted = _wanted_by_cell(periods, months, pad_days)
    parts = [_cell_hours(cache, cell, period, wanted[cell], values) for cell, period in _as_utc(periods).items()]
    parts = [p for p in parts if p is not None]
    columns = ["cell", "t_utc", "lat", "lon", *values]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    return out[columns].sort_values(["cell", "t_utc"]).reset_index(drop=True)


# ---------------------------------------------------------------- local-time wrappers


def fetch_cell(lat: float, lon: float, start, end, variables: Iterable[str] = DEFAULT_VARIABLES,
               utc_offset_hours: float | None = None, *, source: Source = OPEN_METEO) -> pd.DataFrame:
    """Fetch hourly weather for one location directly from the API, without the cache.

    start, end        naive local times, both included
    utc_offset_hours  local time minus UTC in hours (required; e.g. -6 for UTC-6, 0 for UTC)

    Returns the columns t (local time), lat, lon, cell and one column per variable.
    """
    _require_offset(utc_offset_hours)
    variables = _check_variables(variables, source)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    first, last = utc_days(local_to_utc(start, utc_offset_hours), local_to_utc(end, utc_offset_hours))
    out = source.fetch(lat, lon, first, last, variables)
    out = out.rename(columns={"t_utc": "t"})
    out["t"] = to_local(out["t"], utc_offset_hours)
    out = out[(out["t"] >= start.floor("h")) & (out["t"] <= end)].reset_index(drop=True)
    out.insert(1, "lat", float(lat))
    out.insert(2, "lon", float(lon))
    out.insert(3, "cell", cell_key(lat, lon))
    return out


def fetch_weather(
    cells: Iterable[str],
    start=None,
    end=None,
    variables: Iterable[str] = DEFAULT_VARIABLES,
    cache_path: str | Path | None = None,
    utc_offset_hours: float | None = None,
    calls_per_minute: float = 500,
    verbose: bool = True,
    months: Iterable[int] | None = None,
    pad_days: int = 3,
    on_limit: str = "raise",
    max_calls: float | None = None,
    max_minutes: float | None = None,
    periods: dict | None = None,
    return_data: bool = True,
    *,
    source: Source = OPEN_METEO,
) -> pd.DataFrame:
    """Return hourly weather for the cells in local time, fetching only what the cache lacks.

    cells             cells in priority order; a run stopped early has completed the first ones
    start, end        naive local times, both included; required for cells without an entry in ``periods``
    periods           {cell: (start, end)} in local time, a separate period per cell
    cache_path        cache directory; None caches in memory
    utc_offset_hours  local time minus UTC in hours (required; e.g. -6 for UTC-6, 0 for UTC)
    months, pad_days  fetch and return only these calendar months plus pad_days days on each side
    calls_per_minute, max_calls, max_minutes  pacing and budgets, as in ``run_requests``
    on_limit          at a budget or rate limit: "raise" raises RateLimitError; "partial" warns and returns
                      what the cache holds
    return_data       False returns ``cache_status`` of the cells instead of the weather (saves memory in
                      long loading runs)

    Returns the columns cell, t (local time), lat, lon and one column per variable, sorted by cell and t.
    Every completed request is stored at once, so a repeated call resumes where an interrupted one stopped.
    """
    utc_periods = _local_periods_to_utc(cells, start, end, periods, utc_offset_hours)
    if on_limit not in ("raise", "partial"):
        raise ValueError('on_limit must be "raise" or "partial"')
    variables = _check_variables(variables, source)
    cache = WeatherCache.open(cache_path)
    cache.check(source, variables)  # a renamed column, changed unit or other source stops before any request
    plan = plan_requests(cache, utc_periods, variables, source=source, calls_per_minute=calls_per_minute,
                         months=months, pad_days=pad_days)
    _announce(plan, len(utc_periods), calls_per_minute, max_calls, source, verbose)
    run = run_requests(plan, cache, source=source, calls_per_minute=calls_per_minute, max_calls=max_calls,
                       max_minutes=max_minutes, verbose=verbose)
    if run.stopped:
        _stop(run, plan, on_limit)
    if not return_data:
        status = cache.status()
        return status[status["cell"].isin(utc_periods)].reset_index(drop=True)
    never_stored = dict(zip(variables, column_names(variables, source)))  # a stopped run may leave a variable empty
    out = read_weather(cache, utc_periods, variables, months=months, pad_days=pad_days, fallback_columns=never_stored)
    _require_every_cell(out, utc_periods, run.stopped)
    return _to_local(out, utc_offset_hours)


def cache_status(cache_path: str | Path) -> pd.DataFrame:
    """Summarize the cache: one row per cell and variable with the number of days, first and last UTC day.

    Reads only; it never writes.
    """
    return WeatherCache.open(cache_path).status()


# ---------------------------------------------------------------- internals


def _check_variables(variables: Iterable[str], source: Source) -> list[str]:
    variables = list(variables)
    if not variables:
        raise ValueError("no variables requested")
    unknown = set(variables) - set(source.variables)
    if unknown:
        raise ValueError(f"unknown {source.name} variables {sorted(unknown)}; known: {sorted(source.variables)}; "
                         f"add them to source.variables")
    return variables


def _as_utc(periods: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {cell: (as_utc(start), as_utc(end)) for cell, (start, end) in periods.items()}


def _wanted_by_cell(periods: dict, months, pad_days: int) -> dict[str, set[date]]:
    """Return {cell: wanted days}; cells with the same period share one computation."""
    periods = _as_utc(periods)
    by_period = {p: wanted_days(*p, months, pad_days) for p in set(periods.values())}  # each period computed once
    return {cell: by_period[p] for cell, p in periods.items()}


def _lacking_variables(cache: WeatherCache, cell: str, wanted: set[date], variables) -> dict[tuple, tuple[str, ...]]:
    """Missing days -> the variables that lack exactly those days (one request serves them all)."""
    groups: dict[tuple, list[str]] = {}
    for v in variables:
        if days := tuple(sorted(wanted - cache.covered(cell, v))):
            groups.setdefault(days, []).append(v)
    return {days: tuple(vs) for days, vs in groups.items()}


def _complete_days(cache: WeatherCache, cell: str, variables) -> set[date]:
    """Return the days for which every variable is cached."""
    return set.intersection(*(cache.covered(cell, v) for v in variables))


def _lat_lon(cell: str) -> tuple[float, float]:
    lat, lon = map(float, cell.split(","))
    return lat, lon


def _budget_used(spent: float, cost: float, minutes: float, max_calls, max_minutes) -> str | None:
    """Return why the next request must not be sent (max_calls or max_minutes reached), or None."""
    if max_calls is not None and spent + cost > max_calls:
        return f"the budget max_calls={max_calls:,.0f} is used ({spent:,.0f} calls made)"
    if max_minutes is not None and minutes >= max_minutes:
        return f"the time budget max_minutes={max_minutes:g} is used ({spent:,.0f} calls made)"
    return None


def _send(request: Request, source: Source, pacer: _Pacer) -> pd.DataFrame:
    """Wait for the pacer, send one request and return the answer."""
    pacer.wait(request.cost)
    fresh = source.fetch(*_lat_lon(request.cell), request.first, request.last, list(request.variables))
    pacer.record(request.cost)
    return fresh


def _cell_hours(cache: WeatherCache, cell: str, period, wanted: set[date], values: list[str]) -> pd.DataFrame | None:
    """Return the cached hours of one cell inside its period and on wanted days, or None if not cached."""
    table = cache.read(cell)
    if table is None:
        return None
    start, end = period
    t = table["t_utc"]
    keep = t.between(start.floor("h"), end) & t.dt.normalize().isin(pd.to_datetime(sorted(wanted)))
    lat, lon = _lat_lon(cell)
    return table.reindex(columns=["t_utc", *values])[keep].assign(cell=cell, lat=lat, lon=lon)


def _require_offset(utc_offset_hours) -> None:
    if utc_offset_hours is None:
        raise ValueError("pass utc_offset_hours (0 for UTC)")


def _local_periods_to_utc(cells, start, end, periods, utc_offset_hours) -> dict[str, tuple]:
    """{cell: (start, end)} in UTC, cells in the caller's order (the priority order), duplicates dropped."""
    _require_offset(utc_offset_hours)
    cells, periods = list(dict.fromkeys(cells)), periods or {}
    if (start is None or end is None) and any(c not in periods for c in cells):
        raise ValueError("start and end are required for cells without an entry in periods")
    return {c: tuple(local_to_utc(t, utc_offset_hours) for t in periods.get(c, (start, end))) for c in cells}


def _to_local(out: pd.DataFrame, utc_offset_hours: float) -> pd.DataFrame:
    out = out.rename(columns={"t_utc": "t"})
    out["t"] = to_local(out["t"], utc_offset_hours)
    return out


def _announce(plan: list[Request], n_cells: int, calls_per_minute: float, max_calls, source: Source,
              verbose: bool) -> None:
    """Print the size of the plan; warn when it exceeds the source's hourly limit."""
    total = sum(r.cost for r in plan)
    if plan and verbose:
        todo = len({r.cell for r in plan})
        print(f"weather cache: {n_cells - todo} of {n_cells} cells complete; {len(plan)} requests for "
              f"{todo} cells, {total:,.0f} API calls, approximately {total / calls_per_minute:.0f} min "
              f"at {calls_per_minute:.0f} calls/min")
    hourly, daily = getattr(source, "limit_per_hour", None), getattr(source, "limit_per_day", None)
    if hourly and total > hourly and (max_calls is None or max_calls > hourly):
        warnings.warn(f"{total:,.0f} API calls exceed the hourly limit of {hourly:,}"
                      + (f" (daily limit {daily:,})" if daily else "")
                      + "; the run stops at the limit and a repeated call retrieves only what is still missing",
                      stacklevel=3)


def _stop(run: RunResult, plan: list[Request], on_limit: str) -> None:
    """Act on a run that stopped early: raise RateLimitError (on_limit="raise") or warn ("partial")."""
    left = sum(r.cost for r in plan) - run.spent
    if on_limit == "raise":
        raise RateLimitError(f"{run.stopped}; {left:,.0f} API calls remain, the retrieved days are in the cache")
    warnings.warn(f"weather is incomplete: {run.stopped}; {left:,.0f} API calls remain", stacklevel=3)


def _require_every_cell(out: pd.DataFrame, periods: dict, stopped: str | None) -> None:
    """Raise RuntimeError when a cell has no weather although the run completed."""
    absent = set(periods) - set(out["cell"])
    if absent and not stopped:
        raise RuntimeError(f"no weather for cells {sorted(absent)}")


class _Pacer:
    """Keeps the weighted calls of any 60 s window at or below ``per_minute``."""

    def __init__(self, per_minute: float):
        self.per_minute = per_minute
        self.log: collections.deque[tuple[float, float]] = collections.deque()  # (time, calls)

    def wait(self, calls: float) -> None:
        while True:
            now = time.monotonic()
            while self.log and now - self.log[0][0] >= 60:
                self.log.popleft()
            if sum(c for _, c in self.log) + calls <= self.per_minute:
                return
            time.sleep(60 - (now - self.log[0][0]) + 0.5)

    def record(self, calls: float) -> None:
        self.log.append((time.monotonic(), calls))
