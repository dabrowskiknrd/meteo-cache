"""Time handling for the meteo_cache package.

All timestamps are naive (no time zone attached). The library works in UTC; local time appears only in
``fetch_weather`` and ``fetch_cell``, which convert with a fixed ``utc_offset_hours`` (local time minus UTC).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

# The archive lags the present by about five days; more recent days are not requested (7 leaves a margin).
ARCHIVE_DELAY_DAYS = 7


def as_utc(t) -> pd.Timestamp:
    """Return a naive UTC Timestamp: naive input is taken to be UTC already; tz-aware input is converted."""
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def local_to_utc(local, utc_offset_hours: float) -> pd.Timestamp:
    """Convert a naive local time to naive UTC; e.g. 06:00 at offset -6 becomes 12:00. Tz-aware input raises."""
    local = pd.Timestamp(local)
    if local.tzinfo is not None:
        raise ValueError("pass naive local times (the stamps of the data), not tz-aware ones")
    return local - pd.Timedelta(hours=utc_offset_hours)


def to_local(t_utc: pd.Series, utc_offset_hours: float) -> pd.Series:
    """Convert a Series of naive UTC times to naive local times (the inverse of ``local_to_utc``)."""
    return t_utc + pd.Timedelta(hours=utc_offset_hours)


def utc_days(start_utc, end_utc) -> tuple[date, date]:
    """Return the first and last UTC calendar day (``datetime.date``) of the period."""
    return as_utc(start_utc).date(), as_utc(end_utc).date()


def wanted_days(start_utc, end_utc, months=None, pad_days: int = 3, newest: date | None = None) -> set[date]:
    """Return the UTC days (``datetime.date``) to fetch and read for the period.

    months    keep only days in these calendar months, e.g. (11, 12, 1, 2, 3); None keeps every day
    pad_days  also keep this many days before and after each kept month
    newest    drop later days; the default is today minus ARCHIVE_DELAY_DAYS, as the archive is not yet complete
    """
    first, last = utc_days(start_utc, end_utc)
    days = pd.date_range(first, last, freq="D")
    if months is not None:
        keep = np.zeros(len(days), dtype=bool)
        for shift in range(-pad_days, pad_days + 1):
            keep |= (days + pd.Timedelta(days=shift)).month.isin(set(months))
        days = days[keep]
    if newest is None:
        newest = pd.Timestamp.now(tz="UTC").date() - timedelta(days=ARCHIVE_DELAY_DAYS)
    return {d for d in days.date if d <= newest}


def day_ranges(days, max_days: int) -> list[tuple[date, date]]:
    """Group sorted days into runs of consecutive days and return (first, last) of each run.

    A run longer than ``max_days`` is split; e.g. Jan 1-3 and Jan 7 with max_days=2 give
    [(Jan 1, Jan 2), (Jan 3, Jan 3), (Jan 7, Jan 7)].
    """
    out, first, prev = [], None, None
    for d in days:
        if first is None or (d - prev).days != 1 or (d - first).days >= max_days:
            if first is not None:
                out.append((first, prev))
            first = d
        prev = d
    if first is not None:
        out.append((first, prev))
    return out
