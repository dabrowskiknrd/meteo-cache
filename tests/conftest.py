"""Shared test double: a fake weather source (no network)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
import pytest

from meteo_cache.openmeteo import OPEN_METEO, VARIABLES


@dataclass
class FakeSource:
    """Satisfies ``Source``: deterministic values, records ``calls`` as (lat, lon, first, last, variables)."""

    name: str = "fake"
    variables: dict = field(default_factory=lambda: VARIABLES)
    calls: list = field(default_factory=list)
    fail: Callable[[float], Exception | None] = lambda lat: None  # an error to raise for this latitude
    empty: bool = False  # answer with NaN only, like an archive without values

    cost = staticmethod(OPEN_METEO.cost)
    max_days = staticmethod(OPEN_METEO.max_days)

    def fetch(self, lat, lon, first, last, variables):
        self.calls.append((lat, lon, f"{first:%Y-%m-%d}", f"{last:%Y-%m-%d}", tuple(variables)))
        if err := self.fail(lat):
            raise err
        t = pd.date_range(first, pd.Timestamp(last) + pd.Timedelta(hours=23), freq="h")
        out = pd.DataFrame({"t_utc": t})
        for v in variables:
            out[self.variables[v][0]] = float("nan") if self.empty else lat + t.day_of_year + t.hour / 100
        return out


@pytest.fixture
def fake() -> FakeSource:
    return FakeSource()

