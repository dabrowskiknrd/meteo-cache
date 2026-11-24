"""Golden test against a copy of the real cache (skipped when the cache is absent); no network.

The cache directory is read from the environment variable METEO_CACHE_REAL (default: data/cache/weather_cells
under the repository root) and the golden file from METEO_CACHE_GOLDEN (default: local/weather_golden.parquet).
The golden cells are the cells of the golden file; without a golden file, they are read from
METEO_CACHE_GOLDEN_CELLS (cells separated by ";", e.g. "52.2,21.0;50.1,19.9"), or else the first three cells
that cover the period are used.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from meteo_cache import weather as W
from meteo_cache.openmeteo import DEFAULT_VARIABLES, OPEN_METEO

ROOT = Path(__file__).resolve().parents[1]
REAL = Path(os.environ.get("METEO_CACHE_REAL", ROOT / "data" / "cache" / "weather_cells"))
GOLDEN = Path(os.environ.get("METEO_CACHE_GOLDEN", ROOT / "local" / "weather_golden.parquet"))

pytestmark = pytest.mark.skipif(not REAL.is_dir(), reason="real weather cache not present")

NEEDED = (pd.Timestamp("2015-01-01"), pd.Timestamp("2015-04-01"))


def _hashes(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


def _no_network(*a, **k):
    raise AssertionError("network")


def _covers(ranges) -> bool:
    return any(pd.Timestamp(a) <= NEEDED[0] and pd.Timestamp(b) >= NEEDED[1] for a, b in ranges)


def _covered_cells(cov: dict) -> list[str]:
    return sorted(c for c, v in cov.items()
                  if not c.startswith("_") and all(_covers(v.get(x, [])) for x in DEFAULT_VARIABLES))


def _golden_cells(covered: list[str]) -> list[str]:
    """Return the cells of the golden file, else those of METEO_CACHE_GOLDEN_CELLS, else the first three covered."""
    if GOLDEN.exists():
        return sorted(pd.read_parquet(GOLDEN, columns=["cell"])["cell"].unique())
    if cells := os.environ.get("METEO_CACHE_GOLDEN_CELLS"):
        return [c.strip() for c in cells.split(";") if c.strip()]
    return covered[:3]


@pytest.fixture(scope="module")
def cache_copy(tmp_path_factory):
    dst = tmp_path_factory.mktemp("real") / "weather_cells"
    shutil.copytree(REAL, dst)
    return dst, _hashes(dst)


@pytest.fixture(scope="module")
def result(cache_copy):
    dst, _ = cache_copy
    covered = _covered_cells(json.loads((dst / "_coverage.json").read_text()))
    cells = _golden_cells(covered)
    assert cells, "no cell of the cache covers the period"
    assert set(cells) <= set(covered), "the golden cells no longer cover the period"
    source = type("NoNetwork", (type(OPEN_METEO),), {"fetch": staticmethod(_no_network)})()  # a request fails the test
    df = W.fetch_weather(cells, "2015-01-01", "2015-03-31 23:00", cache_path=dst,
                         utc_offset_hours=-6, verbose=False, source=source)
    return cells, df


def test_matches_golden(result):
    cells, df = result
    print("cells:", cells)
    assert len(df) > 0
    if not GOLDEN.exists():
        pytest.skip(f"no golden file {GOLDEN}; create it from a trusted version of the code with "
                    f"W.fetch_weather({cells}, '2015-01-01', '2015-03-31 23:00', cache_path=<copy of {REAL}>, "
                    f"utc_offset_hours=-6).to_parquet('{GOLDEN}')")
    pd.testing.assert_frame_equal(df, pd.read_parquet(GOLDEN))


def test_cache_files_unchanged(cache_copy, result):
    dst, before = cache_copy
    assert _hashes(dst) == before
