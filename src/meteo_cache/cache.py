"""The weather cache: one parquet file per cell, plus a record of which days each file holds.

    cache = WeatherCache.open("data/cache/weather_cells")
    cache.status()                  # what is stored
    cache.read("52.2,21.0")         # one cell's hourly table
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from meteo_cache.days import day_ranges
from meteo_cache.source import Source

class CacheMismatchError(ValueError):
    """The cache was written by another source, or with another column name or unit."""


class WeatherCache:
    """Hourly UTC weather stored in a directory, or in memory when the directory is None.

    Files in the directory:
        <cell>.parquet   column t_utc (naive UTC) and one column per variable
        _coverage.json   {cell: {variable: [[first day, last day], ...]}}; the record
                         of the UTC days retrieved. A day listed here is not fetched again, even when the API
                         returned empty values for it.
        _meta.json       {"time": "UTC", "source": name, "variables": {variable: {"column": .., "unit": ..}}};
                         written by the first fetch that stores data.

    Reading (open, status, read, covered, columns, check) never writes; only ``add`` writes.
    """

    def __init__(self, directory: str | Path | None):
        self.dir = Path(directory) if directory is not None else None
        self.memory: dict[str, pd.DataFrame] = {}
        self.coverage: dict = self._load("_coverage.json") or {}
        self.meta: dict | None = self._load("_meta.json")

    @classmethod
    def open(cls, path: str | Path | None) -> WeatherCache:
        """Open the cache directory at ``path`` (None: in memory); nothing is written."""
        return cls(path)

    def cells(self) -> list[str]:
        """Return the cells recorded in ``_coverage.json``; keys that begin with "_" are not cells."""
        return [c for c in self.coverage if not c.startswith("_")]

    def covered(self, cell: str, variable: str) -> set:
        """Return the UTC days (``datetime.date``) retrieved for the cell and variable.

        Returns an empty set when the cell file has been deleted, so that the cell is fetched again.
        """
        if self.dir is not None and not self._file(cell).exists():
            return set()
        return {d.date() for a, b in self.coverage.get(cell, {}).get(variable, [])
                for d in pd.date_range(a, b, freq="D")}

    def read(self, cell: str) -> pd.DataFrame | None:
        """Return the cell's whole table (t_utc and one column per stored variable), or None if not cached."""
        if self.dir is None:
            return self.memory.get(cell)
        return pd.read_parquet(self._file(cell)) if self._file(cell).exists() else None

    def status(self) -> pd.DataFrame:
        """Return one row per cell and variable with the columns cell, variable, days, first, last (UTC days)."""
        rows = [dict(cell=c, variable=v, days=len(d), first=min(d), last=max(d))
                for c in self.cells() for v in self.coverage[c] if (d := self.covered(c, v))]
        return pd.DataFrame(rows, columns=["cell", "variable", "days", "first", "last"])

    def columns(self, variables, fallback: dict[str, str] | None = None) -> list[str]:
        """Return the column in which the cache stores each variable, e.g. ["temperature_2m"] -> ["air_temperature"].

        ``fallback`` {variable: column} names the column of variables the cache has never stored; without it,
        such a variable raises ValueError.
        """
        stored = {v: d["column"] for v, d in (self.meta or {}).get("variables", {}).items()}
        stored = {**(fallback or {}), **stored}  # what the cache stored takes precedence over the fallback
        if unknown := [v for v in variables if v not in stored]:
            raise ValueError(f"weather cache {self.dir} has never stored {unknown}; stored: {sorted(stored)}")
        return [stored[v] for v in variables]

    def check(self, source: Source, variables) -> None:
        """Raise ``CacheMismatchError`` if storing these variables from ``source`` would corrupt the cache.

        That is the case when the cache was written by another source, when a stored variable now has another
        column or unit, or when a new variable would reuse the column of a stored one. An empty cache passes.
        """
        meta = self.meta
        if meta is None:
            return
        if meta["source"] != source.name:
            raise CacheMismatchError(f"weather cache {self.dir} was written by source {meta['source']!r}, not "
                                     f"{source.name!r}; use another cache directory for {source.name!r}")
        holder = {d["column"]: v for v, d in meta["variables"].items()}  # column -> the variable stored in it
        for v in variables:
            stored, wanted = meta["variables"].get(v), _definition(source, v)
            if stored is not None and stored != wanted:
                raise CacheMismatchError(f"weather cache {self.dir} stores {v} as {stored}, but {source.name!r} "
                                         f"defines {wanted}; use another cache directory or restore the definition")
            if stored is None and holder.get(wanted["column"], v) != v:
                raise CacheMismatchError(f"weather cache {self.dir} stores column {wanted['column']!r} for "
                                         f"{holder[wanted['column']]}, but {source.name!r} defines it for {v}; "
                                         f"give {v} another column name")

    def add(self, cell: str, source: Source, variables, first, last, fresh: pd.DataFrame) -> None:
        """Store one API answer: merge ``fresh`` into the cell file and record the days ``first``-``last`` as retrieved.

        ``fresh`` has the column t_utc and one column per variable. Where old and new values overlap, the new
        ones are kept; other columns of the file are preserved. Calls ``check`` first. Files are written in the
        order _meta.json, cell file, _coverage.json, each through a temporary file, so an interruption causes at
        most a repeated fetch.
        """
        self.check(source, variables)
        old = self.read(cell)
        new = fresh.set_index("t_utc")
        if old is not None:
            new = new.combine_first(old.set_index("t_utc"))  # the new values win; other columns are kept
        new = new.sort_index().reset_index()
        days = set(pd.date_range(first, last, freq="D").date)
        per = dict(self.coverage.get(cell, {}))
        for v in variables:
            both = sorted(self.covered(cell, v) | days) if old is not None else sorted(days)
            per[v] = [[a.isoformat(), b.isoformat()] for a, b in day_ranges(both, 10**9)]
        coverage = {**self.coverage, cell: per}
        meta = self.meta or {"time": "UTC", "source": source.name, "variables": {}}
        meta = {**meta, "variables": {**meta["variables"],
                                      **{v: _definition(source, v) for v in variables if v not in meta["variables"]}}}
        if self.dir is None:
            self.memory[cell], self.coverage, self.meta = new, coverage, meta
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        if meta != self.meta:  # first: a definition without coverage costs only a re-fetch
            self._write("_meta.json", meta)
            self.meta = meta
        _replace(self._file(cell), lambda tmp: new.to_parquet(tmp, index=False))
        self._write("_coverage.json", coverage)
        self.coverage = coverage

    def _file(self, cell: str) -> Path:
        return self.dir / f"{cell}.parquet"

    def _load(self, name: str) -> dict | None:
        path = self.dir / name if self.dir is not None else None
        return json.loads(path.read_text()) if path is not None and path.exists() else None

    def _write(self, name: str, obj: dict) -> None:
        _replace(self.dir / name, lambda tmp: tmp.write_text(json.dumps(obj, indent=1)))

def _definition(source: Source, variable: str) -> dict[str, str]:
    column, unit = source.variables[variable]
    return {"column": column, "unit": unit}


def _replace(target: Path, write) -> None:
    """Write through a temporary file, so that an interrupted write leaves the previous file intact."""
    tmp = target.with_name(target.name + ".tmp")
    write(tmp)
    tmp.replace(target)
