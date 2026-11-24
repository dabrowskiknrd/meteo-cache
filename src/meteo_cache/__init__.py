"""Hourly weather at weather cells, fetched from a weather API and kept in an incremental on-disk cache.

The public interface is defined in ``meteo_cache.weather`` and re-exported here:

    from meteo_cache import fetch_weather
    weather = fetch_weather(cells, start, end, cache_path="data/cache/weather_cells", utc_offset_hours=-6)
"""

from meteo_cache.weather import *  # noqa: F403
from meteo_cache.weather import __all__ as __all__
