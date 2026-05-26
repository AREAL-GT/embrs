"""Open-Meteo ERA5 archive client with Parquet caching.

Pulls hourly weather (six variables) for a centroid + date range and caches
the result as a Parquet file plus a JSON sidecar (plan §4.4, qa C1/C2/C3/C4).

The cache key is ``(round(lat,4), round(lon,4), year, m_start, m_end)``,
identifying one (region, year, fire-season-span) tuple. Manual deletion of
the cache file forces a refresh (spec §"Caching" methodological note).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


OPEN_METEO_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
)

CANONICAL_COLUMNS: tuple[str, ...] = (
    "temp_C",
    "rh_pct",
    "rain_mm_hr",
    "wind_mps",
    "wind_dir_deg",
    "cloud_pct",
)

_OPEN_METEO_TO_CANONICAL: dict[str, str] = {
    "temperature_2m": "temp_C",
    "relative_humidity_2m": "rh_pct",
    "rain": "rain_mm_hr",
    "wind_speed_10m": "wind_mps",
    "wind_direction_10m": "wind_dir_deg",
    "cloud_cover": "cloud_pct",
}

ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
ERA5_LAG_DAYS: int = 7


class OpenMeteoFetchError(RuntimeError):
    """Raised when an Open-Meteo pull fails after retries."""


@dataclass(frozen=True)
class OpenMeteoPullSpec:
    """Inputs to :func:`fetch_history`.

    ``timezone`` accepts the Open-Meteo strings ``"auto"`` or an IANA tz
    string. ``"auto"`` returns times in the centroid's local timezone, which
    is what the .wxs writer expects (qa B6).
    """

    lat: float
    lon: float
    start_date: dt.date
    end_date: dt.date
    timezone: str = "auto"


@dataclass
class OpenMeteoResult:
    """Output of :func:`fetch_history`.

    ``df`` is hourly, indexed by a tz-aware ``DatetimeIndex`` in the
    centroid's local time. Columns follow :data:`CANONICAL_COLUMNS`.
    Missing hours (rare in ERA5) appear as ``NaN`` after reindex.
    """

    df: pd.DataFrame
    elevation_m: float
    timezone: str
    source: str            # 'cache' | 'fetch'
    nan_hour_count: int    # number of NaN hours after reindex


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(spec: OpenMeteoPullSpec) -> str:
    return (
        f"{round(spec.lat, 4):.4f}_{round(spec.lon, 4):.4f}_"
        f"{spec.start_date.year}_"
        f"{spec.start_date.month:02d}_{spec.end_date.month:02d}"
    )


def _cache_paths(cache_dir: str, key: str) -> tuple[str, str]:
    return (
        os.path.join(cache_dir, f"{key}.parquet"),
        os.path.join(cache_dir, f"{key}.meta.json"),
    )


def _load_cached(parquet_path: str, meta_path: str) -> Optional[OpenMeteoResult]:
    if not (os.path.exists(parquet_path) and os.path.exists(meta_path)):
        return None
    try:
        df = pd.read_parquet(parquet_path)
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        # Parquet may not preserve tz; restore it from meta.
        tz = meta.get("timezone")
        if df.index.tz is None and tz:
            df.index = df.index.tz_localize(tz)
        nan_count = int(df[list(CANONICAL_COLUMNS)].isna().any(axis=1).sum())
        return OpenMeteoResult(
            df=df,
            elevation_m=float(meta.get("elevation_m", float("nan"))),
            timezone=str(tz) if tz else "UTC",
            source="cache",
            nan_hour_count=nan_count,
        )
    except Exception as exc:
        logger.warning(
            "Cache load failed for %s (%s); will refetch.", parquet_path, exc
        )
        return None


def _save_cache(
    result: OpenMeteoResult, parquet_path: str, meta_path: str, spec: OpenMeteoPullSpec
) -> None:
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    result.df.to_parquet(parquet_path)
    meta = {
        "schema_version": 1,
        "lat": spec.lat,
        "lon": spec.lon,
        "start_date": spec.start_date.isoformat(),
        "end_date": spec.end_date.isoformat(),
        "timezone": result.timezone,
        "elevation_m": result.elevation_m,
        "variables": list(OPEN_METEO_VARIABLES),
        "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "nan_hour_count": result.nan_hour_count,
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _warn_if_within_era5_lag(end_date: dt.date) -> None:
    today = dt.date.today()
    if end_date > today - dt.timedelta(days=ERA5_LAG_DAYS):
        logger.warning(
            "Requested end_date %s is within the ERA5 archive lag (~%d days); "
            "the most recent rows may be missing or revised later.",
            end_date.isoformat(),
            ERA5_LAG_DAYS,
        )


def _build_canonical_df(response) -> tuple[pd.DataFrame, float]:
    """Convert an openmeteo-sdk response object into the canonical DataFrame.

    Returns ``(df, elevation_m)`` where ``df`` is hourly, tz-aware-local
    indexed (Open-Meteo returns ``Time()`` in seconds since epoch and a
    ``UtcOffsetSeconds()`` accessor).
    """
    hourly = response.Hourly()
    n_vars = hourly.VariablesLength()

    if n_vars < len(OPEN_METEO_VARIABLES):
        raise OpenMeteoFetchError(
            f"Open-Meteo returned {n_vars} hourly variables, expected "
            f"{len(OPEN_METEO_VARIABLES)}: {OPEN_METEO_VARIABLES}"
        )

    # Build a UTC datetime index from Time()/TimeEnd()/Interval().
    start_utc = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    end_utc = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
    interval = pd.Timedelta(seconds=hourly.Interval())
    times_utc = pd.date_range(
        start=start_utc, end=end_utc, freq=interval, inclusive="left"
    )

    # Localize into the centroid's local time using UtcOffsetSeconds().
    offset_seconds = int(response.UtcOffsetSeconds())
    if offset_seconds == 0:
        # Pure UTC return — keep tz='UTC'.
        local_index = times_utc
    else:
        # Convert to a fixed-offset tz so the index is unambiguous and
        # round-trips cleanly. Downstream we treat .tz as informational.
        offset = dt.timezone(dt.timedelta(seconds=offset_seconds))
        local_index = times_utc.tz_convert(offset)

    columns: dict[str, np.ndarray] = {}
    for i, om_name in enumerate(OPEN_METEO_VARIABLES):
        values = np.asarray(hourly.Variables(i).ValuesAsNumpy(), dtype=float)
        if values.shape[0] != local_index.shape[0]:
            raise OpenMeteoFetchError(
                f"Variable {om_name!r} length {values.shape[0]} does not match "
                f"index length {local_index.shape[0]}"
            )
        columns[_OPEN_METEO_TO_CANONICAL[om_name]] = values

    df = pd.DataFrame(columns, index=local_index)
    df.index.name = "datetime"
    df = df.reindex(columns=list(CANONICAL_COLUMNS))

    elevation_m = float(response.Elevation())
    return df, elevation_m


def _reindex_complete_hourly(
    df: pd.DataFrame, start_local: pd.Timestamp, end_local: pd.Timestamp
) -> tuple[pd.DataFrame, int]:
    """Reindex onto a complete hourly grid, flagging NaN holes.

    ``start_local`` / ``end_local`` are tz-aware in the same tz as ``df``.
    """
    full_index = pd.date_range(
        start=start_local, end=end_local, freq="h", inclusive="both"
    )
    reindexed = df.reindex(full_index)
    nan_count = int(reindexed[list(CANONICAL_COLUMNS)].isna().any(axis=1).sum())
    if nan_count > 0:
        logger.info(
            "Reindexed Open-Meteo response onto %d-hour grid; %d hours had NaN.",
            len(full_index),
            nan_count,
        )
    reindexed.index.name = "datetime"
    return reindexed, nan_count


def fetch_history(
    spec: OpenMeteoPullSpec, cache_dir: str
) -> OpenMeteoResult:
    """Pull hourly weather from Open-Meteo ERA5 with Parquet caching.

    Args:
        spec: Pull parameters (lat/lon, dates, timezone).
        cache_dir: Directory for the Parquet + sidecar cache.

    Returns:
        :class:`OpenMeteoResult` with a hourly tz-aware DataFrame and
        ``source='cache'`` if loaded from disk, ``source='fetch'`` if
        freshly pulled.

    Raises:
        OpenMeteoFetchError: After exhausting retries.
    """
    _warn_if_within_era5_lag(spec.end_date)

    key = _cache_key(spec)
    parquet_path, meta_path = _cache_paths(cache_dir, key)
    cached = _load_cached(parquet_path, meta_path)
    if cached is not None:
        logger.info("Open-Meteo cache hit: %s", parquet_path)
        return cached

    # Lazy import to keep the import path light for test environments that
    # don't have the SDK; the BI pipeline already uses the same imports.
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    os.makedirs(cache_dir, exist_ok=True)
    http_cache_path = os.path.join(cache_dir, ".http_cache")
    cache_session = requests_cache.CachedSession(http_cache_path, expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": float(spec.lat),
        "longitude": float(spec.lon),
        "start_date": spec.start_date.isoformat(),
        "end_date": spec.end_date.isoformat(),
        "hourly": list(OPEN_METEO_VARIABLES),
        "timezone": spec.timezone,
    }

    t0 = time.monotonic()
    try:
        responses = client.weather_api(ARCHIVE_URL, params=params)
    except Exception as exc:
        raise OpenMeteoFetchError(
            f"Open-Meteo fetch failed for params={params!r}: {exc!r}"
        ) from exc
    if not responses:
        raise OpenMeteoFetchError(f"Open-Meteo returned no responses for {params!r}")
    response = responses[0]
    df, elevation_m = _build_canonical_df(response)
    logger.info(
        "Open-Meteo pulled %d hours in %.2fs (lat=%.4f, lon=%.4f, %s..%s).",
        len(df),
        time.monotonic() - t0,
        spec.lat,
        spec.lon,
        spec.start_date.isoformat(),
        spec.end_date.isoformat(),
    )

    # Reindex onto a complete hourly grid in the response's local tz.
    tz_obj = df.index.tz
    start_local = pd.Timestamp(spec.start_date).tz_localize(tz_obj or "UTC")
    end_local = (
        pd.Timestamp(spec.end_date).tz_localize(tz_obj or "UTC")
        + pd.Timedelta(hours=23)
    )
    df, nan_count = _reindex_complete_hourly(df, start_local, end_local)

    result = OpenMeteoResult(
        df=df,
        elevation_m=elevation_m,
        timezone=str(tz_obj) if tz_obj else "UTC",
        source="fetch",
        nan_hour_count=nan_count,
    )
    _save_cache(result, parquet_path, meta_path, spec)
    return result
