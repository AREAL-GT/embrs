"""Tests for the Open-Meteo Parquet-caching client.

Network is fully mocked: the openmeteo-sdk response object is faked.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from embrs.weather_candidate_search.openmeteo_client import (
    CANONICAL_COLUMNS,
    OPEN_METEO_VARIABLES,
    OpenMeteoFetchError,
    OpenMeteoPullSpec,
    fetch_history,
)


def _build_fake_response(
    n_hours: int,
    tz_offset_sec: int = -6 * 3600,
    elevation_m: float = 1200.0,
    start_utc: int | None = None,
    start_local_date: dt.date | None = None,
):
    """Create a MagicMock matching the openmeteo-sdk response interface.

    If ``start_local_date`` is provided, ``start_utc`` is computed so that
    the returned local-time index starts at ``start_local_date 00:00`` in
    the centroid's offset (matches what Open-Meteo returns for
    ``timezone='auto'``).
    """
    if start_utc is None:
        if start_local_date is not None:
            start_local_naive = dt.datetime.combine(start_local_date, dt.time.min)
            # local = UTC + offset  =>  UTC = local - offset
            start_utc_dt = start_local_naive - dt.timedelta(seconds=tz_offset_sec)
            start_utc = int(start_utc_dt.replace(tzinfo=dt.timezone.utc).timestamp())
        else:
            start_utc = 1_690_000_000
    response = MagicMock()
    hourly = MagicMock()
    hourly.VariablesLength.return_value = len(OPEN_METEO_VARIABLES)
    hourly.Time.return_value = start_utc
    hourly.TimeEnd.return_value = start_utc + n_hours * 3600
    hourly.Interval.return_value = 3600

    # Build a different array per variable so we can spot misordering.
    arrays = {
        "temperature_2m": np.full(n_hours, 20.0),
        "relative_humidity_2m": np.full(n_hours, 50.0),
        "rain": np.zeros(n_hours),
        "wind_speed_10m": np.full(n_hours, 3.0),
        "wind_direction_10m": np.full(n_hours, 180.0),
        "cloud_cover": np.full(n_hours, 10.0),
    }

    def variables_at(i):
        name = OPEN_METEO_VARIABLES[i]
        var = MagicMock()
        var.ValuesAsNumpy.return_value = arrays[name]
        return var

    hourly.Variables.side_effect = variables_at
    response.Hourly.return_value = hourly
    response.UtcOffsetSeconds.return_value = tz_offset_sec
    response.Elevation.return_value = elevation_m
    return response


def test_fetch_history_canonical_dataframe(tmp_path):
    spec = OpenMeteoPullSpec(
        lat=38.5, lon=-109.6,
        start_date=dt.date(2024, 6, 1),
        end_date=dt.date(2024, 6, 2),
        timezone="auto",
    )
    n_hours = 48
    fake_resp = _build_fake_response(n_hours, start_local_date=dt.date(2024, 6, 1))

    with patch(
        "openmeteo_requests.Client"
    ) as ClientCls, patch(
        "requests_cache.CachedSession", return_value=MagicMock()
    ), patch(
        "retry_requests.retry", side_effect=lambda s, retries, backoff_factor: s
    ):
        client = MagicMock()
        client.weather_api.return_value = [fake_resp]
        ClientCls.return_value = client
        result = fetch_history(spec, cache_dir=str(tmp_path / "cache"))

    assert result.source == "fetch"
    assert list(result.df.columns) == list(CANONICAL_COLUMNS)
    # Reindex extends to 24 hours per day; with start+end inclusive that's 2 full days = 48 hours.
    assert len(result.df) == 48
    assert result.df["temp_C"].iloc[0] == 20.0
    assert result.elevation_m == 1200.0


def test_fetch_history_cache_hit(tmp_path):
    """Second call to fetch_history hits the Parquet cache (source='cache')."""
    spec = OpenMeteoPullSpec(
        lat=38.5, lon=-109.6,
        start_date=dt.date(2024, 6, 1),
        end_date=dt.date(2024, 6, 2),
        timezone="auto",
    )
    cache_dir = str(tmp_path / "cache")
    fake_resp = _build_fake_response(48, start_local_date=dt.date(2024, 6, 1))

    # First call: hits the SDK.
    with patch(
        "openmeteo_requests.Client"
    ) as ClientCls, patch(
        "requests_cache.CachedSession", return_value=MagicMock()
    ), patch(
        "retry_requests.retry", side_effect=lambda s, retries, backoff_factor: s
    ):
        client = MagicMock()
        client.weather_api.return_value = [fake_resp]
        ClientCls.return_value = client
        r1 = fetch_history(spec, cache_dir=cache_dir)
    assert r1.source == "fetch"

    # Second call: no SDK patch needed — must hit cache and not call out.
    with patch("openmeteo_requests.Client") as ClientCls2:
        r2 = fetch_history(spec, cache_dir=cache_dir)
        assert ClientCls2.called is False
    assert r2.source == "cache"
    assert len(r2.df) == 48
    assert list(r2.df.columns) == list(CANONICAL_COLUMNS)


def test_fetch_history_propagates_fetch_failure(tmp_path):
    spec = OpenMeteoPullSpec(
        lat=0.0, lon=0.0,
        start_date=dt.date(2024, 6, 1),
        end_date=dt.date(2024, 6, 2),
    )
    with patch("openmeteo_requests.Client") as ClientCls, patch(
        "requests_cache.CachedSession", return_value=MagicMock()
    ), patch("retry_requests.retry", side_effect=lambda s, retries, backoff_factor: s):
        client = MagicMock()
        client.weather_api.side_effect = RuntimeError("boom")
        ClientCls.return_value = client
        with pytest.raises(OpenMeteoFetchError, match="boom"):
            fetch_history(spec, cache_dir=str(tmp_path / "cache"))


def test_fetch_history_era5_lag_warning(tmp_path, caplog):
    today = dt.date.today()
    spec = OpenMeteoPullSpec(
        lat=38.0, lon=-109.0,
        start_date=today - dt.timedelta(days=10),
        end_date=today - dt.timedelta(days=2),
    )
    fake_resp = _build_fake_response(
        24 * 9, start_local_date=spec.start_date
    )
    with patch("openmeteo_requests.Client") as ClientCls, patch(
        "requests_cache.CachedSession", return_value=MagicMock()
    ), patch("retry_requests.retry", side_effect=lambda s, retries, backoff_factor: s):
        client = MagicMock()
        client.weather_api.return_value = [fake_resp]
        ClientCls.return_value = client
        import logging
        with caplog.at_level(logging.WARNING):
            fetch_history(spec, cache_dir=str(tmp_path / "cache"))
    assert any("ERA5 archive lag" in m for m in caplog.messages)
