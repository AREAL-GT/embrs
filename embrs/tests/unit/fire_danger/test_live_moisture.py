"""Tests for embrs.fire_danger.live_moisture."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from embrs.fire_danger.config import HourlyWeather
from embrs.fire_danger.live_moisture import (
    _gsi_to_live_moisture_pct,
    build_daily_summaries,
    compute_live_moisture,
)
from embrs.utilities.data_classes import GeoInfo


def _make_weather(days: int = 5, temp_F: float = 75.0, rh_pct: float = 30.0,
                  precip_cm_hr: float = 0.0) -> HourlyWeather:
    n = days * 24
    times = pd.date_range("2025-07-01 00:00", periods=n, freq="h", tz="America/Denver")
    df = pd.DataFrame(
        {
            "temp_F": np.full(n, temp_F),
            "temp_C": np.full(n, (temp_F - 32) * 5 / 9),
            "rh_pct": np.full(n, rh_pct),
            "rh_frac": np.full(n, rh_pct / 100.0),
            "wind_mph": np.full(n, 5.0),
            "wind_dir_deg": np.full(n, 180.0),
            "precip_in_hr": np.full(n, precip_cm_hr / 2.54),
            "precip_cm_hr": np.full(n, precip_cm_hr),
            "cloud_cover": np.full(n, 20.0),
        },
        index=times,
    )
    df.index.name = "datetime"
    return HourlyWeather(df=df, ref_elev_m=1200.0, time_step_min=60,
                         raw_start=df.index[0].to_pydatetime(),
                         raw_end=df.index[-1].to_pydatetime())


def _geo() -> GeoInfo:
    return GeoInfo(center_lat=39.5, center_lon=-105.0, timezone="America/Denver")


def test_gsi_to_live_moisture_boundary_values():
    h, w = _gsi_to_live_moisture_pct(0.0)
    assert h == pytest.approx(30.0)
    assert w == pytest.approx(60.0)

    h, w = _gsi_to_live_moisture_pct(0.19)
    assert h == pytest.approx(30.0)  # still dormant just below gu
    assert w == pytest.approx(60.0)

    h, w = _gsi_to_live_moisture_pct(0.20)  # green-up edge
    assert h == pytest.approx(30.0)
    assert w == pytest.approx(60.0)

    h, w = _gsi_to_live_moisture_pct(1.0)
    assert h == pytest.approx(250.0)
    assert w == pytest.approx(200.0)

    # Halfway between gu=0.2 and 1.0 (gsi=0.6)
    h, w = _gsi_to_live_moisture_pct(0.6)
    expected_h = (30.0 + 250.0) / 2
    expected_w = (60.0 + 200.0) / 2
    assert h == pytest.approx(expected_h, rel=1e-9)
    assert w == pytest.approx(expected_w, rel=1e-9)


def test_build_daily_summaries_shape_and_units():
    w = _make_weather(days=5, temp_F=72.0, rh_pct=40.0, precip_cm_hr=0.1)
    summaries = build_daily_summaries(w)
    assert len(summaries) == 5
    s0 = summaries[0]
    # Constant inputs -> min == max
    assert s0.min_temp_F == pytest.approx(72.0)
    assert s0.max_temp_F == pytest.approx(72.0)
    assert s0.min_rh == pytest.approx(40.0)
    # 0.1 cm/h * 24 h = 2.4 cm/day
    assert s0.rain_cm == pytest.approx(2.4, rel=1e-9)


def test_compute_live_moisture_returns_per_day():
    w = _make_weather(days=10)
    out = compute_live_moisture(w, _geo()).df
    assert len(out) == 10
    assert list(out.columns) == ["GSI", "MCHERB", "MCWOOD"]
    # First day has <2 days buffered -> NaN GSI -> dormant
    assert np.isnan(out["GSI"].iloc[0])
    assert out["MCHERB"].iloc[0] == pytest.approx(30.0)
    assert out["MCWOOD"].iloc[0] == pytest.approx(60.0)


def test_compute_live_moisture_dry_hot_keeps_gsi_low():
    """Hot, dry, no rain -> precip sub-index ~ 0 -> GSI ~ 0."""
    w = _make_weather(days=30, temp_F=90.0, rh_pct=15.0, precip_cm_hr=0.0)
    out = compute_live_moisture(w, _geo()).df
    # After spin-up the GSI should be very low (no rain -> iPrcp=0 -> product=0)
    assert out["GSI"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    # Dormant moistures
    assert out["MCHERB"].iloc[-1] == pytest.approx(30.0)
    assert out["MCWOOD"].iloc[-1] == pytest.approx(60.0)
