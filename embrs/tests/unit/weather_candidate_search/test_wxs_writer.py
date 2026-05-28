"""Tests for the .wxs writer — round-trip through the BI reader is the gold."""
from __future__ import annotations

import datetime as dt
import math
from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from embrs.fire_danger.weather_loader import load_wxs
from embrs.weather_candidate_search.config import WindConversionConfig
from embrs.weather_candidate_search.wxs_writer import (
    WxsWriteSpec,
    correct_wind_speed_10m_to_20ft,
    log_profile_factor,
    write_wxs,
)


def _make_om_df(n_hours: int = 48):
    base = pd.Timestamp("2024-07-01 00:00", tz="UTC")
    idx = pd.date_range(base, periods=n_hours, freq="H")
    hours = np.arange(n_hours)
    return pd.DataFrame(
        {
            "temp_C": 20.0 + 5.0 * np.sin(2 * np.pi * hours / 24),
            "rh_pct": 50.0 + 20.0 * np.cos(2 * np.pi * hours / 24),
            "rain_mm_hr": np.where(hours % 12 == 0, 1.0, 0.0),
            "wind_mps": 3.0 + 1.5 * np.sin(2 * np.pi * hours / 24),
            "wind_dir_deg": (180.0 + 30.0 * np.sin(2 * np.pi * hours / 24)) % 360,
            "cloud_pct": 20.0 + 50.0 * (hours % 8 == 0),
        },
        index=idx,
    )


def test_log_profile_factor_default_z0():
    factor = log_profile_factor(0.06)
    assert 0.90 < factor < 0.92


def test_log_profile_factor_requires_positive_z0():
    with pytest.raises(ValueError):
        log_profile_factor(0)


def test_correct_wind_array_matches_scalar():
    arr = np.array([0.0, 5.0, 10.0])
    factor = log_profile_factor(0.06)
    corrected = correct_wind_speed_10m_to_20ft(arr, 0.06)
    np.testing.assert_allclose(corrected, arr * factor)


def test_write_wxs_roundtrip_through_bi_reader(tmp_path):
    df = _make_om_df(48)
    out = tmp_path / "out.wxs"
    spec = WxsWriteSpec(
        df=df,
        elevation_ft=4200,
        wind_correction=WindConversionConfig(enabled=False),  # disable for clean comparison
    )
    write_wxs(spec, str(out))

    hw = load_wxs(str(out))
    assert len(hw.df) == 48
    # Temp round-trip (writer writes 1 decimal °F; reader returns °F)
    expected_F = df["temp_C"].to_numpy() * 9 / 5 + 32
    np.testing.assert_allclose(
        hw.df["temp_F"].to_numpy(), expected_F, atol=0.06
    )
    # RH integer-rounded
    np.testing.assert_allclose(
        hw.df["rh_pct"].to_numpy(),
        np.round(df["rh_pct"].to_numpy()),
        atol=0.6,
    )
    # Wind mph: with correction disabled, just m/s * 2.237 then rounded to 1 dp
    expected_mph = df["wind_mps"].to_numpy() * 2.23693629
    np.testing.assert_allclose(
        hw.df["wind_mph"].to_numpy(), expected_mph, atol=0.06
    )
    # Precip
    expected_in_hr = df["rain_mm_hr"].to_numpy() * 0.0393701
    np.testing.assert_allclose(
        hw.df["precip_in_hr"].to_numpy(), expected_in_hr, atol=0.006
    )
    # Elevation header (BI reader stores it in metres)
    assert abs(hw.ref_elev_m - 4200 * 0.3048) < 0.5


def test_write_wxs_applies_log_profile_correction_by_default(tmp_path):
    df = _make_om_df(24)
    out = tmp_path / "out.wxs"
    spec = WxsWriteSpec(
        df=df,
        elevation_ft=1000,
        wind_correction=WindConversionConfig(enabled=True, surface_roughness_m=0.06),
    )
    write_wxs(spec, str(out))

    hw = load_wxs(str(out))
    factor = log_profile_factor(0.06)
    expected = df["wind_mps"].to_numpy() * factor * 2.23693629
    np.testing.assert_allclose(hw.df["wind_mph"].to_numpy(), expected, atol=0.06)


def test_write_wxs_uses_precomputed_wind_column(tmp_path):
    df = _make_om_df(24)
    df["wind_mph"] = 7.0  # known-good post-correction
    out = tmp_path / "out.wxs"
    spec = WxsWriteSpec(
        df=df,
        elevation_ft=500,
        wind_correction=WindConversionConfig(enabled=True),  # should be ignored
        wind_mph_precomputed="wind_mph",
    )
    write_wxs(spec, str(out))
    hw = load_wxs(str(out))
    np.testing.assert_allclose(hw.df["wind_mph"].to_numpy(), 7.0, atol=0.06)


def test_write_wxs_rejects_missing_columns(tmp_path):
    df = _make_om_df(24).drop(columns=["wind_mps"])
    out = tmp_path / "out.wxs"
    spec = WxsWriteSpec(df=df, elevation_ft=0, wind_correction=WindConversionConfig())
    with pytest.raises(ValueError, match="wind_mps"):
        write_wxs(spec, str(out))


def test_write_wxs_skips_nan_rows(tmp_path):
    df = _make_om_df(24)
    df.iloc[5, df.columns.get_loc("temp_C")] = float("nan")
    out = tmp_path / "out.wxs"
    spec = WxsWriteSpec(df=df, elevation_ft=1000, wind_correction=WindConversionConfig(enabled=False))
    write_wxs(spec, str(out))
    hw = load_wxs(str(out))
    assert len(hw.df) == 23
