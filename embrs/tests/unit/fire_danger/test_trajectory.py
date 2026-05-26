"""Unit tests for embrs.fire_danger.trajectory.

Heavier end-to-end behavior on real data lives in
``embrs/tests/integration/test_bi_trajectory.py``.
"""
from __future__ import annotations

import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from embrs.fire_danger.config import Config
from embrs.fire_danger.trajectory import compute_bi_trajectory


def _write_tiny_lcp(tmp_path, h=10, w=10, fuel_code=181):
    fuel = np.full((h, w), fuel_code, dtype=np.int16)
    slope = np.full((h, w), 10, dtype=np.int16)
    bands = []
    for b in range(1, 5):
        if b == 2:
            bands.append(slope)
        elif b == 4:
            bands.append(fuel)
        else:
            bands.append(np.zeros((h, w), dtype=np.int16))
    transform = from_origin(west=-1_000_000, north=2_000_000, xsize=30.0, ysize=30.0)
    path = str(tmp_path / "lcp.tif")
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=4, dtype="int16", crs="EPSG:5070", transform=transform,
    ) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr, i)
    return path


def _write_short_wxs(tmp_path, n_days: int = 5) -> str:
    lines = [
        "RAWS_UNITS: English",
        "RAWS_ELEVATION: 4200",
        "RAWS: 1",
        "Year  Mth  Day   Time    Temp     RH  HrlyPcp  WindSpd WindDir CloudCov",
    ]
    base = datetime(2025, 7, 1, 0, 0)
    for h in range(n_days * 24):
        d = base + pd.Timedelta(hours=h)
        # Diurnal swing: 60°F overnight, 90°F afternoon; rh inverse
        hour = d.hour
        temp = 75 + 15 * np.cos(2 * np.pi * (hour - 15) / 24)
        rh = 50 - 30 * np.cos(2 * np.pi * (hour - 15) / 24)
        lines.append(
            f"{d.year:4d}  {d.month:<3d}  {d.day:<3d}  {hour:02d}00    {temp:5.1f}    "
            f"{rh:3.0f}    0.00      5.0     180      10"
        )
    path = str(tmp_path / "tiny.wxs")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def test_trajectory_produces_expected_columns(tmp_path):
    lcp = _write_tiny_lcp(tmp_path, fuel_code=181)   # TL1 -> Y (single model)
    wxs = _write_short_wxs(tmp_path, n_days=5)
    cfg = Config(
        landscape_path=lcp, wxs_path=wxs,
        scenario_start=datetime(2025, 7, 3, 0, 0),
        avg_ann_precip_in=20.0,
    )
    result = compute_bi_trajectory(cfg)
    df = result.df
    # Should have 5 days * 24h = 120 rows
    assert len(df) == 120
    # Single-model raster: only BI_Y, SC_Y, ERC_Y per-model columns.
    assert "BI_Y" in df.columns
    assert "BI_area_weighted" in df.columns
    assert "phase" in df.columns
    assert "MC1" in df.columns
    assert "GSI" in df.columns
    assert "KBDI" in df.columns


def test_phase_split_at_scenario_start(tmp_path):
    lcp = _write_tiny_lcp(tmp_path, fuel_code=181)
    wxs = _write_short_wxs(tmp_path, n_days=5)
    scenario_start = datetime(2025, 7, 3, 0, 0)
    cfg = Config(landscape_path=lcp, wxs_path=wxs,
                 scenario_start=scenario_start, avg_ann_precip_in=20.0)
    df = compute_bi_trajectory(cfg).df
    counts = df["phase"].value_counts()
    # 2025-07-01, 02 = conditioning (48h); 03, 04, 05 = scenario (72h)
    assert counts["conditioning"] == 48
    assert counts["scenario"] == 72


def test_area_weighted_bi_bracketed_by_per_model_min_max(tmp_path):
    """Multi-model raster (V + Y) — weighted BI in [min_BI_i, max_BI_i] per hour."""
    # 50% V (101), 50% Y (181)
    fuel = np.full((10, 10), 101, dtype=np.int16)
    fuel[5:] = 181
    slope = np.full((10, 10), 10, dtype=np.int16)
    bands = [np.zeros((10, 10), dtype=np.int16), slope,
             np.zeros((10, 10), dtype=np.int16), fuel]
    transform = from_origin(west=-1_000_000, north=2_000_000, xsize=30.0, ysize=30.0)
    lcp = str(tmp_path / "multi.tif")
    with rasterio.open(lcp, "w", driver="GTiff", height=10, width=10,
                       count=4, dtype="int16", crs="EPSG:5070",
                       transform=transform) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr, i)

    wxs = _write_short_wxs(tmp_path, n_days=5)
    cfg = Config(landscape_path=lcp, wxs_path=wxs,
                 scenario_start=datetime(2025, 7, 3, 0, 0),
                 avg_ann_precip_in=20.0)
    result = compute_bi_trajectory(cfg)
    df = result.df
    assert set(result.fuel_composition.fractions) == {"V", "Y"}
    bi_v = df["BI_V"].to_numpy()
    bi_y = df["BI_Y"].to_numpy()
    bi_w = df["BI_area_weighted"].to_numpy()
    valid = np.isfinite(bi_v) & np.isfinite(bi_y) & np.isfinite(bi_w)
    lo = np.minimum(bi_v[valid], bi_y[valid])
    hi = np.maximum(bi_v[valid], bi_y[valid])
    # Allow ~1e-9 floating slack
    assert (bi_w[valid] >= lo - 1e-9).all()
    assert (bi_w[valid] <= hi + 1e-9).all()


def test_peak_bi_is_97th_percentile_of_scenario(tmp_path):
    lcp = _write_tiny_lcp(tmp_path, fuel_code=181)
    wxs = _write_short_wxs(tmp_path, n_days=5)
    cfg = Config(landscape_path=lcp, wxs_path=wxs,
                 scenario_start=datetime(2025, 7, 3, 0, 0),
                 avg_ann_precip_in=20.0)
    result = compute_bi_trajectory(cfg)
    scenario_bi = result.df.loc[result.df["phase"] == "scenario", "BI_area_weighted"]
    expected = float(np.percentile(scenario_bi.dropna().to_numpy(), 97))
    assert result.peak_bi == pytest.approx(expected)


def test_scenario_start_outside_wxs_raises(tmp_path):
    lcp = _write_tiny_lcp(tmp_path)
    wxs = _write_short_wxs(tmp_path, n_days=5)
    cfg = Config(landscape_path=lcp, wxs_path=wxs,
                 scenario_start=datetime(2030, 1, 1),    # way after
                 avg_ann_precip_in=20.0)
    with pytest.raises(ValueError, match="outside"):
        compute_bi_trajectory(cfg)


def test_deterministic_two_runs_equal(tmp_path):
    """Same inputs must produce identical CSVs — required for the tuning loop."""
    lcp = _write_tiny_lcp(tmp_path)
    wxs = _write_short_wxs(tmp_path, n_days=4)
    cfg = Config(landscape_path=lcp, wxs_path=wxs,
                 scenario_start=datetime(2025, 7, 3, 0, 0),
                 avg_ann_precip_in=20.0)
    r1 = compute_bi_trajectory(cfg)
    r2 = compute_bi_trajectory(cfg)
    pd.testing.assert_frame_equal(r1.df, r2.df)
    assert r1.peak_bi == r2.peak_bi
