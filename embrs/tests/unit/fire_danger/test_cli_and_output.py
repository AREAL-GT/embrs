"""Tests for embrs.fire_danger.cli and embrs.fire_danger.output."""
from __future__ import annotations

import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from embrs.fire_danger.cli import _build_parser, _config_from_args, main
from embrs.fire_danger.config import Config, FuelComposition, TrajectoryResult
from embrs.fire_danger.output import plot_trajectory, write_csv


# ---------------------------------------------------------------------------
# CLI / config round-trip
# ---------------------------------------------------------------------------


def test_argparse_basic_required_fields():
    args = _build_parser().parse_args([
        "--landscape", "x.tif",
        "--wxs", "y.wxs",
        "--scenario-start", "2025-07-22T06:00",
    ])
    cfg = _config_from_args(args)
    assert cfg.landscape_path == "x.tif"
    assert cfg.wxs_path == "y.wxs"
    assert cfg.scenario_start == datetime(2025, 7, 22, 6, 0)
    assert cfg.cloud_scale == "percent"
    assert cfg.snow_mode == "none"


def test_argparse_missing_required_errors():
    args = _build_parser().parse_args([])
    with pytest.raises(SystemExit, match="missing required"):
        _config_from_args(args)


def test_config_ini_round_trip(tmp_path):
    ini = tmp_path / "run.cfg"
    ini.write_text(textwrap.dedent("""\
        [fire_danger]
        landscape_path = lcp.tif
        wxs_path = w.wxs
        scenario_start = 2025-08-01T12:00
        avg_ann_precip_in = 22.5
        slope_class = 3
        out_csv = out.csv
        out_plot = plot.png
    """))
    args = _build_parser().parse_args(["--config", str(ini)])
    cfg = _config_from_args(args)
    assert cfg.landscape_path == "lcp.tif"
    assert cfg.wxs_path == "w.wxs"
    assert cfg.scenario_start == datetime(2025, 8, 1, 12, 0)
    assert cfg.avg_ann_precip_in == 22.5
    assert cfg.slope_class == 3
    assert cfg.out_csv == "out.csv"
    assert cfg.out_plot == "plot.png"


def test_cli_flag_overrides_ini(tmp_path):
    ini = tmp_path / "run.cfg"
    ini.write_text(textwrap.dedent("""\
        [fire_danger]
        landscape_path = ini.tif
        wxs_path = ini.wxs
        scenario_start = 2025-08-01
    """))
    args = _build_parser().parse_args([
        "--config", str(ini),
        "--landscape", "cli.tif",
    ])
    cfg = _config_from_args(args)
    # CLI overrides; INI provides others
    assert cfg.landscape_path == "cli.tif"
    assert cfg.wxs_path == "ini.wxs"


# ---------------------------------------------------------------------------
# CSV + plot writers, end-to-end via main()
# ---------------------------------------------------------------------------


def _write_tiny_lcp(tmp_path):
    fuel = np.full((10, 10), 181, dtype=np.int16)   # TL1 -> Y
    slope = np.full((10, 10), 10, dtype=np.int16)
    bands = [np.zeros((10, 10), dtype=np.int16), slope,
             np.zeros((10, 10), dtype=np.int16), fuel]
    transform = from_origin(west=-1_000_000, north=2_000_000, xsize=30.0, ysize=30.0)
    path = str(tmp_path / "lcp.tif")
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10,
                       count=4, dtype="int16", crs="EPSG:5070",
                       transform=transform) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr, i)
    return path


def _write_short_wxs(tmp_path, n_days=4):
    lines = [
        "RAWS_UNITS: English",
        "RAWS_ELEVATION: 4200",
        "RAWS: 1",
        "Year  Mth  Day   Time    Temp     RH  HrlyPcp  WindSpd WindDir CloudCov",
    ]
    base = datetime(2025, 7, 1, 0, 0)
    for h in range(n_days * 24):
        d = base + pd.Timedelta(hours=h)
        temp = 75 + 15 * np.cos(2 * np.pi * (d.hour - 15) / 24)
        rh = 50 - 30 * np.cos(2 * np.pi * (d.hour - 15) / 24)
        lines.append(
            f"{d.year:4d}  {d.month:<3d}  {d.day:<3d}  {d.hour:02d}00    {temp:5.1f}    "
            f"{rh:3.0f}    0.00      5.0     180      10"
        )
    path = str(tmp_path / "tiny.wxs")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def test_main_end_to_end_writes_csv_and_plot(tmp_path):
    lcp = _write_tiny_lcp(tmp_path)
    wxs = _write_short_wxs(tmp_path, n_days=4)
    out_csv = str(tmp_path / "bi.csv")
    out_plot = str(tmp_path / "bi.png")
    rc = main([
        "--landscape", lcp,
        "--wxs", wxs,
        "--scenario-start", "2025-07-03T00:00",
        "--avg-ann-precip", "20.0",
        "--out-csv", out_csv,
        "--out-plot", out_plot,
    ])
    assert rc == 0
    assert os.path.exists(out_csv)
    assert os.path.exists(out_plot)
    csv_df = pd.read_csv(out_csv, parse_dates=[0])
    assert "BI_Y" in csv_df.columns
    assert "BI_area_weighted" in csv_df.columns
    assert "phase" in csv_df.columns
    assert len(csv_df) == 96  # 4 days * 24h
