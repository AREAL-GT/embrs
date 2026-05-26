"""Tests for the extract_candidate_wxs helper.

The full flow: write a full-season .wxs + a metadata.json pointing to it,
call extract_candidate_wxs, then re-load the result with the BI reader.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from embrs.fire_danger.weather_loader import load_wxs
from embrs.weather_candidate_search.config import WindConversionConfig
from embrs.weather_candidate_search.extract_candidate_wxs import extract_candidate_wxs
from embrs.weather_candidate_search.wxs_writer import WxsWriteSpec, write_wxs


def _build_full_season_wxs(path: str, start_date: datetime, n_days: int):
    idx = pd.date_range(start_date, periods=n_days * 24, freq="H")
    n = len(idx)
    df = pd.DataFrame(
        {
            "temp_C": 20.0 + 5 * np.sin(np.arange(n) / 6),
            "rh_pct": 50.0,
            "rain_mm_hr": np.zeros(n),
            "wind_mph": np.full(n, 5.0),
            "wind_dir_deg": 180.0,
            "cloud_pct": 20.0,
        },
        index=idx,
    )
    write_wxs(
        WxsWriteSpec(
            df=df,
            elevation_ft=4200,
            wind_correction=WindConversionConfig(enabled=False),
            wind_mph_precomputed="wind_mph",
        ),
        path,
    )


def test_extract_candidate_wxs_roundtrip(tmp_path):
    cell_dir = tmp_path / "r1_moderate"
    cell_dir.mkdir()
    full_path = cell_dir / "full_season.wxs"
    _build_full_season_wxs(str(full_path), datetime(2024, 6, 1), n_days=10)

    candidate_dir = cell_dir / "candidates" / "01_2024-06-08T0600"
    candidate_dir.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "source": {"full_season_wxs": "../../full_season.wxs"},
        "window": {
            "start_local": "2024-06-08T06:00:00",
            "end_local": "2024-06-08T18:00:00",
        },
    }
    (candidate_dir / "metadata.json").write_text(json.dumps(metadata))

    out_path = extract_candidate_wxs(
        candidate_dir=str(candidate_dir),
        conditioning_days=5,
    )
    assert os.path.exists(out_path)

    hw = load_wxs(out_path)
    # 5-day conditioning + 12-hour window = 5*24 + 12 = 132 rows
    assert len(hw.df) == 132
    # Span starts at 2024-06-03 06:00 (start - 5 days) and ends just before 18:00 on the 8th.
    assert hw.df.index[0] == pd.Timestamp("2024-06-03 06:00")
    assert hw.df.index[-1] == pd.Timestamp("2024-06-08 17:00")
