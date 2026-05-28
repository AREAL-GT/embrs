"""Tests for output artifacts (metadata.json, plot, summary.csv, report.md)."""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import pytest

from embrs.weather_candidate_search import artifacts
from embrs.weather_candidate_search.config import Config
from embrs.weather_candidate_search.lull_detection import Lull
from embrs.weather_candidate_search.ranking import RankedCandidate


def _minimal_cfg(output_dir):
    return Config(
        landscape_tif="/tmp/foo.tif",
        year=2024,
        fire_season_start_month=5,
        fire_season_end_month=10,
        scenario_length_hours=12,
        bi_target_band=(60.0, 80.0),
        output_dir=str(output_dir),
        region_tag="r1",
        volatility_class="moderate",
    )


def _stub_candidate(rank=1, peak_bi=70.0, n_lulls=2, total_lull_hours=8,
                    mean_daily_1pm_bi=65.0):
    start = pd.Timestamp("2024-07-15 06:00", tz="UTC")
    end = start + pd.Timedelta(hours=12)
    lulls = [
        Lull(start + pd.Timedelta(hours=2), start + pd.Timedelta(hours=5), 4),
        Lull(start + pd.Timedelta(hours=8), start + pd.Timedelta(hours=10), 4),
    ]
    return RankedCandidate(
        window_id=start.strftime("%Y-%m-%dT%H%M"),
        start=start, end=end,
        peak_bi=peak_bi,
        time_of_peak=start + pd.Timedelta(hours=6),
        mean_bi=50.0,
        mean_daily_1pm_bi=mean_daily_1pm_bi,
        lulls=lulls,
        n_lulls=n_lulls,
        total_lull_hours=total_lull_hours,
        score=0.5,
        bi_distance_normalized=0.5,
        rank=rank,
    )


def _stub_window_df(candidate):
    idx = pd.date_range(candidate.start, candidate.end, freq="H", inclusive="left")
    n = len(idx)
    return pd.DataFrame(
        {
            "temp_C": 20.0 + 5 * np.sin(np.arange(n) / 6),
            "rh_pct": 50.0,
            "wind_mph": np.linspace(2, 8, n),
            "wind_dir_deg": 180.0,
            "BI_area_weighted": np.linspace(60, 80, n),
        },
        index=idx,
    )


def test_write_candidate_dir_creates_files(tmp_path):
    cfg = _minimal_cfg(tmp_path)
    c = _stub_candidate()
    window_df = _stub_window_df(c)
    bi_traj = window_df[["BI_area_weighted"]]
    full_season = tmp_path / "r1_moderate" / "full_season.wxs"
    full_season.parent.mkdir(parents=True, exist_ok=True)
    full_season.write_text("RAWS_UNITS: English\nRAWS_ELEVATION: 1000\nRAWS: 1\n")

    sub = artifacts.write_candidate_dir(
        cfg=cfg, candidate=c,
        window_df=window_df, bi_trajectory=bi_traj,
        geo_lat=38.9, geo_lon=-109.6, geo_tz="America/Denver",
        full_season_wxs_path=str(full_season),
        query_timestamp_utc=dt.datetime(2026, 5, 26, tzinfo=dt.timezone.utc),
    )
    assert os.path.exists(os.path.join(sub, "metadata.json"))
    assert os.path.exists(os.path.join(sub, "plot.png"))
    metadata = json.loads(open(os.path.join(sub, "metadata.json")).read())
    assert metadata["schema_version"] == 1
    assert metadata["source"]["region_tag"] == "r1"
    assert metadata["source"]["volatility_class"] == "moderate"
    assert metadata["window"]["window_id"] == c.window_id
    assert metadata["bi"]["peak_bi"] == 70.0
    assert metadata["bi"]["peak_percentile"] == 97
    assert metadata["bi"]["mean_daily_1pm_bi"] == 65.0
    assert len(metadata["lulls"]) == 2
    assert "Peak BI 70.0" in metadata["synoptic_summary"]
    assert "mean daily 1 PM BI 65.0" in metadata["synoptic_summary"]


def test_write_summary_csv(tmp_path):
    cfg = _minimal_cfg(tmp_path)
    artifacts.ensure_cell_dir(cfg)
    per_window = pd.DataFrame(
        {
            "start": [pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-02")],
            "end": [pd.Timestamp("2024-07-01 12:00"), pd.Timestamp("2024-07-02 12:00")],
            "peak_bi": [70.0, 40.0],
            "time_of_peak": [pd.Timestamp("2024-07-01 06:00"), pd.Timestamp("2024-07-02 06:00")],
            "mean_bi": [55.0, 30.0],
            "mean_daily_1pm_bi": [65.0, 35.0],
            "n_daily_1pm_samples": [1, 1],
            "n_hours": [12, 12],
            "n_valid_hours": [12, 12],
        },
        index=pd.Index(["w_in", "w_out"], name="window_id"),
    )
    scored_passing = pd.DataFrame(
        {"score": [0.4], "bi_distance_normalized": [0.5]},
        index=pd.Index(["w_in"]),
    )
    selected = {
        "w_in": RankedCandidate(
            window_id="w_in",
            start=per_window.loc["w_in", "start"],
            end=per_window.loc["w_in", "end"],
            peak_bi=70.0, time_of_peak=per_window.loc["w_in", "time_of_peak"],
            mean_bi=55.0, mean_daily_1pm_bi=65.0,
            lulls=[], n_lulls=0, total_lull_hours=0,
            score=0.4, bi_distance_normalized=0.5, rank=1,
        )
    }
    out_path = artifacts.write_summary_csv(
        cfg, per_window, scored_passing, selected, {}
    )
    df = pd.read_csv(out_path)
    assert set(df["window_id"]) == {"w_in", "w_out"}
    in_row = df[df["window_id"] == "w_in"].iloc[0]
    assert bool(in_row["passed_band_filter"]) is True
    assert bool(in_row["selected"]) is True
    assert int(in_row["rank"]) == 1
    assert in_row["mean_daily_1pm_bi"] == 65.0
    out_row = df[df["window_id"] == "w_out"].iloc[0]
    assert bool(out_row["passed_band_filter"]) is False
    assert int(out_row["rank"]) == -1
    assert out_row["mean_daily_1pm_bi"] == 35.0


def test_write_report_md_no_candidates(tmp_path):
    cfg = _minimal_cfg(tmp_path)
    artifacts.ensure_cell_dir(cfg)
    out_path = artifacts.write_report_md(
        cfg=cfg,
        pull_stats={"n_hours": 100, "nan_hour_count": 0, "source": "fetch"},
        n_windows_enumerated=50,
        n_windows_skipped=2,
        n_passing=0,
        selected=[],
    )
    text = open(out_path).read()
    assert "Windows passing BI band filter: 0" in text
    assert "No candidates passed" in text


def test_write_report_md_with_candidates(tmp_path):
    cfg = _minimal_cfg(tmp_path)
    artifacts.ensure_cell_dir(cfg)
    c = _stub_candidate()
    out_path = artifacts.write_report_md(
        cfg=cfg,
        pull_stats={"n_hours": 100, "nan_hour_count": 0, "source": "fetch"},
        n_windows_enumerated=50,
        n_windows_skipped=2,
        n_passing=5,
        selected=[c],
    )
    text = open(out_path).read()
    assert "| 1 |" in text
    assert "70.0" in text
