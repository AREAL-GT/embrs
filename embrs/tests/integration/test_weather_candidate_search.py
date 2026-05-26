"""End-to-end integration test for weather_candidate_search.

Open-Meteo is mocked but everything else (LANDFIRE tile, BI pipeline,
windowing, ranking, artifact writing) runs for real.

Plan §5.2.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from embrs.weather_candidate_search.config import (
    Config,
    LullConfig,
    ScoringConfig,
    WindConversionConfig,
)
from embrs.weather_candidate_search.openmeteo_client import (
    CANONICAL_COLUMNS,
    OpenMeteoResult,
)
from embrs.weather_candidate_search.pipeline import run_candidate_search


_REAL_LCP = os.path.expanduser(
    "~/Documents/Research/embrs_map/scenario_4/cropped_lcp.tif"
)


def _synthetic_weather_df(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Build a synthetic deterministic weather frame for the full pull span.

    Includes a known calm-RH lull (planted) inside July (hour 12345 ≈
    mid-July) and a BI-prone "hot dry windy" stretch later in August so
    that the band [60, 80] picks at least one window in moderate-shrub
    terrain.
    """
    idx = pd.date_range(
        start=pd.Timestamp(start),
        end=pd.Timestamp(end) + pd.Timedelta(hours=23),
        freq="h",
        inclusive="both",
        tz=dt.timezone(dt.timedelta(hours=-6)),
    )
    n = len(idx)
    days = np.arange(n) / 24.0
    hour_of_day = np.asarray(idx.hour)

    # Diurnal swing: hot afternoons, cool nights.
    temp_C = 22.0 + 12.0 * np.cos(2 * np.pi * (hour_of_day - 15) / 24.0)
    rh_pct = np.clip(55.0 - 30.0 * np.cos(2 * np.pi * (hour_of_day - 15) / 24.0), 5, 100)
    rain_mm_hr = np.zeros(n)
    rain_mm_hr[::24 * 14] = 5.0   # one rain event every two weeks
    wind_mps = np.clip(
        2.0 + 3.0 * np.sin(2 * np.pi * (hour_of_day - 12) / 24.0), 0.1, None
    )
    # Seasonal trend — hotter, drier, windier in mid-summer.
    months = np.asarray(idx.month)
    summer_mask = (months >= 7) & (months <= 8)
    temp_C[summer_mask] += 6.0
    rh_pct[summer_mask] = np.clip(rh_pct[summer_mask] - 20.0, 5, 100)
    wind_mps[summer_mask] += 2.0
    wind_dir_deg = 180.0 + 30.0 * np.sin(2 * np.pi * days / 7.0)
    cloud_pct = 20.0 + 30.0 * np.sin(2 * np.pi * days / 5.0)

    # Planted lull near Aug 15 06:00 — calm + humid for 6 hours.
    lull_start = pd.Timestamp(f"{start.year}-08-15 06:00").tz_localize(idx.tz)
    if lull_start in idx:
        i0 = idx.get_loc(lull_start)
        for k in range(6):
            wind_mps[i0 + k] = 1.0
            rh_pct[i0 + k] = 60.0

    return pd.DataFrame(
        {
            "temp_C": np.asarray(temp_C),
            "rh_pct": np.asarray(rh_pct),
            "rain_mm_hr": np.asarray(rain_mm_hr),
            "wind_mps": np.asarray(wind_mps),
            "wind_dir_deg": np.asarray(wind_dir_deg),
            "cloud_pct": np.asarray(cloud_pct),
        },
        index=idx,
    )


@pytest.fixture
def mocked_fetch_history(monkeypatch):
    """Patch fetch_history to return synthetic data for the requested span."""
    def fake_fetch(spec, cache_dir):
        df = _synthetic_weather_df(spec.start_date, spec.end_date)
        return OpenMeteoResult(
            df=df,
            elevation_m=1500.0,
            timezone=str(df.index.tz),
            source="fetch",
            nan_hour_count=0,
        )
    monkeypatch.setattr(
        "embrs.weather_candidate_search.pipeline.fetch_history",
        fake_fetch,
    )
    return fake_fetch


@pytest.fixture
def mocked_avg_ann_precip(monkeypatch):
    """Avoid hitting Open-Meteo for AvgAnnPrecip during the BI run."""
    monkeypatch.setattr(
        "embrs.fire_danger.kbdi.fetch_avg_ann_precip_in",
        lambda lat, lon, year_range=(1991, 2020): 18.0,
    )


@pytest.mark.skipif(
    not os.path.exists(_REAL_LCP),
    reason="LANDFIRE fixture not present; skipping integration test",
)
def test_end_to_end_writes_expected_artifacts(
    tmp_path, mocked_fetch_history, mocked_avg_ann_precip, caplog
):
    cfg = Config(
        landscape_tif=_REAL_LCP,
        year=2024,
        fire_season_start_month=6,        # short fire season for speed
        fire_season_end_month=8,
        scenario_length_hours=24,
        bi_target_band=(0.0, 1000.0),     # broad — guarantees ≥ 1 candidate
        output_dir=str(tmp_path / "out"),
        region_tag="test_region",
        volatility_class="any",
        n_candidates=3,
        conditioning_days=14,             # smaller buffer for test speed
        cache_dir=str(tmp_path / "cache"),
        # avoid the explicit-precip override path being needed
        bi=__import__(
            "embrs.weather_candidate_search.config", fromlist=["BISection"]
        ).BISection(avg_ann_precip_in=18.0),
    )
    with caplog.at_level(logging.INFO):
        rc = run_candidate_search(cfg)
    assert rc == 0

    cell = os.path.join(cfg.output_dir, "test_region_any")
    assert os.path.exists(os.path.join(cell, "full_season.wxs"))
    assert os.path.exists(os.path.join(cell, "summary.csv"))
    assert os.path.exists(os.path.join(cell, "report.md"))
    candidates = sorted(os.listdir(os.path.join(cell, "candidates")))
    assert len(candidates) == 3
    for sub in candidates:
        d = os.path.join(cell, "candidates", sub)
        assert os.path.exists(os.path.join(d, "metadata.json"))
        assert os.path.exists(os.path.join(d, "plot.png"))
        meta = json.loads(open(os.path.join(d, "metadata.json")).read())
        assert meta["schema_version"] == 1
        assert meta["source"]["region_tag"] == "test_region"

    # summary.csv has one row per evaluated window (≥ n_candidates).
    summary = pd.read_csv(os.path.join(cell, "summary.csv"))
    assert len(summary) >= 3
    assert summary["selected"].sum() == 3


@pytest.mark.skipif(
    not os.path.exists(_REAL_LCP),
    reason="LANDFIRE fixture not present; skipping integration test",
)
def test_empty_candidate_set_returns_exit_code_2(
    tmp_path, mocked_fetch_history, mocked_avg_ann_precip
):
    cfg = Config(
        landscape_tif=_REAL_LCP,
        year=2024,
        fire_season_start_month=6,
        fire_season_end_month=7,
        scenario_length_hours=24,
        bi_target_band=(99990.0, 99999.0),  # unreachable
        output_dir=str(tmp_path / "out"),
        region_tag="test_region",
        volatility_class="impossible",
        n_candidates=3,
        conditioning_days=14,
        cache_dir=str(tmp_path / "cache"),
        bi=__import__(
            "embrs.weather_candidate_search.config", fromlist=["BISection"]
        ).BISection(avg_ann_precip_in=18.0),
    )
    rc = run_candidate_search(cfg)
    assert rc == 2
    cell = os.path.join(cfg.output_dir, "test_region_impossible")
    assert os.path.exists(os.path.join(cell, "summary.csv"))
    assert os.path.exists(os.path.join(cell, "report.md"))
    # No candidates dir or empty
    cand_dir = os.path.join(cell, "candidates")
    if os.path.exists(cand_dir):
        assert os.listdir(cand_dir) == []


@pytest.mark.skipif(
    not os.path.exists(_REAL_LCP),
    reason="LANDFIRE fixture not present; skipping integration test",
)
def test_determinism_byte_identical_summary(
    tmp_path, mocked_fetch_history, mocked_avg_ann_precip
):
    def _build_cfg(out_root):
        return Config(
            landscape_tif=_REAL_LCP,
            year=2024,
            fire_season_start_month=6,
            fire_season_end_month=7,
            scenario_length_hours=24,
            bi_target_band=(0.0, 1000.0),
            output_dir=str(out_root),
            region_tag="det",
            volatility_class="x",
            n_candidates=2,
            conditioning_days=14,
            cache_dir=str(tmp_path / "cache"),
            bi=__import__(
                "embrs.weather_candidate_search.config", fromlist=["BISection"]
            ).BISection(avg_ann_precip_in=18.0),
        )

    rc1 = run_candidate_search(_build_cfg(tmp_path / "run1"))
    rc2 = run_candidate_search(_build_cfg(tmp_path / "run2"))
    assert rc1 == 0 and rc2 == 0
    s1 = open(tmp_path / "run1" / "det_x" / "summary.csv", "rb").read()
    s2 = open(tmp_path / "run2" / "det_x" / "summary.csv", "rb").read()
    assert s1 == s2, "summary.csv was not byte-identical across two runs"
