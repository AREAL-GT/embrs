"""Tests for the climatology calibration tool.

Open-Meteo is mocked. A real (synthetic) LANDFIRE tile is built so the
BI pipeline can run end-to-end across two years.
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from embrs.weather_candidate_search.calibrate_bands import (
    CalibrationConfig,
    _default_breakpoints,
    _extract_daily_1pm_in_season,
    _rolling_fortnight_mean,
    calibrate_bands,
    parse_months_spec,
)
from embrs.weather_candidate_search.openmeteo_client import OpenMeteoResult


def _write_synthetic_lcp(tmp_path, fuel_code=181):
    """Same shape as the BI integration tests use."""
    h = w = 10
    fuel = np.full((h, w), fuel_code, dtype=np.int16)
    slope = np.full((h, w), 10, dtype=np.int16)
    elev = np.full((h, w), 400, dtype=np.int16)
    bands = [elev, slope, np.zeros((h, w), dtype=np.int16), fuel]
    transform = from_origin(west=-1_000_000, north=2_000_000, xsize=30.0, ysize=30.0)
    path = str(tmp_path / "lcp.tif")
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=4, dtype="int16",
        crs="EPSG:5070", transform=transform,
    ) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr, i)
    return path


def _synthetic_weather_for(spec):
    """Build a deterministic OpenMeteoResult for a given pull span.

    Diurnal swing + summer hot/dry trend so BI varies through the season.
    """
    idx = pd.date_range(
        start=pd.Timestamp(spec.start_date),
        end=pd.Timestamp(spec.end_date) + pd.Timedelta(hours=23),
        freq="h", inclusive="both",
        tz=dt.timezone(dt.timedelta(hours=-6)),
    )
    n = len(idx)
    hours = np.asarray(idx.hour)
    months = np.asarray(idx.month)
    days = np.arange(n) / 24.0
    temp_C = 22.0 + 12.0 * np.cos(2 * np.pi * (hours - 15) / 24.0)
    rh_pct = np.clip(55.0 - 30.0 * np.cos(2 * np.pi * (hours - 15) / 24.0), 5, 100)
    rain_mm_hr = np.zeros(n)
    rain_mm_hr[::24 * 14] = 5.0
    wind_mps = np.clip(2.0 + 3.0 * np.sin(2 * np.pi * (hours - 12) / 24.0), 0.1, None)
    summer_mask = (months >= 6) & (months <= 8)
    temp_C[summer_mask] += 6.0
    rh_pct[summer_mask] = np.clip(rh_pct[summer_mask] - 20.0, 5, 100)
    wind_mps[summer_mask] += 2.0
    df = pd.DataFrame(
        {
            "temp_C": temp_C,
            "rh_pct": rh_pct,
            "rain_mm_hr": rain_mm_hr,
            "wind_mps": wind_mps,
            "wind_dir_deg": 180.0 + 30.0 * np.sin(2 * np.pi * days / 7.0),
            "cloud_pct": 20.0 + 30.0 * np.sin(2 * np.pi * days / 5.0),
        },
        index=idx,
    )
    return OpenMeteoResult(
        df=df, elevation_m=400.0,
        timezone=str(idx.tz), source="fetch", nan_hour_count=0,
    )


@pytest.fixture
def mocked_fetch_history(monkeypatch):
    def fake_fetch(spec, cache_dir):
        return _synthetic_weather_for(spec)
    monkeypatch.setattr(
        "embrs.weather_candidate_search.calibrate_bands.fetch_history",
        fake_fetch,
    )


@pytest.fixture
def mocked_avg_ann_precip(monkeypatch):
    monkeypatch.setattr(
        "embrs.fire_danger.kbdi.fetch_avg_ann_precip_in",
        lambda lat, lon, year_range=(1991, 2020): 25.0,
    )


# ---------------------------------------------------------------------------
# Pure-function tests (no BI pipeline)
# ---------------------------------------------------------------------------


def test_default_breakpoints():
    bp = _default_breakpoints()
    assert set(bp) == {"mild", "moderate", "extreme"}
    assert bp["mild"] == (20.0, 30.0)
    assert bp["moderate"] == (55.0, 65.0)
    assert bp["extreme"] == (90.0, 95.0)


def test_extract_daily_1pm_in_season_contiguous():
    idx = pd.date_range("2020-01-01", periods=24 * 365, freq="h", tz="UTC")
    bi = pd.Series(np.linspace(0, 80, len(idx)), index=idx, name="BI_area_weighted")
    phase = pd.Series("scenario", index=idx)
    df = pd.DataFrame({"BI_area_weighted": bi, "phase": phase})

    out = _extract_daily_1pm_in_season(df, fire_season_months=(4, 5, 6, 7, 8, 9, 10))
    # All values should be from hour=13 and month in {4..10}
    assert (out.index.hour == 13).all()
    months = np.asarray(out.index.month)
    assert set(months) <= {4, 5, 6, 7, 8, 9, 10}
    # Roughly 7 months × ~30 days = ~214 daily samples
    assert 200 <= len(out) <= 220


def test_extract_daily_1pm_in_season_non_contiguous():
    """Appalachian-style: keep March-May and October-November only."""
    idx = pd.date_range("2020-01-01", periods=24 * 365, freq="h", tz="UTC")
    bi = pd.Series(np.linspace(0, 80, len(idx)), index=idx, name="BI_area_weighted")
    phase = pd.Series("scenario", index=idx)
    df = pd.DataFrame({"BI_area_weighted": bi, "phase": phase})

    months_kept = (3, 4, 5, 10, 11)
    out = _extract_daily_1pm_in_season(df, fire_season_months=months_kept)
    assert (out.index.hour == 13).all()
    months = set(np.asarray(out.index.month).tolist())
    assert months == {3, 4, 5, 10, 11}
    # No values from June..September
    assert not any(m in months for m in (6, 7, 8, 9))


# ---------------------------------------------------------------------------
# parse_months_spec
# ---------------------------------------------------------------------------


def test_parse_months_spec_contiguous_range():
    assert parse_months_spec("5-10") == (5, 6, 7, 8, 9, 10)


def test_parse_months_spec_non_contiguous():
    assert parse_months_spec("3-5,10-11") == (3, 4, 5, 10, 11)


def test_parse_months_spec_mixed():
    assert parse_months_spec("3,4-5,10,11") == (3, 4, 5, 10, 11)


def test_parse_months_spec_full_year():
    assert parse_months_spec("1-12") == tuple(range(1, 13))


def test_parse_months_spec_dedupes_overlap():
    # Overlapping ranges collapse without duplication.
    assert parse_months_spec("3-5,4-6") == (3, 4, 5, 6)


def test_parse_months_spec_rejects_out_of_range():
    with pytest.raises(ValueError, match="1..12"):
        parse_months_spec("3-13")
    with pytest.raises(ValueError, match="1..12"):
        parse_months_spec("0-4")


def test_parse_months_spec_rejects_bad_range():
    with pytest.raises(ValueError, match="lo > hi"):
        parse_months_spec("5-3")


def test_parse_months_spec_rejects_garbage():
    with pytest.raises(ValueError):
        parse_months_spec("foo")
    with pytest.raises(ValueError, match="empty"):
        parse_months_spec("")


def test_rolling_fortnight_mean_basic():
    # Build 30 days of daily 1pm values 0, 1, 2, ..., 29
    idx = pd.date_range("2024-04-01 13:00", periods=30, freq="D", tz="UTC")
    s = pd.Series(np.arange(30, dtype=float), index=idx)
    rolling = _rolling_fortnight_mean(s, window_days=14)
    # First valid value is the mean of days 0..13 = 6.5, on day index 13.
    assert len(rolling) == 30 - 14 + 1
    assert rolling.iloc[0] == pytest.approx(6.5)
    assert rolling.iloc[-1] == pytest.approx((30 - 14 + (30 - 1)) / 2)


def test_rolling_fortnight_mean_handles_empty():
    s = pd.Series([], dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    rolling = _rolling_fortnight_mean(s, window_days=14)
    assert rolling.empty


# ---------------------------------------------------------------------------
# End-to-end test with mocked Open-Meteo
# ---------------------------------------------------------------------------


def test_calibrate_bands_end_to_end(
    tmp_path, mocked_fetch_history, mocked_avg_ann_precip
):
    lcp = _write_synthetic_lcp(tmp_path)
    output_yaml = str(tmp_path / "bands.yaml")
    cfg = CalibrationConfig(
        landscape_tif=lcp,
        year_start=2021,
        year_end=2022,                       # 2 years for speed
        fire_season_months=(5, 6, 7, 8, 9),
        window_length_hours=24 * 7,          # 7-day window
        conditioning_days=14,
        cache_dir=str(tmp_path / "cache"),
        output_yaml=output_yaml,
        region_tag="testreg",
    )
    result = calibrate_bands(cfg)

    # Band keys
    assert set(result.band_breakpoints) == {"mild", "moderate", "extreme"}
    # Lo <= hi for each band
    for level, (lo, hi) in result.band_breakpoints.items():
        assert lo <= hi, f"{level} band has lo > hi: {lo}, {hi}"
    # Bands monotone increasing across mild → moderate → extreme
    assert result.band_breakpoints["mild"][1] <= result.band_breakpoints["moderate"][1]
    assert result.band_breakpoints["moderate"][1] <= result.band_breakpoints["extreme"][1]

    # Distribution stats sensible
    stats = result.distribution_stats
    assert stats["n_windows"] > 0
    assert stats["min"] >= 0
    assert stats["max"] >= stats["min"]

    # Metadata is complete
    md = result.metadata
    assert md["region_tag"] == "testreg"
    assert md["year_range"] == [2021, 2022]
    assert md["n_years_used"] == 2
    assert md["window_length_days"] == 7

    # YAML written
    assert os.path.exists(output_yaml)
    import yaml
    doc = yaml.safe_load(open(output_yaml))
    assert "climatology_meta" in doc
    assert "bands" in doc
    assert set(doc["bands"]) == {"mild", "moderate", "extreme"}
    for level in ("mild", "moderate", "extreme"):
        band = doc["bands"][level]
        assert "bi_target_band" in band
        assert "percentile_range" in band
        assert len(band["bi_target_band"]) == 2
        assert band["bi_target_band"][0] <= band["bi_target_band"][1]


def test_calibrate_bands_uses_trajectory_cache_on_second_run(
    tmp_path, mocked_fetch_history, mocked_avg_ann_precip, monkeypatch
):
    """Second invocation with the same cache_dir must hit the BI trajectory
    cache (no re-run of the BI pipeline)."""
    lcp = _write_synthetic_lcp(tmp_path)
    cfg = CalibrationConfig(
        landscape_tif=lcp,
        year_start=2022, year_end=2022,
        fire_season_months=(5, 6, 7, 8, 9),
        window_length_hours=24 * 7,
        conditioning_days=14,
        cache_dir=str(tmp_path / "cache"),
        output_yaml=str(tmp_path / "bands.yaml"),
        region_tag="testreg",
    )
    calibrate_bands(cfg)        # warm the cache

    # Monkeypatch run_bi to blow up — if it gets called the cache was missed.
    def boom(*args, **kwargs):
        raise AssertionError("run_bi should not be called on cached run")
    monkeypatch.setattr(
        "embrs.weather_candidate_search.calibrate_bands.run_bi", boom
    )
    cfg2 = CalibrationConfig(
        landscape_tif=lcp,
        year_start=2022, year_end=2022,
        fire_season_months=(5, 6, 7, 8, 9),
        window_length_hours=24 * 7,
        conditioning_days=14,
        cache_dir=str(tmp_path / "cache"),
        output_yaml=str(tmp_path / "bands2.yaml"),
        region_tag="testreg",
    )
    result = calibrate_bands(cfg2)
    assert os.path.exists(cfg2.output_yaml)
    assert result.distribution_stats["n_windows"] > 0


def test_calibrate_bands_rejects_invalid_config():
    with pytest.raises(ValueError, match="year_start"):
        CalibrationConfig(
            landscape_tif="/tmp/x.tif",
            year_start=2024, year_end=2020,
            fire_season_months=tuple(range(5, 11)),
        )
    with pytest.raises(ValueError, match="sorted ascending"):
        CalibrationConfig(
            landscape_tif="/tmp/x.tif",
            year_start=2020, year_end=2021,
            fire_season_months=(11, 3),
        )
    with pytest.raises(ValueError, match="duplicates"):
        CalibrationConfig(
            landscape_tif="/tmp/x.tif",
            year_start=2020, year_end=2021,
            fire_season_months=(3, 4, 4, 5),
        )
    with pytest.raises(ValueError, match="non-empty"):
        CalibrationConfig(
            landscape_tif="/tmp/x.tif",
            year_start=2020, year_end=2021,
            fire_season_months=(),
        )
    with pytest.raises(ValueError, match="outside 1..12"):
        CalibrationConfig(
            landscape_tif="/tmp/x.tif",
            year_start=2020, year_end=2021,
            fire_season_months=(0, 1, 2),
        )
    with pytest.raises(ValueError, match="window_length_hours"):
        CalibrationConfig(
            landscape_tif="/tmp/x.tif",
            year_start=2020, year_end=2021,
            fire_season_months=tuple(range(5, 11)),
            window_length_hours=25,           # not multiple of 24
        )
    with pytest.raises(ValueError, match="percentile"):
        CalibrationConfig(
            landscape_tif="/tmp/x.tif",
            year_start=2020, year_end=2021,
            fire_season_months=tuple(range(5, 11)),
            percentile_breakpoints={"mild": (105, 110)},
        )
    with pytest.raises(ValueError, match="lo > hi"):
        CalibrationConfig(
            landscape_tif="/tmp/x.tif",
            year_start=2020, year_end=2021,
            fire_season_months=tuple(range(5, 11)),
            percentile_breakpoints={"mild": (30, 20)},
        )


def test_calibrate_bands_end_to_end_non_contiguous_months(
    tmp_path, mocked_fetch_history, mocked_avg_ann_precip
):
    """Appalachian-style split season (spring + fall): BI runs over the
    full Mar..Nov span; climatology filter keeps only spring + fall days;
    no rolling fortnight straddles the quiet summer gap.
    """
    lcp = _write_synthetic_lcp(tmp_path)
    output_yaml = str(tmp_path / "bands.yaml")
    cfg = CalibrationConfig(
        landscape_tif=lcp,
        year_start=2022, year_end=2022,
        fire_season_months=(3, 4, 5, 10, 11),
        window_length_hours=24 * 7,
        conditioning_days=14,
        cache_dir=str(tmp_path / "cache"),
        output_yaml=output_yaml,
        region_tag="appalachian_test",
    )
    result = calibrate_bands(cfg)

    # Climatology contains only spring + fall months.
    months_in = set(np.asarray(result.rolling_mean_series.index.month).tolist())
    assert months_in <= {3, 4, 5, 10, 11}
    # No rolling-window mean lands on a date whose trailing window crossed
    # the quiet summer (we don't have a direct hook to verify this, but
    # an absence of June-September dates in the rolling series implies it
    # was correctly handled by the resample-then-min_periods path).
    assert not (months_in & {6, 7, 8, 9})

    # Bands still emitted; YAML records the explicit month list.
    import yaml
    doc = yaml.safe_load(open(output_yaml))
    assert doc["climatology_meta"]["fire_season_months"] == [3, 4, 5, 10, 11]


def test_calibrate_bands_skips_failed_year(
    tmp_path, mocked_avg_ann_precip, monkeypatch
):
    """If one year's pull fails, the tool warns and continues with the others."""
    lcp = _write_synthetic_lcp(tmp_path)

    call_count = {"n": 0}
    def flaky_fetch(spec, cache_dir):
        call_count["n"] += 1
        if spec.start_date.year == 2021:
            raise RuntimeError("simulated Open-Meteo failure")
        return _synthetic_weather_for(spec)
    monkeypatch.setattr(
        "embrs.weather_candidate_search.calibrate_bands.fetch_history",
        flaky_fetch,
    )

    cfg = CalibrationConfig(
        landscape_tif=lcp,
        year_start=2021, year_end=2022,
        fire_season_months=(5, 6, 7, 8, 9),
        window_length_hours=24 * 7,
        conditioning_days=14,
        cache_dir=str(tmp_path / "cache"),
        output_yaml=str(tmp_path / "bands.yaml"),
        region_tag="testreg",
    )
    result = calibrate_bands(cfg)
    # Both years attempted, only 2022 used.
    assert result.metadata["n_years_attempted"] == 2
    assert result.metadata["n_years_used"] == 1
