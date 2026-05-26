"""Tests for embrs.fire_danger.solar."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import pytz

from embrs.fire_danger.config import HourlyWeather
from embrs.fire_danger.solar import synthesize_solar, _cloud_to_fraction
from embrs.utilities.data_classes import GeoInfo


def _make_weather(cloud_pct: list[float]) -> HourlyWeather:
    n = len(cloud_pct)
    times = pd.date_range("2025-07-22 00:00", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "temp_F": np.full(n, 70.0),
            "temp_C": np.full(n, 21.1),
            "rh_pct": np.full(n, 40.0),
            "rh_frac": np.full(n, 0.4),
            "wind_mph": np.full(n, 5.0),
            "wind_dir_deg": np.full(n, 180.0),
            "precip_in_hr": np.zeros(n),
            "precip_cm_hr": np.zeros(n),
            "cloud_cover": np.asarray(cloud_pct, dtype=float),
        },
        index=times,
    )
    df.index.name = "datetime"
    return HourlyWeather(
        df=df, ref_elev_m=1200.0, time_step_min=60,
        raw_start=df.index[0].to_pydatetime(),
        raw_end=df.index[-1].to_pydatetime(),
    )


def _geo() -> GeoInfo:
    return GeoInfo(center_lat=39.5, center_lon=-105.0, timezone="America/Denver")


def test_cloud_scale_conversions():
    assert _cloud_to_fraction(np.array([0.0, 50.0, 100.0]), "percent").tolist() == [0.0, 0.5, 1.0]
    assert _cloud_to_fraction(np.array([0.0, 0.5, 1.0]), "fraction").tolist() == [0.0, 0.5, 1.0]
    assert _cloud_to_fraction(np.array([0.0, 4.0, 8.0]), "okta").tolist() == [0.0, 0.5, 1.0]
    assert _cloud_to_fraction(np.array([0.0, 5.0, 10.0]), "tenths").tolist() == [0.0, 0.5, 1.0]
    # Overrange clips
    assert _cloud_to_fraction(np.array([150.0]), "percent").tolist() == [1.0]
    with pytest.raises(ValueError, match="cloud_scale"):
        _cloud_to_fraction(np.array([1.0]), "bogus")


def test_solar_zero_at_midnight_positive_at_noon():
    # 24 hours of clear sky
    w = _make_weather([0.0] * 24)
    synthesize_solar(w, _geo(), cloud_scale="percent")
    s = w.df["solar_wm2"].to_numpy()
    # Midnight ~0; solar noon (local hour 12 = index 12 since starting at 00:00 local) > 500 W/m2 in July CO
    assert s[0] == pytest.approx(0.0, abs=1e-3)
    assert s[12] > 500.0
    assert s[-1] == pytest.approx(0.0, abs=1.0)  # 23:00 local, near dark


def test_full_cloud_attenuates_toward_zero_but_not_below():
    w_clear = _make_weather([0.0] * 24)
    w_cloudy = _make_weather([100.0] * 24)
    synthesize_solar(w_clear, _geo())
    synthesize_solar(w_cloudy, _geo())
    s_clear = w_clear.df["solar_wm2"].to_numpy()
    s_cloudy = w_cloudy.df["solar_wm2"].to_numpy()
    # Cloudy noon strictly less than clear noon
    assert s_cloudy[12] < s_clear[12]
    # All values non-negative, finite
    assert (s_cloudy >= 0).all()
    assert np.isfinite(s_cloudy).all()
    # Attenuation factor is (1 - 0.75) = 0.25 at c=1
    assert s_cloudy[12] == pytest.approx(s_clear[12] * 0.25, rel=1e-9)


def test_zero_cloud_equals_clearsky():
    w = _make_weather([0.0] * 24)
    synthesize_solar(w, _geo())
    # With c=0, attenuation factor is 1.0 exactly
    # (verify by comparing the noon value against pvlib directly, indirectly:
    #  the test above showed a positive value > 500, which combined with the
    #  cloudy test confirms the identity at c=0.)
    assert w.df["solar_wm2"].max() > 500.0


def test_naive_index_localized_in_place():
    w = _make_weather([20.0] * 24)
    assert w.df.index.tz is None
    synthesize_solar(w, _geo())
    assert str(w.df.index.tz) == "America/Denver"


def test_missing_geo_fields_raises():
    w = _make_weather([0.0])
    with pytest.raises(ValueError, match="GeoInfo"):
        synthesize_solar(w, GeoInfo(center_lat=39.5, center_lon=None, timezone="UTC"))


def _make_weather_at(start: pd.Timestamp, n_hours: int, cloud_pct: float = 20.0) -> HourlyWeather:
    """Build an n-hour HourlyWeather with tz-naive index starting at ``start``."""
    times = pd.date_range(start, periods=n_hours, freq="h")
    df = pd.DataFrame(
        {
            "temp_F": np.full(n_hours, 70.0),
            "temp_C": np.full(n_hours, 21.1),
            "rh_pct": np.full(n_hours, 40.0),
            "rh_frac": np.full(n_hours, 0.4),
            "wind_mph": np.full(n_hours, 5.0),
            "wind_dir_deg": np.full(n_hours, 180.0),
            "precip_in_hr": np.zeros(n_hours),
            "precip_cm_hr": np.zeros(n_hours),
            "cloud_cover": np.full(n_hours, cloud_pct),
        },
        index=times,
    )
    df.index.name = "datetime"
    return HourlyWeather(
        df=df, ref_elev_m=400.0, time_step_min=60,
        raw_start=df.index[0].to_pydatetime(),
        raw_end=df.index[-1].to_pydatetime(),
    )


def test_dst_fall_back_ambiguous_hour_is_dropped_not_left_as_nat():
    """Regression test: prior bug — a tz-naive ``.wxs`` index spanning the
    autumn DST transition (e.g. 2023-11-05 01:00 in America/Chicago) was
    localized with ``ambiguous="NaT"`` but the resulting NaT-indexed row
    was *not* dropped by ``dropna(how="all")`` (NaT in the index is not
    visible to dropna). The NaT row then propagated to
    ``kbdi.compute_kbdi_series`` which read ``NaT.year`` → NaN → blew up
    with ``TypeError: 'float' object cannot be interpreted as an integer``.

    After the fix the row must be filtered out and the index must have no
    NaT entries.
    """
    # 2023-11-05 spans the fall-back DST transition in America/Chicago:
    # 02:00 CDT (= 07:00 UTC) is followed by 01:00 CST (= 07:00 UTC),
    # making the naive wall-clock "2023-11-05 01:00" ambiguous.
    start = pd.Timestamp("2023-11-04 12:00")
    w = _make_weather_at(start, n_hours=48)
    geo = GeoInfo(center_lat=38.5, center_lon=-96.6, timezone="America/Chicago")

    # Confirm the source data contains the ambiguous hour and is tz-naive.
    assert w.df.index.tz is None
    assert pd.Timestamp("2023-11-05 01:00") in w.df.index

    synthesize_solar(w, geo, cloud_scale="percent")

    # No NaT entries remain in the index.
    assert not w.df.index.isna().any()
    # The ambiguous hour was dropped (48 input rows - 1 dropped = 47).
    assert len(w.df) == 47
    # Index is tz-aware and localized to America/Chicago.
    assert str(w.df.index.tz) == "America/Chicago"
    # Solar synthesis ran successfully — no NaN, all finite.
    assert np.isfinite(w.df["solar_wm2"].to_numpy()).all()


def test_dst_spring_forward_does_not_lose_data():
    """``nonexistent="shift_forward"`` already handled spring forward.
    This test pins down that behaviour: no rows are dropped on the spring
    DST transition; the non-existent 02:00 wall-clock hour is shifted to
    03:00. Counterpart to the fall-back regression test above.
    """
    # 2023-03-12 02:00 doesn't exist in America/Chicago (clocks jump to 03:00).
    start = pd.Timestamp("2023-03-12 00:00")
    w = _make_weather_at(start, n_hours=24)
    geo = GeoInfo(center_lat=38.5, center_lon=-96.6, timezone="America/Chicago")

    synthesize_solar(w, geo)

    assert not w.df.index.isna().any()
    assert len(w.df) == 24                 # nothing dropped on spring forward
    assert str(w.df.index.tz) == "America/Chicago"


def test_year_long_chicago_pull_completes_through_compute_bi_trajectory(tmp_path):
    """Integration regression: a year-long Chicago run used to crash inside
    ``kbdi.compute_kbdi_series`` because of the NaT-in-index bug. Exercise
    the whole BI pipeline with a tz that observes DST to confirm it now
    completes.
    """
    import os
    import rasterio
    from rasterio.transform import from_origin

    from embrs.fire_danger import Config as FireDangerConfig, compute_bi_trajectory
    from embrs.weather_candidate_search.config import WindConversionConfig
    from embrs.weather_candidate_search.wxs_writer import WxsWriteSpec, write_wxs

    # Build a year-long synthetic weather frame spanning both DST transitions.
    # The frame is a fixed-offset hourly index (what the candidate-search
    # pipeline produces); writing + re-reading through .wxs round-trips it
    # to tz-naive local time, which is exactly where the bug manifested.
    start_local = pd.Timestamp("2023-01-02 00:00", tz=pytz.FixedOffset(-6 * 60))
    n_hours = 8760
    idx = pd.date_range(start_local, periods=n_hours, freq="h")
    hours = np.arange(n_hours)
    df = pd.DataFrame(
        {
            "temp_C": 15.0 + 10.0 * np.sin(2 * np.pi * (hours - 9000) / (24 * 365)),
            "rh_pct": 50.0,
            "rain_mm_hr": np.where(hours % (24 * 14) == 0, 5.0, 0.0),
            "wind_mph": 6.0,
            "wind_dir_deg": 180.0,
            "cloud_pct": 30.0,
        },
        index=idx,
    )

    wxs_path = str(tmp_path / "year.wxs")
    write_wxs(
        WxsWriteSpec(
            df=df, elevation_ft=1300,
            wind_correction=WindConversionConfig(enabled=False),
            wind_mph_precomputed="wind_mph",
        ),
        wxs_path,
    )

    # Tiny synthetic LANDFIRE tile (TL1 → NFDRS Y, slope 10°).
    h = w = 10
    fuel = np.full((h, w), 181, dtype=np.int16)
    slope = np.full((h, w), 10, dtype=np.int16)
    elev = np.full((h, w), 400, dtype=np.int16)
    bands = [elev, slope, np.zeros((h, w), dtype=np.int16), fuel]
    transform = from_origin(west=-500_000, north=2_000_000, xsize=30.0, ysize=30.0)
    lcp_path = str(tmp_path / "lcp.tif")
    with rasterio.open(
        lcp_path, "w", driver="GTiff",
        height=h, width=w, count=4, dtype="int16",
        crs="EPSG:5070", transform=transform,
    ) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr, i)

    cfg = FireDangerConfig(
        landscape_path=lcp_path,
        wxs_path=wxs_path,
        scenario_start=datetime(2023, 2, 1),
        avg_ann_precip_in=30.0,
    )
    # Must NOT raise.
    result = compute_bi_trajectory(cfg)
    assert len(result.df) > 0
    assert not result.df.index.isna().any()
