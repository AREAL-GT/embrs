"""Tests for the backburn-feasibility proxy (spec §6). No simulation is run."""
import numpy as np
import pandas as pd
import pytest

from embrs.scenario_weather.backburn_check import _angular_distance_deg, validate
from embrs.scenario_weather.config import BackburnProxyConfig
from embrs.weather_candidate_search.config import WindConversionConfig
from embrs.weather_candidate_search.wxs_writer import WxsWriteSpec, write_wxs

_MPS_TO_MPH = 2.23693629


def _write_wxs(path, wind_mps, wind_dir_deg):
    n = len(wind_mps)
    idx = pd.date_range("2022-07-08", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "temp_C": np.full(n, 25.0),
            "rh_pct": np.full(n, 30.0),
            "rain_mm_hr": np.zeros(n),
            "wind_mph_native": np.asarray(wind_mps) * _MPS_TO_MPH,
            "wind_dir_deg": np.asarray(wind_dir_deg, dtype=float),
            "cloud_pct": np.zeros(n),
        },
        index=idx,
    )
    write_wxs(
        WxsWriteSpec(df=df, elevation_ft=1000,
                     wind_correction=WindConversionConfig(enabled=False),
                     wind_mph_precomputed="wind_mph_native"),
        path,
    )
    return path


def test_angular_distance_wraps():
    d = _angular_distance_deg(np.array([350.0, 10.0, 180.0]), 0.0)
    assert d[0] == 10.0
    assert d[1] == 10.0
    assert d[2] == 180.0


def test_suitable_requires_low_wind_and_alignment(tmp_path):
    # 6 aligned-low, 6 aligned-high, 6 low-misaligned, 6 low-aligned.
    wind = [3] * 6 + [15] * 6 + [3] * 6 + [3] * 6
    bear = [180] * 6 + [180] * 6 + [0] * 6 + [180] * 6
    wxs = _write_wxs(str(tmp_path / "bb.wxs"), wind, bear)
    rep = validate(wxs, BackburnProxyConfig(fireline_bearing_deg=180.0,
                                            hi_wind_speed_thresh_m_s=10.0,
                                            wind_angle_tol_deg=45.0,
                                            min_window_hours=2.0))
    # High-wind hours and misaligned hours are excluded.
    assert rep.suitable_hours == 12          # first 6 + last 6
    assert rep.wind_under_thresh_fraction == pytest.approx(18 / 24)
    # Two distinct windows (separated by the high-wind + misaligned blocks).
    assert rep.n_windows == 2
    assert not rep.flagged


def test_flag_when_no_usable_window(tmp_path):
    # Wind always above threshold -> no lulls at all.
    wxs = _write_wxs(str(tmp_path / "windy.wxs"), [20] * 24, [180] * 24)
    rep = validate(wxs, BackburnProxyConfig(fireline_bearing_deg=180.0))
    assert rep.suitable_hours == 0
    assert rep.flagged
    assert any("never drops below" in n for n in rep.notes)


def test_direction_coverage(tmp_path):
    # All low wind, half aligned half not.
    wxs = _write_wxs(str(tmp_path / "dir.wxs"), [3] * 24, [180] * 12 + [0] * 12)
    rep = validate(wxs, BackburnProxyConfig(fireline_bearing_deg=180.0))
    assert rep.wind_under_thresh_fraction == 1.0
    assert rep.directional_coverage == pytest.approx(0.5)
