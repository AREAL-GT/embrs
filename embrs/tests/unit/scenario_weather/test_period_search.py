"""Tests for the temp/RH backdrop period search (spec §4.1.1)."""
import numpy as np
import pytest

from embrs.scenario_weather.config import SearchConfig
from embrs.scenario_weather.period_search import (
    find_windows,
    saturation_vapor_pressure_kpa,
    vapor_pressure_deficit_kpa,
)
from embrs.tests.unit.scenario_weather._synth import write_season_wxs


def test_vpd_basics():
    # Hotter air holds more moisture -> higher es; drier air -> higher VPD.
    es = saturation_vapor_pressure_kpa(np.array([10.0, 30.0]))
    assert es[1] > es[0]
    vpd = vapor_pressure_deficit_kpa(np.array([30.0, 30.0]), np.array([100.0, 20.0]))
    assert vpd[0] == pytest.approx(0.0, abs=1e-9)
    assert vpd[1] > 0.0


def test_severity_ordering_extreme_hotter_than_mild(tmp_path):
    # A season that ramps hotter/drier each day -> later windows more severe.
    wxs = write_season_wxs(str(tmp_path / "season.wxs"), "2022-07-01", 60,
                           daily_temp_trend_c=0.4)
    res = find_windows(wxs, SearchConfig(window_days=14, stride_days=1))
    mild = res["mild"][0]
    extreme = res["extreme"][0]
    assert extreme.severity_score > mild.severity_score
    assert extreme.mean_daily_max_temp_F > mild.mean_daily_max_temp_F
    # extreme should sit later in the (warming) season than mild.
    assert extreme.start > mild.start


def test_wet_guard_rejects_soaking_windows(tmp_path):
    # Put a big rain event on day 5; windows covering it should be rejected.
    wxs = write_season_wxs(str(tmp_path / "wet.wxs"), "2022-07-01", 40,
                           daily_temp_trend_c=0.2,
                           precip_in_per_day={5: 3.0})
    cfg = SearchConfig(window_days=14, max_total_precip_in=1.0)
    res = find_windows(wxs, cfg)
    for windows in res.values():
        for w in windows:
            # No returned window may straddle the soaking day (2022-07-06).
            assert not (w.start <= "2022-07-06" <= w.end)


def test_season_guard(tmp_path):
    wxs = write_season_wxs(str(tmp_path / "season.wxs"), "2022-07-01", 40)
    # Restrict to August only -> a July season has no valid windows.
    cfg = SearchConfig(window_days=14, fire_season_months=(8,))
    with pytest.raises(ValueError):
        find_windows(wxs, cfg)


def test_dst_guard_rejects_spanning_windows(tmp_path):
    # US spring-forward 2022 is 2022-03-13 in America/Chicago.
    wxs = write_season_wxs(str(tmp_path / "march.wxs"), "2022-03-01", 30)
    cfg = SearchConfig(window_days=14, local_tz="America/Chicago")
    res = find_windows(wxs, cfg)
    for windows in res.values():
        for w in windows:
            assert not (w.start <= "2022-03-13" <= w.end)


def test_too_short_season_raises(tmp_path):
    wxs = write_season_wxs(str(tmp_path / "short.wxs"), "2022-07-01", 10)
    with pytest.raises(ValueError):
        find_windows(wxs, SearchConfig(window_days=14))
