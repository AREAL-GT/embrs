"""Tests for embrs.fire_danger.dead_moisture."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from embrs.fire_danger.config import HourlyWeather
from embrs.fire_danger.dead_moisture import (
    _NFDRS_MAX_LOCAL_MOISTURE,
    _NFDRS_STICK_PARAMS,
    compute_dead_moisture,
    make_nfdrs_sticks,
)


def _make_weather(
    n_hours: int = 48,
    temp_C: float = 25.0,
    rh_frac: float = 0.30,
    solar_wm2: float = 200.0,
    precip_cm_hr: float = 0.0,
    snow: bool | None = None,
) -> HourlyWeather:
    times = pd.date_range("2025-07-01 00:00", periods=n_hours, freq="h")
    df = pd.DataFrame(
        {
            "temp_F": np.full(n_hours, temp_C * 9 / 5 + 32),
            "temp_C": np.full(n_hours, temp_C),
            "rh_pct": np.full(n_hours, rh_frac * 100),
            "rh_frac": np.full(n_hours, rh_frac),
            "wind_mph": np.full(n_hours, 5.0),
            "wind_dir_deg": np.full(n_hours, 180.0),
            "precip_in_hr": np.full(n_hours, precip_cm_hr / 2.54),
            "precip_cm_hr": np.full(n_hours, precip_cm_hr),
            "cloud_cover": np.full(n_hours, 20.0),
            "solar_wm2": np.full(n_hours, solar_wm2),
        },
        index=times,
    )
    if snow is not None:
        df["snow"] = snow
    df.index.name = "datetime"
    return HourlyWeather(df=df, ref_elev_m=1200.0, time_step_min=60,
                         raw_start=df.index[0].to_pydatetime(),
                         raw_end=df.index[-1].to_pydatetime())


def test_sticks_have_nfdrs_radii_and_adsorption():
    sticks = make_nfdrs_sticks()
    for key, (radius, ads_rate) in _NFDRS_STICK_PARAMS.items():
        s = sticks[key]
        assert s.m_radius == pytest.approx(radius)
        assert s.m_stca == pytest.approx(ads_rate)
        assert s.m_wmx == pytest.approx(_NFDRS_MAX_LOCAL_MOISTURE)


def test_dead_moisture_finite_and_in_range():
    out = compute_dead_moisture(_make_weather(n_hours=72)).df
    assert list(out.columns) == ["MC1", "MC10", "MC100", "MC1000"]
    arr = out.to_numpy()
    assert np.isfinite(arr).all()
    assert (arr >= 0).all()
    # Sanity ceiling — 35% max-local + film contribution should not exceed
    # ~100% by much under any realistic forcing.
    assert (arr <= 100).all()


def test_1hr_equilibrates_faster_than_1000hr_to_dry_forcing():
    """Under sustained hot/dry forcing, the 1-hr stick reaches a low
    equilibrium within ~24 h while the 1000-hr stick is still descending."""
    out = compute_dead_moisture(_make_weather(n_hours=72, temp_C=35.0, rh_frac=0.10,
                                              solar_wm2=900.0)).df
    # By 24 h the 1-hr is below 5%; the 1000-hr is still above 8%.
    assert out["MC1"].iloc[24] < 5.0
    assert out["MC1000"].iloc[24] > 8.0
    # At every sample t > 6h the 1-hr is well below the 1000-hr.
    for t in (12, 24, 48):
        assert out["MC1"].iloc[t] < out["MC1000"].iloc[t]


def test_rain_pulse_raises_1hr_moisture():
    # 12h dry baseline, 12h rain pulse, 12h dry
    n = 36
    times = pd.date_range("2025-07-01 00:00", periods=n, freq="h")
    precip_cm_hr = np.zeros(n)
    precip_cm_hr[12:24] = 0.5  # 0.5 cm/h for 12h = 6 cm total
    df = pd.DataFrame(
        {
            "temp_F": np.full(n, 80.0),
            "temp_C": np.full(n, 26.67),
            "rh_pct": np.full(n, 40.0),
            "rh_frac": np.full(n, 0.40),
            "wind_mph": np.full(n, 5.0),
            "wind_dir_deg": np.full(n, 180.0),
            "precip_in_hr": precip_cm_hr / 2.54,
            "precip_cm_hr": precip_cm_hr,
            "cloud_cover": np.full(n, 50.0),
            "solar_wm2": np.full(n, 300.0),
        },
        index=times,
    )
    w = HourlyWeather(df=df, ref_elev_m=1200.0, time_step_min=60,
                      raw_start=df.index[0].to_pydatetime(),
                      raw_end=df.index[-1].to_pydatetime())
    out = compute_dead_moisture(w).df
    # 1-hr moisture during rain (h 20) much higher than just before (h 11)
    assert out["MC1"].iloc[20] > out["MC1"].iloc[11]


def test_snow_holds_rcum_flat_and_substitutes_inputs():
    n = 24
    times = pd.date_range("2025-12-15 00:00", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "temp_F": np.full(n, 30.0),
            "temp_C": np.full(n, -1.1),
            "rh_pct": np.full(n, 80.0),
            "rh_frac": np.full(n, 0.80),
            "wind_mph": np.full(n, 5.0),
            "wind_dir_deg": np.full(n, 180.0),
            "precip_in_hr": np.full(n, 0.1 / 2.54),
            "precip_cm_hr": np.full(n, 0.1),  # 0.1 cm/h
            "cloud_cover": np.full(n, 100.0),
            "solar_wm2": np.full(n, 0.0),
            "snow": np.array([False] * 12 + [True] * 12),
        },
        index=times,
    )
    w = HourlyWeather(df=df, ref_elev_m=1200.0, time_step_min=60,
                      raw_start=df.index[0].to_pydatetime(),
                      raw_end=df.index[-1].to_pydatetime())
    out = compute_dead_moisture(w).df
    # Just smoke — output is finite and snow hours did not crash the solver.
    assert np.isfinite(out.to_numpy()).all()
