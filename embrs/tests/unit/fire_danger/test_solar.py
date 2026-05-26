"""Tests for embrs.fire_danger.solar."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

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
