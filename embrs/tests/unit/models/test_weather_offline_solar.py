"""Tests for offline solar irradiance synthesis in WeatherStream.

Validates that ``WeatherStream._synthesize_irradiance`` produces a physically
sensible GHI/DNI/DHI triple from cloud cover alone, with no network access —
the path used when ``solar_source='offline'`` (e.g. on PACE compute nodes).
"""

import numpy as np
import pandas as pd
import pvlib
import pytest

from embrs.models.weather import WeatherStream
from embrs.utilities.data_classes import GeoInfo


def _make_stream(timezone="America/Denver", lat=39.74, lon=-104.99, elev=1600.0):
    """Build a bare WeatherStream with just the attrs the helper needs."""
    ws = WeatherStream.__new__(WeatherStream)
    ws.geo = GeoInfo(center_lat=lat, center_lon=lon, timezone=timezone)
    ws.ref_elev = elev
    return ws


def _frame(timezone, cloud_percent, day="2025-07-15"):
    """One clear summer day at hourly resolution with constant cloud cover."""
    idx = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="1h",
                        tz=timezone)
    return pd.DataFrame({"cloud_cover": np.full(len(idx), cloud_percent)}, index=idx)


def _solpos(df, ws):
    return pvlib.solarposition.get_solarposition(
        df.index, ws.geo.center_lat, ws.geo.center_lon)


class TestSynthesizeIrradiance:
    def test_no_network_call(self):
        """Synthesis must not touch openmeteo_requests."""
        import embrs.models.weather as weather_mod

        ws = _make_stream()
        df = _frame(ws.geo.timezone, 0.0)

        def _boom(*a, **k):  # pragma: no cover - should never run
            raise AssertionError("offline path made a network client")

        original = weather_mod.openmeteo_requests.Client
        weather_mod.openmeteo_requests.Client = _boom
        try:
            ghi, dni, dhi = ws._synthesize_irradiance(df, _solpos(df, ws))
        finally:
            weather_mod.openmeteo_requests.Client = original

        assert len(ghi) == len(df)

    def test_outputs_finite_and_nonnegative(self):
        ws = _make_stream()
        df = _frame(ws.geo.timezone, 30.0)
        ghi, dni, dhi = ws._synthesize_irradiance(df, _solpos(df, ws))

        for arr in (ghi, dni, dhi):
            assert np.all(np.isfinite(arr))
            assert np.all(arr >= 0.0)

    def test_daytime_positive_night_zero(self):
        ws = _make_stream()
        df = _frame(ws.geo.timezone, 0.0)
        ghi, _, _ = ws._synthesize_irradiance(df, _solpos(df, ws))

        ghi = pd.Series(ghi, index=df.index)
        # Local midnight: no sun. Local noon (summer, Denver): strong sun.
        assert ghi.iloc[0] == pytest.approx(0.0, abs=1.0)
        assert ghi.between_time("12:00", "13:00").max() > 500.0

    def test_clouds_reduce_irradiance(self):
        ws = _make_stream()
        clear_df = _frame(ws.geo.timezone, 0.0)
        cloudy_df = _frame(ws.geo.timezone, 90.0)

        clear_ghi, _, _ = ws._synthesize_irradiance(clear_df, _solpos(clear_df, ws))
        cloudy_ghi, _, _ = ws._synthesize_irradiance(cloudy_df, _solpos(cloudy_df, ws))

        # Total daily GHI must drop substantially under heavy cloud.
        assert cloudy_ghi.sum() < clear_ghi.sum()
        assert cloudy_ghi.sum() < 0.6 * clear_ghi.sum()
