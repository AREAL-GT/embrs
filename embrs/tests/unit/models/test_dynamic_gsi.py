"""Tests for dynamic GSI updates: update_curing() and GSITracker.

Covers the new functionality for recomputing live fuel moisture during
simulation via daily GSI recomputation and in-place fuel curing updates.
"""

import math
from datetime import datetime, date, timedelta

import numpy as np
import pytest

from embrs.models.fuel_models import ScottBurgan40, Anderson13
from embrs.models.weather import GSITracker, DailySummary
from embrs.utilities.data_classes import WeatherEntry, GeoInfo


# ============================================================================
# Helpers
# ============================================================================

def _make_weather_entry(temp_F=77.0, rel_humidity=30.0, rain_cum=0.0):
    """Create a WeatherEntry with sim-runtime units (F, percent, cm cumulative)."""
    return WeatherEntry(
        wind_speed=4.5, wind_dir_deg=180.0,
        temp=temp_F, rel_humidity=rel_humidity,
        cloud_cover=0.0, rain=rain_cum,
        dni=800.0, dhi=100.0, ghi=850.0,
        solar_zenith=30.0, solar_azimuth=180.0,
    )


def _make_daily_summary(day_offset=0, min_temp_F=50.0, max_temp_F=85.0,
                        min_rh=25.0, rain_cm=0.01):
    """Create a DailySummary for a given day offset from a reference date."""
    return DailySummary(
        date=date(2024, 7, 1) + timedelta(days=day_offset),
        min_temp_F=min_temp_F,
        max_temp_F=max_temp_F,
        min_rh=min_rh,
        rain_cm=rain_cm,
    )


def _make_geo():
    """GeoInfo for Boulder, CO — mid-latitude with ~15h summer days."""
    return GeoInfo(
        center_lat=40.0, center_lon=-105.3,
        timezone="America/Denver",
    )


# ============================================================================
# update_curing() tests
# ============================================================================

class TestUpdateCuring:
    """Verify ScottBurgan40.update_curing() produces correct in-place updates."""

    def test_matches_fresh_construction(self):
        """update_curing(new_mf) must produce the same constants as
        constructing a fresh ScottBurgan40 with that moisture."""
        # GR2 (model 102) is dynamic
        original = ScottBurgan40(102, live_h_mf=0.5)
        original.update_curing(1.5)

        fresh = ScottBurgan40(102, live_h_mf=1.5)

        np.testing.assert_array_almost_equal(original.w_0, fresh.w_0)
        np.testing.assert_array_almost_equal(original.load, fresh.load)
        assert original.w_n_dead == pytest.approx(fresh.w_n_dead)
        assert original.w_n_live == pytest.approx(fresh.w_n_live)
        assert original.beta == pytest.approx(fresh.beta)
        assert original.rho_b == pytest.approx(fresh.rho_b)
        assert original.gamma == pytest.approx(fresh.gamma)
        assert original.flux_ratio == pytest.approx(fresh.flux_ratio)
        assert original.W == pytest.approx(fresh.W)
        assert original.w_n_dead_nominal == pytest.approx(fresh.w_n_dead_nominal)

    def test_round_trip(self):
        """Updating curing back to the original moisture restores original state."""
        fuel = ScottBurgan40(102, live_h_mf=0.8)
        orig_w0 = fuel.w_0.copy()
        orig_beta = fuel.beta

        fuel.update_curing(1.5)
        assert not np.allclose(fuel.w_0, orig_w0)

        fuel.update_curing(0.8)
        np.testing.assert_array_almost_equal(fuel.w_0, orig_w0)
        assert fuel.beta == pytest.approx(orig_beta)

    def test_noop_for_static_sb40(self):
        """Non-dynamic SB40 models should be unchanged by update_curing."""
        # TL1 (model 181) is static
        fuel = ScottBurgan40(181)
        orig_w0 = fuel.w_0.copy()
        orig_beta = fuel.beta

        fuel.update_curing(1.5)

        np.testing.assert_array_equal(fuel.w_0, orig_w0)
        assert fuel.beta == orig_beta

    def test_noop_for_anderson13(self):
        """Anderson 13 models are always non-dynamic — no-op."""
        fuel = Anderson13(1)
        orig_w0 = fuel.w_0.copy()

        fuel.update_curing(1.5)

        np.testing.assert_array_equal(fuel.w_0, orig_w0)

    def test_noop_for_non_burnable(self):
        """Non-burnable SB40 models should not crash."""
        fuel = ScottBurgan40(91)  # Urban
        fuel.update_curing(1.5)  # Should not raise

    def test_preserves_sav_constants(self):
        """SAV-derived constants must not change after update_curing."""
        fuel = ScottBurgan40(102, live_h_mf=0.5)
        orig_sav = fuel.sav_ratio
        orig_A = fuel.A
        orig_beta_op = fuel.beta_op
        orig_gammax = fuel.gammax

        fuel.update_curing(1.5)

        assert fuel.sav_ratio == orig_sav
        assert fuel.A == orig_A
        assert fuel.beta_op == orig_beta_op
        assert fuel.gammax == orig_gammax

    def test_multiple_dynamic_models(self):
        """Verify update_curing works across several dynamic SB40 models."""
        # GR1=101, GR2=102, GS1=121, GS2=122 are all dynamic
        for model_num in [101, 102, 121, 122]:
            fuel = ScottBurgan40(model_num, live_h_mf=0.5)
            fuel.update_curing(1.0)
            fresh = ScottBurgan40(model_num, live_h_mf=1.0)
            np.testing.assert_array_almost_equal(fuel.w_0, fresh.w_0,
                err_msg=f"Model {model_num} mismatch")


# ============================================================================
# GSITracker tests
# ============================================================================

class TestGSITracker:
    """Verify GSI tracker accumulation and computation."""

    def test_seed_and_compute(self):
        """Seeded tracker with 28+ days should return a valid GSI."""
        summaries = [_make_daily_summary(i) for i in range(30)]
        tracker = GSITracker(_make_geo(), summaries)

        gsi = tracker.compute_gsi()
        assert 0.0 <= gsi <= 1.0

    def test_insufficient_data_returns_negative(self):
        """Fewer than 2 days should return -1."""
        tracker = GSITracker(_make_geo(), [_make_daily_summary(0)])
        assert tracker.compute_gsi() == -1.0

    def test_empty_buffer_returns_negative(self):
        """Empty tracker should return -1."""
        tracker = GSITracker(_make_geo(), [])
        assert tracker.compute_gsi() == -1.0

    def test_rolling_window_cap(self):
        """Buffer should never exceed 56 entries."""
        summaries = [_make_daily_summary(i) for i in range(60)]
        tracker = GSITracker(_make_geo(), summaries)
        assert len(tracker._daily_buffer) == 56

    def test_accumulate_and_finalize(self):
        """Feed 24 hourly entries for one day, finalize, check summary."""
        tracker = GSITracker(_make_geo(), [])
        base_dt = datetime(2024, 7, 15, 0, 0, 0)

        # Simulate 24 hours of weather with varying temps
        temps = [60 + i for i in range(24)]  # 60F to 83F
        cum_rain = 0.0
        for h in range(24):
            cum_rain += 0.01  # 0.01 cm/hr
            entry = _make_weather_entry(temp_F=temps[h], rel_humidity=40.0,
                                        rain_cum=cum_rain)
            dt = base_dt + timedelta(hours=h)
            tracker.ingest_hourly(entry, dt)

        # Finalize by crossing to next day
        next_day_entry = _make_weather_entry(temp_F=65.0, rain_cum=cum_rain + 0.01)
        day_changed = tracker.ingest_hourly(next_day_entry,
                                            base_dt + timedelta(days=1))
        assert day_changed is True
        assert len(tracker._daily_buffer) == 1

        summary = tracker._daily_buffer[0]
        assert summary.min_temp_F == 60.0
        assert summary.max_temp_F == 83.0
        assert summary.min_rh == 40.0
        assert summary.rain_cm == pytest.approx(24 * 0.01)

    def test_ingest_same_day_no_finalize(self):
        """Hourly entries within the same day should not trigger finalization."""
        tracker = GSITracker(_make_geo(), [])
        base_dt = datetime(2024, 7, 15, 0, 0, 0)

        for h in range(12):
            entry = _make_weather_entry()
            changed = tracker.ingest_hourly(entry, base_dt + timedelta(hours=h))
            assert changed is False

        assert len(tracker._daily_buffer) == 0

    def test_rain_delta_computation(self):
        """Cumulative rain should be correctly converted to daily deltas."""
        tracker = GSITracker(_make_geo(), [], initial_cum_rain=10.0)
        base_dt = datetime(2024, 7, 15, 0, 0, 0)

        # First entry: cumulative rain=10.5 → delta = 0.5
        entry1 = _make_weather_entry(rain_cum=10.5)
        tracker.ingest_hourly(entry1, base_dt)

        # Second entry: cumulative rain=11.0 → delta = 0.5
        entry2 = _make_weather_entry(rain_cum=11.0)
        tracker.ingest_hourly(entry2, base_dt + timedelta(hours=1))

        assert tracker._hourly_rain_cm == pytest.approx(1.0)

    def test_compute_gsi_deterministic(self):
        """Same input should produce same GSI."""
        summaries = [_make_daily_summary(i) for i in range(30)]
        t1 = GSITracker(_make_geo(), summaries)
        t2 = GSITracker(_make_geo(), summaries)

        assert t1.compute_gsi() == t2.compute_gsi()

    def test_gsi_responds_to_new_data(self):
        """GSI should change when buffer content changes substantially."""
        warm_summaries = [_make_daily_summary(i, min_temp_F=50.0, max_temp_F=85.0,
                                              min_rh=25.0, rain_cm=0.05)
                          for i in range(28)]
        tracker = GSITracker(_make_geo(), warm_summaries)
        gsi_warm = tracker.compute_gsi()

        # Replace with cold/wet days — very different GSI
        cold_summaries = [_make_daily_summary(i, min_temp_F=-10.0, max_temp_F=15.0,
                                              min_rh=90.0, rain_cm=0.0)
                          for i in range(28)]
        tracker2 = GSITracker(_make_geo(), cold_summaries)
        gsi_cold = tracker2.compute_gsi()

        assert gsi_warm != gsi_cold


# ============================================================================
# set_live_moistures boundary tests
# ============================================================================

class TestSetLiveMoistures:
    """Verify GSI-to-moisture mapping at key boundaries."""

    def _call(self, gsi):
        """Call set_live_moistures as unbound method with a dummy self."""
        from unittest.mock import MagicMock
        from embrs.models.weather import WeatherStream
        mock_ws = MagicMock()
        return WeatherStream.set_live_moistures(mock_ws, gsi)

    def test_dormant_at_zero(self):
        h, w = self._call(0.0)
        assert h == 0.3
        assert w == 0.6

    def test_dormant_below_greenup(self):
        h, w = self._call(0.19)
        assert h == 0.3
        assert w == 0.6

    def test_greenup_threshold(self):
        h, w = self._call(0.2)
        # At exactly gu, the else branch runs but interpolates to near-dormant
        assert h == pytest.approx(0.3, abs=0.01)

    def test_max_at_one(self):
        h, w = self._call(1.0)
        assert h == pytest.approx(2.5)
        assert w == pytest.approx(2.0)

    def test_mid_range(self):
        h, w = self._call(0.6)
        assert 0.3 < h < 2.5
        assert 0.6 < w < 2.0
