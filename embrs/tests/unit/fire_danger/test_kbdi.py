"""Tests for embrs.fire_danger.kbdi."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from embrs.fire_danger.config import HourlyWeather
from embrs.fire_danger.kbdi import (
    DroughtAdjustedLoads,
    apply_drought_load_transfer,
    calc_kbdi,
    compute_kbdi_series,
    resolve_avg_ann_precip_in,
)
from embrs.fire_danger.nfdrs_fuel_models import CTA, NFDRS_FUEL_MODELS


# ---------------------------------------------------------------------------
# calc_kbdi
# ---------------------------------------------------------------------------


def test_kbdi_no_change_under_cool_dry():
    # max_temp_F <= 50 => no drying; no precip => no reduction; cum reset to 0.
    kbdi, cum = calc_kbdi(precip_in_day=0.0, max_temp_F=45.0,
                          cum_precip_in=0.5, prev_kbdi=200, avg_ann_precip_in=30)
    assert kbdi == 200.0
    assert cum == 0.0


def test_kbdi_rises_on_hot_dry_day():
    kbdi, cum = calc_kbdi(precip_in_day=0.0, max_temp_F=95.0,
                          cum_precip_in=0.0, prev_kbdi=200, avg_ann_precip_in=30)
    assert kbdi > 200.0
    assert kbdi <= 800.0


def test_kbdi_rain_above_threshold_reduces():
    """A > 0.20" rain event nets pptnet = rain - 0.20 when cum was below."""
    kbdi, cum = calc_kbdi(precip_in_day=1.0, max_temp_F=45.0,
                          cum_precip_in=0.0, prev_kbdi=300, avg_ann_precip_in=30)
    # pptnet = 1.0 - 0.20 = 0.80; net = int(80 + 0.0005) = 80; KBDI = 300 - 80 = 220.
    assert kbdi == 220.0
    assert cum == pytest.approx(1.0)


def test_kbdi_floors_at_zero():
    kbdi, _ = calc_kbdi(precip_in_day=10.0, max_temp_F=45.0,
                        cum_precip_in=0.0, prev_kbdi=50, avg_ann_precip_in=30)
    assert kbdi == 0.0


def test_kbdi_subsequent_rain_no_threshold_subtraction():
    """If cum_precip already > 0.20 entering the day, pptnet is full precip."""
    kbdi, cum = calc_kbdi(precip_in_day=0.5, max_temp_F=45.0,
                          cum_precip_in=0.5, prev_kbdi=300, avg_ann_precip_in=30)
    # pptnet = 0.5; net = int(50 + 0.0005) = 50; KBDI = 300 - 50 = 250.
    assert kbdi == 250.0
    assert cum == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# apply_drought_load_transfer
# ---------------------------------------------------------------------------


def test_drought_identity_below_threshold():
    fuel = NFDRS_FUEL_MODELS["Y"]
    adj = apply_drought_load_transfer(fuel, kbdi=100.0)
    assert adj.W1 == pytest.approx(fuel.L1 * CTA)
    assert adj.W10 == pytest.approx(fuel.L10 * CTA)
    assert adj.W100 == pytest.approx(fuel.L100 * CTA)
    assert adj.W1000 == pytest.approx(fuel.L1000 * CTA)
    assert adj.depth == pytest.approx(fuel.Depth)


def test_drought_increases_loadings_above_threshold():
    fuel = NFDRS_FUEL_MODELS["Y"]
    base = apply_drought_load_transfer(fuel, kbdi=100.0)
    drier = apply_drought_load_transfer(fuel, kbdi=500.0)
    assert drier.W1 > base.W1
    assert drier.W10 > base.W10
    assert drier.W100 > base.W100
    assert drier.W1000 > base.W1000


def test_drought_zero_drought_load_model():
    """Model V has LDrought=0 -> no change even when KBDI is high."""
    fuel = NFDRS_FUEL_MODELS["V"]
    base = apply_drought_load_transfer(fuel, kbdi=100.0)
    high = apply_drought_load_transfer(fuel, kbdi=800.0)
    assert high.W1 == pytest.approx(base.W1)
    assert high.W10 == pytest.approx(base.W10)


# ---------------------------------------------------------------------------
# compute_kbdi_series
# ---------------------------------------------------------------------------


def _hot_dry_weather(n_days: int = 7) -> HourlyWeather:
    n = n_days * 24
    times = pd.date_range("2025-07-01 00:00", periods=n, freq="h",
                          tz="America/Denver")
    df = pd.DataFrame(
        {
            "temp_F": np.full(n, 95.0),
            "temp_C": np.full(n, 35.0),
            "rh_pct": np.full(n, 15.0),
            "rh_frac": np.full(n, 0.15),
            "wind_mph": np.full(n, 5.0),
            "wind_dir_deg": np.full(n, 180.0),
            "precip_in_hr": np.zeros(n),
            "precip_cm_hr": np.zeros(n),
            "cloud_cover": np.zeros(n),
            "solar_wm2": np.full(n, 500.0),
        },
        index=times,
    )
    df.index.name = "datetime"
    return HourlyWeather(df=df, ref_elev_m=1200.0, time_step_min=60,
                         raw_start=df.index[0].to_pydatetime(),
                         raw_end=df.index[-1].to_pydatetime())


def test_kbdi_series_rises_monotonically_under_hot_dry():
    out = compute_kbdi_series(_hot_dry_weather(7), avg_ann_precip_in=30.0).df
    assert "KBDI" in out.columns
    # First valid day's KBDI is greater than seed (100) because of drying.
    assert out["KBDI"].iloc[0] > 100.0
    # Monotonically non-decreasing across the run.
    assert (out["KBDI"].diff().fillna(0) >= 0).all()
    # Hard bound.
    assert (out["KBDI"] <= 800).all()


def test_kbdi_series_skips_partial_first_window():
    """The first day usually lacks the full 24-h trailing window ending at 13:00."""
    n = 25  # only a single full 24-h window will be available
    times = pd.date_range("2025-07-01 00:00", periods=n, freq="h",
                          tz="America/Denver")
    df = pd.DataFrame(
        {
            "temp_F": np.full(n, 80.0),
            "temp_C": np.full(n, 26.7),
            "rh_pct": np.full(n, 30.0),
            "rh_frac": np.full(n, 0.3),
            "wind_mph": np.full(n, 5.0),
            "wind_dir_deg": np.full(n, 180.0),
            "precip_in_hr": np.zeros(n),
            "precip_cm_hr": np.zeros(n),
            "cloud_cover": np.zeros(n),
            "solar_wm2": np.full(n, 300.0),
        },
        index=times,
    )
    df.index.name = "datetime"
    w = HourlyWeather(df=df, ref_elev_m=1200.0, time_step_min=60,
                      raw_start=df.index[0].to_pydatetime(),
                      raw_end=df.index[-1].to_pydatetime())
    out = compute_kbdi_series(w, avg_ann_precip_in=30.0).df
    # Need 24 hours up to 13:00 on 2025-07-01 — starts at 14:00 prior day,
    # not available. So expect 0 valid days (or just the 2nd if RegObsHr falls
    # within a full window).
    assert len(out) <= 1


# ---------------------------------------------------------------------------
# fetch_avg_ann_precip_in (mocked) + resolve_avg_ann_precip_in
# ---------------------------------------------------------------------------


def test_resolve_explicit_short_circuits():
    val, src = resolve_avg_ann_precip_in(explicit=42.5, lat=39.5, lon=-105.0)
    assert val == 42.5
    assert src == "explicit"


def test_resolve_fallback_on_network_failure(monkeypatch):
    """If the fetch raises, resolve falls back to the default (30 in)."""
    from embrs.fire_danger import kbdi as kbdi_mod

    def boom(lat, lon, **kwargs):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(kbdi_mod, "fetch_avg_ann_precip_in", boom)
    with pytest.warns(RuntimeWarning, match="falling back"):
        val, src = resolve_avg_ann_precip_in(explicit=None, lat=39.5, lon=-105.0)
    assert val == 30.0
    assert src == "default"


def test_resolve_no_geo_means_default():
    val, src = resolve_avg_ann_precip_in(explicit=None, lat=None, lon=None)
    assert val == 30.0
    assert src == "default"
