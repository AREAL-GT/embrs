"""Tests for the synthetic wind model (spec §4.2-4.4)."""
import numpy as np
import pandas as pd
import pytest

from embrs.scenario_weather.config import WindModelConfig
from embrs.scenario_weather.wind_model import (
    diurnal_mean_ms,
    generate_wind,
    _ou_series,
)


def test_diurnal_peak_and_floor():
    cfg = WindModelConfig(w_min_ms=1.5, peak_scale_ms=4.0)
    t_peak = cfg.rise_start_hr + cfg.peak_frac * cfg.daytime_span_hr
    # Peak == floor + amplitude.
    assert diurnal_mean_ms(np.array([t_peak]), cfg)[0] == pytest.approx(5.5, abs=1e-9)
    # Deep night == floor.
    assert diurnal_mean_ms(np.array([3.0]), cfg)[0] == pytest.approx(1.5, abs=1e-9)
    assert diurnal_mean_ms(np.array([23.5]), cfg)[0] == pytest.approx(1.5, abs=1e-9)


def test_diurnal_peak_is_afternoon():
    cfg = WindModelConfig()
    hod = np.arange(0, 24, 0.1)
    peak_hour = hod[int(np.argmax(diurnal_mean_ms(hod, cfg)))]
    assert 13.0 <= peak_hour <= 16.0


def test_peak_scale_override_is_monotonic():
    cfg = WindModelConfig(w_min_ms=1.0)
    t_peak = cfg.rise_start_hr + cfg.peak_frac * cfg.daytime_span_hr
    vals = [diurnal_mean_ms(np.array([t_peak]), cfg, peak_scale=ps)[0]
            for ps in (2.0, 4.0, 6.0, 8.0)]
    assert vals == sorted(vals)
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_ou_stationarity_and_zero_sigma():
    rng = np.random.default_rng(0)
    e = _ou_series(50000, phi=0.8, sigma=0.5, rng=rng)
    # Theoretical stationary std = sigma / sqrt(1 - phi^2).
    expected = 0.5 / np.sqrt(1 - 0.8 ** 2)
    assert np.std(e[1000:]) == pytest.approx(expected, rel=0.1)
    # Zero sigma => identically zero perturbation.
    assert np.all(_ou_series(100, 0.8, 0.0, rng) == 0.0)


def test_generate_wind_is_seed_reproducible():
    cfg = WindModelConfig(noise_seed=123)
    idx = pd.date_range("2022-07-08", periods=120, freq="h")
    s1, d1 = generate_wind(idx, cfg)
    s2, d2 = generate_wind(idx, cfg)
    assert np.array_equal(s1, s2)
    assert np.array_equal(d1, d2)


def test_speed_and_direction_independent():
    idx = pd.date_range("2022-07-08", periods=120, freq="h")
    base = WindModelConfig(noise_seed=7)
    only_dir_changed = WindModelConfig(noise_seed=7, dir_sigma_deg=30.0)
    s_a, _ = generate_wind(idx, base)
    s_b, _ = generate_wind(idx, only_dir_changed)
    # Changing a direction-only parameter must not perturb the speed series.
    assert np.array_equal(s_a, s_b)


def test_direction_stays_near_prevailing_and_varies():
    cfg = WindModelConfig(prevailing_dir_deg=200.0, dir_sigma_deg=8.0, noise_seed=5)
    idx = pd.date_range("2022-07-08", periods=336, freq="h")
    _, d = generate_wind(idx, cfg)
    assert d.min() >= 0.0 and d.max() < 360.0
    # Mean-reverting => stays in a band around prevailing, but does vary.
    signed = ((d - 200.0 + 180.0) % 360.0) - 180.0
    assert abs(np.mean(signed)) < 30.0
    assert np.std(signed) > 1.0


def test_wind_never_negative():
    cfg = WindModelConfig(w_min_ms=0.5, ou_sigma_ms=2.0, noise_seed=99)
    idx = pd.date_range("2022-07-08", periods=500, freq="h")
    s, _ = generate_wind(idx, cfg)
    assert np.all(s >= 0.0)
