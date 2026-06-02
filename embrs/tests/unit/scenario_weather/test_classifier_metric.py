"""Tests for the classifier metric internals (spec §2): flame conversion and
the streaming daily-peak histogram. No simulation is run here."""
import numpy as np
import pytest

from embrs.scenario_weather.classifier import (
    _DayHistogram,
    i_ss_btu_to_flame_ft,
)
from embrs.utilities.unit_conversions import BTU_ft_min_to_kW_m


def test_flame_length_byram_hand_value():
    # 2000 BTU/ft/min -> kW/m -> Byram flame length (ft).
    i_kw = BTU_ft_min_to_kW_m(2000.0)
    flame_m = 0.0775 * i_kw ** 0.46
    flame_ft = flame_m * 3.28084
    assert i_ss_btu_to_flame_ft(np.array([2000.0]))[0] == pytest.approx(flame_ft, rel=1e-9)


def test_flame_length_zero_and_monotonic():
    out = i_ss_btu_to_flame_ft(np.array([0.0, 500.0, 1000.0, 2000.0]))
    assert out[0] == 0.0
    assert np.all(np.diff(out) > 0)


def test_histogram_percentile_matches_numpy():
    rng = np.random.default_rng(0)
    data = np.abs(rng.normal(5.0, 2.0, size=300000))
    h = _DayHistogram(max_ft=40.0, bin_ft=0.05)
    for chunk in np.array_split(data, 41):
        h.add(chunk, np.array([1.0]))
    assert h.n_samples == data.size
    for q in (50, 90, 97, 99):
        assert h.percentile(q) == pytest.approx(np.percentile(data, q), abs=0.05)


def test_histogram_tracks_max_ros_and_overflow():
    h = _DayHistogram(max_ft=10.0, bin_ft=0.1)
    h.add(np.array([1.0, 2.0]), np.array([3.0, 5.0]))
    h.add(np.array([50.0]), np.array([1.0]))  # above max_ft -> overflow, top bin
    assert h.max_ros_m_min == 5.0
    assert h.overflow == 1
    # The overflow sample still counts toward the distribution (top bin).
    assert h.n_samples == 3


def test_histogram_empty_is_zero():
    h = _DayHistogram(max_ft=40.0, bin_ft=0.05)
    assert h.percentile(97) == 0.0
    assert h.n_samples == 0
