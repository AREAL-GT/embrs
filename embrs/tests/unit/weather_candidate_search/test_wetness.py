"""Tests for the precipitation-based wetness guard."""
from __future__ import annotations

import pandas as pd
import pytest

from embrs.weather_candidate_search.config import WetnessGuard
from embrs.weather_candidate_search.wetness import evaluate_wetness


class _W:
    def __init__(self, wid, start, end):
        self.window_id = wid
        self.start = start
        self.end = end


def _weather(rain_in_by_hour):
    """rain_in_by_hour: dict {timestamp_str: inches}; builds an hourly mm frame."""
    idx = pd.date_range("2022-06-20 00:00", "2022-07-12 23:00", freq="h", tz="UTC")
    rain_in = pd.Series(0.0, index=idx)
    for ts, inches in rain_in_by_hour.items():
        rain_in.loc[pd.Timestamp(ts, tz="UTC")] = inches
    return pd.DataFrame({"rain_mm_hr": rain_in / 0.0393701}, index=idx)


_WIN = _W("w0", pd.Timestamp("2022-07-06 05:00", tz="UTC"),
          pd.Timestamp("2022-07-10 05:00", tz="UTC"))


def test_antecedent_rain_fails_window():
    # 2.0 in of rain in the 10 days before a 2022-07-06 start.
    wx = _weather({"2022-07-02 12:00": 2.0})
    res = evaluate_wetness(wx, [_WIN], WetnessGuard(max_antecedent_precip_in=1.0))
    assert res["w0"].passed is False
    assert "antecedent" in res["w0"].reason
    assert res["w0"].antecedent_precip_in == pytest.approx(2.0, abs=1e-3)


def test_dry_window_passes():
    wx = _weather({"2022-06-25 12:00": 0.3})  # before the 10-day lookback
    res = evaluate_wetness(wx, [_WIN], WetnessGuard(max_antecedent_precip_in=1.0))
    assert res["w0"].passed is True
    assert res["w0"].reason == ""


def test_soaking_in_window_day_fails():
    wx = _weather({"2022-07-08 14:00": 1.5})  # heavy rain mid-window
    res = evaluate_wetness(wx, [_WIN], WetnessGuard(max_daily_precip_in=1.0))
    assert res["w0"].passed is False
    assert "daily" in res["w0"].reason


def test_disabled_guard_passes_everything():
    wx = _weather({"2022-07-02 12:00": 5.0})
    res = evaluate_wetness(wx, [_WIN], WetnessGuard(enabled=False))
    assert res["w0"].passed is True
    assert res["w0"].antecedent_precip_in == pytest.approx(5.0, abs=1e-3)
