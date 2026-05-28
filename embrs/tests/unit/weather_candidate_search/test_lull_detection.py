"""Tests for lull (backburn-window) detection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from embrs.weather_candidate_search.config import LullConfig
from embrs.weather_candidate_search.lull_detection import detect_lulls, summarize_lulls


def _frame_from_pattern(pattern_calm: list[bool]) -> pd.DataFrame:
    """Build a frame where ``pattern_calm[i]`` controls whether hour i is calm."""
    n = len(pattern_calm)
    idx = pd.date_range("2024-07-01 00:00", periods=n, freq="H", tz="UTC")
    wind = np.where(pattern_calm, 4.0, 12.0)
    rh = np.where(pattern_calm, 50.0, 25.0)
    return pd.DataFrame({"wind_mph": wind, "rh_pct": rh}, index=idx)


def test_single_lull_strict():
    pattern = [False, False, False] + [True] * 5 + [False] * 4
    df = _frame_from_pattern(pattern)
    lulls = detect_lulls(df, LullConfig(wind_threshold_mph=8, rh_threshold_pct=40, min_consecutive_hours=4))
    assert len(lulls) == 1
    assert lulls[0].duration_hours == 5
    assert lulls[0].start == df.index[3]
    assert lulls[0].end == df.index[7]


def test_short_run_below_minimum_dropped():
    pattern = [False] + [True] * 3 + [False] * 6   # 3-hour run, min 4 → dropped
    df = _frame_from_pattern(pattern)
    lulls = detect_lulls(df, LullConfig(min_consecutive_hours=4))
    assert lulls == []


def test_strict_breach_splits_lull():
    # 4 calm, 1 non-calm, 4 calm; strict tolerance=0 must split.
    pattern = [True] * 4 + [False] + [True] * 4
    df = _frame_from_pattern(pattern)
    lulls = detect_lulls(df, LullConfig(min_consecutive_hours=4, tolerance_hours=0))
    assert len(lulls) == 2
    assert all(l.duration_hours == 4 for l in lulls)


def test_tolerance_allows_brief_breach():
    pattern = [True] * 4 + [False] + [True] * 4
    df = _frame_from_pattern(pattern)
    lulls = detect_lulls(df, LullConfig(min_consecutive_hours=4, tolerance_hours=1))
    assert len(lulls) == 1
    assert lulls[0].duration_hours == 9


def test_all_calm():
    pattern = [True] * 12
    df = _frame_from_pattern(pattern)
    lulls = detect_lulls(df, LullConfig(min_consecutive_hours=4))
    assert len(lulls) == 1
    assert lulls[0].duration_hours == 12


def test_no_calm():
    pattern = [False] * 12
    df = _frame_from_pattern(pattern)
    lulls = detect_lulls(df, LullConfig(min_consecutive_hours=4))
    assert lulls == []


def test_summarize_lulls():
    pattern = [True] * 5 + [False] + [True] * 6 + [False] + [True] * 4
    df = _frame_from_pattern(pattern)
    lulls = detect_lulls(df, LullConfig(min_consecutive_hours=4, tolerance_hours=0))
    summary = summarize_lulls(lulls)
    assert summary["n_lulls"] == 3
    assert summary["total_lull_hours"] == 5 + 6 + 4
