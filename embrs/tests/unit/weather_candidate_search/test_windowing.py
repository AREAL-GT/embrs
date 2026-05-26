"""Tests for the sliding-window iterator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from embrs.weather_candidate_search.windowing import iter_windows, Window


def _make_frame(n: int = 48, with_bi: bool = True):
    idx = pd.date_range("2024-07-01 00:00", periods=n, freq="H", tz="UTC")
    cols = {
        "temp_C": np.full(n, 20.0),
        "rh_pct": np.full(n, 50.0),
        "rain_mm_hr": np.zeros(n),
        "wind_mps": np.full(n, 3.0),
        "wind_mph": np.full(n, 6.0),
        "wind_dir_deg": np.full(n, 180.0),
        "cloud_pct": np.full(n, 10.0),
    }
    if with_bi:
        cols["BI_area_weighted"] = 40.0 + np.arange(n) * 0.1
    return pd.DataFrame(cols, index=idx)


def test_iter_windows_count():
    df = _make_frame(48)
    windows = list(
        iter_windows(
            weather_df=df,
            scenario_start=df.index[0],
            scenario_end=df.index[-1] + pd.Timedelta(hours=1),
            scenario_length_hours=12,
            stride_hours=1,
        )
    )
    # 48 - 12 + 1 = 37 valid starts
    assert len(windows) == 37
    assert all(isinstance(w, Window) for w in windows)
    assert windows[0].start == df.index[0]
    assert windows[-1].start == df.index[36]


def test_iter_windows_skips_nan():
    df = _make_frame(24)
    df.iloc[10, df.columns.get_loc("BI_area_weighted")] = float("nan")
    windows = list(
        iter_windows(
            weather_df=df,
            scenario_start=df.index[0],
            scenario_end=df.index[-1] + pd.Timedelta(hours=1),
            scenario_length_hours=6,
            stride_hours=1,
        )
    )
    # Windows starting at hours 5..10 each include hour 10 → invalidated.
    expected_skipped = {5, 6, 7, 8, 9, 10}
    valid_starts = {w.start.hour for w in windows}
    assert valid_starts.isdisjoint(expected_skipped)
    assert 4 in valid_starts                       # window 4-9 is clean
    assert 11 in valid_starts                      # window 11-16 is clean


def test_iter_windows_window_id_is_iso():
    df = _make_frame(12)
    w = next(
        iter_windows(
            weather_df=df,
            scenario_start=df.index[0],
            scenario_end=df.index[-1] + pd.Timedelta(hours=1),
            scenario_length_hours=4,
        )
    )
    assert w.window_id == "2024-07-01T0000"


def test_iter_windows_rejects_bad_inputs():
    df = _make_frame(12)
    with pytest.raises(ValueError):
        list(iter_windows(df, df.index[0], df.index[-1], 0))
    with pytest.raises(ValueError):
        list(iter_windows(df, df.index[0], df.index[-1], 4, stride_hours=0))


def test_iter_windows_empty_when_too_long():
    df = _make_frame(4)
    windows = list(
        iter_windows(
            weather_df=df,
            scenario_start=df.index[0],
            scenario_end=df.index[-1] + pd.Timedelta(hours=1),
            scenario_length_hours=10,
        )
    )
    assert windows == []
