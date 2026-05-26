"""Tests for per-window BI peak extraction (the BI pipeline itself is mocked)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from embrs.weather_candidate_search.bi_search import per_window_peaks


class _StubWindow:
    def __init__(self, wid, start, end):
        self.window_id = wid
        self.start = start
        self.end = end


def _bi_frame(values: list[float]):
    idx = pd.date_range("2024-07-01 00:00", periods=len(values), freq="H", tz="UTC")
    return pd.DataFrame({"BI_area_weighted": values}, index=idx)


def test_per_window_peaks_constant_series():
    df = _bi_frame([50.0] * 24)
    end_after_last = df.index[-1] + pd.Timedelta(hours=1)
    windows = [
        _StubWindow("w0", df.index[0], df.index[12]),
        _StubWindow("w1", df.index[12], end_after_last),
    ]
    out = per_window_peaks(df, windows)
    assert list(out.index) == ["w0", "w1"]
    assert all(out["peak_bi"] == 50.0)
    assert all(out["mean_bi"] == 50.0)
    assert all(out["n_hours"] == 12)


def test_per_window_peaks_matches_numpy_percentile():
    df = _bi_frame(list(np.linspace(0, 100, 24)))
    end_after_last = df.index[-1] + pd.Timedelta(hours=1)
    win = _StubWindow("w0", df.index[0], end_after_last)
    out = per_window_peaks(df, [win])
    expected = np.percentile(df["BI_area_weighted"].to_numpy(), 97)
    np.testing.assert_allclose(out.loc["w0", "peak_bi"], expected)


def test_per_window_peaks_handles_all_nan_window():
    values = [float("nan")] * 12 + [50.0] * 12
    df = _bi_frame(values)
    end_after_last = df.index[-1] + pd.Timedelta(hours=1)
    win_left = _StubWindow("w_left", df.index[0], df.index[12])
    win_right = _StubWindow("w_right", df.index[12], end_after_last)
    out = per_window_peaks(df, [win_left, win_right])
    assert pd.isna(out.loc["w_left", "peak_bi"])
    assert out.loc["w_left", "n_valid_hours"] == 0
    assert out.loc["w_right", "peak_bi"] == 50.0
    assert out.loc["w_right", "n_valid_hours"] == 12


def test_per_window_peaks_empty():
    df = _bi_frame([])
    out = per_window_peaks(df, [])
    assert out.empty


def test_per_window_peaks_includes_mean_daily_1pm_bi():
    """Two full days where 13:00 BI is 60 and 80 → mean_daily_1pm_bi = 70."""
    # Build a 48-h frame where BI at 13:00 is fixed and other hours are 0.
    n = 48
    idx = pd.date_range("2024-07-01 00:00", periods=n, freq="h", tz="UTC")
    values = np.zeros(n)
    # 13:00 local on day 1 (hour 13) and day 2 (hour 37).
    values[13] = 60.0
    values[37] = 80.0
    df = pd.DataFrame({"BI_area_weighted": values}, index=idx)
    end_after_last = idx[-1] + pd.Timedelta(hours=1)
    win = _StubWindow("w0", idx[0], end_after_last)
    out = per_window_peaks(df, [win])
    assert out.loc["w0", "mean_daily_1pm_bi"] == pytest.approx(70.0)
    assert out.loc["w0", "n_daily_1pm_samples"] == 2


def test_per_window_peaks_mean_daily_1pm_nan_when_no_1pm_rows():
    """A window that misses 13:00 entirely returns NaN for the daily mean."""
    # 12 hours starting at 00:00 → does not include hour 13.
    n = 12
    idx = pd.date_range("2024-07-01 00:00", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"BI_area_weighted": np.full(n, 50.0)}, index=idx)
    end_after_last = idx[-1] + pd.Timedelta(hours=1)
    win = _StubWindow("w0", idx[0], end_after_last)
    out = per_window_peaks(df, [win])
    assert pd.isna(out.loc["w0", "mean_daily_1pm_bi"])
    assert out.loc["w0", "n_daily_1pm_samples"] == 0


def test_per_window_peaks_reg_obs_hour_override():
    """The afternoon sampling hour is configurable."""
    n = 24
    idx = pd.date_range("2024-07-01 00:00", periods=n, freq="h", tz="UTC")
    values = np.zeros(n)
    values[15] = 90.0   # 15:00
    df = pd.DataFrame({"BI_area_weighted": values}, index=idx)
    end_after_last = idx[-1] + pd.Timedelta(hours=1)
    win = _StubWindow("w0", idx[0], end_after_last)
    out = per_window_peaks(df, [win], reg_obs_hour=15)
    assert out.loc["w0", "mean_daily_1pm_bi"] == 90.0
    assert out.loc["w0", "n_daily_1pm_samples"] == 1
