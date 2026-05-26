"""Lull (backburn-window) detection.

A row is "calm" if wind speed is below ``wind_threshold_mph`` AND relative
humidity is above ``rh_threshold_pct``. Consecutive calm rows form a lull;
at most ``tolerance_hours`` non-calm rows are allowed inside an otherwise
calm run (strict by default — qa E2). All hours count equally regardless
of time of day (qa E3).

Plan §4.8.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from embrs.weather_candidate_search.config import LullConfig


@dataclass(frozen=True)
class Lull:
    """A detected backburn window.

    ``end`` is the timestamp of the LAST calm hour (inclusive). ``duration_hours``
    counts the number of hours in the run (calm + tolerated non-calm).
    """

    start: pd.Timestamp
    end: pd.Timestamp
    duration_hours: int


def _is_calm(window_df: pd.DataFrame, lull: LullConfig) -> np.ndarray:
    """Per-row calm mask using the corrected wind column."""
    wind_mph = window_df["wind_mph"].to_numpy(dtype=float)
    rh_pct = window_df["rh_pct"].to_numpy(dtype=float)
    return (wind_mph < lull.wind_threshold_mph) & (rh_pct > lull.rh_threshold_pct)


def detect_lulls(window_df: pd.DataFrame, lull: LullConfig) -> List[Lull]:
    """Detect lulls within a single window's DataFrame.

    Args:
        window_df: Window slice with at least ``wind_mph`` and ``rh_pct``
            columns; index must be tz-aware hourly.
        lull: Threshold config.

    Returns:
        List of :class:`Lull` ordered by start time.
    """
    if "wind_mph" not in window_df.columns or "rh_pct" not in window_df.columns:
        raise ValueError("detect_lulls: window_df must have 'wind_mph' and 'rh_pct' columns")

    calm = _is_calm(window_df, lull)
    if calm.size == 0:
        return []

    lulls: List[Lull] = []
    tolerance = max(0, int(lull.tolerance_hours))

    n = len(calm)
    i = 0
    while i < n:
        if not calm[i]:
            i += 1
            continue
        # Start of a potential lull.
        run_start = i
        run_end = i
        non_calm_remaining = tolerance
        j = i + 1
        while j < n:
            if calm[j]:
                run_end = j
                j += 1
                continue
            # Non-calm hour: consume tolerance if available; only do so when
            # there is at least one further calm hour ahead (otherwise we
            # would let trailing non-calm hours pad the run).
            if non_calm_remaining > 0 and _has_calm_ahead(calm, j + 1):
                non_calm_remaining -= 1
                j += 1
                continue
            break
        duration = run_end - run_start + 1
        if duration >= lull.min_consecutive_hours:
            start_ts = pd.Timestamp(window_df.index[run_start])
            end_ts = pd.Timestamp(window_df.index[run_end])
            lulls.append(Lull(start=start_ts, end=end_ts, duration_hours=duration))
        i = max(run_end + 1, j)

    return lulls


def _has_calm_ahead(calm: np.ndarray, start: int) -> bool:
    """Whether any calm hour exists at index ``>= start``."""
    return bool(calm[start:].any()) if start < calm.size else False


def summarize_lulls(lulls: List[Lull]) -> dict:
    """Return ``{n_lulls, total_lull_hours}`` for a candidate."""
    if not lulls:
        return {"n_lulls": 0, "total_lull_hours": 0}
    return {
        "n_lulls": len(lulls),
        "total_lull_hours": int(sum(l.duration_hours for l in lulls)),
    }
