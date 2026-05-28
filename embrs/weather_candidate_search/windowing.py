"""Sliding-window iteration over the fire-season weather frame.

Plan §4.6. Windows are emitted only when fully inside ``[scenario_start,
scenario_end]`` AND every required column has finite values across the
entire window (qa D3).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Columns required to be finite in every window emitted by ``iter_windows``.
# Includes both weather inputs and the joined BI trajectory.
DEFAULT_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "temp_C",
    "rh_pct",
    "rain_mm_hr",
    "wind_mps",
    "wind_dir_deg",
    "cloud_pct",
    "wind_mph",          # post-log-profile correction (pipeline-added)
    "BI_area_weighted",  # joined from the BI trajectory
)


@dataclass(frozen=True)
class Window:
    """One sliding window. ``end`` is exclusive."""

    window_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    df: pd.DataFrame


def _format_window_id(ts: pd.Timestamp) -> str:
    """ISO-8601-ish window ID (deterministic, filename-safe)."""
    return ts.strftime("%Y-%m-%dT%H%M")


def iter_windows(
    weather_df: pd.DataFrame,
    scenario_start: pd.Timestamp,
    scenario_end: pd.Timestamp,
    scenario_length_hours: int,
    stride_hours: int = 1,
    required_columns: Sequence[str] = DEFAULT_REQUIRED_COLUMNS,
) -> Iterator[Window]:
    """Yield sliding-window slices of the merged weather+BI frame.

    Args:
        weather_df: Hourly frame indexed by tz-aware local datetime.
        scenario_start: Earliest allowed window start (inclusive,
            tz-aware).
        scenario_end: Latest allowed window end (exclusive, tz-aware).
        scenario_length_hours: Window length.
        stride_hours: Step between successive window starts.
        required_columns: Columns that must be finite across every hour
            of an emitted window. Missing columns are silently ignored
            (so the same iterator works pre- and post-BI-join in tests).

    Yields:
        :class:`Window` instances in chronological order.
    """
    if scenario_length_hours <= 0:
        raise ValueError(f"scenario_length_hours must be > 0, got {scenario_length_hours}")
    if stride_hours <= 0:
        raise ValueError(f"stride_hours must be > 0, got {stride_hours}")

    df = weather_df.sort_index()
    cols_present = [c for c in required_columns if c in df.columns]
    cols_missing = [c for c in required_columns if c not in df.columns]
    if cols_missing:
        logger.debug(
            "iter_windows: required columns not present (ignored): %s", cols_missing
        )

    length = pd.Timedelta(hours=scenario_length_hours)
    stride = pd.Timedelta(hours=stride_hours)

    # Latest start such that start + length <= scenario_end.
    latest_start = scenario_end - length

    candidate_starts = pd.date_range(
        start=scenario_start, end=latest_start, freq=stride, inclusive="both"
    )

    n_emitted = 0
    n_skipped_missing = 0
    for start_ts in candidate_starts:
        start_ts = pd.Timestamp(start_ts)
        end_ts = start_ts + length
        sliced = df.loc[(df.index >= start_ts) & (df.index < end_ts)]
        if len(sliced) < scenario_length_hours:
            # Gap in the index — window not fully populated.
            n_skipped_missing += 1
            continue
        # Reject any NaN in required columns.
        if cols_present:
            invalid = sliced[cols_present].isna().any(axis=None)
            if bool(invalid):
                n_skipped_missing += 1
                continue
        n_emitted += 1
        yield Window(
            window_id=_format_window_id(start_ts),
            start=start_ts,
            end=end_ts,
            df=sliced,
        )

    logger.info(
        "iter_windows: emitted %d / %d (skipped %d for missing data)",
        n_emitted,
        len(candidate_starts),
        n_skipped_missing,
    )
