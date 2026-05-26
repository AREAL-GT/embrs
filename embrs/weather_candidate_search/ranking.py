"""Filter candidate windows by target BI band and apply a composite score.

Default filter is **mean-only** (``mean_daily_1pm_bi`` in band): the NFDRS
daily afternoon mean over the window is the recommended cross-region
equivalence metric. The peak (97th-pct hourly) is reported as descriptive
metadata but does not gate selection. See ``config.BI_FILTER_MODES`` and
the field docstring on ``Config.bi_filter_mode`` for the rationale.

Composite score:

    bi_distance     = |<score_distance_column> - center(band)| / half_width(band)
    score           = -bi_distance_weight * bi_distance
                       + lulls_weight       * n_lulls
                       + lull_hours_weight  * total_lull_hours

Higher score wins; tie-break by earliest window start (qa F2). The
distance column matches the filter metric except in ``dual`` mode where
both metrics are filtered but only the smoother (mean) is scored.

Plan §4.9.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from embrs.weather_candidate_search.config import ScoringConfig
from embrs.weather_candidate_search.lull_detection import Lull

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    """One ranked candidate window with all its diagnostics.

    ``peak_bi`` is the 97th percentile of hourly ``BI_area_weighted`` over
    the window (BI scope OQ-15); ``mean_daily_1pm_bi`` is the NFDRS
    standard daily-1 PM mean across the window and is the *equivalence*
    metric used by the score's BI-distance term.
    """

    window_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    peak_bi: float
    time_of_peak: pd.Timestamp
    mean_bi: float
    mean_daily_1pm_bi: float
    lulls: List[Lull]
    n_lulls: int
    total_lull_hours: int
    score: float
    bi_distance_normalized: float
    rank: int = -1                  # filled by ``select_top_n``


def _band_geometry(band: Tuple[float, float]) -> Tuple[float, float]:
    lo, hi = float(band[0]), float(band[1])
    center = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)
    if half_width <= 0:
        # Degenerate point-target; use 1.0 so absolute distance is the
        # signal (consistent with plan §4.9).
        half_width = 1.0
    return center, half_width


_DEFAULT_BAND_FILTER_COLS: Tuple[str, ...] = ("mean_daily_1pm_bi",)


def filter_by_target_band(
    per_window: pd.DataFrame,
    target_band: Tuple[float, float],
    columns: Tuple[str, ...] = _DEFAULT_BAND_FILTER_COLS,
) -> pd.DataFrame:
    """Keep rows where EVERY ``columns`` value falls in the target band.

    By default this is the dual ``(peak_bi, mean_daily_1pm_bi)`` filter:
    a window must hit the band on *both* its 97th-pct hourly peak AND its
    NFDRS-daily-1pm mean. Pass ``columns=("peak_bi",)`` for the legacy
    peak-only filter.

    Returns a sub-frame of ``per_window`` (preserves index).
    """
    lo, hi = float(target_band[0]), float(target_band[1])
    missing = [c for c in columns if c not in per_window.columns]
    if missing:
        raise ValueError(
            f"filter_by_target_band: required column(s) missing: {missing}"
        )
    mask = pd.Series(True, index=per_window.index)
    for col in columns:
        mask &= (per_window[col] >= lo) & (per_window[col] <= hi)
    out = per_window.loc[mask].copy()
    logger.info(
        "BI band filter [%.1f, %.1f] on %s: %d / %d windows passed",
        lo,
        hi,
        list(columns),
        len(out),
        len(per_window),
    )
    return out


def score_windows(
    filtered: pd.DataFrame,
    lulls_by_window: Dict[str, List[Lull]],
    target_band: Tuple[float, float],
    scoring: ScoringConfig,
    bi_distance_column: str = "mean_daily_1pm_bi",
) -> pd.DataFrame:
    """Add ``score``, ``bi_distance_normalized``, ``n_lulls``,
    ``total_lull_hours`` columns to ``filtered``.

    The BI-distance term uses ``bi_distance_column`` (default
    ``mean_daily_1pm_bi`` — the NFDRS afternoon mean) so the score varies
    smoothly across overlapping windows. Pass ``"peak_bi"`` to recover
    the legacy peak-distance behaviour.

    Returns a copy with the new columns plus the existing ones, indexed
    the same way as ``filtered``.
    """
    if bi_distance_column not in filtered.columns:
        raise ValueError(
            f"score_windows: bi_distance_column={bi_distance_column!r} "
            f"not in filtered.columns={list(filtered.columns)}"
        )
    center, half_width = _band_geometry(target_band)
    out = filtered.copy()
    bi_distance = (out[bi_distance_column] - center).abs() / half_width
    n_lulls = np.array(
        [len(lulls_by_window.get(wid, [])) for wid in out.index], dtype=float
    )
    total_lull_hours = np.array(
        [
            sum(l.duration_hours for l in lulls_by_window.get(wid, []))
            for wid in out.index
        ],
        dtype=float,
    )
    score = (
        -scoring.bi_distance_weight * bi_distance.to_numpy()
        + scoring.lulls_weight * n_lulls
        + scoring.lull_hours_weight * total_lull_hours
    )
    out["bi_distance_normalized"] = bi_distance
    out["n_lulls"] = n_lulls.astype(int)
    out["total_lull_hours"] = total_lull_hours.astype(int)
    out["score"] = score
    return out


def select_top_n(
    scored: pd.DataFrame,
    lulls_by_window: Dict[str, List[Lull]],
    n: int,
    min_separation_hours: int = 0,
) -> List[RankedCandidate]:
    """Sort by ``(-score, start)`` and select the top ``n`` with NMS.

    Greedy temporal non-maximum suppression: walk the sorted list in
    descending-score order; accept a window only if its start is at least
    ``min_separation_hours`` away from every already-selected window's
    start. Set ``min_separation_hours = 0`` to disable suppression entirely
    (legacy behaviour; identical to a strict top-N by score).

    The motivation: with a 1-hour stride over a 168-hour window length,
    consecutive candidates share > 99% of their data and so produce
    near-identical scores. Without suppression the top-N collapse onto a
    single weather episode (5 windows starting one hour apart). NMS forces
    diversity across the fire season.

    Ties on ``(score, start)`` are broken by earliest start (qa F2);
    ``mergesort`` keeps the order deterministic.

    Returns:
        Up to ``n`` :class:`RankedCandidate` with ``rank`` filled (1-based).
        May be shorter than ``n`` if suppression leaves fewer than ``n``
        windows non-overlapping.
    """
    if scored.empty or n <= 0:
        return []
    ordered = scored.sort_values(
        by=["score", "start"],
        ascending=[False, True],
        kind="mergesort",          # stable sort for full determinism
    )
    min_sep = pd.Timedelta(hours=max(0, int(min_separation_hours)))
    candidates: List[RankedCandidate] = []
    chosen_starts: List[pd.Timestamp] = []
    for window_id, row in ordered.iterrows():
        start = pd.Timestamp(row["start"])
        if min_sep > pd.Timedelta(0) and any(
            abs(start - s) < min_sep for s in chosen_starts
        ):
            continue
        candidates.append(
            RankedCandidate(
                window_id=str(window_id),
                start=start,
                end=pd.Timestamp(row["end"]),
                peak_bi=float(row["peak_bi"]),
                time_of_peak=pd.Timestamp(row["time_of_peak"]),
                mean_bi=float(row["mean_bi"]),
                mean_daily_1pm_bi=float(row.get("mean_daily_1pm_bi", float("nan"))),
                lulls=list(lulls_by_window.get(window_id, [])),
                n_lulls=int(row["n_lulls"]),
                total_lull_hours=int(row["total_lull_hours"]),
                score=float(row["score"]),
                bi_distance_normalized=float(row["bi_distance_normalized"]),
                rank=len(candidates) + 1,
            )
        )
        chosen_starts.append(start)
        if len(candidates) >= n:
            break
    return candidates
