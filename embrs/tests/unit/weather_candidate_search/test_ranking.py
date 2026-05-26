"""Tests for ranking + filtering."""
from __future__ import annotations

import pandas as pd
import pytest

from embrs.weather_candidate_search.config import ScoringConfig
from embrs.weather_candidate_search.lull_detection import Lull
from embrs.weather_candidate_search.ranking import (
    filter_by_target_band,
    score_windows,
    select_top_n,
)


def _per_window(rows, mean_daily_1pm_bi=None):
    """rows = list of (window_id, start_hour, peak_bi, mean_bi).

    Defaults ``mean_daily_1pm_bi = peak_bi`` so "in-band peak" rows are
    also "in-band mean" under the dual filter — keeps the legacy tests
    focused on what they test (filtering, scoring, NMS) without forcing
    every row to specify both metrics. Override via the
    ``mean_daily_1pm_bi`` dict ({window_id: value}) when a test needs
    them to disagree.
    """
    df = pd.DataFrame(
        [
            {
                "window_id": wid,
                "start": pd.Timestamp("2024-07-01") + pd.Timedelta(hours=h),
                "end": pd.Timestamp("2024-07-01") + pd.Timedelta(hours=h + 12),
                "peak_bi": peak,
                "time_of_peak": pd.Timestamp("2024-07-01") + pd.Timedelta(hours=h + 6),
                "mean_bi": mean,
                "mean_daily_1pm_bi": (
                    mean_daily_1pm_bi[wid]
                    if mean_daily_1pm_bi is not None and wid in mean_daily_1pm_bi
                    else peak
                ),
                "n_daily_1pm_samples": 1,
                "n_hours": 12,
                "n_valid_hours": 12,
            }
            for (wid, h, peak, mean) in rows
        ]
    ).set_index("window_id")
    return df


def test_filter_by_target_band_dual():
    """Default dual filter: BOTH peak_bi and mean_daily_1pm_bi must be in band."""
    df = _per_window([("a", 0, 30, 20), ("b", 1, 70, 50), ("c", 2, 90, 60)])
    # With the helper's default mean_daily_1pm_bi = peak_bi, only b's peak
    # (70) is in [60, 80] AND its mean_daily_1pm_bi (70) is in [60, 80].
    out = filter_by_target_band(df, (60, 80))
    assert list(out.index) == ["b"]


def test_filter_by_target_band_rejects_peak_in_band_mean_out():
    """A window whose peak hits the band but mean is below it is dropped."""
    df = _per_window(
        [("spike", 0, 70, 30)],
        mean_daily_1pm_bi={"spike": 30.0},
    )
    out = filter_by_target_band(df, (60, 80))
    assert out.empty


def test_filter_by_target_band_legacy_single_column():
    """``columns=("peak_bi",)`` recovers the pre-dual-metric behaviour."""
    df = _per_window(
        [("spike", 0, 70, 30)],
        mean_daily_1pm_bi={"spike": 30.0},
    )
    out = filter_by_target_band(df, (60, 80), columns=("peak_bi",))
    assert list(out.index) == ["spike"]


def test_score_increases_with_lulls():
    df = _per_window([("a", 0, 70, 50), ("b", 1, 70, 50)])
    lulls = {
        "a": [Lull(pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-01 03:00"), 4)],
        "b": [],
    }
    scored = score_windows(df, lulls, (60, 80), ScoringConfig())
    assert scored.loc["a", "score"] > scored.loc["b", "score"]
    assert scored.loc["a", "n_lulls"] == 1
    assert scored.loc["b", "n_lulls"] == 0


def test_score_distance_dominates_lulls_when_band_distance_large():
    """BI distance (now via daily 1 PM mean) dominates when the gap is wide."""
    # a: mean at band center, no lulls. b: mean way off, one lull.
    df = _per_window([("a", 0, 70, 70), ("b", 1, 100, 70)])
    lulls = {
        "a": [],
        "b": [Lull(pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-01 04:00"), 5)],
    }
    scored = score_windows(df, lulls, (60, 80), ScoringConfig())
    assert scored.loc["a", "score"] > scored.loc["b", "score"]


def test_score_uses_mean_daily_1pm_not_peak():
    """Two windows with identical peak but different daily-1pm means are
    ranked by the mean (the new BI-distance column)."""
    # Both peak=70; window 'centered' has mean=70 (at band center),
    # 'biased' has mean=80 (off-center).
    df = _per_window(
        [("centered", 0, 70, 70), ("biased", 1, 70, 80)],
        mean_daily_1pm_bi={"centered": 70.0, "biased": 80.0},
    )
    lulls = {"centered": [], "biased": []}
    scored = score_windows(df, lulls, (60, 80), ScoringConfig())
    assert scored.loc["centered", "bi_distance_normalized"] == 0.0
    assert scored.loc["biased", "bi_distance_normalized"] == pytest.approx(1.0)
    assert scored.loc["centered", "score"] > scored.loc["biased", "score"]


def test_select_top_n_tie_breaks_by_earlier_start():
    # Two windows with identical bi distance and lulls → earlier start wins.
    df = _per_window([("late", 5, 70, 50), ("early", 0, 70, 50)])
    lulls = {"late": [], "early": []}
    scored = score_windows(df, lulls, (60, 80), ScoringConfig())
    candidates = select_top_n(scored, lulls, 2)  # NMS disabled (default 0)
    assert candidates[0].window_id == "early"
    assert candidates[0].rank == 1
    assert candidates[1].window_id == "late"


def test_select_top_n_caps_at_available():
    df = _per_window([("a", 0, 70, 50)])
    scored = score_windows(df, {"a": []}, (60, 80), ScoringConfig())
    candidates = select_top_n(scored, {"a": []}, 5)
    assert len(candidates) == 1
    assert candidates[0].rank == 1


def test_select_top_n_empty():
    df = pd.DataFrame(
        columns=[
            "start", "end", "peak_bi", "time_of_peak", "mean_bi",
            "n_hours", "n_valid_hours", "score", "bi_distance_normalized",
            "n_lulls", "total_lull_hours",
        ]
    )
    assert select_top_n(df, {}, 5) == []


def test_select_top_n_nms_suppresses_near_duplicates():
    """Five identical-score windows starting one hour apart → NMS collapses to one."""
    df = _per_window(
        [
            (f"w_{h:02d}", h, 70.0, 50.0)
            for h in range(0, 5)            # 0..4 starts, 1 h apart
        ]
    )
    lulls = {wid: [] for wid in df.index}
    scored = score_windows(df, lulls, (60, 80), ScoringConfig())
    candidates = select_top_n(scored, lulls, n=5, min_separation_hours=12)
    # All 5 windows start within 4 hours of the first; min_sep=12 ⇒ only 1 survives.
    assert len(candidates) == 1
    assert candidates[0].window_id == "w_00"


def test_select_top_n_nms_allows_separated_windows():
    """Two clusters of identical windows, 48 h apart → one per cluster."""
    df = _per_window(
        [
            ("cluster1_a", 0, 70.0, 50.0),
            ("cluster1_b", 1, 70.0, 50.0),
            ("cluster2_a", 48, 70.0, 50.0),
            ("cluster2_b", 49, 70.0, 50.0),
        ]
    )
    lulls = {wid: [] for wid in df.index}
    scored = score_windows(df, lulls, (60, 80), ScoringConfig())
    candidates = select_top_n(scored, lulls, n=5, min_separation_hours=12)
    # First from each cluster (earlier start tie-break) survives.
    assert [c.window_id for c in candidates] == ["cluster1_a", "cluster2_a"]


def test_select_top_n_nms_zero_disables():
    df = _per_window(
        [(f"w_{h}", h, 70.0, 50.0) for h in range(3)]
    )
    lulls = {wid: [] for wid in df.index}
    scored = score_windows(df, lulls, (60, 80), ScoringConfig())
    candidates = select_top_n(scored, lulls, n=3, min_separation_hours=0)
    assert len(candidates) == 3
