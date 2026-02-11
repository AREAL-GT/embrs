"""Tests for _aggregate_ensemble_predictions.

Verifies that the incremental O(n) aggregation correctly computes
cumulative burn probabilities and per-cell statistics.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional

from embrs.utilities.data_classes import PredictionOutput
from embrs.tools.fire_predictor import _aggregate_ensemble_predictions


def _make_prediction(spread, flame_len_m=None, fli_kw_m=None, ros_ms=None,
                     spread_dir=None, crown_fire=None, hold_probs=None,
                     breaches=None):
    """Helper to build a PredictionOutput with sensible defaults."""
    all_locs = set()
    for locs in spread.values():
        all_locs.update(locs)

    if flame_len_m is None:
        flame_len_m = {loc: 1.0 for loc in all_locs}
    if fli_kw_m is None:
        fli_kw_m = {loc: 10.0 for loc in all_locs}
    if ros_ms is None:
        ros_ms = {loc: 0.5 for loc in all_locs}
    if spread_dir is None:
        spread_dir = {loc: 0.0 for loc in all_locs}
    if crown_fire is None:
        crown_fire = {}
    if hold_probs is None:
        hold_probs = {}
    if breaches is None:
        breaches = {}

    return PredictionOutput(
        spread=spread,
        flame_len_m=flame_len_m,
        fli_kw_m=fli_kw_m,
        ros_ms=ros_ms,
        spread_dir=spread_dir,
        crown_fire=crown_fire,
        hold_probs=hold_probs,
        breaches=breaches,
        active_fire_front={},
        burnt_spread={},
    )


class TestAggregateBurnProbability:
    """Tests for cumulative burn probability computation."""

    def test_single_member_all_burn(self):
        """Single ensemble member: every burned cell has probability 1.0."""
        pred = _make_prediction(spread={
            10: [(0, 0)],
            20: [(1, 1)],
        })
        result = _aggregate_ensemble_predictions([pred])

        assert result.burn_probability[10] == {(0, 0): 1.0}
        assert result.burn_probability[20] == {(0, 0): 1.0, (1, 1): 1.0}

    def test_two_members_partial_overlap(self):
        """Two members with partial overlap give correct probabilities."""
        pred_a = _make_prediction(spread={
            10: [(0, 0), (1, 1)],
        })
        pred_b = _make_prediction(spread={
            10: [(0, 0)],
            20: [(2, 2)],
        })
        result = _aggregate_ensemble_predictions([pred_a, pred_b])

        # At t=10: (0,0) burned in both => 1.0, (1,1) only in A => 0.5
        assert result.burn_probability[10][(0, 0)] == 1.0
        assert result.burn_probability[10][(1, 1)] == 0.5

        # At t=20: cumulative, so (0,0)=1.0, (1,1)=0.5, (2,2)=0.5
        assert result.burn_probability[20][(0, 0)] == 1.0
        assert result.burn_probability[20][(1, 1)] == 0.5
        assert result.burn_probability[20][(2, 2)] == 0.5

    def test_cumulative_semantics(self):
        """Burn probability is cumulative — cells burned earlier stay burned."""
        pred = _make_prediction(spread={
            10: [(0, 0)],
            30: [(1, 1)],
        })
        result = _aggregate_ensemble_predictions([pred])

        # (0,0) burned at t=10, should still show at t=30
        assert (0, 0) in result.burn_probability[30]
        assert result.burn_probability[30][(0, 0)] == 1.0
        assert result.burn_probability[30][(1, 1)] == 1.0

    def test_duplicate_locations_across_timesteps(self):
        """A location appearing in multiple timesteps of the same member
        should only be counted once."""
        pred = _make_prediction(spread={
            10: [(0, 0)],
            20: [(0, 0), (1, 1)],  # (0,0) repeated
        })
        result = _aggregate_ensemble_predictions([pred])

        assert result.burn_probability[10][(0, 0)] == 1.0
        assert result.burn_probability[20][(0, 0)] == 1.0
        assert result.burn_probability[20][(1, 1)] == 1.0

    def test_three_members_varying_agreement(self):
        """Three members with varying overlap."""
        pred_a = _make_prediction(spread={10: [(0, 0), (1, 1), (2, 2)]})
        pred_b = _make_prediction(spread={10: [(0, 0), (1, 1)]})
        pred_c = _make_prediction(spread={10: [(0, 0)]})
        result = _aggregate_ensemble_predictions([pred_a, pred_b, pred_c])

        probs = result.burn_probability[10]
        assert abs(probs[(0, 0)] - 1.0) < 1e-9
        assert abs(probs[(1, 1)] - 2.0 / 3.0) < 1e-9
        assert abs(probs[(2, 2)] - 1.0 / 3.0) < 1e-9

    def test_empty_predictions(self):
        """Members with no spread produce empty probability maps."""
        pred = _make_prediction(spread={})
        result = _aggregate_ensemble_predictions([pred])
        assert result.burn_probability == {}


class TestAggregateStatistics:
    """Tests for per-cell statistics aggregation."""

    def test_flame_len_stats(self):
        """Flame length statistics are computed correctly."""
        pred_a = _make_prediction(
            spread={10: [(0, 0)]},
            flame_len_m={(0, 0): 2.0},
        )
        pred_b = _make_prediction(
            spread={10: [(0, 0)]},
            flame_len_m={(0, 0): 4.0},
        )
        result = _aggregate_ensemble_predictions([pred_a, pred_b])

        stats = result.flame_len_m_stats[(0, 0)]
        assert stats.mean == pytest.approx(3.0)
        assert stats.min == pytest.approx(2.0)
        assert stats.max == pytest.approx(4.0)
        assert stats.count == 2

    def test_n_ensemble(self):
        """n_ensemble matches number of predictions."""
        preds = [_make_prediction(spread={10: [(0, 0)]}) for _ in range(5)]
        result = _aggregate_ensemble_predictions(preds)
        assert result.n_ensemble == 5
