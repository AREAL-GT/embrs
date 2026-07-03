"""Tests for FirePredictor._roll_breach and the firebreaks_always_hold mode.

The default mode draws a Bernoulli sample against the Mees hold probability
(``breached = rand > hold_prob``). With ``firebreaks_always_hold`` set, the
break never breaches and no breach-RNG draw is consumed — hold probabilities
are still recorded so consumers can score breach risk analytically
(``1 - prod(hold_i)`` over ignited fireline cells).
"""

import numpy as np
import pytest

from embrs.fire_simulator.cell import Cell
from embrs.models.fuel_models import ScottBurgan40
from embrs.tools.fire_predictor import FirePredictor
from embrs.utilities.data_classes import CellData, PredictorParams
from embrs.utilities.fire_util import UtilFuncs


BREAK_WIDTH_M = 2.5
FLAME_LEN_M = 2.6  # hold ~0.81 on a 2.5 m break — comfortably in (0, 1)


class _FixedRng:
    """RNG stub returning a fixed value and counting draws."""

    def __init__(self, value: float):
        self.value = value
        self.n_draws = 0

    def random(self) -> float:
        self.n_draws += 1
        return self.value


def _make_break_cell() -> Cell:
    fuel = ScottBurgan40(105)
    cd = CellData(
        fuel_type=fuel, elevation=0.0, aspect=0.0, slope_deg=0.0,
        canopy_cover=0.0, canopy_height=0.0, canopy_base_height=0.0,
        canopy_bulk_density=0.0, init_dead_mf=[0.08] * 3,
        live_h_mf=0.3, live_w_mf=0.6,
    )
    cell = Cell(0, 15.0, 15.0, 30)
    cell._set_cell_data(cd)
    cell.directions, _, cell.end_pts = UtilFuncs.get_ign_parameters(
        0, cell.cell_size
    )
    cell._break_width = BREAK_WIDTH_M
    return cell


def _make_predictor(always_hold: bool, rng) -> FirePredictor:
    predictor = FirePredictor.__new__(FirePredictor)
    predictor.hold_probs = {}
    predictor.breaches = {}
    predictor.firebreaks_always_hold = always_hold
    predictor._rng_breach = rng
    return predictor


class TestDefaultBernoulliMode:
    """Default mode preserves the roll semantics: breached = rand > hold."""

    def test_high_draw_breaches(self):
        cell = _make_break_cell()
        rng = _FixedRng(0.999)
        predictor = _make_predictor(always_hold=False, rng=rng)

        predictor._roll_breach(cell, FLAME_LEN_M)

        assert bool(cell.breached) is True
        assert rng.n_draws == 1
        assert bool(predictor.breaches[(cell.x_pos, cell.y_pos)]) is True

    def test_low_draw_holds(self):
        cell = _make_break_cell()
        rng = _FixedRng(0.0)
        predictor = _make_predictor(always_hold=False, rng=rng)

        predictor._roll_breach(cell, FLAME_LEN_M)

        assert bool(cell.breached) is False
        assert rng.n_draws == 1

    def test_hold_prob_matches_mees(self):
        cell = _make_break_cell()
        predictor = _make_predictor(always_hold=False, rng=_FixedRng(0.5))

        predictor._roll_breach(cell, FLAME_LEN_M)

        expected = cell.calc_hold_prob(FLAME_LEN_M)
        assert predictor.hold_probs[(cell.x_pos, cell.y_pos)] == pytest.approx(
            expected)
        assert 0.0 < expected < 1.0  # test setup keeps it non-degenerate


class TestFirebreaksAlwaysHold:
    """Force-hold mode: never breaches, no RNG draw, holds still recorded."""

    def test_never_breaches_and_consumes_no_rng(self):
        cell = _make_break_cell()
        rng = _FixedRng(0.999)  # would breach in default mode
        predictor = _make_predictor(always_hold=True, rng=rng)

        predictor._roll_breach(cell, FLAME_LEN_M)

        assert bool(cell.breached) is False
        assert rng.n_draws == 0
        assert bool(predictor.breaches[(cell.x_pos, cell.y_pos)]) is False

    def test_hold_prob_still_recorded(self):
        cell = _make_break_cell()
        predictor = _make_predictor(always_hold=True, rng=_FixedRng(0.5))

        predictor._roll_breach(cell, FLAME_LEN_M)

        assert predictor.hold_probs[(cell.x_pos, cell.y_pos)] == pytest.approx(
            cell.calc_hold_prob(FLAME_LEN_M))


class TestResolveFirelineIgnition:
    """_resolve_fireline_ignition: roll + diagnostics for ignited break cells.

    Used by BOTH branches of _init_iteration. Starting cells (iters == 0)
    previously bypassed the breach roll entirely — they kept the default
    breached=True and propagated through the line unrolled and unscored."""

    def _make_predictor(self, always_hold, rng):
        predictor = _make_predictor(always_hold=always_hold, rng=rng)
        predictor.fireline_cell_diagnostics = {}
        return predictor

    def test_records_hold_and_diagnostics(self):
        cell = _make_break_cell()
        predictor = self._make_predictor(always_hold=False, rng=_FixedRng(0.5))

        predictor._resolve_fireline_ignition(cell, FLAME_LEN_M)

        key = (cell.x_pos, cell.y_pos)
        assert key in predictor.hold_probs
        assert key in predictor.breaches
        diag = predictor.fireline_cell_diagnostics[key]
        assert diag[0] == 105          # fuel model
        assert diag[1] == pytest.approx(0.08)  # dead_mf_1hr

    def test_force_hold_never_breaches(self):
        cell = _make_break_cell()
        rng = _FixedRng(0.999)
        predictor = self._make_predictor(always_hold=True, rng=rng)

        predictor._resolve_fireline_ignition(cell, FLAME_LEN_M)

        assert bool(cell.breached) is False
        assert rng.n_draws == 0


class TestParamsPlumbing:
    """firebreaks_always_hold flows through the member-params path."""

    def test_apply_member_params_sets_flag(self):
        predictor = FirePredictor.__new__(FirePredictor)
        # _apply_member_params needs c_size for its cell-size warning check
        predictor.c_size = 30.0
        params = PredictorParams(cell_size_m=30.0, firebreaks_always_hold=True)

        predictor._apply_member_params(params)

        assert predictor.firebreaks_always_hold is True

    def test_default_is_off(self):
        assert PredictorParams().firebreaks_always_hold is False
