"""Regression tests for FirePredictor._set_states cell-state contract.

Pins two bugs found in the prediction stack (2026-07-02):

1. Fire-break width doubling: worker-deserialized predictors set
   ``_skip_first_reset`` and skipped the ``reset_to_fuel()`` loop (which is
   what zeros ``_break_width``), but ``_set_states`` still called
   ``_set_firebreaks()`` unconditionally. ``_apply_firebreak`` accumulates
   (``+=``), so every fire break came out at twice its configured width —
   pushing Mees hold probabilities to ~1 and silencing predicted fireline
   breaches. The contract: after any sequence of ``_set_states()`` calls,
   with or without the skip flag, each break cell's ``_break_width`` equals
   the configured width exactly.

2. Per-member dead fuel moisture never reaching cells: cell ``fmois`` arrays
   are baked from ``sim_params.init_mf`` at grid construction and
   ``reset_to_fuel()`` restores that template, so the per-member ``dead_mf``
   set by ``_apply_member_params`` (particle filter / IRPG samples) was
   never used by the fire behavior calculations. The contract: after
   ``_set_states()``, every burnable cell's dead moisture classes equal the
   predictor's current ``dead_mf``.

Also covers the related reset-path contract that road break widths are
re-applied after ``reset_to_fuel()`` zeroes them (previously they were
silently dropped on any same-process re-run).
"""

import numpy as np
import pytest

from embrs.base_classes.control_handler import ControlActionHandler
from embrs.fire_simulator.cell import Cell
from embrs.models.fuel_models import ScottBurgan40
from embrs.tools.fire_predictor import FirePredictor
from embrs.utilities.data_classes import CellData, StateEstimate
from embrs.utilities.fire_util import UtilFuncs

from shapely.geometry import LineString


BREAK_WIDTH_M = 2.5
ROAD_WIDTH_M = 1.5
TEMPLATE_DEAD_MF = 0.08


def _make_cell(cell_id: int, x: float, y: float, fuel_num: int = 105) -> Cell:
    """Build a real, standalone Cell with ScottBurgan fuel."""
    fuel = ScottBurgan40(fuel_num)
    cd = CellData(
        fuel_type=fuel, elevation=0.0, aspect=0.0, slope_deg=0.0,
        canopy_cover=0.0, canopy_height=0.0, canopy_base_height=0.0,
        canopy_bulk_density=0.0, init_dead_mf=[TEMPLATE_DEAD_MF] * 3,
        live_h_mf=0.3, live_w_mf=0.6,
    )
    cell = Cell(cell_id, x, y, 30)
    cell._set_cell_data(cd)
    cell.directions, _, cell.end_pts = UtilFuncs.get_ign_parameters(
        0, cell.cell_size
    )
    return cell


class _StubGridManager:
    """Grid manager stub: maps any fire-break geometry to fixed cells."""

    def __init__(self, break_cells):
        self._break_cells = list(break_cells)

    def get_cells_at_geometry(self, geom):
        return list(self._break_cells)


@pytest.fixture
def predictor_with_cells():
    """A minimally-populated FirePredictor with real cells.

    Bypasses __init__ (which needs a full FireSim) and sets exactly the
    state _set_states touches: the cell templates, a real
    ControlActionHandler for fire-break application, one fire break, one
    road, and the member fuel moisture scalar.

    Returns:
        Tuple of (predictor, break_cell, road_cell, plain_cell).
    """
    break_cell = _make_cell(0, 15.0, 15.0)
    road_cell = _make_cell(1, 75.0, 15.0, fuel_num=103)
    plain_cell = _make_cell(2, 135.0, 15.0)
    cells = [break_cell, road_cell, plain_cell]

    predictor = FirePredictor.__new__(FirePredictor)

    # Cell templates, exactly as PredictorSerializer.set_state provides them.
    predictor.orig_grid = np.empty((1, 3), dtype=object)
    predictor.orig_dict = {c.id: c for c in cells}
    predictor._grid_manager = None

    # Real control handler so _set_firebreaks exercises the real
    # accumulating _apply_firebreak.
    predictor._control_handler = ControlActionHandler(
        grid_manager=_StubGridManager([break_cell]),
        cell_size=30.0,
        time_step=30.0,
        fuel_class_factory=ScottBurgan40,
    )
    predictor._fire_breaks = [
        (LineString([(0.0, 15.0), (30.0, 15.0)]), BREAK_WIDTH_M, "fb0"),
    ]

    # One narrow road through road_cell; _set_roads resolves cells via
    # get_cell_from_xy on the predictor.
    predictor._roads = [(([75.0], [15.0]), None, ROAD_WIDTH_M)]
    predictor._cell_size = 30.0
    predictor.get_cell_from_xy = (
        lambda x, y, oob_ok=False: road_cell
        if (x, y) == (75.0, 15.0) else None
    )

    # Member fuel moisture (what _apply_member_params would set).
    predictor.dead_mf = 0.13

    for cell in cells:
        cell.set_parent(predictor)

    # Simulate the template state after grid construction: breaks and
    # roads applied once by BaseFireSim.__init__.
    break_cell._break_width = BREAK_WIDTH_M
    road_cell._break_width = ROAD_WIDTH_M

    return predictor, break_cell, road_cell, plain_cell


def _empty_estimate() -> StateEstimate:
    return StateEstimate(burnt_polys=[], burning_polys=[], start_time_s=0.0)


def _dead_moistures(cell: Cell) -> list:
    """The cell's dead-class moisture values (loaded classes only)."""
    idx = cell._fuel.rel_indices
    return list(cell.fmois[idx[idx < 4]])


class TestBreakWidthReset:
    """Break widths must equal configured widths after every _set_states."""

    def test_skip_first_reset_does_not_double_break_widths(
        self, predictor_with_cells
    ):
        """Worker path: template cells already carry break widths."""
        predictor, break_cell, road_cell, _ = predictor_with_cells
        predictor._skip_first_reset = True

        predictor._set_states(_empty_estimate())

        assert break_cell._break_width == pytest.approx(BREAK_WIDTH_M)
        assert road_cell._break_width == pytest.approx(ROAD_WIDTH_M)

    def test_reset_path_reapplies_breaks_and_roads_once(
        self, predictor_with_cells
    ):
        """Reset path: reset_to_fuel zeroes widths, both get re-applied."""
        predictor, break_cell, road_cell, _ = predictor_with_cells

        predictor._set_states(_empty_estimate())

        assert break_cell._break_width == pytest.approx(BREAK_WIDTH_M)
        assert road_cell._break_width == pytest.approx(ROAD_WIDTH_M)

    def test_widths_stable_across_repeated_calls(self, predictor_with_cells):
        """Worker first run followed by re-runs never accumulates width."""
        predictor, break_cell, road_cell, _ = predictor_with_cells
        predictor._skip_first_reset = True

        for _ in range(3):
            predictor._set_states(_empty_estimate())
            assert break_cell._break_width == pytest.approx(BREAK_WIDTH_M)
            assert road_cell._break_width == pytest.approx(ROAD_WIDTH_M)


class TestMemberMoistureApplied:
    """Cell fmois dead classes must track the predictor's dead_mf."""

    def test_member_dead_mf_reaches_cells_on_skip_path(
        self, predictor_with_cells
    ):
        predictor, break_cell, road_cell, plain_cell = predictor_with_cells
        predictor._skip_first_reset = True

        predictor._set_states(_empty_estimate())

        for cell in (break_cell, road_cell, plain_cell):
            assert _dead_moistures(cell) == pytest.approx([0.13] * len(
                _dead_moistures(cell)))

    def test_member_dead_mf_reaches_cells_on_reset_path(
        self, predictor_with_cells
    ):
        predictor, _, _, plain_cell = predictor_with_cells

        predictor._set_states(_empty_estimate())

        assert _dead_moistures(plain_cell) == pytest.approx(
            [0.13] * len(_dead_moistures(plain_cell)))

    def test_dead_mf_tracks_member_changes(self, predictor_with_cells):
        """A new member's dead_mf replaces the previous member's value."""
        predictor, _, _, plain_cell = predictor_with_cells

        predictor._set_states(_empty_estimate())
        predictor.dead_mf = 0.11
        predictor._set_states(_empty_estimate())

        assert _dead_moistures(plain_cell) == pytest.approx(
            [0.11] * len(_dead_moistures(plain_cell)))

    def test_live_moisture_and_template_untouched(self, predictor_with_cells):
        """Live classes keep template values; _init_fmois stays intact."""
        predictor, _, _, plain_cell = predictor_with_cells
        template = plain_cell._init_fmois.copy()
        live_before = plain_cell.fmois[4:].copy()

        predictor._set_states(_empty_estimate())

        assert np.array_equal(plain_cell._init_fmois, template)
        assert np.array_equal(plain_cell.fmois[4:], live_before)


class TestStaleWindForecastCleared:
    """Template cells carry a 1-entry zero wind forecast from the
    prediction-mode placeholder wind field at grid construction.
    Cell.curr_wind's lazy refresh (len(series) <= w_idx) never replaces a
    1-entry series at w_idx=0, so _set_states must clear it on every path —
    otherwise cells igniting in a member's first weather interval burn at
    zero wind."""

    def _seed_stale_forecast(self, cells):
        for cell in cells:
            cell.forecast_wind_speeds = np.zeros(1)
            cell.forecast_wind_dirs = np.zeros(1)

    def test_cleared_on_skip_path(self, predictor_with_cells):
        predictor, break_cell, road_cell, plain_cell = predictor_with_cells
        cells = (break_cell, road_cell, plain_cell)
        self._seed_stale_forecast(cells)
        predictor._skip_first_reset = True

        predictor._set_states(_empty_estimate())

        for cell in cells:
            assert len(cell.forecast_wind_speeds) == 0
            assert len(cell.forecast_wind_dirs) == 0

    def test_cleared_on_reset_path(self, predictor_with_cells):
        predictor, _, _, plain_cell = predictor_with_cells
        self._seed_stale_forecast([plain_cell])

        predictor._set_states(_empty_estimate())

        assert len(plain_cell.forecast_wind_speeds) == 0
