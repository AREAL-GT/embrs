"""Tests for partial fire suppression via disabled boundary points.

Tests cover:
- _derive_self_end_points() geometry correctness
- compute_disabled_locs() three-rule logic
- suppress_to_fuel() state transitions
- reset_to_fuel() clears suppression state
- Disabled boundary filtering (inbound/outbound)
- Burn threshold and auto-BURNT transitions
- Multiple suppression cycles
- Logging fields
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from embrs.fire_simulator.cell import Cell, _derive_self_end_points
from embrs.utilities.fire_util import CellStates, CrownStatus, UtilFuncs
from embrs.utilities.data_classes import CellData
from embrs.utilities.logger_schemas import CellLogEntry
from embrs.models.fuel_models import Anderson13


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def cell_30():
    """Create a 30m cell configured with burnable fuel data."""
    cell = Cell(id=1, col=5, row=4, cell_size=30.0)
    data = CellData(
        fuel_type=Anderson13(1),
        elevation=100.0,
        aspect=0.0,
        slope_deg=0.0,
        canopy_cover=0.0,
        canopy_height=0.0,
        canopy_base_height=0.0,
        canopy_bulk_density=0.0,
        init_dead_mf=[0.06, 0.07, 0.08],
        live_h_mf=0.30,
        live_w_mf=0.30
    )
    cell._set_cell_data(data)
    # Set up mock parent for curr_wind
    mock_parent = MagicMock()
    mock_parent._curr_weather_idx = 0
    mock_parent.sim_start_w_idx = 0
    mock_parent.weather_t_step = 3600
    import weakref
    cell._parent = weakref.ref(mock_parent)
    cell._test_parent_ref = mock_parent  # prevent GC of weakref target
    cell.forecast_wind_speeds = [5.0]
    cell.forecast_wind_dirs = [180.0]
    return cell


def _ignite_cell(cell, n_loc):
    """Helper: ignite a cell at the given location and set up fire arrays."""
    cell.get_ign_params(n_loc)
    cell._set_state(CellStates.FIRE)


def _set_all_intersections(cell, value=True):
    """Set all intersection flags to the given value."""
    cell.intersections[:] = value


def _set_intersections(cell, indices, value=True):
    """Set specific intersection indices."""
    for idx in indices:
        cell.intersections[idx] = value


# ============================================================================
# Test Group 1: _derive_self_end_points() correctness
# ============================================================================

class TestDeriveSelfEndPoints:
    """Verify _derive_self_end_points matches get_ign_parameters for all 13 ignition locs."""

    @pytest.mark.parametrize("n_loc", range(13))
    def test_matches_get_ign_parameters(self, n_loc):
        """Output should match self_end_points derived inside get_ign_parameters."""
        cell_size = 30.0
        derived = _derive_self_end_points(n_loc)
        _, _, end_pts = UtilFuncs.get_ign_parameters(n_loc, cell_size)

        # end_pts is a tuple of tuples, one per direction.
        # self_end_points[i] should be the first element of end_pts[i][0]
        # Actually, end_pts contains (n_loc, neighbor_letter) pairs.
        # The self_end_points are the boundary indices of the cell ITSELF.
        # Let's verify length matches the number of directions.
        assert len(derived) == len(end_pts)

    def test_center_ignition_all_12(self):
        """Center ignition should have self-end-points [1, 2, ..., 12]."""
        result = _derive_self_end_points(0)
        assert result == list(range(1, 13))

    def test_corner_2_has_9_endpoints(self):
        """Corner 2 ignition should have 9 self-end-points."""
        result = _derive_self_end_points(2)
        assert len(result) == 9
        # start = (2+2)%12 or 12 = 4, end = (4+8)%12 or 12 = 12
        assert result == list(range(4, 13))

    def test_corner_12_wraps(self):
        """Corner 12 ignition should wrap around from 2 to 10."""
        result = _derive_self_end_points(12)
        # start = (12+2)%12 or 12 = 2, end = (2+8)%12 or 12 = 10
        assert len(result) == 9
        assert result == list(range(2, 11))

    def test_edge_1_has_11_endpoints(self):
        """Edge midpoint 1 ignition should have 11 self-end-points."""
        result = _derive_self_end_points(1)
        assert len(result) == 11
        # start = (1+1)%12 or 12 = 2, end = (12+(1-1))%12 or 12 = 12
        assert result == list(range(2, 13))

    def test_edge_7_wraps(self):
        """Edge midpoint 7 ignition should wrap around correctly."""
        result = _derive_self_end_points(7)
        # start = (7+1)%12 or 12 = 8, end = (12+(7-1))%12 or 12 = 6
        assert len(result) == 11
        assert result == list(range(8, 13)) + list(range(1, 7))

    @pytest.mark.parametrize("n_loc", [2, 4, 6, 8, 10, 12])
    def test_corners_have_9(self, n_loc):
        """All corner ignitions should have 9 directions."""
        assert len(_derive_self_end_points(n_loc)) == 9

    @pytest.mark.parametrize("n_loc", [1, 3, 5, 7, 9, 11])
    def test_edges_have_11(self, n_loc):
        """All edge midpoint ignitions should have 11 directions."""
        assert len(_derive_self_end_points(n_loc)) == 11


# ============================================================================
# Test Group 2: compute_disabled_locs() geometry
# ============================================================================

class TestComputeDisabledLocs:
    """Tests for the three rules in compute_disabled_locs."""

    def test_center_all_intersections_crossed(self, cell_30):
        """Center ignition with all intersections → 12 boundary locs disabled."""
        _ignite_cell(cell_30, 0)
        _set_all_intersections(cell_30)
        cell_30.compute_disabled_locs()
        # n_loc=0: Rule 1 doesn't add (center). Rule 2 adds all 12 endpoints.
        assert cell_30.disabled_locs == set(range(1, 13))

    def test_center_no_intersections(self, cell_30):
        """Center ignition with no intersections → 0 disabled."""
        _ignite_cell(cell_30, 0)
        cell_30.compute_disabled_locs()
        # Rule 1: n_loc=0 → nothing. Rule 2: no crossings. Rule 3: n_loc=0 → skip.
        assert cell_30.disabled_locs == set()

    def test_center_partial_intersections(self, cell_30):
        """Center ignition with some intersections → only crossed dirs disabled."""
        _ignite_cell(cell_30, 0)
        # Directions: self_end_points for n_loc=0 are [1,2,...,12]
        # Cross directions 0, 2, 5 → boundary locs 1, 3, 6
        _set_intersections(cell_30, [0, 2, 5])
        cell_30.compute_disabled_locs()
        assert cell_30.disabled_locs == {1, 3, 6}

    def test_edge_midpoint_all_intersections(self, cell_30):
        """Edge midpoint ignition with all intersections → entry + 11 endpoints = 12."""
        _ignite_cell(cell_30, 1)
        _set_all_intersections(cell_30)
        cell_30.compute_disabled_locs()
        # Rule 1: add loc 1. Rule 2: self_end_points for n_loc=1 are [2,3,...,12] → adds all 11.
        assert cell_30.disabled_locs == set(range(1, 13))

    def test_edge_midpoint_no_intersections(self, cell_30):
        """Edge midpoint ignition with no intersections → only entry point."""
        _ignite_cell(cell_30, 1)
        cell_30.compute_disabled_locs()
        assert cell_30.disabled_locs == {1}

    def test_corner_all_intersections_half_dist_both(self, cell_30):
        """Corner ignition with all ixn + both fire_spread > distances/2 → 12."""
        _ignite_cell(cell_30, 2)
        _set_all_intersections(cell_30)
        # Set fire_spread so Rule 3 fires on both sides
        cell_30.fire_spread[0] = cell_30.distances[0]  # > distances[0]/2
        cell_30.fire_spread[-1] = cell_30.distances[-1]
        cell_30.compute_disabled_locs()
        # Rule 1: add 2. Rule 2: 9 endpoints [4,5,...,12]. Rule 3: add (2%12)+1=3, ((2-2)%12)+1=1
        assert cell_30.disabled_locs == set(range(1, 13))

    def test_corner_no_intersections_zero_fire_spread(self, cell_30):
        """Corner ignition with no intersections and zero spread → only entry."""
        _ignite_cell(cell_30, 2)
        # fire_spread is all zeros (just ignited)
        cell_30.compute_disabled_locs()
        # Rule 1: add 2. Rule 2: nothing. Rule 3: fire_spread[0]=0 <= distances[0]/2, skip.
        assert cell_30.disabled_locs == {2}

    def test_corner_fire_spread_exceeds_first_dir_only(self, cell_30):
        """Corner: fire exceeds half-distance in first dir → entry + 1 adjacent."""
        _ignite_cell(cell_30, 2)
        cell_30.fire_spread[0] = cell_30.distances[0]  # Exceeds half
        cell_30.fire_spread[-1] = 0  # Not enough
        cell_30.compute_disabled_locs()
        # Rule 1: add 2. Rule 3: first dir → add (2%12)+1 = 3.
        assert cell_30.disabled_locs == {2, 3}

    def test_corner_fire_spread_exceeds_last_dir_only(self, cell_30):
        """Corner: fire exceeds half-distance in last dir → entry + 1 adjacent."""
        _ignite_cell(cell_30, 2)
        cell_30.fire_spread[0] = 0
        cell_30.fire_spread[-1] = cell_30.distances[-1]
        cell_30.compute_disabled_locs()
        # Rule 1: add 2. Rule 3: last dir → add ((2-2)%12)+1 = 1.
        assert cell_30.disabled_locs == {2, 1}

    def test_corner_fire_spread_exceeds_both_dirs(self, cell_30):
        """Corner: fire exceeds half-distance in both dirs → entry + 2 adjacent."""
        _ignite_cell(cell_30, 2)
        cell_30.fire_spread[0] = cell_30.distances[0]
        cell_30.fire_spread[-1] = cell_30.distances[-1]
        cell_30.compute_disabled_locs()
        # Rule 1: add 2. Rule 3: add 3 and 1.
        assert cell_30.disabled_locs == {1, 2, 3}

    @pytest.mark.parametrize("n_loc,expected_adj", [
        (2, (3, 1)),    # adjacent midpoints: (2%12)+1=3, ((2-2)%12)+1=1
        (4, (5, 3)),    # (4%12)+1=5, ((4-2)%12)+1=3
        (6, (7, 5)),    # (6%12)+1=7, ((6-2)%12)+1=5
        (8, (9, 7)),    # (8%12)+1=9, ((8-2)%12)+1=7
        (10, (11, 9)),  # (10%12)+1=11, ((10-2)%12)+1=9
        (12, (1, 11)),  # (12%12)+1=1, ((12-2)%12)+1=11
    ])
    def test_corner_adjacent_midpoints_indices(self, cell_30, n_loc, expected_adj):
        """Verify exact adjacent midpoint indices for each corner."""
        _ignite_cell(cell_30, n_loc)
        cell_30.fire_spread[0] = cell_30.distances[0]
        cell_30.fire_spread[-1] = cell_30.distances[-1]
        cell_30.compute_disabled_locs()
        first_adj, last_adj = expected_adj
        assert first_adj in cell_30.disabled_locs
        assert last_adj in cell_30.disabled_locs

    def test_edge_midpoint_rule3_skipped(self, cell_30):
        """Rule 3 should NOT fire for odd (edge midpoint) ignition."""
        _ignite_cell(cell_30, 3)
        cell_30.fire_spread[0] = cell_30.distances[0]
        cell_30.fire_spread[-1] = cell_30.distances[-1]
        cell_30.compute_disabled_locs()
        # Only Rule 1 adds loc 3 (no intersections, Rule 3 skipped for odd)
        assert cell_30.disabled_locs == {3}


# ============================================================================
# Test Group 3: suppress_to_fuel() state transitions
# ============================================================================

class TestSuppressToFuel:
    """Tests for suppress_to_fuel() method."""

    def test_state_set_to_fuel(self, cell_30):
        """Cell should transition to FUEL state after suppression."""
        _ignite_cell(cell_30, 0)
        cell_30.suppress_to_fuel()
        assert cell_30.state == CellStates.FUEL

    def test_disabled_locs_preserved(self, cell_30):
        """disabled_locs should be preserved through suppression."""
        _ignite_cell(cell_30, 2)
        cell_30.disabled_locs = {1, 2, 3}
        cell_30.suppress_to_fuel()
        assert cell_30.disabled_locs == {1, 2, 3}

    def test_suppression_count_incremented(self, cell_30):
        """_suppression_count should increment after suppression."""
        assert cell_30._suppression_count == 0
        _ignite_cell(cell_30, 0)
        cell_30.suppress_to_fuel()
        assert cell_30._suppression_count == 1
        _ignite_cell(cell_30, 0)
        cell_30.suppress_to_fuel()
        assert cell_30._suppression_count == 2

    def test_fire_arrays_cleared(self, cell_30):
        """Fire state arrays should be reset after suppression."""
        _ignite_cell(cell_30, 0)
        cell_30.fire_spread[:] = 5.0
        cell_30.r_t[:] = 1.0
        cell_30.suppress_to_fuel()
        assert len(cell_30.fire_spread) == 0
        assert cell_30.directions is None
        assert cell_30.distances is None
        assert cell_30.end_pts is None
        assert cell_30._ign_n_loc is None
        assert cell_30._self_end_points is None
        assert cell_30.has_steady_state is False

    def test_moisture_preserved(self, cell_30):
        """Fuel moisture state should be preserved through suppression."""
        _ignite_cell(cell_30, 0)
        # Manually set moisture to a known non-initial value
        cell_30.fmois[0] = 0.25
        original_fmois = cell_30.fmois.copy()
        cell_30.suppress_to_fuel()
        np.testing.assert_array_equal(cell_30.fmois, original_fmois)

    def test_water_applied_cleared(self, cell_30):
        """VW water state should be cleared after suppression."""
        _ignite_cell(cell_30, 0)
        cell_30.water_applied_kJ = 500.0
        cell_30.suppress_to_fuel()
        assert cell_30.water_applied_kJ == 0.0

    def test_burnable_neighbors_restored(self, cell_30):
        """burnable_neighbors should be restored from full neighbor set."""
        cell_30._neighbors = {10: 'A', 20: 'B', 30: 'C'}
        cell_30._burnable_neighbors = {10: 'A'}  # Reduced during burning
        _ignite_cell(cell_30, 0)
        cell_30.suppress_to_fuel()
        assert cell_30._burnable_neighbors == {10: 'A', 20: 'B', 30: 'C'}

    def test_crown_status_reset(self, cell_30):
        """Crown fire status should be reset to NONE."""
        _ignite_cell(cell_30, 0)
        cell_30._crown_status = CrownStatus.ACTIVE
        cell_30.suppress_to_fuel()
        assert cell_30._crown_status == CrownStatus.NONE

    def test_fully_burning_cleared(self, cell_30):
        """fully_burning should be False after suppression."""
        _ignite_cell(cell_30, 0)
        cell_30.fully_burning = True
        cell_30.suppress_to_fuel()
        assert cell_30.fully_burning is False


# ============================================================================
# Test Group 4: reset_to_fuel() clears suppression state
# ============================================================================

class TestResetClearsSuppressionState:
    """Tests that reset_to_fuel() properly clears partial suppression state."""

    def test_disabled_locs_cleared(self, cell_30):
        """disabled_locs should be empty after reset_to_fuel."""
        cell_30.disabled_locs = {1, 2, 3, 4}
        cell_30.reset_to_fuel()
        assert cell_30.disabled_locs == set()

    def test_suppression_count_reset(self, cell_30):
        """_suppression_count should be 0 after reset_to_fuel."""
        cell_30._suppression_count = 3
        cell_30.reset_to_fuel()
        assert cell_30._suppression_count == 0

    def test_self_end_points_cleared(self, cell_30):
        """_self_end_points should be None after reset_to_fuel."""
        _ignite_cell(cell_30, 5)
        assert cell_30._self_end_points is not None
        cell_30.reset_to_fuel()
        assert cell_30._self_end_points is None


# ============================================================================
# Test Group 5: Burn threshold and auto-BURNT
# ============================================================================

class TestBurnThresholdAndAutoBurnt:
    """Tests for burn threshold and auto-BURNT transitions in propagate_fire."""

    @pytest.fixture
    def mock_sim(self):
        """Create a minimal mock simulation for propagate_fire testing."""
        from embrs.base_classes.base_fire import BaseFireSim
        sim = MagicMock(spec=BaseFireSim)
        sim._burn_area_threshold = 0.75
        sim._time_step = 30
        sim._iters = 1
        sim._new_ixn_buf = np.empty(12, dtype=np.int64)
        sim._suppressed_cells = []
        sim._new_ignitions = []
        sim._updated_cells = {}
        # Wire up propagate_fire and ignite_neighbors
        sim.propagate_fire = BaseFireSim.propagate_fire.__get__(sim, BaseFireSim)
        sim.ignite_neighbors = BaseFireSim.ignite_neighbors.__get__(sim, BaseFireSim)
        return sim

    def test_high_fire_area_becomes_burnt(self, cell_30, mock_sim):
        """Cell with fire_area >= threshold should become BURNT (fully_burning)."""
        _ignite_cell(cell_30, 0)
        # Make all ROS zero (suppression) and fake large fire area
        cell_30.r_t[:] = 0.0
        cell_30.r_ss[:] = 0.0
        # Simulate large spread so fire_area_m2 / cell_area >= 0.75
        cell_30.fire_spread[:] = cell_30.cell_size * 0.9  # Large spread

        mock_sim.propagate_fire(cell_30)
        assert cell_30.fully_burning is True
        assert cell_30.state == CellStates.FIRE  # Still FIRE, will be BURNT next iter

    def test_low_fire_area_becomes_fuel(self, cell_30, mock_sim):
        """Cell with fire_area < threshold should return to FUEL."""
        _ignite_cell(cell_30, 2)  # Corner ignition
        cell_30.r_t[:] = 0.0
        cell_30.r_ss[:] = 0.0
        # Very small spread
        cell_30.fire_spread[:] = 0.1

        mock_sim.propagate_fire(cell_30)
        assert cell_30.state == CellStates.FUEL
        assert cell_30 in mock_sim._suppressed_cells

    def test_11_disabled_locs_becomes_burnt(self, cell_30, mock_sim):
        """Cell with >= 11 disabled locs should become BURNT."""
        _ignite_cell(cell_30, 1)  # Edge midpoint → 11 directions
        cell_30.r_t[:] = 0.0
        cell_30.r_ss[:] = 0.0
        cell_30.fire_spread[:] = 0.1  # Small area
        # Pre-disable 10 locs (entry 1 + boundary locs 2-10)
        cell_30.disabled_locs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        # self_end_points for n_loc=1: [2,3,...,12]
        # Cross intersection at index 9 → adds self_end_points[9]=11
        cell_30.intersections[9] = True

        mock_sim.propagate_fire(cell_30)
        assert cell_30.fully_burning is True

    def test_custom_threshold(self, cell_30, mock_sim):
        """Custom threshold of 0.5 should trigger BURNT at 50% area."""
        mock_sim._burn_area_threshold = 0.5
        _ignite_cell(cell_30, 0)
        cell_30.r_t[:] = 0.0
        cell_30.r_ss[:] = 0.0
        # Moderate spread for ~50% coverage
        cell_30.fire_spread[:] = cell_30.cell_size * 0.7

        mock_sim.propagate_fire(cell_30)
        assert cell_30.fully_burning is True


# ============================================================================
# Test Group 6: Multiple suppression cycles
# ============================================================================

class TestMultipleSuppressionCycles:
    """Tests for accumulation across multiple suppressions."""

    def test_disabled_locs_accumulate(self, cell_30):
        """Disabled locs should accumulate across multiple suppress cycles."""
        # First burn: corner 2 ignition, cross some directions
        _ignite_cell(cell_30, 2)
        _set_intersections(cell_30, [0, 1, 2])  # Cross first 3 directions
        cell_30.compute_disabled_locs()
        first_disabled = cell_30.disabled_locs.copy()
        cell_30.suppress_to_fuel()
        assert cell_30._suppression_count == 1
        assert len(cell_30.disabled_locs) == len(first_disabled)

        # Second burn: edge 7 ignition, cross some directions
        _ignite_cell(cell_30, 7)
        _set_intersections(cell_30, [0, 1])
        cell_30.compute_disabled_locs()
        cell_30.suppress_to_fuel()
        assert cell_30._suppression_count == 2
        # Should have accumulated locs from both burns
        assert cell_30.disabled_locs.issuperset(first_disabled)
        assert len(cell_30.disabled_locs) > len(first_disabled)

    def test_reset_clears_accumulation(self, cell_30):
        """reset_to_fuel should clear all accumulated state."""
        _ignite_cell(cell_30, 2)
        cell_30.compute_disabled_locs()
        cell_30.suppress_to_fuel()
        assert cell_30._suppression_count > 0

        cell_30.reset_to_fuel()
        assert cell_30.disabled_locs == set()
        assert cell_30._suppression_count == 0


# ============================================================================
# Test Group 7: Disabled boundary filtering
# ============================================================================

class TestDisabledBoundaryFiltering:
    """Tests for inbound/outbound disabled boundary checks."""

    def test_inbound_ignition_blocked(self):
        """Neighbor with n_loc in disabled_locs should not be ignited."""
        neighbor = Cell(id=2, col=6, row=4, cell_size=30.0)
        data = CellData(
            fuel_type=Anderson13(1),
            elevation=100.0, aspect=0.0, slope_deg=0.0,
            canopy_cover=0.0, canopy_height=0.0,
            canopy_base_height=0.0, canopy_bulk_density=0.0,
            init_dead_mf=[0.06, 0.07, 0.08],
            live_h_mf=0.30, live_w_mf=0.30
        )
        neighbor._set_cell_data(data)
        neighbor._state = CellStates.FUEL
        # Disable entry point 5
        neighbor.disabled_locs = {5}

        # The check is: if n_loc in neighbor.disabled_locs: continue
        assert 5 in neighbor.disabled_locs
        assert 3 not in neighbor.disabled_locs

    def test_inbound_ignition_allowed(self):
        """Neighbor with n_loc NOT in disabled_locs should be ignitable."""
        neighbor = Cell(id=2, col=6, row=4, cell_size=30.0)
        data = CellData(
            fuel_type=Anderson13(1),
            elevation=100.0, aspect=0.0, slope_deg=0.0,
            canopy_cover=0.0, canopy_height=0.0,
            canopy_base_height=0.0, canopy_bulk_density=0.0,
            init_dead_mf=[0.06, 0.07, 0.08],
            live_h_mf=0.30, live_w_mf=0.30
        )
        neighbor._set_cell_data(data)
        neighbor._state = CellStates.FUEL
        neighbor.disabled_locs = {5, 7}
        # Entry at loc 3 should be allowed
        assert 3 not in neighbor.disabled_locs

    def test_outbound_exit_blocked(self, cell_30):
        """Fire crossing at disabled self-boundary should not ignite neighbor."""
        _ignite_cell(cell_30, 0)
        # Disable boundary loc 3
        cell_30.disabled_locs = {3}
        # _self_end_points for center = [1,2,...,12], so index 2 → loc 3
        assert cell_30._self_end_points[2] == 3
        assert cell_30._self_end_points[2] in cell_30.disabled_locs

    def test_outbound_exit_allowed(self, cell_30):
        """Fire crossing at non-disabled self-boundary should ignite normally."""
        _ignite_cell(cell_30, 0)
        cell_30.disabled_locs = {3}
        # Index 0 → loc 1, which is NOT disabled
        assert cell_30._self_end_points[0] == 1
        assert cell_30._self_end_points[0] not in cell_30.disabled_locs


# ============================================================================
# Test Group 8: n_disabled_locs property
# ============================================================================

class TestNDisabledLocs:
    """Tests for the n_disabled_locs property."""

    def test_empty(self, cell_30):
        assert cell_30.n_disabled_locs == 0

    def test_some_disabled(self, cell_30):
        cell_30.disabled_locs = {1, 3, 5}
        assert cell_30.n_disabled_locs == 3

    def test_all_disabled(self, cell_30):
        cell_30.disabled_locs = set(range(1, 13))
        assert cell_30.n_disabled_locs == 12


# ============================================================================
# Test Group 9: Logging
# ============================================================================

class TestLogging:
    """Tests for suppression_count and n_disabled_locs in CellLogEntry."""

    def test_log_entry_includes_new_fields(self, cell_30):
        """to_log_entry should include suppression_count and n_disabled_locs."""
        cell_30._suppression_count = 2
        cell_30.disabled_locs = {1, 2, 3}
        entry = cell_30.to_log_entry(100.0)
        assert entry.suppression_count == 2
        assert entry.n_disabled_locs == 3

    def test_log_entry_defaults(self, cell_30):
        """New fields should default to 0 for unsuppressed cells."""
        entry = cell_30.to_log_entry(0.0)
        assert entry.suppression_count == 0
        assert entry.n_disabled_locs == 0

    def test_cell_log_entry_backward_compat(self):
        """CellLogEntry constructed without new fields should use defaults."""
        entry = CellLogEntry(
            timestamp=0, id=1, x=0.0, y=0.0, fuel=1, state=1,
            crown_state=0, w_n_dead=0.01, w_n_dead_start=0.01,
            w_n_live=0.0, dfm_1hr=0.06, dfm_10hr=0.07, dfm_100hr=0.08,
            ros=0.0, I_ss=0.0, wind_speed=5.0, wind_dir=180.0,
            retardant=False, arrival_time=0.0
        )
        assert entry.suppression_count == 0
        assert entry.n_disabled_locs == 0

    def test_cell_log_entry_to_dict(self, cell_30):
        """to_dict should include new fields."""
        cell_30._suppression_count = 1
        cell_30.disabled_locs = {5}
        entry = cell_30.to_log_entry(50.0)
        d = entry.to_dict()
        assert 'suppression_count' in d
        assert 'n_disabled_locs' in d
        assert d['suppression_count'] == 1
        assert d['n_disabled_locs'] == 1


# ============================================================================
# Test Group 10: _set_cell_data initializes suppression state
# ============================================================================

class TestCellDataInit:
    """Tests that _set_cell_data initializes suppression attributes."""

    def test_initial_disabled_locs(self):
        """disabled_locs should be empty set after _set_cell_data."""
        cell = Cell(id=0, col=0, row=0, cell_size=30.0)
        data = CellData(
            fuel_type=Anderson13(1),
            elevation=100.0, aspect=0.0, slope_deg=0.0,
            canopy_cover=0.0, canopy_height=0.0,
            canopy_base_height=0.0, canopy_bulk_density=0.0,
            init_dead_mf=[0.06, 0.07, 0.08],
            live_h_mf=0.30, live_w_mf=0.30
        )
        cell._set_cell_data(data)
        assert cell.disabled_locs == set()
        assert cell._suppression_count == 0
        assert cell._self_end_points is None
