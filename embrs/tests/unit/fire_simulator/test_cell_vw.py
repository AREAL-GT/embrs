"""Tests for Van Wagner binary-threshold suppression methods on Cell.

Tests cover:
- decay_water_energy() exponential decay behavior
- fire_front_length_m property for center and boundary ignitions
- check_vw_extinguishment() threshold check
- extinguish_vw() moisture push to dead_mx
- water_drop_vw() new signature (efficiency=None default)
"""

import math
import pytest
import numpy as np
from unittest.mock import MagicMock
import weakref

from embrs.fire_simulator.cell import Cell
from embrs.utilities.fire_util import CellStates
from embrs.utilities.data_classes import CellData
from embrs.models.fuel_models import Anderson13


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def cell_30():
    """Create a 30m cell configured with burnable fuel data and mock parent."""
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
    mock_parent = MagicMock()
    mock_parent._curr_weather_idx = 0
    mock_parent.sim_start_w_idx = 0
    mock_parent.weather_t_step = 3600
    mock_parent.curr_time_s = 0.0
    mock_parent._vw_decay_tau = 120.0
    cell._parent = weakref.ref(mock_parent)
    cell._test_parent_ref = mock_parent  # prevent GC
    cell.forecast_wind_speeds = [5.0]
    cell.forecast_wind_dirs = [180.0]
    return cell


def _ignite_cell(cell, n_loc):
    """Helper: ignite a cell at the given location and set up fire arrays."""
    cell.get_ign_params(n_loc)
    cell._set_state(CellStates.FIRE)


# ============================================================================
# TestDecayWaterEnergy
# ============================================================================

class TestDecayWaterEnergy:
    """Tests for Cell.decay_water_energy()."""

    def test_basic_decay(self, cell_30):
        """Energy decays correctly over known dt with known tau."""
        cell_30._vw_tau = 120.0
        cell_30.water_applied_kJ = 1000.0
        cell_30._vw_last_decay_time_s = 0.0

        cell_30.decay_water_energy(60.0)  # 60s elapsed

        expected = 1000.0 * math.exp(-60.0 / 120.0)
        assert cell_30.water_applied_kJ == pytest.approx(expected, rel=1e-6)
        assert cell_30._vw_last_decay_time_s == 60.0

    def test_clears_below_threshold(self, cell_30):
        """Energy clears to 0 when below 1 kJ threshold."""
        cell_30._vw_tau = 120.0
        cell_30.water_applied_kJ = 2.0  # Small amount
        cell_30._vw_last_decay_time_s = 0.0

        # After long enough time, should decay below 1 kJ and be cleared
        cell_30.decay_water_energy(300.0)  # 5 minutes
        assert cell_30.water_applied_kJ == 0.0

    def test_no_decay_when_dt_zero(self, cell_30):
        """No decay when dt=0."""
        cell_30._vw_tau = 120.0
        cell_30.water_applied_kJ = 1000.0
        cell_30._vw_last_decay_time_s = 50.0

        cell_30.decay_water_energy(50.0)  # Same time
        assert cell_30.water_applied_kJ == 1000.0

    def test_multiple_decay_steps(self, cell_30):
        """Multiple decay steps accumulate correctly."""
        cell_30._vw_tau = 120.0
        cell_30.water_applied_kJ = 1000.0
        cell_30._vw_last_decay_time_s = 0.0

        # Two 30s steps should equal one 60s step
        cell_30.decay_water_energy(30.0)
        cell_30.decay_water_energy(60.0)

        expected = 1000.0 * math.exp(-30.0 / 120.0) * math.exp(-30.0 / 120.0)
        assert cell_30.water_applied_kJ == pytest.approx(expected, rel=1e-6)

    def test_reads_tau_from_parent(self, cell_30):
        """tau is read from parent on first call."""
        cell_30._vw_tau = None
        cell_30._test_parent_ref._vw_decay_tau = 60.0
        cell_30.water_applied_kJ = 1000.0
        cell_30._vw_last_decay_time_s = 0.0

        cell_30.decay_water_energy(60.0)

        assert cell_30._vw_tau == 60.0
        expected = 1000.0 * math.exp(-60.0 / 60.0)
        assert cell_30.water_applied_kJ == pytest.approx(expected, rel=1e-6)

    def test_fallback_tau_without_parent(self, cell_30):
        """Falls back to 120s when parent has no _vw_decay_tau."""
        cell_30._vw_tau = None
        del cell_30._test_parent_ref._vw_decay_tau
        cell_30.water_applied_kJ = 1000.0
        cell_30._vw_last_decay_time_s = 0.0

        cell_30.decay_water_energy(60.0)
        assert cell_30._vw_tau == 120.0


# ============================================================================
# TestFireFrontLength
# ============================================================================

class TestFireFrontLength:
    """Tests for Cell.fire_front_length_m property."""

    def test_non_burning_returns_zero(self, cell_30):
        """Non-burning cell → 0.0."""
        assert cell_30.fire_front_length_m == 0.0

    def test_center_ignition_computed_perimeter(self, cell_30):
        """Center ignition with known spread → computed perimeter (no floor)."""
        _ignite_cell(cell_30, 0)
        # Set uniform spread in all directions (circle-like)
        r = 10.0
        cell_30.fire_spread = np.full(len(cell_30.directions), r)

        perimeter = cell_30.fire_front_length_m

        # For uniform radial spread, chord-length perimeter of regular polygon
        dirs_rad = np.deg2rad(cell_30.directions)
        n = len(dirs_rad)
        expected = 0.0
        for i in range(n - 1):
            dx = r * np.cos(dirs_rad[i+1]) - r * np.cos(dirs_rad[i])
            dy = r * np.sin(dirs_rad[i+1]) - r * np.sin(dirs_rad[i])
            expected += np.sqrt(dx*dx + dy*dy)
        # Closing segment
        dx = r * np.cos(dirs_rad[0]) - r * np.cos(dirs_rad[-1])
        dy = r * np.sin(dirs_rad[0]) - r * np.sin(dirs_rad[-1])
        expected += np.sqrt(dx*dx + dy*dy)

        assert perimeter == pytest.approx(expected, rel=1e-6)
        assert perimeter > 0.0

    def test_center_ignition_small_spread_no_floor(self, cell_30):
        """Center ignition with very small spread → small perimeter (NOT floored)."""
        _ignite_cell(cell_30, 0)
        r = 0.5  # Very small spread
        cell_30.fire_spread = np.full(len(cell_30.directions), r)

        perimeter = cell_30.fire_front_length_m

        # Should be small, NOT floored to cell_size
        assert perimeter < cell_30._cell_size
        assert perimeter > 0.0

    def test_boundary_ignition_small_spread_floored(self, cell_30):
        """Boundary ignition with small spread → floored to cell_size."""
        _ignite_cell(cell_30, 1)  # Boundary ignition
        # Set small spread values
        cell_30.fire_spread = np.full(len(cell_30.directions), 0.5)

        perimeter = cell_30.fire_front_length_m
        assert perimeter == pytest.approx(cell_30._cell_size)

    def test_boundary_ignition_large_spread(self, cell_30):
        """Boundary ignition with large spread → computed perimeter (exceeds floor)."""
        _ignite_cell(cell_30, 1)
        # Set large spread so computed perimeter exceeds cell_size
        cell_30.fire_spread = np.full(len(cell_30.directions), 50.0)

        perimeter = cell_30.fire_front_length_m
        assert perimeter >= cell_30._cell_size

    def test_no_spread_data(self, cell_30):
        """Burning cell with empty fire_spread → 0.0."""
        _ignite_cell(cell_30, 0)
        cell_30.fire_spread = np.array([], dtype=np.float64)

        assert cell_30.fire_front_length_m == 0.0

    def test_boundary_single_direction(self, cell_30):
        """Boundary ignition with single spread value → cell_size."""
        _ignite_cell(cell_30, 1)
        cell_30.fire_spread = np.array([5.0])
        cell_30.directions = np.array([0.0])

        assert cell_30.fire_front_length_m == cell_30._cell_size


# ============================================================================
# TestCheckVWExtinguishment
# ============================================================================

class TestCheckVWExtinguishment:
    """Tests for Cell.check_vw_extinguishment()."""

    def _setup_burning_cell(self, cell, I_btu=500.0):
        """Set up cell as burning with known intensity."""
        _ignite_cell(cell, 0)
        # Set spread so fire_area_m2 and fire_front_length_m are nonzero
        cell.fire_spread = np.full(len(cell.directions), 10.0)
        cell.I_t = np.array([I_btu])

    def test_sufficient_energy_returns_true(self, cell_30):
        """Sufficient water energy → returns True."""
        self._setup_burning_cell(cell_30)
        # Apply very large amount of water energy
        cell_30.water_applied_kJ = 1e9
        cell_30._vw_efficiency = 2.5

        assert cell_30.check_vw_extinguishment() is True

    def test_insufficient_energy_returns_false(self, cell_30):
        """Insufficient water energy → returns False."""
        self._setup_burning_cell(cell_30)
        cell_30.water_applied_kJ = 0.001  # Tiny amount
        cell_30._vw_efficiency = 2.5

        assert cell_30.check_vw_extinguishment() == False

    def test_zero_energy_returns_false(self, cell_30):
        """Zero water energy → returns False."""
        self._setup_burning_cell(cell_30)
        cell_30.water_applied_kJ = 0.0

        assert cell_30.check_vw_extinguishment() is False

    def test_higher_intensity_needs_more_water(self, cell_30):
        """Higher intensity fire needs more water to extinguish."""
        # Low intensity: should pass with moderate water
        self._setup_burning_cell(cell_30, I_btu=100.0)
        cell_30._vw_efficiency = None  # Use intensity-dependent table

        from embrs.models.van_wagner_water import (
            heat_to_extinguish_kJ, burning_zone_area_m2,
            efficiency_for_intensity,
        )
        from embrs.utilities.unit_conversions import (
            BTU_ft_min_to_kW_m, BTU_ft_min_to_kcal_s_m, Lbsft2_to_KiSq,
        )

        # Compute heat needed for low intensity
        I_kW_low = BTU_ft_min_to_kW_m(100.0)
        I_kcal_low = BTU_ft_min_to_kcal_s_m(100.0)
        W_1 = Lbsft2_to_KiSq(cell_30.fuel.w_n_dead)
        bz_low = burning_zone_area_m2(I_kcal_low, cell_30.fire_front_length_m)
        eff_low = efficiency_for_intensity(I_kW_low)
        area_low = max(bz_low, cell_30.fire_area_m2)
        heat_low = heat_to_extinguish_kJ(I_kW_low, W_1, area_low, eff_low)

        # Compute heat needed for high intensity
        cell_30.I_t = np.array([2000.0])
        I_kW_high = BTU_ft_min_to_kW_m(2000.0)
        I_kcal_high = BTU_ft_min_to_kcal_s_m(2000.0)
        bz_high = burning_zone_area_m2(I_kcal_high, cell_30.fire_front_length_m)
        eff_high = efficiency_for_intensity(I_kW_high)
        area_high = max(bz_high, cell_30.fire_area_m2)
        heat_high = heat_to_extinguish_kJ(I_kW_high, W_1, area_high, eff_high)

        assert heat_high > heat_low


# ============================================================================
# TestExtinguishVW
# ============================================================================

class TestExtinguishVW:
    """Tests for Cell.extinguish_vw()."""

    def test_dead_fuel_set_to_dead_mx(self, cell_30):
        """Dead fuel classes (0-3) should be set to dead_mx."""
        _ignite_cell(cell_30, 0)
        # Ensure initial moisture is below dead_mx
        dead_mx = cell_30.fuel.dead_mx
        for i in range(4):
            cell_30.fmois[i] = 0.05

        cell_30.extinguish_vw()

        for i in range(min(4, len(cell_30.fmois))):
            assert cell_30.fmois[i] == pytest.approx(dead_mx)

    def test_live_fuel_unchanged(self, cell_30):
        """Live fuel classes (4-5) should remain unchanged."""
        _ignite_cell(cell_30, 0)
        original_live = cell_30.fmois[4:].copy()

        cell_30.extinguish_vw()

        np.testing.assert_array_almost_equal(cell_30.fmois[4:], original_live)

    def test_has_steady_state_false(self, cell_30):
        """has_steady_state should be set to False."""
        _ignite_cell(cell_30, 0)
        cell_30.has_steady_state = True

        cell_30.extinguish_vw()

        assert cell_30.has_steady_state is False


# ============================================================================
# TestWaterDropVWNewSignature
# ============================================================================

class TestWaterDropVWNewSignature:
    """Tests for updated water_drop_vw() signature."""

    def test_default_efficiency_is_none(self, cell_30):
        """Default efficiency=None stores None (table lookup deferred)."""
        _ignite_cell(cell_30, 0)
        cell_30.water_drop_vw(10.0)

        assert cell_30._vw_efficiency is None

    def test_explicit_efficiency_stored(self, cell_30):
        """Explicit efficiency=2.5 stores 2.5."""
        _ignite_cell(cell_30, 0)
        cell_30.water_drop_vw(10.0, efficiency=2.5)

        assert cell_30._vw_efficiency == 2.5

    def test_energy_accumulates(self, cell_30):
        """Energy still accumulates correctly across multiple drops."""
        _ignite_cell(cell_30, 0)
        from embrs.models.van_wagner_water import volume_L_to_energy_kJ

        cell_30.water_drop_vw(10.0)
        e1 = cell_30.water_applied_kJ

        cell_30.water_drop_vw(5.0)
        e2 = cell_30.water_applied_kJ

        expected_total = volume_L_to_energy_kJ(10.0) + volume_L_to_energy_kJ(5.0)
        assert e2 == pytest.approx(expected_total, rel=1e-6)
        assert e2 > e1

    def test_negative_volume_raises(self, cell_30):
        """Negative volume raises ValueError."""
        with pytest.raises(ValueError):
            cell_30.water_drop_vw(-1.0)

    def test_initializes_decay_time_from_parent(self, cell_30):
        """First drop initializes _vw_last_decay_time_s from parent time."""
        _ignite_cell(cell_30, 0)
        cell_30._vw_last_decay_time_s = 0.0
        cell_30._test_parent_ref.curr_time_s = 100.0

        cell_30.water_drop_vw(10.0)

        assert cell_30._vw_last_decay_time_s == 100.0
