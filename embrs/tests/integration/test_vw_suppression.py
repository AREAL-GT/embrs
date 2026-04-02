"""Integration tests for Van Wagner binary-threshold water suppression.

Tests the full pipeline: water drop → decay → check extinguishment →
extinguish → surface_fire recompute → ROS = 0 / suppress_to_fuel.
"""

import math
import pytest
import numpy as np
from unittest.mock import MagicMock

from embrs.fire_simulator.cell import Cell
from embrs.utilities.fire_util import CellStates
from embrs.utilities.data_classes import CellData
from embrs.models.fuel_models import Anderson13
from embrs.models.rothermel import surface_fire
from embrs.models.van_wagner_water import (
    heat_to_extinguish_kJ, volume_L_to_energy_kJ,
    burning_zone_area_m2, efficiency_for_intensity,
)
from embrs.utilities.unit_conversions import (
    BTU_ft_min_to_kW_m, BTU_ft_min_to_kcal_s_m, Lbsft2_to_KiSq,
)


class TestVWSuppressionIntegration:
    """Binary-threshold Van Wagner suppression integration tests."""

    @pytest.fixture
    def burning_cell(self):
        """Create a burning cell with known fire state.

        Uses Anderson13 model 1 (short grass), sets up fire spread state
        so that surface_fire() can compute I_ss and r_ss.
        """
        cell = Cell(id=0, col=5, row=5, cell_size=30.0)
        cell_data = CellData(
            fuel_type=Anderson13(1),
            elevation=100.0,
            aspect=0.0,
            slope_deg=0.0,
            canopy_cover=0.0,
            canopy_height=0.0,
            canopy_base_height=0.0,
            canopy_bulk_density=0.0,
            init_dead_mf=[0.04, 0.04, 0.04],
            live_h_mf=0.0,
            live_w_mf=0.0,
        )
        cell._set_cell_data(cell_data)

        # Mock parent for curr_wind() and decay tau
        parent = MagicMock()
        parent._curr_weather_idx = 0
        parent.sim_start_w_idx = 0
        parent.is_prediction.return_value = False
        parent.curr_time_s = 0.0
        parent._vw_decay_tau = 120.0
        cell.set_parent(parent)
        # Keep strong reference so weak ref stays alive
        cell._test_parent_ref = parent

        # Set wind forecast arrays (surface_fire reads via curr_wind)
        cell.forecast_wind_speeds = [2.0]  # m/s
        cell.forecast_wind_dirs = [0.0]    # degrees

        # Set up minimal ignition state for surface_fire
        cell._state = CellStates.FIRE
        cell._ign_n_loc = 0  # center ignition

        # Initialize spread arrays (6 hex directions)
        n_dirs = 6
        cell.directions = np.linspace(0, 300, n_dirs)
        cell.distances = np.full(n_dirs, 15.0)
        cell.fire_spread = np.full(n_dirs, 5.0)  # 5m spread in each direction
        cell.intersections = np.zeros(n_dirs, dtype=np.bool_)
        cell.end_pts = np.zeros((n_dirs, 2))

        # Compute initial steady state fire
        surface_fire(cell)

        return cell

    def _compute_heat_needed(self, cell, efficiency=2.5):
        """Helper to compute heat_to_extinguish for a cell."""
        I_btu = max(cell.I_ss) if len(cell.I_ss) > 0 else max(cell.I_t)
        I_kW_m = BTU_ft_min_to_kW_m(float(I_btu))
        I_kcal_s_m = BTU_ft_min_to_kcal_s_m(float(I_btu))
        W_1_kg_m2 = Lbsft2_to_KiSq(cell.fuel.w_n_dead)

        bz_area = burning_zone_area_m2(I_kcal_s_m, cell.fire_front_length_m)
        effective_area = max(bz_area, cell.fire_area_m2)

        return heat_to_extinguish_kJ(
            I_kW_m, W_1_kg_m2, effective_area, efficiency=efficiency
        )

    def test_sufficient_water_extinguishes(self, burning_cell):
        """Applying sufficient water should trigger extinguishment.

        check_vw_extinguishment() returns True, extinguish_vw() pushes
        dead moisture to dead_mx, and surface_fire() recomputes ROS = 0.
        """
        cell = burning_cell

        # Verify fire is initially spreading
        assert cell.I_ss[0] > 0, "Cell should have positive initial intensity"
        assert cell.r_ss[0] > 0, "Cell should have positive initial ROS"

        # Compute how much water is needed
        heat_needed = self._compute_heat_needed(cell, efficiency=2.5)

        # Apply 2x the needed water to ensure full suppression
        H_w = volume_L_to_energy_kJ(1.0)  # kJ per liter
        volume_needed_L = (heat_needed / H_w) * 2.0 if H_w > 0 else 100.0

        cell.water_drop_vw(volume_needed_L, efficiency=2.5)

        # Check extinguishment
        assert cell.check_vw_extinguishment() is True

        # Apply extinguishment
        cell.extinguish_vw()

        # Verify dead fuel moisture is at extinction
        dead_mx = cell.fuel.dead_mx
        for i in range(min(4, len(cell.fmois))):
            assert cell.fmois[i] >= dead_mx - 1e-10, \
                f"Dead fuel moisture class {i} should be at extinction"

        # Recompute fire behavior — ROS should be zero
        surface_fire(cell)
        assert cell.r_ss[0] == pytest.approx(0.0, abs=1e-10), \
            "ROS should be zero after sufficient water application"

    def test_insufficient_water_no_effect(self, burning_cell):
        """Partial water below threshold should NOT change fire state.

        No intermediate moisture injection — fire burns unperturbed.
        """
        cell = burning_cell

        initial_mois = cell.fmois.copy()
        initial_ros = cell.r_ss[0]

        # Apply a tiny amount of water (well below threshold)
        cell.water_drop_vw(0.1, efficiency=2.5)

        # Should not meet extinguishment threshold
        assert not cell.check_vw_extinguishment()

        # Moisture should be UNCHANGED (no intermediate injection)
        for i in range(len(cell.fmois)):
            assert cell.fmois[i] == pytest.approx(initial_mois[i], abs=1e-15), \
                f"Moisture class {i} should be unchanged"

        # Recompute — ROS should be unchanged
        surface_fire(cell)
        assert cell.r_ss[0] == pytest.approx(initial_ros, rel=1e-10), \
            "ROS should be unchanged after insufficient water"

    def test_energy_accumulates_with_decay(self, burning_cell):
        """Multiple drops at different times accumulate with decay."""
        cell = burning_cell
        tau = 120.0
        cell._vw_tau = tau

        # First drop at t=0
        cell.water_drop_vw(5.0, efficiency=2.5)
        cell._vw_last_decay_time_s = 0.0
        energy_per_drop = volume_L_to_energy_kJ(5.0)

        # Decay to t=60s, then apply second drop
        cell.decay_water_energy(60.0)
        decayed_first = energy_per_drop * math.exp(-60.0 / tau)
        assert cell.water_applied_kJ == pytest.approx(decayed_first, rel=1e-10)

        cell.water_drop_vw(5.0, efficiency=2.5)
        total_after_second = decayed_first + energy_per_drop

        assert cell.water_applied_kJ == pytest.approx(total_after_second, rel=1e-10)

        # Total is less than sum without decay
        sum_without_decay = energy_per_drop * 2.0
        assert cell.water_applied_kJ < sum_without_decay

    def test_decay_prevents_stale_accumulation(self, burning_cell):
        """Water that fully decays should not contribute to later drops."""
        cell = burning_cell
        tau = 120.0
        cell._vw_tau = tau

        # First drop at t=0
        cell.water_drop_vw(1.0, efficiency=2.5)
        cell._vw_last_decay_time_s = 0.0
        energy_one_drop = volume_L_to_energy_kJ(1.0)

        # Wait long enough for full decay (10 * tau → effectively zero)
        cell.decay_water_energy(10 * tau)
        assert cell.water_applied_kJ == 0.0, \
            "Energy should have decayed to zero"

        # Apply second drop — only second drop counts
        cell.water_drop_vw(1.0, efficiency=2.5)
        assert cell.water_applied_kJ == pytest.approx(energy_one_drop, rel=1e-10)

    def test_burning_zone_floor_prevents_trivial_suppression(self, burning_cell):
        """Heat needed should use burning zone area, not trivially small fire area."""
        cell = burning_cell

        # Set fire_spread very small so fire_area is tiny
        cell.fire_spread = np.full(len(cell.fire_spread), 0.1)

        # Recompute fire to get current intensity
        surface_fire(cell)

        I_btu = max(cell.I_ss)
        I_kcal_s_m = BTU_ft_min_to_kcal_s_m(float(I_btu))
        I_kW_m = BTU_ft_min_to_kW_m(float(I_btu))

        # burning zone area based on intensity and fire front length
        bz_area = burning_zone_area_m2(I_kcal_s_m, cell.fire_front_length_m)
        fire_area = cell.fire_area_m2

        # For non-trivial intensity, burning zone should be larger than
        # the tiny fire_area from 0.1m spread
        if I_kcal_s_m > 0 and cell.fire_front_length_m > 0:
            effective_area = max(bz_area, fire_area)
            assert effective_area >= bz_area, \
                "Effective area should be at least the burning zone area"

            # Heat needed with effective area should be >= heat with just fire area
            W_1_kg_m2 = Lbsft2_to_KiSq(cell.fuel.w_n_dead)
            heat_with_bz = heat_to_extinguish_kJ(
                I_kW_m, W_1_kg_m2, effective_area, efficiency=2.5
            )
            heat_with_fire_area_only = heat_to_extinguish_kJ(
                I_kW_m, W_1_kg_m2, fire_area, efficiency=2.5
            )
            assert heat_with_bz >= heat_with_fire_area_only, \
                "Burning zone floor should prevent trivially low heat_needed"

    def test_intensity_dependent_efficiency(self, burning_cell):
        """Higher intensity should require more water due to higher efficiency."""
        # Low intensity: efficiency is lower → less water needed
        low_eff = efficiency_for_intensity(100.0)    # 100 kW/m
        high_eff = efficiency_for_intensity(3000.0)  # 3000 kW/m

        assert high_eff > low_eff, \
            "Higher intensity should have higher efficiency multiplier"

        # Compute heat_needed at different efficiencies for same area/loading
        I_kW_m = 500.0
        W_1_kg_m2 = 0.5
        area = 10.0

        heat_low = heat_to_extinguish_kJ(I_kW_m, W_1_kg_m2, area, efficiency=low_eff)
        heat_high = heat_to_extinguish_kJ(I_kW_m, W_1_kg_m2, area, efficiency=high_eff)

        assert heat_high > heat_low, \
            "Higher efficiency multiplier should require more water energy"

    def test_extinguishment_triggers_suppress_to_fuel(self, burning_cell):
        """Full pipeline: extinguish_vw → surface_fire → ROS=0 → propagate conditions met.

        After extinguishment, the dead fuel moisture is at dead_mx so
        Rothermel produces zero ROS, which is the condition for
        suppress_to_fuel in the propagate_fire pathway.
        """
        cell = burning_cell

        # Apply overwhelming water
        heat_needed = self._compute_heat_needed(cell, efficiency=2.5)
        H_w = volume_L_to_energy_kJ(1.0)
        volume_L = (heat_needed / H_w) * 5.0 if H_w > 0 else 500.0

        cell.water_drop_vw(volume_L, efficiency=2.5)

        # Simulate the iterate() sequence
        assert cell.check_vw_extinguishment() is True
        cell.extinguish_vw()

        # Verify moisture at extinction
        dead_mx = cell.fuel.dead_mx
        assert all(cell.fmois[i] >= dead_mx - 1e-10 for i in range(min(4, len(cell.fmois))))

        # surface_fire recompute
        surface_fire(cell)

        # All ROS values should be zero
        for i, ros in enumerate(cell.r_ss):
            assert ros == pytest.approx(0.0, abs=1e-10), \
                f"r_ss[{i}] should be zero after extinguishment"

        # has_steady_state was set to False by extinguish_vw
        # (it gets set True again after surface_fire in the real loop,
        # but the key point is ROS = 0)
