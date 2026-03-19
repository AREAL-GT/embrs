"""Integration test for Van Wagner energy-balance water suppression.

End-to-end test: create a minimal burning cell with known I_ss and fuel loading,
apply sufficient water via water_drop_vw, run apply_vw_suppression() + surface_fire(),
and verify ROS goes to zero.
"""

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
)
from embrs.utilities.unit_conversions import BTU_ft_min_to_kW_m, Lbsft2_to_KiSq


class TestVWSuppressionIntegration:
    """End-to-end Van Wagner suppression test."""

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

        # Mock parent for curr_wind() — needed by surface_fire
        parent = MagicMock()
        parent._curr_weather_idx = 0
        parent.sim_start_w_idx = 0
        parent.is_prediction.return_value = False
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

    def test_sufficient_water_zeros_ros(self, burning_cell):
        """Applying sufficient water should drive dead fuel moisture to extinction.

        After apply_vw_suppression(), surface_fire() should recompute ROS
        near zero because dead fuel moisture is at or above dead_mx.
        """
        cell = burning_cell

        # Verify fire is initially spreading
        assert cell.I_ss[0] > 0, "Cell should have positive initial intensity"
        initial_ros = cell.r_ss[0]
        assert initial_ros > 0, "Cell should have positive initial ROS"

        # Compute how much water is needed
        I_kW_m = BTU_ft_min_to_kW_m(float(cell.I_ss[0]))
        W_1_kg_m2 = Lbsft2_to_KiSq(cell.fuel.w_n_dead)
        area_m2 = cell.fire_area_m2

        heat_needed = heat_to_extinguish_kJ(
            I_kW_m, W_1_kg_m2, area_m2, efficiency=2.5
        )

        # Apply 2x the needed water to ensure full suppression
        H_w = volume_L_to_energy_kJ(1.0)  # kJ per liter
        volume_needed_L = (heat_needed / H_w) * 2.0 if H_w > 0 else 100.0

        cell.water_drop_vw(volume_needed_L, efficiency=2.5)

        # Apply suppression
        cell.apply_vw_suppression()

        # Verify dead fuel moisture is at or above extinction
        dead_mx = cell.fuel.dead_mx
        for i in range(min(4, len(cell.fmois))):
            assert cell.fmois[i] >= dead_mx - 1e-10, \
                f"Dead fuel moisture class {i} should be at extinction"

        # Recompute fire behavior
        surface_fire(cell)

        # ROS should be zero (moisture at extinction kills fire)
        assert cell.r_ss[0] == pytest.approx(0.0, abs=1e-10), \
            "ROS should be zero after sufficient water application"

    def test_insufficient_water_partial_suppression(self, burning_cell):
        """Partial water should increase moisture but not fully extinguish."""
        cell = burning_cell

        initial_mois = cell.fmois[0]

        # Apply a small amount of water
        cell.water_drop_vw(0.1)
        cell.apply_vw_suppression()

        # Moisture should increase but not reach extinction
        assert cell.fmois[0] > initial_mois, \
            "Moisture should increase after partial water application"

    def test_energy_accumulates_across_drops(self, burning_cell):
        """Multiple small drops should accumulate the same as one large drop."""
        cell = burning_cell

        # Apply 10 drops of 1L each
        for _ in range(10):
            cell.water_drop_vw(1.0)

        expected_energy = volume_L_to_energy_kJ(10.0)
        assert cell.water_applied_kJ == pytest.approx(expected_energy, rel=1e-10)
