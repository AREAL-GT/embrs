"""Tests for Rothermel fire spread model.

These tests validate the Rothermel surface fire spread calculations
against published results from RMRS-GTR-371.
"""

import pytest
import numpy as np
from embrs.models.rothermel import (
    calc_r_0,
    get_characteristic_moistures,
    calc_live_mx,
    calc_I_r,
    calc_heat_sink,
    calc_wind_factor,
    calc_slope_factor,
    calc_moisture_damping,
    calc_mineral_damping,
    calc_effective_wind_factor,
    calc_effective_wind_speed,
    calc_eccentricity,
    calc_flame_len,
    calc_r_h,
    calc_vals_for_all_directions
)
from embrs.models.fuel_models import Anderson13, ScottBurgan40
from embrs.fire_simulator.cell import Cell
from embrs.utilities.data_classes import CellData
from embrs.utilities.unit_conversions import mph_to_ft_min
from embrs.utilities.fire_util import UtilFuncs
from unittest.mock import MagicMock, patch


class TestCharacteristicMoistures:
    """Tests for fuel moisture characteristic calculations."""

    def test_anderson_model_2_dead_live(self):
        """Anderson Model 2 should return correct dead and live moisture."""
        fuel = Anderson13(2)
        m_f = np.array([0.08, 0.10, 0.12, 0.08, 0.35, 0.6])
        result = get_characteristic_moistures(fuel, m_f)
        expected = (0.0804, 0.95)
        assert result == pytest.approx(expected, abs=0.05) # TODO: Not sure where this expected value coming from

    def test_anderson_model_13_no_live(self):
        """Anderson Model 13 (no live fuel) should return zero live moisture."""
        fuel = Anderson13(13)
        m_f = np.array([0.08, 0.10, 0.12, 0.08, 0.35, 0.6])
        result = get_characteristic_moistures(fuel, m_f)
        expected = (0.086, 0.0)
        assert result == pytest.approx(expected, abs=0.05)


class TestLiveMoistureExtinction:
    """Tests for live fuel moisture extinction calculations.

    Expected results are from RMRS-GTR-371 p. 67.
    """

    def test_sh2_low_dead_moisture(self):
        """SH2 at 6% dead moisture should have high live M_x."""
        fuel = ScottBurgan40(142)  # SH2
        result = calc_live_mx(fuel, 0.06)
        assert result == pytest.approx(0.98, abs=0.02)

    def test_sh2_medium_dead_moisture(self):
        """SH2 at 10% dead moisture."""
        fuel = ScottBurgan40(142)
        result = calc_live_mx(fuel, 0.10)
        assert result == pytest.approx(0.44, abs=0.02)

    def test_sh2_high_dead_moisture(self):
        """SH2 at 14% dead moisture should have low live M_x."""
        fuel = ScottBurgan40(142)
        result = calc_live_mx(fuel, 0.14)
        assert result == pytest.approx(0.15, abs=0.02)

    def test_sh6_low_dead_moisture(self):
        """SH6 at 6% dead moisture."""
        fuel = ScottBurgan40(146)  # SH6
        result = calc_live_mx(fuel, 0.06)
        assert result == pytest.approx(6.16, abs=0.02)

    def test_sh6_medium_dead_moisture(self):
        """SH6 at 10% dead moisture."""
        fuel = ScottBurgan40(146)
        result = calc_live_mx(fuel, 0.10)
        assert result == pytest.approx(5.10, abs=0.02)

    def test_sh6_high_dead_moisture(self):
        """SH6 at 14% dead moisture."""
        fuel = ScottBurgan40(146)
        result = calc_live_mx(fuel, 0.14)
        assert result == pytest.approx(4.03, abs=0.02)


class TestWindFactor:
    """Tests for wind factor calculations."""

    def test_model_2_6mph(self):
        """Anderson Model 2 at 6 mph wind."""
        fuel = Anderson13(2)
        wind_speed_mph = 6
        wind_speed = wind_speed_mph * 88  # Convert to ft/min
        result = calc_wind_factor(fuel, wind_speed)
        assert result == pytest.approx(20.5, abs=0.05)

    def test_model_2_12mph(self):
        """Anderson Model 2 at 12 mph wind."""
        fuel = Anderson13(2)
        wind_speed_mph = 12
        wind_speed = wind_speed_mph * 88
        result = calc_wind_factor(fuel, wind_speed)
        assert result == pytest.approx(72.8, abs=0.05)

    def test_model_9_6mph(self):
        """Anderson Model 9 at 6 mph wind."""
        fuel = Anderson13(9)
        wind_speed_mph = 6
        wind_speed = wind_speed_mph * 88
        result = calc_wind_factor(fuel, wind_speed)
        assert result == pytest.approx(12.9, abs=0.1)

    def test_model_9_12mph(self):
        """Anderson Model 9 at 12 mph wind."""
        fuel = Anderson13(9)
        wind_speed_mph = 12
        wind_speed = wind_speed_mph * 88
        result = calc_wind_factor(fuel, wind_speed)
        assert result == pytest.approx(42.7, abs=0.1)


class TestSlopeFactor:
    """Tests for slope factor calculations."""

    @pytest.fixture
    def slope_angles(self):
        """Common slope angles for testing."""
        return {
            'deg_11': np.arctan(0.2),   # ~11 degrees
            'deg_27': np.arctan(0.5),   # ~27 degrees
            'deg_45': np.arctan(1),     # 45 degrees
        }

    def test_model_2_gentle_slope(self, slope_angles):
        """Anderson Model 2 on gentle slope (~11 deg)."""
        fuel = Anderson13(2)
        result = calc_slope_factor(fuel, slope_angles['deg_11'])
        assert result == pytest.approx(1.0, abs=0.05)

    def test_model_2_moderate_slope(self, slope_angles):
        """Anderson Model 2 on moderate slope (~27 deg)."""
        fuel = Anderson13(2)
        result = calc_slope_factor(fuel, slope_angles['deg_27'])
        assert result == pytest.approx(6.2, abs=0.05)

    def test_model_2_steep_slope(self, slope_angles):
        """Anderson Model 2 on steep slope (45 deg)."""
        fuel = Anderson13(2)
        result = calc_slope_factor(fuel, slope_angles['deg_45'])
        assert result == pytest.approx(24.9, abs=0.1)

    def test_model_9_gentle_slope(self, slope_angles):
        """Anderson Model 9 on gentle slope (~11 deg)."""
        fuel = Anderson13(9)
        result = calc_slope_factor(fuel, slope_angles['deg_11'])
        assert result == pytest.approx(0.6, abs=0.05)

    def test_model_9_moderate_slope(self, slope_angles):
        """Anderson Model 9 on moderate slope (~27 deg)."""
        fuel = Anderson13(9)
        result = calc_slope_factor(fuel, slope_angles['deg_27'])
        assert result == pytest.approx(4.0, abs=0.05)

    def test_model_9_steep_slope(self, slope_angles):
        """Anderson Model 9 on steep slope (45 deg)."""
        fuel = Anderson13(9)
        result = calc_slope_factor(fuel, slope_angles['deg_45'])
        assert result == pytest.approx(16.0, abs=0.05)


class TestMoistureDamping:
    """Tests for moisture damping coefficient calculations."""

    def test_dead_fuel_below_extinction(self):
        """Dead fuel below extinction moisture."""
        m_f = 0.20
        m_x = 0.25
        result = calc_moisture_damping(m_f, m_x)
        assert result == pytest.approx(0.4, abs=0.02)

    def test_dead_fuel_well_below_extinction(self):
        """Dead fuel well below extinction moisture."""
        m_f = 0.20
        m_x = 0.45
        result = calc_moisture_damping(m_f, m_x)
        assert result == pytest.approx(0.55, abs=0.02)

    def test_dead_fuel_above_extinction(self):
        """Dead fuel above extinction moisture should return 0."""
        m_f = 0.27
        m_x = 0.25
        result = calc_moisture_damping(m_f, m_x)
        assert result == pytest.approx(0.0, abs=0.02)

    def test_live_fuel_sh2(self):
        """Live fuel damping for SH2 model."""
        fuel = ScottBurgan40(142)
        live_mx = calc_live_mx(fuel, 0.10)
        m_f = 0.30
        result = calc_moisture_damping(m_f, live_mx)
        assert result == pytest.approx(0.49, abs=0.02)

    def test_live_fuel_at_extinction(self):
        """Live fuel at extinction moisture should return 0."""
        fuel = ScottBurgan40(142)
        live_mx = calc_live_mx(fuel, 0.10)
        m_f = 0.45
        result = calc_moisture_damping(m_f, live_mx)
        assert result == pytest.approx(0.0, abs=0.02)

    def test_live_fuel_sh6(self):
        """Live fuel damping for SH6 model with high live moisture."""
        fuel = ScottBurgan40(146)
        live_mx = calc_live_mx(fuel, 0.10)
        m_f = 2.0
        result = calc_moisture_damping(m_f, live_mx)
        assert result == pytest.approx(0.56, abs=0.02)


class TestFlameLength:
    """Tests for flame length calculations.

    Expected results from RMRS-GTR-371 p. 47.
    """

    def test_model_4_no_wind(self):
        """Anderson Model 4 (chaparral) with no wind should produce expected flame length."""
        fuel = Anderson13(4)

        cell_data = CellData(
            fuel_type=fuel,
            elevation=0,
            aspect=0,
            slope_deg=0,
            canopy_cover=0,
            canopy_height=0.0,
            canopy_base_height=0.0,
            canopy_bulk_density=0.0,
            init_dead_mf=[0.05, 0.06, 0.07],
            live_h_mf=0,
            live_w_mf=0.0
        )
        cell = Cell(0, 0, 0, 30)
        cell._set_cell_data(cell_data)
        
        parent = MagicMock()
        parent._curr_weather_idx = 0
        parent.sim_start_w_idx = 0
        cell.set_parent(parent)
        
        cell.directions, distances, cell.end_pts = UtilFuncs.get_ign_parameters(0, cell.cell_size)


        wind_speed_mph = 0
        wind_speed_ft_min = mph_to_ft_min(wind_speed_mph)
        cell.forecast_wind_speeds = [wind_speed_ft_min]
        cell.forecast_wind_dirs = [0]

        # According to RMRS-GTR-371 p. 47
        cell.fmois = np.array([0.06, 0.07, 0.08, 0.06, 0.60, 0.90])

        R_h, R_0, I_r, _ = calc_r_h(cell)
        e = calc_eccentricity(fuel, R_h, R_0)

        cell.r_ss, cell.I_ss = calc_vals_for_all_directions(cell, R_h, I_r, 0.0, e)

        result = calc_flame_len(cell)
        expected = 5.3
        assert result == pytest.approx(expected, abs=0.05)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_wind_speed(self):
        """Wind factor with zero wind should be zero."""
        fuel = Anderson13(1)
        result = calc_wind_factor(fuel, 0.0)
        assert result == 0.0

    def test_zero_slope(self):
        """Slope factor with zero slope should be zero."""
        fuel = Anderson13(1)
        result = calc_slope_factor(fuel, 0.0)
        assert result == 0.0

    def test_moisture_at_zero(self):
        """Moisture damping with zero moisture should be 1."""
        m_f = 0.0
        m_x = 0.25
        result = calc_moisture_damping(m_f, m_x)
        assert result == pytest.approx(1.0, abs=0.02)

    def test_extinction_moisture_zero(self):
        """Moisture damping with zero extinction should return 0."""
        m_f = 0.10
        m_x = 0.0
        result = calc_moisture_damping(m_f, m_x)
        assert result == 0.0
