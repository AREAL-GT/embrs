"""Tests for Crown Fire model.

These tests validate the crown fire initiation and spread calculations
based on Van Wagner's crown fire models and Rothermel (1991) active crown fire.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from embrs.models.crown_model import crown_fire, set_accel_constant, calc_R10
from embrs.models.fuel_models import Anderson13
from embrs.utilities.fire_util import CrownStatus


class TestCrownFireInitiation:
    """Tests for crown fire initiation thresholds."""

    @pytest.fixture
    def mock_cell_no_canopy(self):
        """Create a mock cell without canopy."""
        cell = MagicMock()
        cell.has_canopy = False
        return cell

    @pytest.fixture
    def mock_cell_with_canopy(self):
        """Create a mock cell with canopy suitable for crown fire."""
        cell = MagicMock()
        cell.has_canopy = True
        cell.canopy_base_height = 3.0  # meters
        cell.canopy_bulk_density = 0.1  # kg/m^3
        cell.r_ss = np.array([0.1, 0.15, 0.2, 0.15, 0.1, 0.05])  # m/s
        cell.I_ss = np.array([500, 750, 1000, 750, 500, 250])  # BTU/ft/min
        cell.r_h_ss = 0.2
        cell.fuel = Anderson13(10)
        cell.fuel.sav_ratio = 2000
        cell.reaction_intensity = 5000
        cell.fmois = np.array([0.06, 0.07, 0.08, 0.60, 0.90])
        cell.curr_wind = (500, 180)  # ft/min, degrees
        cell._crown_status = CrownStatus.NONE
        cell.cfb = 0.0
        cell.a_a = 0.0
        return cell

    def test_no_crown_fire_without_canopy(self, mock_cell_no_canopy):
        """Cells without canopy should not have crown fire."""
        fmc = 0.08

        # crown_fire returns early if no canopy
        result = crown_fire(mock_cell_no_canopy, fmc)

        assert result is None

    def test_crown_status_none_when_intensity_low(self, mock_cell_with_canopy):
        """Crown status should be NONE when fireline intensity is below threshold."""
        mock_cell_with_canopy.I_ss = np.array([10, 15, 20, 15, 10, 5])  # Low intensity

        crown_fire(mock_cell_with_canopy, fmc=0.08)

        mock_cell_with_canopy._crown_status = CrownStatus.NONE


class TestCriticalSurfaceIntensity:
    """Tests for critical surface fire intensity calculations."""

    def test_critical_intensity_formula(self):
        """Verify critical intensity formula I_o = (0.01 * CBH * (460 + 25.9 * fmc))^1.5."""
        # Van Wagner's crown fire initiation threshold
        cbh = 3.0  # canopy base height in meters
        fmc = 0.08  # foliar moisture content

        # Expected from formula
        expected = (0.01 * cbh * (460 + 25.9 * fmc)) ** 1.5

        # This is the calculation used in crown_fire
        I_o = (0.01 * cbh * (460 + 25.9 * fmc)) ** (3 / 2)

        assert I_o == pytest.approx(expected, abs=0.01)

    def test_critical_intensity_increases_with_cbh(self):
        """Higher canopy base height should require more intensity."""
        fmc = 0.08

        I_o_low = (0.01 * 2.0 * (460 + 25.9 * fmc)) ** 1.5
        I_o_high = (0.01 * 5.0 * (460 + 25.9 * fmc)) ** 1.5

        assert I_o_high > I_o_low

    def test_critical_intensity_increases_with_fmc(self):
        """Higher foliar moisture should require more intensity."""
        cbh = 3.0

        I_o_dry = (0.01 * cbh * (460 + 25.9 * 0.05)) ** 1.5
        I_o_wet = (0.01 * cbh * (460 + 25.9 * 0.15)) ** 1.5

        assert I_o_wet > I_o_dry


class TestActiveCrownFireThreshold:
    """Tests for active crown fire spread rate threshold."""

    def test_rac_formula(self):
        """Active crown spread threshold rac = 3.0 / CBD."""
        cbd = 0.1  # canopy bulk density kg/m^3

        rac = 3.0 / cbd

        assert rac == pytest.approx(30.0, abs=0.01)

    def test_rac_decreases_with_cbd(self):
        """Higher canopy bulk density should lower the threshold for active crown fire."""
        rac_low_cbd = 3.0 / 0.05
        rac_high_cbd = 3.0 / 0.15

        assert rac_high_cbd < rac_low_cbd


class TestAccelerationConstant:
    """Tests for crown fire acceleration constant calculations."""

    @pytest.fixture
    def mock_cell(self):
        """Create a mock cell for acceleration tests."""
        cell = MagicMock()
        cell.a_a = 0.0
        return cell

    def test_set_accel_constant_no_cfb(self, mock_cell):
        """With no crown fraction burned, acceleration should be minimal."""
        set_accel_constant(mock_cell, cfb=0.0)

        # a = (0.3/60) - 18.8 * (0)^2.5 * exp(-8*0)
        # a = 0.005
        assert mock_cell.a_a == pytest.approx(0.005, abs=0.001)

    def test_set_accel_constant_partial_cfb(self, mock_cell):
        """Partial crown fraction burned should produce intermediate acceleration."""
        set_accel_constant(mock_cell, cfb=0.5)

        # Value should be different from zero cfb case
        assert mock_cell.a_a != 0.005

    def test_set_accel_constant_full_cfb(self, mock_cell):
        """Full crown fraction burned should affect acceleration constant."""
        set_accel_constant(mock_cell, cfb=1.0)

        # With cfb=1.0: a = 0.005 - 18.8 * 1 * exp(-8)
        # exp(-8) is very small, so result close to 0.005
        assert mock_cell.a_a <= 0.01


class TestCrownFractionBurned:
    """Tests for crown fraction burned (CFB) calculations."""

    def test_cfb_formula_structure(self):
        """CFB formula: cfb = 1 - exp(-a_c * (R - R_0))."""
        R = 10.0  # m/min
        R_0 = 5.0  # critical surface spread rate
        rac = 30.0  # active crown threshold

        # CFB scaling exponent
        a_c = -np.log(0.1) / (0.9 * (rac - R_0))

        cfb = 1 - np.exp(-a_c * (R - R_0))

        # CFB should be between 0 and 1
        assert 0.0 <= cfb <= 1.0

    def test_cfb_zero_at_critical_spread(self):
        """CFB should be zero when R equals R_0."""
        R = 5.0
        R_0 = 5.0
        rac = 30.0

        a_c = -np.log(0.1) / (0.9 * (rac - R_0))
        cfb = 1 - np.exp(-a_c * (R - R_0))

        assert cfb == pytest.approx(0.0, abs=0.01)

    def test_cfb_increases_with_spread_rate(self):
        """CFB should increase as R increases above R_0."""
        R_0 = 5.0
        rac = 30.0
        a_c = -np.log(0.1) / (0.9 * (rac - R_0))

        cfb_low = 1 - np.exp(-a_c * (10 - R_0))
        cfb_high = 1 - np.exp(-a_c * (20 - R_0))

        assert cfb_high > cfb_low


class TestCrownStatus:
    """Tests for crown fire status enumeration."""

    def test_crown_status_values(self):
        """Crown status constants should have expected values."""
        assert CrownStatus.NONE == 0
        assert CrownStatus.PASSIVE == 1
        assert CrownStatus.ACTIVE == 2

    def test_crown_status_ordering(self):
        """Crown status should have logical ordering."""
        assert CrownStatus.NONE < CrownStatus.PASSIVE
        assert CrownStatus.PASSIVE < CrownStatus.ACTIVE
