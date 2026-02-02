"""Tests for Dead Fuel Moisture model.

These tests validate the Nelson dead fuel moisture model implementation
which simulates moisture diffusion through fuel sticks.
"""

import pytest
import numpy as np
from embrs.models.dead_fuel_moisture import DeadFuelMoisture


class TestDeadFuelMoistureInitialization:
    """Tests for DeadFuelMoisture initialization and parameter derivation."""

    def test_1hr_fuel_initialization(self):
        """1-hour fuel (radius 0.5cm) should initialize correctly."""
        # Parameters for 1-hr fuel class
        radius = 0.5  # cm
        stv = 22.0    # surface-to-volume ratio
        wmx = 0.35    # max fiber saturation
        wfilmk = 0.5  # film moisture constant

        dfm = DeadFuelMoisture(radius, stv, wmx, wfilmk)

        assert dfm.m_radius == radius
        assert dfm.m_density == 0.4
        assert dfm.m_nodes >= 11  # Should have adequate nodes
        assert len(dfm.m_w) == dfm.m_nodes
        assert len(dfm.m_t) == dfm.m_nodes
        assert len(dfm.m_s) == dfm.m_nodes

    def test_10hr_fuel_initialization(self):
        """10-hour fuel (radius 1.27cm) should initialize correctly."""
        radius = 1.27  # cm (1/2 inch diameter)
        stv = 22.0
        wmx = 0.35
        wfilmk = 0.5

        dfm = DeadFuelMoisture(radius, stv, wmx, wfilmk)

        assert dfm.m_radius == radius
        # Larger radius should have different node count
        assert dfm.m_nodes >= 11

    def test_100hr_fuel_initialization(self):
        """100-hour fuel (radius 3.8cm) should initialize correctly."""
        radius = 3.8  # cm (3 inch diameter)
        stv = 22.0
        wmx = 0.35
        wfilmk = 0.5

        dfm = DeadFuelMoisture(radius, stv, wmx, wfilmk)

        assert dfm.m_radius == radius

    def test_initial_moisture_content(self):
        """Initial moisture should be half the maximum."""
        radius = 0.5
        stv = 22.0
        wmx = 0.35
        wfilmk = 0.5

        dfm = DeadFuelMoisture(radius, stv, wmx, wfilmk)

        # Initial moisture should be 0.5 * wmx
        expected_initial = 0.5 * wmx
        for w in dfm.m_w:
            assert w == pytest.approx(expected_initial, abs=0.001)

    def test_initial_temperature(self):
        """Initial temperature should be 20C (ambient)."""
        radius = 0.5
        stv = 22.0
        wmx = 0.35
        wfilmk = 0.5

        dfm = DeadFuelMoisture(radius, stv, wmx, wfilmk)

        for t in dfm.m_t:
            assert t == 20.0


class TestParameterDerivation:
    """Tests for derived parameter calculations."""

    def test_derive_stick_nodes_odd(self):
        """Number of nodes should always be odd."""
        dfm = DeadFuelMoisture(0.5, 22.0, 0.35, 0.5)

        # Nodes should be odd for symmetric finite difference
        assert dfm.m_nodes % 2 == 1

    def test_derive_moisture_steps_small_radius(self):
        """Small radius should require more moisture steps."""
        dfm_small = DeadFuelMoisture(0.3, 22.0, 0.35, 0.5)
        dfm_large = DeadFuelMoisture(2.0, 22.0, 0.35, 0.5)

        # Smaller radius needs more steps for numerical stability
        assert dfm_small.m_mSteps > dfm_large.m_mSteps

    def test_derive_diffusivity_steps(self):
        """Diffusivity steps should increase for smaller radii."""
        dfm_small = DeadFuelMoisture(0.3, 22.0, 0.35, 0.5)
        dfm_large = DeadFuelMoisture(2.0, 22.0, 0.35, 0.5)

        assert dfm_small.m_dSteps > dfm_large.m_dSteps

    def test_internodal_distance(self):
        """Internodal distance should equal radius/(nodes-1)."""
        radius = 0.5
        dfm = DeadFuelMoisture(radius, 22.0, 0.35, 0.5)

        expected_dx = radius / (dfm.m_nodes - 1)
        assert dfm.m_dx == pytest.approx(expected_dx, abs=1e-6)

    def test_nodal_radial_distances(self):
        """Nodal radial distances should span from radius to 0."""
        radius = 0.5
        dfm = DeadFuelMoisture(radius, 22.0, 0.35, 0.5)

        # First node should be at surface (radius)
        assert dfm.m_x[0] == pytest.approx(radius, abs=1e-6)

        # Last node should be at center (0)
        assert dfm.m_x[-1] == pytest.approx(0.0, abs=1e-6)


class TestMoistureContentBounds:
    """Tests for moisture content validity."""

    def test_moisture_content_positive(self):
        """All moisture values should be non-negative."""
        dfm = DeadFuelMoisture(0.5, 22.0, 0.35, 0.5)

        for w in dfm.m_w:
            assert w >= 0.0

    def test_moisture_below_maximum(self):
        """All moisture values should be below maximum."""
        dfm = DeadFuelMoisture(0.5, 22.0, 0.35, 0.5)

        for w in dfm.m_w:
            assert w <= dfm.m_wmax

    def test_maximum_moisture_calculation(self):
        """Maximum moisture should be based on density."""
        dfm = DeadFuelMoisture(0.5, 22.0, 0.35, 0.5)

        # m_wmax = (1/density) - (1/1.53)
        expected_wmax = (1.0 / 0.4) - (1.0 / 1.53)
        assert dfm.m_wmax == pytest.approx(expected_wmax, abs=1e-6)


class TestMoistureMethods:
    """Tests for fuel moisture retrieval methods."""

    def test_mean_moisture_returns_float(self):
        """meanMoisture should return a float value."""
        dfm = DeadFuelMoisture(0.5, 22.0, 0.35, 0.5)

        result = dfm.meanMoisture()

        assert isinstance(result, float)

    def test_mean_moisture_in_valid_range(self):
        """meanMoisture should return value in valid range."""
        dfm = DeadFuelMoisture(0.5, 22.0, 0.35, 0.5)

        result = dfm.meanMoisture()

        # Moisture as fraction should be between 0 and wmax (~2.0)
        assert 0.0 <= result <= dfm.m_wmax

    def test_mean_weighted_moisture_returns_float(self):
        """meanWtdMoisture should return a float value."""
        dfm = DeadFuelMoisture(0.5, 22.0, 0.35, 0.5)

        result = dfm.meanWtdMoisture()

        assert isinstance(result, float)

    def test_mean_weighted_temperature_returns_float(self):
        """meanWtdTemperature should return a float value."""
        dfm = DeadFuelMoisture(0.5, 22.0, 0.35, 0.5)

        result = dfm.meanWtdTemperature()

        assert isinstance(result, float)

    def test_mean_weighted_temperature_reasonable(self):
        """meanWtdTemperature should return reasonable temperature."""
        dfm = DeadFuelMoisture(0.5, 22.0, 0.35, 0.5)

        result = dfm.meanWtdTemperature()

        # Initial temperature is 20C
        assert result == pytest.approx(20.0, abs=0.1)


class TestPhysicalConstants:
    """Tests for physical constants used in the model."""

    def test_density_constant(self):
        """Fuel density should be 0.4 g/cm^3."""
        dfm = DeadFuelMoisture(0.5, 22.0, 0.35, 0.5)
        assert dfm.m_density == 0.4

    def test_stick_length_constant(self):
        """Stick length should be 41.0 cm."""
        dfm = DeadFuelMoisture(0.5, 22.0, 0.35, 0.5)
        assert dfm.m_length == 41.0
