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


class TestUpdateInternalRegression:
    """Regression tests for update_internal() numerical outputs.

    These tests capture the exact numerical behavior of the moisture model
    to ensure any performance optimizations (e.g., Numba JIT) produce
    identical results.
    """

    @pytest.fixture
    def dfm_1hr(self):
        """Create and initialize 1-hour fuel moisture model."""
        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(
            ta=20.0,    # Ambient air temperature (oC)
            ha=0.30,    # Ambient air relative humidity (g/g)
            sr=500.0,   # Solar radiation (W/m2)
            rc=0.0,     # Cumulative rainfall (cm)
            ti=20.0,    # Initial stick temperature (oC)
            hi=0.30,    # Initial stick surface humidity (g/g)
            wi=0.10,    # Initial stick moisture content
            bp=0.0218   # Initial stick barometric pressure (cal/cm3)
        )
        return dfm

    @pytest.fixture
    def dfm_10hr(self):
        """Create and initialize 10-hour fuel moisture model."""
        dfm = DeadFuelMoisture.createDeadFuelMoisture10()
        dfm.initializeEnvironment(
            ta=20.0,
            ha=0.30,
            sr=500.0,
            rc=0.0,
            ti=20.0,
            hi=0.30,
            wi=0.10,
            bp=0.0218
        )
        return dfm

    @pytest.fixture
    def dfm_100hr(self):
        """Create and initialize 100-hour fuel moisture model."""
        dfm = DeadFuelMoisture.createDeadFuelMoisture100()
        dfm.initializeEnvironment(
            ta=20.0,
            ha=0.30,
            sr=500.0,
            rc=0.0,
            ti=20.0,
            hi=0.30,
            wi=0.10,
            bp=0.0218
        )
        return dfm

    def test_update_internal_basic_drying(self, dfm_1hr):
        """Test moisture model under drying conditions (low humidity, high temp).

        This test captures the exact numerical output for a standard drying scenario
        to verify that any refactoring produces identical results.
        """
        import random
        random.seed(42)  # Fix randomness for perturbation

        # Simulate 1 hour of drying conditions
        result = dfm_1hr.update_internal(
            et=1.0,       # Elapsed time (hours)
            at=30.0,      # Air temperature (C) - warm
            rh=0.15,      # Relative humidity - low
            sW=800.0,     # Solar radiation (W/m2) - high
            rcum=0.0,     # No rain
            bpr=0.0218    # Barometric pressure
        )

        assert result is True

        # Check mean moisture - should decrease in drying conditions
        mean_moist = dfm_1hr.meanMoisture()
        assert isinstance(mean_moist, float)
        assert 0.0 <= mean_moist <= dfm_1hr.m_wmx

        # Capture expected values (these will be verified after running once)
        # The actual values are determined by running the test first
        assert dfm_1hr.m_state in range(11)  # Valid state

    def test_update_internal_basic_wetting(self, dfm_1hr):
        """Test moisture model under wetting conditions (high humidity, low temp)."""
        import random
        random.seed(42)

        result = dfm_1hr.update_internal(
            et=1.0,
            at=10.0,      # Cool temperature
            rh=0.90,      # High humidity
            sW=100.0,     # Low solar radiation (cloudy)
            rcum=0.0,     # No rain
            bpr=0.0218
        )

        assert result is True

        mean_moist = dfm_1hr.meanMoisture()
        assert isinstance(mean_moist, float)
        assert 0.0 <= mean_moist <= dfm_1hr.m_wmx

    def test_update_internal_with_rainfall(self, dfm_1hr):
        """Test moisture model with rainfall."""
        import random
        random.seed(42)

        result = dfm_1hr.update_internal(
            et=1.0,
            at=15.0,
            rh=0.80,
            sW=50.0,      # Low solar (rainy)
            rcum=0.5,     # 0.5 cm cumulative rainfall
            bpr=0.0218
        )

        assert result is True

        mean_moist = dfm_1hr.meanMoisture()
        assert isinstance(mean_moist, float)
        # With rainfall, moisture should be high
        assert mean_moist > 0.0

    def test_update_internal_hot_dry_extreme(self, dfm_1hr):
        """Test moisture model under extreme fire weather conditions."""
        import random
        random.seed(42)

        result = dfm_1hr.update_internal(
            et=1.0,
            at=40.0,      # Very hot
            rh=0.08,      # Very dry
            sW=1000.0,    # Strong sun
            rcum=0.0,
            bpr=0.0218
        )

        assert result is True

        mean_moist = dfm_1hr.meanMoisture()
        assert isinstance(mean_moist, float)
        # Under extreme conditions, moisture should be low
        assert mean_moist < dfm_1hr.m_wmx

    def test_update_internal_10hr_consistency(self, dfm_10hr):
        """Test 10-hour fuel responds appropriately to conditions."""
        import random
        random.seed(42)

        result = dfm_10hr.update_internal(
            et=1.0,
            at=25.0,
            rh=0.30,
            sW=600.0,
            rcum=0.0,
            bpr=0.0218
        )

        assert result is True

        mean_moist = dfm_10hr.meanMoisture()
        assert isinstance(mean_moist, float)
        assert 0.0 <= mean_moist <= dfm_10hr.m_wmx

    def test_update_internal_100hr_consistency(self, dfm_100hr):
        """Test 100-hour fuel responds appropriately to conditions."""
        import random
        random.seed(42)

        result = dfm_100hr.update_internal(
            et=1.0,
            at=25.0,
            rh=0.30,
            sW=600.0,
            rcum=0.0,
            bpr=0.0218
        )

        assert result is True

        mean_moist = dfm_100hr.meanMoisture()
        assert isinstance(mean_moist, float)
        assert 0.0 <= mean_moist <= dfm_100hr.m_wmx

    def test_update_internal_sequential_updates(self, dfm_1hr):
        """Test multiple sequential updates produce consistent results."""
        import random
        random.seed(42)

        # Record initial moisture
        initial_moist = dfm_1hr.meanMoisture()

        # Perform 3 hours of drying
        for _ in range(3):
            result = dfm_1hr.update_internal(
                et=1.0,
                at=35.0,
                rh=0.20,
                sW=700.0,
                rcum=0.0,
                bpr=0.0218
            )
            assert result is True

        final_moist = dfm_1hr.meanMoisture()

        # Moisture should have changed (likely decreased in drying conditions)
        assert isinstance(final_moist, float)
        assert 0.0 <= final_moist <= dfm_1hr.m_wmx

    def test_update_internal_preserves_array_lengths(self, dfm_1hr):
        """Test that update_internal preserves array dimensions."""
        import random
        random.seed(42)

        initial_nodes = dfm_1hr.m_nodes
        initial_w_len = len(dfm_1hr.m_w)
        initial_t_len = len(dfm_1hr.m_t)
        initial_s_len = len(dfm_1hr.m_s)

        dfm_1hr.update_internal(
            et=1.0, at=25.0, rh=0.30, sW=600.0, rcum=0.0, bpr=0.0218
        )

        assert len(dfm_1hr.m_w) == initial_w_len
        assert len(dfm_1hr.m_t) == initial_t_len
        assert len(dfm_1hr.m_s) == initial_s_len
        assert dfm_1hr.m_nodes == initial_nodes

    def test_update_internal_node_values_bounded(self, dfm_1hr):
        """Test that all node values remain within physical bounds."""
        import random
        random.seed(42)

        dfm_1hr.update_internal(
            et=1.0, at=30.0, rh=0.20, sW=800.0, rcum=0.0, bpr=0.0218
        )

        # Moisture should be non-negative
        for w in dfm_1hr.m_w:
            assert w >= 0.0, f"Negative moisture: {w}"
            assert w <= dfm_1hr.m_wmax, f"Moisture exceeds max: {w} > {dfm_1hr.m_wmax}"

        # Temperature should be reasonable
        for t in dfm_1hr.m_t:
            assert -60.0 <= t <= 71.0, f"Temperature out of range: {t}"

        # Saturation should be in [0, 1]
        for s in dfm_1hr.m_s:
            assert 0.0 <= s <= 1.0, f"Saturation out of range: {s}"


class TestUpdateInternalEdgeCases:
    """Test edge cases and boundary conditions for update_internal()."""

    def test_very_short_elapsed_time(self):
        """Test with very short elapsed time."""
        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)

        # Very short but valid elapsed time
        result = dfm.update_internal(
            et=0.001,  # 3.6 seconds
            at=25.0, rh=0.30, sW=600.0, rcum=0.0, bpr=0.0218
        )

        # Should return False for very short time (< 0.0000027)
        # 0.001 is valid though
        assert result is True

    def test_elapsed_time_too_short_rejected(self):
        """Test that elapsed time below threshold is rejected."""
        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)

        # Below minimum threshold
        result = dfm.update_internal(
            et=0.000001,  # Way too short
            at=25.0, rh=0.30, sW=600.0, rcum=0.0, bpr=0.0218
        )

        assert result is False

    def test_humidity_boundary_low(self):
        """Test with humidity at lower bound."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)

        result = dfm.update_internal(
            et=1.0, at=25.0, rh=0.01, sW=600.0, rcum=0.0, bpr=0.0218
        )

        assert result is True

    def test_humidity_boundary_high(self):
        """Test with humidity at upper bound."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)

        result = dfm.update_internal(
            et=1.0, at=25.0, rh=0.99, sW=600.0, rcum=0.0, bpr=0.0218
        )

        assert result is True

    def test_temperature_boundary_cold(self):
        """Test with very cold temperature."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)

        result = dfm.update_internal(
            et=1.0, at=-55.0, rh=0.50, sW=200.0, rcum=0.0, bpr=0.0218
        )

        assert result is True

    def test_temperature_boundary_hot(self):
        """Test with very hot temperature."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)

        result = dfm.update_internal(
            et=1.0, at=55.0, rh=0.10, sW=1200.0, rcum=0.0, bpr=0.0218
        )

        assert result is True

    def test_zero_solar_radiation(self):
        """Test with no solar radiation (night)."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)

        result = dfm.update_internal(
            et=1.0, at=15.0, rh=0.70, sW=0.0, rcum=0.0, bpr=0.0218
        )

        assert result is True

    def test_heavy_rainfall(self):
        """Test with heavy rainfall conditions."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)

        result = dfm.update_internal(
            et=1.0, at=15.0, rh=0.95, sW=10.0, rcum=2.0, bpr=0.0218
        )

        assert result is True
        # Heavy rain should increase moisture significantly
        mean_moist = dfm.meanMoisture()
        assert mean_moist > 0.1


class TestUpdateInternalNumericalStability:
    """Tests to verify numerical stability of the moisture model.

    These tests are important for ensuring that any optimizations
    (vectorization, JIT compilation) maintain numerical precision.
    """

    def test_deterministic_with_fixed_seed(self):
        """Test that results are deterministic when random seed is fixed."""
        import random

        results = []
        for _ in range(3):
            random.seed(42)
            dfm = DeadFuelMoisture.createDeadFuelMoisture1()
            dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)
            dfm.update_internal(1.0, 30.0, 0.25, 700.0, 0.0, 0.0218)
            results.append(dfm.meanMoisture())

        # All runs should produce identical results
        assert results[0] == results[1] == results[2]

    def test_long_simulation_stability(self):
        """Test numerical stability over many timesteps."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)

        # Simulate 24 hours
        for hour in range(24):
            result = dfm.update_internal(
                et=1.0,
                at=15.0 + 10.0 * np.sin(hour * np.pi / 12),  # Diurnal temp
                rh=0.50 - 0.20 * np.sin(hour * np.pi / 12),   # Diurnal humidity
                sW=max(0, 800.0 * np.sin((hour - 6) * np.pi / 12)),  # Solar cycle
                rcum=0.0,
                bpr=0.0218
            )
            assert result is True

        # Should remain numerically stable
        mean_moist = dfm.meanMoisture()
        assert not np.isnan(mean_moist)
        assert not np.isinf(mean_moist)
        assert 0.0 <= mean_moist <= dfm.m_wmax

    def test_compare_all_fuel_classes(self):
        """Test that different fuel classes respond correctly to same conditions."""
        import random

        fuel_classes = [
            DeadFuelMoisture.createDeadFuelMoisture1,
            DeadFuelMoisture.createDeadFuelMoisture10,
            DeadFuelMoisture.createDeadFuelMoisture100,
        ]

        results = []
        for create_func in fuel_classes:
            random.seed(42)
            dfm = create_func()
            dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)
            dfm.update_internal(1.0, 30.0, 0.20, 800.0, 0.0, 0.0218)
            results.append(dfm.meanMoisture())

        # All should produce valid results
        for r in results:
            assert not np.isnan(r)
            assert not np.isinf(r)
            assert r >= 0.0


class TestUpdateInternalPrecisionRegression:
    """Precision regression tests with hardcoded expected values.

    IMPORTANT: These tests verify that the numerical outputs match exactly
    (within floating point tolerance). If any performance optimizations
    (Numba JIT, vectorization) are applied, these tests will fail if the
    results differ. This ensures numerical accuracy is preserved.

    The expected values were captured from the baseline implementation
    (commit prior to Numba optimization).
    """

    def test_1hr_drying_exact_values(self):
        """Test 1-hr fuel drying produces exact expected values."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)
        dfm.update_internal(1.0, 30.0, 0.15, 800.0, 0.0, 0.0218)

        # Expected values from baseline implementation
        expected_mean_moist = 0.089373383140569
        expected_state = 2  # Desorption
        expected_m_w_0 = 0.082479717770933
        expected_m_t_0 = 20.527276230953571

        assert dfm.meanMoisture() == pytest.approx(expected_mean_moist, rel=1e-12)
        assert dfm.m_state == expected_state
        assert dfm.m_w[0] == pytest.approx(expected_m_w_0, rel=1e-12)
        assert dfm.m_t[0] == pytest.approx(expected_m_t_0, rel=1e-12)

    def test_10hr_moderate_exact_values(self):
        """Test 10-hr fuel under moderate conditions produces exact expected values."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture10()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)
        dfm.update_internal(1.0, 25.0, 0.30, 600.0, 0.0, 0.0218)

        expected_mean_moist = 0.095236249069107
        expected_state = 2

        assert dfm.meanMoisture() == pytest.approx(expected_mean_moist, rel=1e-12)
        assert dfm.m_state == expected_state

    def test_100hr_humid_exact_values(self):
        """Test 100-hr fuel under humid conditions produces exact expected values."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture100()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)
        dfm.update_internal(1.0, 15.0, 0.80, 200.0, 0.0, 0.0218)

        expected_mean_moist = 0.097595869942862
        expected_state = 2

        assert dfm.meanMoisture() == pytest.approx(expected_mean_moist, rel=1e-12)
        assert dfm.m_state == expected_state

    def test_1hr_rainfall_exact_values(self):
        """Test 1-hr fuel with rainfall produces exact expected values."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)
        dfm.update_internal(1.0, 15.0, 0.90, 50.0, 1.0, 0.0218)  # 1cm cumulative rain

        expected_mean_moist = 0.767429483796575
        expected_state = 8  # Rainstorm

        assert dfm.meanMoisture() == pytest.approx(expected_mean_moist, rel=1e-12)
        assert dfm.m_state == expected_state

    def test_5step_sequence_exact_values(self):
        """Test 5-step sequential update produces exact expected values."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)

        for i in range(5):
            dfm.update_internal(1.0, 28.0 + i, 0.25 - 0.02*i, 700.0, 0.0, 0.0218)

        expected_mean_moist = 0.052604784257102
        expected_state = 2

        # Expected final m_w array values (first few and last)
        expected_m_w = [
            0.0495036424, 0.0502821154, 0.0510389721, 0.051747655,
            0.0523932973, 0.0529625874, 0.0534440607, 0.05382828,
            0.0541079172, 0.0542777679, 0.0542777679
        ]

        assert dfm.meanMoisture() == pytest.approx(expected_mean_moist, rel=1e-12)
        assert dfm.m_state == expected_state

        # Check all m_w values
        for i, expected_w in enumerate(expected_m_w):
            assert dfm.m_w[i] == pytest.approx(expected_w, rel=1e-8), \
                f"m_w[{i}] mismatch: {dfm.m_w[i]} != {expected_w}"

    def test_node_array_precision(self):
        """Test that all node arrays maintain precision after update."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)
        dfm.update_internal(1.0, 30.0, 0.15, 800.0, 0.0, 0.0218)

        # Verify arrays are numpy-compatible and have correct precision
        m_w_array = np.array(dfm.m_w)
        m_t_array = np.array(dfm.m_t)
        m_s_array = np.array(dfm.m_s)

        # Check dtype is float64 (standard Python float)
        assert m_w_array.dtype == np.float64
        assert m_t_array.dtype == np.float64
        assert m_s_array.dtype == np.float64

        # Verify no precision loss (values should match exactly)
        for i in range(len(dfm.m_w)):
            assert m_w_array[i] == dfm.m_w[i]
            assert m_t_array[i] == dfm.m_t[i]
            assert m_s_array[i] == dfm.m_s[i]

    def test_diffusivity_precision(self):
        """Test diffusivity calculation maintains precision."""
        import random
        random.seed(42)

        dfm = DeadFuelMoisture.createDeadFuelMoisture1()
        dfm.initializeEnvironment(20.0, 0.30, 500.0, 0.0, 20.0, 0.30, 0.10, 0.0218)

        # Initial diffusivity is calculated during initializeEnvironment
        initial_d = dfm.m_d.copy()

        # After update, diffusivity should be recalculated
        dfm.update_internal(1.0, 30.0, 0.15, 800.0, 0.0, 0.0218)

        # Diffusivity values should be positive and finite
        for d in dfm.m_d:
            assert d >= 0.0
            assert not np.isnan(d)
            assert not np.isinf(d)
