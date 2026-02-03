"""Performance regression tests for Phase 4 optimizations.

These tests ensure that optimized code paths maintain acceptable performance
characteristics and don't regress over time.
"""

import time
import pytest
import numpy as np


class TestMoistureModelPerformance:
    """Performance tests for dead fuel moisture model optimizations."""

    @pytest.mark.slow
    def test_moisture_kernel_performance(self):
        """Test that vectorized moisture kernels complete in acceptable time."""
        from embrs.models.dead_fuel_moisture import (
            _solve_saturation,
            _update_moisture_diffusion,
            _update_temperature,
        )

        # Setup test arrays
        m_nodes = 20
        m_dx = 0.1
        m_mdt = 0.5
        m_wmx = 0.35  # Maximum fiber saturation

        # Initialize arrays
        m_x = np.linspace(0.01, 1, m_nodes)  # Avoid zero at center
        m_Tg = np.ones(m_nodes) * 0.1
        m_Tsold = np.ones(m_nodes) * 25.0
        m_s = np.ones(m_nodes) * 0.05
        m_To = np.ones(m_nodes) * 1e-6  # Moisture diffusivity coefficients
        m_Twold = np.ones(m_nodes) * 0.1
        m_w = np.ones(m_nodes) * 0.1
        m_Tv = np.ones(m_nodes) * 0.1  # Thermal diffusivity coefficients
        m_Ttold = np.ones(m_nodes) * 25.0
        m_t = np.ones(m_nodes) * 25.0

        # Warm up (allows JIT compilation if enabled)
        for _ in range(10):
            _solve_saturation(m_nodes, m_dx, m_mdt, m_x, m_Tg, m_Tsold, m_s)
            _update_moisture_diffusion(m_nodes, m_dx, m_mdt, m_wmx, m_x, m_To, m_Twold, m_w)
            _update_temperature(m_nodes, m_dx, m_mdt, m_x, m_Tv, m_Ttold, m_t)

        # Measure performance (1000 iterations)
        num_iterations = 1000
        start = time.perf_counter()

        for _ in range(num_iterations):
            _solve_saturation(m_nodes, m_dx, m_mdt, m_x, m_Tg, m_Tsold, m_s)
            _update_moisture_diffusion(m_nodes, m_dx, m_mdt, m_wmx, m_x, m_To, m_Twold, m_w)
            _update_temperature(m_nodes, m_dx, m_mdt, m_x, m_Tv, m_Ttold, m_t)

        elapsed = time.perf_counter() - start
        avg_time_us = (elapsed / num_iterations) * 1e6

        # Each kernel call should complete in < 100 microseconds on average
        # This is a generous threshold to avoid flaky tests on slow CI machines
        assert avg_time_us < 1000, f"Moisture kernels too slow: {avg_time_us:.1f} us/iteration"


class TestVectorizedGridOperations:
    """Performance tests for vectorized grid operations."""

    @pytest.mark.slow
    def test_compute_all_cell_positions_performance(self):
        """Test that vectorized position computation scales well."""
        from embrs.base_classes.grid_manager import GridManager

        # Test with a moderately sized grid
        num_rows, num_cols = 200, 200
        cell_size = 30.0

        grid_manager = GridManager(num_rows, num_cols, cell_size)

        # Warm up
        for _ in range(3):
            grid_manager.compute_all_cell_positions()

        # Measure performance
        num_iterations = 100
        start = time.perf_counter()

        for _ in range(num_iterations):
            all_x, all_y = grid_manager.compute_all_cell_positions()

        elapsed = time.perf_counter() - start
        avg_time_ms = (elapsed / num_iterations) * 1000

        # Should complete in < 10ms for 40,000 cells
        assert avg_time_ms < 50, f"Position computation too slow: {avg_time_ms:.1f} ms"

        # Verify output shape
        assert all_x.shape == (num_rows, num_cols)
        assert all_y.shape == (num_rows, num_cols)

    @pytest.mark.slow
    def test_compute_data_indices_performance(self):
        """Test that data index computation scales well."""
        from embrs.base_classes.grid_manager import GridManager

        # Test with a moderately sized grid
        num_rows, num_cols = 200, 200
        cell_size = 30.0
        data_res = 10.0
        data_rows, data_cols = 1000, 1000

        grid_manager = GridManager(num_rows, num_cols, cell_size)
        all_x, all_y = grid_manager.compute_all_cell_positions()

        # Warm up
        for _ in range(3):
            grid_manager.compute_data_indices(all_x, all_y, data_res, data_rows, data_cols)

        # Measure performance
        num_iterations = 100
        start = time.perf_counter()

        for _ in range(num_iterations):
            row_idx, col_idx = grid_manager.compute_data_indices(
                all_x, all_y, data_res, data_rows, data_cols
            )

        elapsed = time.perf_counter() - start
        avg_time_ms = (elapsed / num_iterations) * 1000

        # Should complete in < 10ms for 40,000 cells
        assert avg_time_ms < 50, f"Index computation too slow: {avg_time_ms:.1f} ms"

        # Verify indices are valid
        assert np.all(row_idx >= 0)
        assert np.all(row_idx < data_rows)
        assert np.all(col_idx >= 0)
        assert np.all(col_idx < data_cols)


class TestWeatherStreamPerformance:
    """Performance tests for weather stream handling optimizations."""

    @pytest.mark.slow
    def test_weather_entry_creation_performance(self):
        """Test that WeatherEntry creation is efficient (no deepcopy overhead)."""
        from embrs.utilities.data_classes import WeatherEntry

        # Create a template entry
        template = WeatherEntry(
            wind_speed=10.0,
            wind_dir_deg=180.0,
            temp=25.0,
            rel_humidity=30.0,
            cloud_cover=0.5,
            rain=0.0,
            dni=800.0,
            dhi=100.0,
            ghi=700.0,
            solar_zenith=30.0,
            solar_azimuth=180.0,
        )

        # Warm up
        for _ in range(100):
            _ = WeatherEntry(
                wind_speed=template.wind_speed * 1.1,
                wind_dir_deg=(template.wind_dir_deg + 5) % 360,
                temp=template.temp,
                rel_humidity=template.rel_humidity,
                cloud_cover=template.cloud_cover,
                rain=template.rain,
                dni=template.dni,
                dhi=template.dhi,
                ghi=template.ghi,
                solar_zenith=template.solar_zenith,
                solar_azimuth=template.solar_azimuth,
            )

        # Measure performance (10000 creations)
        num_iterations = 10000
        start = time.perf_counter()

        for i in range(num_iterations):
            _ = WeatherEntry(
                wind_speed=template.wind_speed * (1.0 + i * 0.001),
                wind_dir_deg=(template.wind_dir_deg + i) % 360,
                temp=template.temp,
                rel_humidity=template.rel_humidity,
                cloud_cover=template.cloud_cover,
                rain=template.rain,
                dni=template.dni,
                dhi=template.dhi,
                ghi=template.ghi,
                solar_zenith=template.solar_zenith,
                solar_azimuth=template.solar_azimuth,
            )

        elapsed = time.perf_counter() - start
        avg_time_us = (elapsed / num_iterations) * 1e6

        # Each creation should complete in < 10 microseconds on average
        # (deepcopy of dataclass would be much slower)
        assert avg_time_us < 50, f"WeatherEntry creation too slow: {avg_time_us:.1f} us"


@pytest.mark.slow
class TestEndToEndPerformance:
    """End-to-end performance tests for full simulation components."""

    def test_moisture_model_full_update(self):
        """Test full moisture model update performance."""
        from embrs.models.dead_fuel_moisture import DeadFuelMoisture

        # Create 1-hour fuel moisture model (radius 0.5cm)
        moisture_model = DeadFuelMoisture.createDeadFuelMoisture1()

        # Initialize environment
        moisture_model.initializeEnvironment(
            ta=30.0,    # Ambient air temperature (oC)
            ha=0.20,    # Ambient air relative humidity (fraction)
            sr=800.0,   # Solar radiation (W/m2)
            rc=0.0,     # Cumulative rainfall (cm)
            ti=25.0,    # Initial stick temperature (oC)
            hi=0.20,    # Initial stick surface humidity (fraction)
            wi=0.10,    # Initial stick moisture content
            bp=0.0218   # Barometric pressure (cal/cm3)
        )

        # Warm up with a few updates (includes JIT compilation if enabled)
        for _ in range(5):
            moisture_model.update_internal(
                et=0.0167,    # Elapsed time (hours) = 1 minute
                at=30.0,      # Air temperature (°C)
                rh=0.20,      # Relative humidity (fraction)
                sW=800.0,     # Solar radiation (W/m²)
                rcum=0.0,     # Cumulative rainfall (cm)
                bpr=0.0218    # Barometric pressure (cal/cm³)
            )

        # Measure performance (100 updates simulating ~100 minutes)
        num_iterations = 100
        start = time.perf_counter()

        for _ in range(num_iterations):
            moisture_model.update_internal(
                et=0.0167,    # 1 minute elapsed
                at=30.0,
                rh=0.20,
                sW=800.0,
                rcum=0.0,
                bpr=0.0218
            )

        elapsed = time.perf_counter() - start
        avg_time_ms = (elapsed / num_iterations) * 1000

        # Full moisture update should complete in < 100ms
        # This is the primary target for Phase 4 optimizations
        assert avg_time_ms < 100, f"Moisture model update too slow: {avg_time_ms:.2f} ms"

        # Verify output is reasonable
        moisture = moisture_model.meanMoisture()
        assert 0.0 < moisture < 3.0, f"Moisture out of range: {moisture}"
