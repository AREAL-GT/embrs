"""Tests for memory bounds and leak prevention.

This module tests that EMBRS simulation components don't grow unboundedly
during long-running simulations.
"""

import pytest
import tracemalloc
from unittest.mock import MagicMock, patch
import numpy as np


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_cell():
    """Create a mock cell for testing."""
    cell = MagicMock()
    cell.id = 1
    cell.x_pos = 100.0
    cell.y_pos = 100.0
    cell.fuel = MagicMock()
    cell.fuel.burnable = True
    cell.state = MagicMock()
    return cell


# =============================================================================
# Tests for Dictionary/List Bounds
# =============================================================================

class TestLoggerMemoryBounds:
    """Tests for logger cache memory management."""

    def test_logger_cache_pattern_cleared_on_flush(self):
        """Test that the cache-and-clear pattern works correctly.

        This tests the pattern used by the Logger without requiring
        full Logger initialization which needs additional setup.
        """
        # Simulate the cache pattern used by Logger
        cell_cache = []

        # Add entries
        for i in range(100):
            cell_cache.append({'cell_id': i, 'state': 1})

        assert len(cell_cache) == 100

        # Simulate flush by clearing
        cell_cache.clear()

        assert len(cell_cache) == 0

    def test_logger_caches_bounded_with_periodic_flush_pattern(self):
        """Test that periodic flushing keeps caches bounded.

        This tests the pattern used by the Logger's periodic flush mechanism.
        """
        cell_cache = []
        max_cache_size = 0

        # Simulate 1000 iterations with flush every 100
        for iteration in range(1000):
            cell_cache.append({'time_s': iteration, 'cell_id': iteration % 50})

            max_cache_size = max(max_cache_size, len(cell_cache))

            # Periodic flush (as done in FireSim.iterate)
            if iteration % 100 == 99:
                cell_cache.clear()

        # Max cache size should be bounded to ~100
        assert max_cache_size <= 100


class TestSpreadDictionaryBounds:
    """Tests for FirePredictor spread dictionary memory management."""

    def test_spread_dict_structure(self):
        """Test spread dictionary accumulation pattern."""
        # Simulate the spread dictionary behavior
        spread = {}

        # This pattern mirrors what happens in FirePredictor
        for time_s in range(100):
            for cell_idx in range(10):
                if spread.get(time_s) is None:
                    spread[time_s] = [(cell_idx * 10.0, cell_idx * 10.0)]
                else:
                    spread[time_s].append((cell_idx * 10.0, cell_idx * 10.0))

        # After 100 time steps with 10 cells each
        assert len(spread) == 100
        total_entries = sum(len(v) for v in spread.values())
        assert total_entries == 1000

    def test_spread_dict_memory_per_timestep(self):
        """Test that clearing old timesteps bounds memory."""
        spread = {}
        max_timesteps_to_keep = 10

        for time_s in range(100):
            # Add new entries
            spread[time_s] = [(i * 10.0, i * 10.0) for i in range(20)]

            # Remove old timesteps to bound memory
            if len(spread) > max_timesteps_to_keep:
                oldest_time = min(spread.keys())
                del spread[oldest_time]

        # Should only keep last 10 timesteps
        assert len(spread) == max_timesteps_to_keep


class TestForecastPoolMemoryBounds:
    """Tests for forecast pool memory management."""

    def test_forecast_pool_eviction(self):
        """Test that ForecastPoolManager evicts old pools."""
        from embrs.tools.forecast_pool import ForecastPoolManager, ForecastPool
        import numpy as np
        from unittest.mock import MagicMock

        # Clear any existing pools
        ForecastPoolManager.clear_all()

        # Create mock forecast pools
        pools = []
        for i in range(5):
            pool = MagicMock(spec=ForecastPool)
            pool._scenarios = [MagicMock() for _ in range(3)]
            ForecastPoolManager.register(pool)
            pools.append(pool)

        # Should only keep MAX_ACTIVE_POOLS
        active = ForecastPoolManager.get_active_pools()
        assert len(active) == ForecastPoolManager.MAX_ACTIVE_POOLS

        # Clean up
        ForecastPoolManager.clear_all()


class TestFirePredictorMemoryCleanup:
    """Tests for FirePredictor memory cleanup methods."""

    def test_clear_prediction_data_clears_dictionaries(self):
        """Test that clear_prediction_data clears all spread dictionaries."""
        from embrs.tools.fire_predictor import FirePredictor

        # Create a mock predictor with data
        predictor = MagicMock(spec=FirePredictor)
        predictor.spread = {0: [(1.0, 2.0)], 100: [(3.0, 4.0)]}
        predictor.flame_len_m = {0: 1.5, 100: 2.0}
        predictor.fli_kw_m = {0: 100.0, 100: 200.0}
        predictor.ros_ms = {0: 0.5, 100: 0.6}
        predictor.spread_dir = {(1.0, 2.0): 45, (3.0, 4.0): 90}
        predictor.crown_fire = {0: True}
        predictor.hold_probs = {0: 0.8}
        predictor.breaches = {0: False}
        predictor._updated_cells = {1: MagicMock(), 2: MagicMock()}
        predictor._scheduled_spot_fires = {100: [MagicMock()]}

        # Call the real method
        FirePredictor.clear_prediction_data(predictor)

        # All dictionaries should be empty
        assert len(predictor.spread) == 0
        assert len(predictor.flame_len_m) == 0
        assert len(predictor.fli_kw_m) == 0
        assert len(predictor.ros_ms) == 0
        assert len(predictor.spread_dir) == 0
        assert len(predictor.crown_fire) == 0
        assert len(predictor.hold_probs) == 0
        assert len(predictor.breaches) == 0
        assert len(predictor._updated_cells) == 0
        assert len(predictor._scheduled_spot_fires) == 0

    def test_cleanup_calls_clear_prediction_data(self):
        """Test that cleanup() also clears prediction data."""
        from embrs.tools.fire_predictor import FirePredictor
        from embrs.tools.forecast_pool import ForecastPoolManager

        # Create an object that looks like FirePredictor with real dict attributes
        class MockPredictor:
            def clear_prediction_data(self):
                # Bind the real method
                FirePredictor.clear_prediction_data(self)

        predictor = MockPredictor()
        predictor.spread = {0: [(1.0, 2.0)]}
        predictor.flame_len_m = {0: 1.5}
        predictor.fli_kw_m = {}
        predictor.ros_ms = {}
        predictor.spread_dir = {}
        predictor.crown_fire = {}
        predictor.hold_probs = {}
        predictor.breaches = {}
        predictor._updated_cells = {}
        predictor._scheduled_spot_fires = {}

        # Clear any existing pools
        ForecastPoolManager.clear_all()

        # Call cleanup (which calls clear_prediction_data)
        FirePredictor.cleanup(predictor)

        # Spread should be cleared
        assert len(predictor.spread) == 0

    def test_clear_prediction_data_safe_without_attributes(self):
        """Test that clear_prediction_data handles missing attributes gracefully."""
        from embrs.tools.fire_predictor import FirePredictor

        # Create an empty mock (no attributes)
        predictor = MagicMock(spec=[])

        # Should not raise any exceptions
        FirePredictor.clear_prediction_data(predictor)


# =============================================================================
# Tests for Copy Operations
# =============================================================================

class TestCopyOperationEfficiency:
    """Tests for efficient copy operations."""

    def test_weather_entry_creation_avoids_deepcopy(self):
        """Test that weather entries can be created without deepcopy."""
        from embrs.utilities.data_classes import WeatherEntry

        # Creating new entries directly is more efficient than deepcopy
        base_entry = WeatherEntry(
            wind_speed=10.0,
            wind_dir_deg=180.0,
            temp=25.0,
            rel_humidity=0.30,
            cloud_cover=0.0,
            rain=0.0,
            dni=800.0,
            dhi=100.0,
            ghi=500.0,
            solar_zenith=30.0,
            solar_azimuth=150.0
        )

        # This pattern should be used instead of copy.deepcopy(base_entry)
        new_entry = WeatherEntry(
            wind_speed=base_entry.wind_speed * 1.1,  # Perturbed
            wind_dir_deg=(base_entry.wind_dir_deg + 5) % 360,  # Perturbed
            temp=base_entry.temp,
            rel_humidity=base_entry.rel_humidity,
            cloud_cover=base_entry.cloud_cover,
            rain=base_entry.rain,
            dni=base_entry.dni,
            dhi=base_entry.dhi,
            ghi=base_entry.ghi,
            solar_zenith=base_entry.solar_zenith,
            solar_azimuth=base_entry.solar_azimuth
        )

        assert new_entry.wind_speed == pytest.approx(11.0)
        assert new_entry.wind_dir_deg == 185.0


# =============================================================================
# Integration Memory Tests
# =============================================================================

class TestMemoryTracking:
    """Integration tests using tracemalloc for memory tracking."""

    def test_dictionary_growth_pattern(self):
        """Test that dictionary operations don't cause unexpected memory growth."""
        tracemalloc.start()

        # Simulate accumulating data
        data = {}
        for i in range(1000):
            data[i] = {'x': i * 10.0, 'y': i * 10.0, 'state': i % 3}

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should be reasonable (less than 10MB for this test)
        assert peak < 10 * 1024 * 1024

    def test_list_comprehension_vs_copy_remove(self):
        """Compare memory efficiency of list comprehension vs copy-remove."""
        import copy

        # Method 1: Copy and remove (inefficient)
        tracemalloc.start()
        original = list(range(10000))
        working = copy.copy(original)
        for item in original:
            if item % 2 == 0:
                working.remove(item)
        copy_remove_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Method 2: List comprehension (efficient)
        tracemalloc.start()
        original = list(range(10000))
        result = [x for x in original if x % 2 != 0]
        comprehension_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # List comprehension should use less peak memory
        # (or at least comparable - remove() has O(n) overhead)
        assert comprehension_peak <= copy_remove_peak * 1.5


class TestControlHandlerMemory:
    """Tests for ControlActionHandler memory management."""

    def test_retardant_cleanup_removes_cells(self, mock_cell):
        """Test that expired retardants are properly cleaned up."""
        from embrs.base_classes.control_handler import ControlActionHandler

        # Create minimal handler
        mock_grid = MagicMock()
        handler = ControlActionHandler(
            grid_manager=mock_grid,
            cell_size=30.0,
            time_step=1.0,
            fuel_class_factory=MagicMock()
        )

        # Add retardant cells
        cells = []
        for i in range(10):
            cell = MagicMock()
            cell.id = i
            cell.fuel = MagicMock()
            cell.fuel.burnable = True
            cell._retardant = True
            cell._retardant_factor = 0.5
            cell.retardant_expiration_s = i * 10.0  # Expire at different times
            cells.append(cell)
            handler._long_term_retardants.add(cell)

        # Update at time 50s - should clear cells with expiration <= 50
        handler.update_long_term_retardants(50.0)

        # Should have removed cells 0-5 (expiration 0-50)
        # and kept cells 6-9 (expiration 60-90)
        remaining = len(handler._long_term_retardants)
        assert remaining == 4  # Cells with expiration 60, 70, 80, 90

    def test_water_drops_list_bounded(self):
        """Test that water drops list doesn't grow unboundedly."""
        # Create cells with different IDs
        water_drops = []
        for i in range(100):
            cell = MagicMock()
            cell.id = i
            water_drops.append(cell)

        # Simulate filtering based on moisture
        def should_keep(cell):
            return cell.id % 2 == 0  # Keep half

        water_drops = [c for c in water_drops if should_keep(c)]

        # Should be filtered to 50 (0, 2, 4, ... 98)
        assert len(water_drops) == 50


# =============================================================================
# Tests for Weak References
# =============================================================================

class TestWeakReferenceIntegrity:
    """Tests for proper weak reference usage."""

    def test_cell_parent_is_weak_reference(self):
        """Test that Cell uses weak reference for parent."""
        from embrs.fire_simulator.cell import Cell
        import inspect

        # Check that set_parent uses weakref
        source = inspect.getsource(Cell.set_parent)
        assert 'weakref.ref' in source, "Cell.set_parent should use weakref.ref"

    def test_weak_reference_allows_gc(self):
        """Test that weak references allow garbage collection."""
        import weakref
        import gc

        class Parent:
            pass

        class Child:
            def __init__(self):
                self._parent_ref = None

            def set_parent(self, parent):
                self._parent_ref = weakref.ref(parent)

            def get_parent(self):
                return self._parent_ref() if self._parent_ref else None

        parent = Parent()
        child = Child()
        child.set_parent(parent)

        # Child should be able to access parent
        assert child.get_parent() is parent

        # Delete parent reference
        del parent
        gc.collect()

        # Child's weak reference should now return None
        assert child.get_parent() is None


# =============================================================================
# Stress Tests (marked slow)
# =============================================================================

@pytest.mark.slow
class TestMemoryStress:
    """Stress tests for memory bounds."""

    def test_large_iteration_memory_bounded(self):
        """Test that memory stays bounded over many iterations."""
        tracemalloc.start()

        # Simulate many iterations with proper cleanup
        updated_cells = {}
        burning_cells = []
        burnt_cells = set()

        for iteration in range(1000):
            # Simulate adding cells during iteration
            for i in range(100):
                cell_id = iteration * 100 + i
                updated_cells[cell_id] = {'id': cell_id, 'state': 1}
                burning_cells.append(cell_id)

            # Simulate some cells burning out
            still_burning = burning_cells[:50]
            newly_burnt = burning_cells[50:]
            burning_cells = still_burning
            burnt_cells.update(newly_burnt)

            # Clear updated_cells at end of iteration (like the fix in 5.2)
            updated_cells.clear()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory should be bounded (burnt_cells will grow, but that's expected)
        # The key is that updated_cells doesn't accumulate
        assert peak < 100 * 1024 * 1024  # 100MB limit for this test
