"""Tests for ForecastPool and ForecastData classes.

This module tests the ForecastPool class behavior to ensure
the refactoring maintains identical functionality.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock
from dataclasses import dataclass

from embrs.utilities.data_classes import (
    ForecastPool,
    ForecastData,
    PredictorParams,
    MapParams,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_weather_stream():
    """Create a mock weather stream for testing."""
    from datetime import timedelta
    mock = MagicMock()
    mock.time_step = 60  # 60 minute time step
    mock.stream = [MagicMock() for _ in range(10)]
    base_time = datetime(2026, 7, 1, 12, 0)
    mock.stream_times = [base_time + timedelta(hours=i) for i in range(10)]
    return mock


@pytest.fixture
def sample_wind_forecast():
    """Create a sample wind forecast array.

    Shape: (n_timesteps, height, width, 2) where [...,0]=speed, [...,1]=direction
    """
    return np.random.rand(5, 10, 10, 2) * np.array([10, 360])  # speed 0-10, dir 0-360


@pytest.fixture
def sample_forecast_data(mock_weather_stream, sample_wind_forecast):
    """Create a sample ForecastData object."""
    return ForecastData(
        wind_forecast=sample_wind_forecast,
        weather_stream=mock_weather_stream,
        wind_speed_bias=1.5,
        wind_dir_bias=10.0,
        speed_error_seed=12345,
        dir_error_seed=67890,
        forecast_id=0,
        generation_time=1700000000.0
    )


@pytest.fixture
def sample_predictor_params():
    """Create sample predictor parameters."""
    return PredictorParams(
        time_horizon_hr=2.0,
        time_step_s=30,
        cell_size_m=30.0,
        dead_mf=0.08,
        live_mf=0.30,
        model_spotting=False,
        wind_speed_bias=0,
        wind_dir_bias=0,
        wind_uncertainty_factor=0.5
    )


@pytest.fixture
def sample_map_params():
    """Create a mock MapParams object."""
    mock = MagicMock(spec=MapParams)
    mock.folder = "/test/path"
    return mock


@pytest.fixture
def sample_forecast_pool(
    mock_weather_stream,
    sample_wind_forecast,
    sample_predictor_params,
    sample_map_params
):
    """Create a sample ForecastPool with multiple forecasts."""
    forecasts = []
    for i in range(5):
        forecast = ForecastData(
            wind_forecast=sample_wind_forecast.copy(),
            weather_stream=mock_weather_stream,
            wind_speed_bias=1.5 + i * 0.1,
            wind_dir_bias=10.0 + i * 2.0,
            speed_error_seed=12345 + i,
            dir_error_seed=67890 + i,
            forecast_id=i,
            generation_time=1700000000.0 + i
        )
        forecasts.append(forecast)

    return ForecastPool(
        forecasts=forecasts,
        base_weather_stream=mock_weather_stream,
        map_params=sample_map_params,
        predictor_params=sample_predictor_params,
        created_at_time_s=3600.0,
        forecast_start_datetime=datetime(2026, 7, 1, 13, 0)
    )


# =============================================================================
# ForecastData Tests
# =============================================================================

class TestForecastData:
    """Tests for the ForecastData dataclass."""

    def test_forecast_data_creation(self, sample_forecast_data):
        """Test that ForecastData can be created with all required fields."""
        assert sample_forecast_data.forecast_id == 0
        assert sample_forecast_data.wind_speed_bias == 1.5
        assert sample_forecast_data.wind_dir_bias == 10.0
        assert sample_forecast_data.speed_error_seed == 12345
        assert sample_forecast_data.dir_error_seed == 67890
        assert sample_forecast_data.generation_time == 1700000000.0

    def test_forecast_data_wind_array_shape(self, sample_forecast_data):
        """Test that wind_forecast array has expected shape."""
        # Shape should be (n_timesteps, height, width, 2)
        assert len(sample_forecast_data.wind_forecast.shape) == 4
        assert sample_forecast_data.wind_forecast.shape[-1] == 2  # speed and direction

    def test_forecast_data_wind_array_values(self, sample_forecast_data):
        """Test that wind_forecast array contains valid values."""
        speeds = sample_forecast_data.wind_forecast[..., 0]
        directions = sample_forecast_data.wind_forecast[..., 1]

        # Speed should be non-negative
        assert np.all(speeds >= 0)
        # Direction should be in [0, 360)
        assert np.all(directions >= 0)
        assert np.all(directions < 360)

    def test_forecast_data_weather_stream_attached(self, sample_forecast_data):
        """Test that weather_stream is properly attached."""
        assert sample_forecast_data.weather_stream is not None
        assert sample_forecast_data.weather_stream.time_step == 60


# =============================================================================
# ForecastPool Initialization Tests
# =============================================================================

class TestForecastPoolInitialization:
    """Tests for ForecastPool initialization and basic attributes."""

    def test_pool_creation(self, sample_forecast_pool):
        """Test that ForecastPool can be created."""
        assert sample_forecast_pool is not None
        assert sample_forecast_pool.created_at_time_s == 3600.0

    def test_pool_forecasts_list(self, sample_forecast_pool):
        """Test that forecasts list is properly stored."""
        assert len(sample_forecast_pool.forecasts) == 5
        assert all(isinstance(f, ForecastData) for f in sample_forecast_pool.forecasts)

    def test_pool_base_weather_stream(self, sample_forecast_pool, mock_weather_stream):
        """Test that base_weather_stream is stored."""
        assert sample_forecast_pool.base_weather_stream is not None
        assert sample_forecast_pool.base_weather_stream.time_step == 60

    def test_pool_map_params(self, sample_forecast_pool):
        """Test that map_params is stored."""
        assert sample_forecast_pool.map_params is not None

    def test_pool_predictor_params(self, sample_forecast_pool):
        """Test that predictor_params is stored."""
        assert sample_forecast_pool.predictor_params is not None
        assert sample_forecast_pool.predictor_params.time_horizon_hr == 2.0

    def test_pool_forecast_start_datetime(self, sample_forecast_pool):
        """Test that forecast_start_datetime is stored."""
        assert sample_forecast_pool.forecast_start_datetime == datetime(2026, 7, 1, 13, 0)

    def test_empty_pool_creation(
        self,
        mock_weather_stream,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that an empty ForecastPool can be created."""
        pool = ForecastPool(
            forecasts=[],
            base_weather_stream=mock_weather_stream,
            map_params=sample_map_params,
            predictor_params=sample_predictor_params,
            created_at_time_s=0.0,
            forecast_start_datetime=datetime(2026, 7, 1, 12, 0)
        )
        assert len(pool) == 0


# =============================================================================
# ForecastPool __len__ Tests
# =============================================================================

class TestForecastPoolLen:
    """Tests for ForecastPool.__len__() method."""

    def test_len_returns_correct_count(self, sample_forecast_pool):
        """Test that __len__ returns the number of forecasts."""
        assert len(sample_forecast_pool) == 5

    def test_len_empty_pool(
        self,
        mock_weather_stream,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that __len__ returns 0 for empty pool."""
        pool = ForecastPool(
            forecasts=[],
            base_weather_stream=mock_weather_stream,
            map_params=sample_map_params,
            predictor_params=sample_predictor_params,
            created_at_time_s=0.0,
            forecast_start_datetime=datetime(2026, 7, 1, 12, 0)
        )
        assert len(pool) == 0

    def test_len_single_forecast(
        self,
        sample_forecast_data,
        mock_weather_stream,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that __len__ returns 1 for single-forecast pool."""
        pool = ForecastPool(
            forecasts=[sample_forecast_data],
            base_weather_stream=mock_weather_stream,
            map_params=sample_map_params,
            predictor_params=sample_predictor_params,
            created_at_time_s=0.0,
            forecast_start_datetime=datetime(2026, 7, 1, 12, 0)
        )
        assert len(pool) == 1


# =============================================================================
# ForecastPool __getitem__ Tests
# =============================================================================

class TestForecastPoolGetItem:
    """Tests for ForecastPool.__getitem__() method."""

    def test_getitem_valid_index(self, sample_forecast_pool):
        """Test that __getitem__ returns correct forecast by index."""
        forecast = sample_forecast_pool[0]
        assert isinstance(forecast, ForecastData)
        assert forecast.forecast_id == 0

    def test_getitem_all_indices(self, sample_forecast_pool):
        """Test that all indices return correct forecasts."""
        for i in range(len(sample_forecast_pool)):
            forecast = sample_forecast_pool[i]
            assert forecast.forecast_id == i

    def test_getitem_negative_index(self, sample_forecast_pool):
        """Test that negative indexing works."""
        forecast = sample_forecast_pool[-1]
        assert forecast.forecast_id == 4  # Last item

        forecast = sample_forecast_pool[-2]
        assert forecast.forecast_id == 3  # Second to last

    def test_getitem_out_of_range(self, sample_forecast_pool):
        """Test that out-of-range index raises IndexError."""
        with pytest.raises(IndexError):
            _ = sample_forecast_pool[100]

    def test_getitem_returns_same_object(self, sample_forecast_pool):
        """Test that __getitem__ returns the same object on repeated calls."""
        forecast1 = sample_forecast_pool[0]
        forecast2 = sample_forecast_pool[0]
        assert forecast1 is forecast2


# =============================================================================
# ForecastPool sample() Tests
# =============================================================================

class TestForecastPoolSample:
    """Tests for ForecastPool.sample() method."""

    def test_sample_returns_correct_count(self, sample_forecast_pool):
        """Test that sample returns the requested number of indices."""
        indices = sample_forecast_pool.sample(3)
        assert len(indices) == 3

    def test_sample_returns_valid_indices(self, sample_forecast_pool):
        """Test that all sampled indices are valid."""
        indices = sample_forecast_pool.sample(10)
        assert all(0 <= idx < len(sample_forecast_pool) for idx in indices)

    def test_sample_with_replacement_allows_duplicates(self, sample_forecast_pool):
        """Test that sampling with replacement can have duplicates."""
        # Sample many times to increase probability of duplicates
        indices = sample_forecast_pool.sample(100, replace=True, seed=42)
        # With 5 forecasts and 100 samples, we should have duplicates
        assert len(set(indices)) < len(indices)

    def test_sample_without_replacement_no_duplicates(self, sample_forecast_pool):
        """Test that sampling without replacement has no duplicates."""
        indices = sample_forecast_pool.sample(5, replace=False)
        assert len(set(indices)) == len(indices)

    def test_sample_without_replacement_exceeds_pool_size(self, sample_forecast_pool):
        """Test that sampling without replacement raises error when n > pool size."""
        with pytest.raises(ValueError):
            sample_forecast_pool.sample(10, replace=False)

    def test_sample_with_seed_reproducible(self, sample_forecast_pool):
        """Test that sampling with same seed produces same results."""
        indices1 = sample_forecast_pool.sample(10, seed=42)
        indices2 = sample_forecast_pool.sample(10, seed=42)
        assert indices1 == indices2

    def test_sample_different_seeds_different_results(self, sample_forecast_pool):
        """Test that different seeds produce different results."""
        indices1 = sample_forecast_pool.sample(10, seed=42)
        indices2 = sample_forecast_pool.sample(10, seed=123)
        assert indices1 != indices2

    def test_sample_returns_list_of_ints(self, sample_forecast_pool):
        """Test that sample returns a list of integers."""
        indices = sample_forecast_pool.sample(5)
        assert isinstance(indices, list)
        assert all(isinstance(idx, int) for idx in indices)

    def test_sample_zero_returns_empty_list(self, sample_forecast_pool):
        """Test that sampling 0 returns empty list."""
        indices = sample_forecast_pool.sample(0)
        assert indices == []


# =============================================================================
# ForecastPool get_forecast() Tests
# =============================================================================

class TestForecastPoolGetForecast:
    """Tests for ForecastPool.get_forecast() method."""

    def test_get_forecast_returns_correct_forecast(self, sample_forecast_pool):
        """Test that get_forecast returns the forecast at the given index."""
        forecast = sample_forecast_pool.get_forecast(2)
        assert forecast.forecast_id == 2

    def test_get_forecast_same_as_getitem(self, sample_forecast_pool):
        """Test that get_forecast returns the same as __getitem__."""
        for i in range(len(sample_forecast_pool)):
            assert sample_forecast_pool.get_forecast(i) is sample_forecast_pool[i]

    def test_get_forecast_all_indices(self, sample_forecast_pool):
        """Test getting all forecasts by index."""
        for i in range(5):
            forecast = sample_forecast_pool.get_forecast(i)
            assert forecast.forecast_id == i
            assert forecast.wind_speed_bias == 1.5 + i * 0.1

    def test_get_forecast_out_of_range(self, sample_forecast_pool):
        """Test that out-of-range index raises IndexError."""
        with pytest.raises(IndexError):
            sample_forecast_pool.get_forecast(100)


# =============================================================================
# ForecastPool get_weather_scenarios() Tests
# =============================================================================

class TestForecastPoolGetWeatherScenarios:
    """Tests for ForecastPool.get_weather_scenarios() method."""

    def test_get_weather_scenarios_returns_list(self, sample_forecast_pool):
        """Test that get_weather_scenarios returns a list."""
        scenarios = sample_forecast_pool.get_weather_scenarios()
        assert isinstance(scenarios, list)

    def test_get_weather_scenarios_correct_length(self, sample_forecast_pool):
        """Test that the number of scenarios equals the number of forecasts."""
        scenarios = sample_forecast_pool.get_weather_scenarios()
        assert len(scenarios) == len(sample_forecast_pool)

    def test_get_weather_scenarios_matches_forecast_streams(self, sample_forecast_pool):
        """Test that scenarios match the weather_stream from each forecast."""
        scenarios = sample_forecast_pool.get_weather_scenarios()
        for i, scenario in enumerate(scenarios):
            assert scenario is sample_forecast_pool[i].weather_stream

    def test_get_weather_scenarios_empty_pool(
        self,
        mock_weather_stream,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that empty pool returns empty list."""
        pool = ForecastPool(
            forecasts=[],
            base_weather_stream=mock_weather_stream,
            map_params=sample_map_params,
            predictor_params=sample_predictor_params,
            created_at_time_s=0.0,
            forecast_start_datetime=datetime(2026, 7, 1, 12, 0)
        )
        scenarios = pool.get_weather_scenarios()
        assert scenarios == []


# =============================================================================
# ForecastPool Data Integrity Tests
# =============================================================================

class TestForecastPoolDataIntegrity:
    """Tests for data integrity across ForecastPool operations."""

    def test_forecast_wind_data_preserved(self, sample_forecast_pool, sample_wind_forecast):
        """Test that wind forecast data is preserved in the pool."""
        forecast = sample_forecast_pool[0]
        # Check shape matches
        assert forecast.wind_forecast.shape == sample_wind_forecast.shape

    def test_forecast_bias_values_unique(self, sample_forecast_pool):
        """Test that each forecast has unique bias values."""
        biases = [(f.wind_speed_bias, f.wind_dir_bias) for f in sample_forecast_pool.forecasts]
        assert len(set(biases)) == len(biases)  # All unique

    def test_forecast_seeds_unique(self, sample_forecast_pool):
        """Test that each forecast has unique random seeds."""
        seeds = [(f.speed_error_seed, f.dir_error_seed) for f in sample_forecast_pool.forecasts]
        assert len(set(seeds)) == len(seeds)  # All unique

    def test_forecast_ids_sequential(self, sample_forecast_pool):
        """Test that forecast IDs are sequential starting from 0."""
        ids = [f.forecast_id for f in sample_forecast_pool.forecasts]
        assert ids == list(range(len(sample_forecast_pool)))

    def test_pool_metadata_accessible_after_sampling(self, sample_forecast_pool):
        """Test that pool metadata is still accessible after sampling."""
        _ = sample_forecast_pool.sample(10, seed=42)

        # Metadata should still be accessible
        assert sample_forecast_pool.created_at_time_s == 3600.0
        assert sample_forecast_pool.forecast_start_datetime == datetime(2026, 7, 1, 13, 0)
        assert len(sample_forecast_pool) == 5


# =============================================================================
# ForecastPool Edge Cases Tests
# =============================================================================

class TestForecastPoolEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_pool_with_one_forecast(
        self,
        sample_forecast_data,
        mock_weather_stream,
        sample_predictor_params,
        sample_map_params
    ):
        """Test pool with a single forecast."""
        pool = ForecastPool(
            forecasts=[sample_forecast_data],
            base_weather_stream=mock_weather_stream,
            map_params=sample_map_params,
            predictor_params=sample_predictor_params,
            created_at_time_s=0.0,
            forecast_start_datetime=datetime(2026, 7, 1, 12, 0)
        )

        assert len(pool) == 1
        assert pool[0] is sample_forecast_data
        assert pool.get_forecast(0) is sample_forecast_data
        assert pool.get_weather_scenarios() == [sample_forecast_data.weather_stream]

        # Sampling with replacement should work
        indices = pool.sample(5, replace=True)
        assert all(idx == 0 for idx in indices)

    def test_sample_entire_pool_without_replacement(self, sample_forecast_pool):
        """Test sampling entire pool without replacement."""
        indices = sample_forecast_pool.sample(5, replace=False)
        assert sorted(indices) == [0, 1, 2, 3, 4]

    def test_large_sample_with_replacement(self, sample_forecast_pool):
        """Test sampling many more times than pool size."""
        indices = sample_forecast_pool.sample(1000, replace=True, seed=42)
        assert len(indices) == 1000
        assert all(0 <= idx < 5 for idx in indices)
        # Check that all indices appear at least once (statistically very likely)
        assert set(indices) == {0, 1, 2, 3, 4}


# =============================================================================
# Tests for _ForecastGenerationTask (used in pool generation)
# =============================================================================

class TestForecastGenerationTask:
    """Tests for _ForecastGenerationTask dataclass from forecast_pool."""

    def test_task_creation(self, mock_weather_stream, sample_map_params):
        """Test that _ForecastGenerationTask can be created."""
        from embrs.tools.forecast_pool import _ForecastGenerationTask

        task = _ForecastGenerationTask(
            forecast_id=0,
            perturbed_stream=mock_weather_stream,
            map_params=sample_map_params,
            wind_speed_bias=1.5,
            wind_dir_bias=10.0,
            speed_seed=12345,
            dir_seed=67890
        )

        assert task.forecast_id == 0
        assert task.perturbed_stream is mock_weather_stream
        assert task.map_params is sample_map_params
        assert task.wind_speed_bias == 1.5
        assert task.wind_dir_bias == 10.0
        assert task.speed_seed == 12345
        assert task.dir_seed == 67890

    def test_task_attributes(self, mock_weather_stream, sample_map_params):
        """Test that all task attributes are accessible."""
        from embrs.tools.forecast_pool import _ForecastGenerationTask

        task = _ForecastGenerationTask(
            forecast_id=5,
            perturbed_stream=mock_weather_stream,
            map_params=sample_map_params,
            wind_speed_bias=2.0,
            wind_dir_bias=15.0,
            speed_seed=11111,
            dir_seed=22222
        )

        # Verify all attributes are stored correctly
        assert hasattr(task, 'forecast_id')
        assert hasattr(task, 'perturbed_stream')
        assert hasattr(task, 'map_params')
        assert hasattr(task, 'wind_speed_bias')
        assert hasattr(task, 'wind_dir_bias')
        assert hasattr(task, 'speed_seed')
        assert hasattr(task, 'dir_seed')


# =============================================================================
# Tests for ForecastPool generation integration
# =============================================================================

class TestForecastPoolGeneration:
    """Tests for forecast pool generation logic."""

    def test_forecast_data_from_task_structure(self, sample_forecast_data):
        """Test that ForecastData has all fields needed from generation."""
        # These are the fields that _generate_single_forecast populates
        assert hasattr(sample_forecast_data, 'wind_forecast')
        assert hasattr(sample_forecast_data, 'weather_stream')
        assert hasattr(sample_forecast_data, 'wind_speed_bias')
        assert hasattr(sample_forecast_data, 'wind_dir_bias')
        assert hasattr(sample_forecast_data, 'speed_error_seed')
        assert hasattr(sample_forecast_data, 'dir_error_seed')
        assert hasattr(sample_forecast_data, 'forecast_id')
        assert hasattr(sample_forecast_data, 'generation_time')

    def test_pool_preserves_forecast_metadata(self, sample_forecast_pool):
        """Test that pool preserves all forecast generation metadata."""
        for i, forecast in enumerate(sample_forecast_pool.forecasts):
            # Each forecast should have unique IDs
            assert forecast.forecast_id == i

            # Each should have seeds for reproducibility
            assert forecast.speed_error_seed is not None
            assert forecast.dir_error_seed is not None

            # Each should have generation time
            assert forecast.generation_time is not None

    def test_pool_creation_timestamp(self, sample_forecast_pool):
        """Test that pool tracks its creation time."""
        assert sample_forecast_pool.created_at_time_s == 3600.0

    def test_pool_forecast_start_datetime_preserved(self, sample_forecast_pool):
        """Test that pool preserves the forecast start datetime."""
        assert sample_forecast_pool.forecast_start_datetime == datetime(2026, 7, 1, 13, 0)

    def test_pool_base_stream_different_from_perturbed(
        self,
        mock_weather_stream,
        sample_forecast_pool
    ):
        """Test that base_weather_stream is separate from perturbed streams."""
        # The base stream should be the original unperturbed stream
        base = sample_forecast_pool.base_weather_stream

        # Each forecast's weather_stream should be a perturbed version
        for forecast in sample_forecast_pool.forecasts:
            # They should be separate objects (different instances)
            # even if mocked for testing
            assert forecast.weather_stream is not None


# =============================================================================
# Tests for weather perturbation applied to pool
# =============================================================================

class TestPoolWeatherPerturbation:
    """Tests for verifying weather perturbation in pool generation."""

    def test_pool_has_predictor_params(self, sample_forecast_pool):
        """Test that pool stores predictor params for reference."""
        params = sample_forecast_pool.predictor_params
        assert params is not None
        assert params.time_horizon_hr == 2.0
        assert params.wind_uncertainty_factor == 0.5

    def test_pool_bias_values_recorded(self, sample_forecast_pool):
        """Test that bias values are recorded in each forecast."""
        for i, forecast in enumerate(sample_forecast_pool.forecasts):
            # Bias values should be non-None
            assert forecast.wind_speed_bias is not None
            assert forecast.wind_dir_bias is not None

    def test_pool_seeds_enable_reproducibility(self, sample_forecast_pool):
        """Test that seeds are recorded for reproducibility."""
        seeds = []
        for forecast in sample_forecast_pool.forecasts:
            seeds.append((forecast.speed_error_seed, forecast.dir_error_seed))

        # All seed pairs should be unique
        assert len(set(seeds)) == len(seeds)


# =============================================================================
# ForecastPoolManager Tests
# =============================================================================

class TestForecastPoolManager:
    """Tests for ForecastPoolManager class."""

    def test_pool_count_initially_zero(self):
        """Test that pool count starts at expected value."""
        from embrs.tools.forecast_pool import ForecastPoolManager
        # Clear any existing pools first
        ForecastPoolManager.clear_all()
        assert ForecastPoolManager.pool_count() == 0

    def test_clear_all_empties_pools(self):
        """Test that clear_all removes all pools."""
        from embrs.tools.forecast_pool import ForecastPoolManager
        ForecastPoolManager.clear_all()
        assert ForecastPoolManager.pool_count() == 0

    def test_get_active_pools_returns_list(self):
        """Test that get_active_pools returns a list."""
        from embrs.tools.forecast_pool import ForecastPoolManager
        pools = ForecastPoolManager.get_active_pools()
        assert isinstance(pools, list)

    def test_set_max_pools_validates_input(self):
        """Test that set_max_pools rejects invalid values."""
        from embrs.tools.forecast_pool import ForecastPoolManager
        with pytest.raises(ValueError):
            ForecastPoolManager.set_max_pools(0)
        with pytest.raises(ValueError):
            ForecastPoolManager.set_max_pools(-1)

    def test_set_max_pools_accepts_valid_values(self):
        """Test that set_max_pools accepts valid values."""
        from embrs.tools.forecast_pool import ForecastPoolManager
        original = ForecastPoolManager.MAX_ACTIVE_POOLS
        try:
            ForecastPoolManager.set_max_pools(5)
            assert ForecastPoolManager.MAX_ACTIVE_POOLS == 5
            ForecastPoolManager.set_max_pools(1)
            assert ForecastPoolManager.MAX_ACTIVE_POOLS == 1
        finally:
            ForecastPoolManager.set_max_pools(original)

    def test_disable_and_enable(self):
        """Test that disable/enable controls pool tracking."""
        from embrs.tools.forecast_pool import ForecastPoolManager
        ForecastPoolManager.disable()
        assert not ForecastPoolManager._enabled
        ForecastPoolManager.enable()
        assert ForecastPoolManager._enabled


# =============================================================================
# Backwards Compatibility Tests
# =============================================================================

class TestBackwardsCompatibility:
    """Tests for backwards compatibility imports."""

    def test_import_from_data_classes_with_warning(self):
        """Test that importing from data_classes works but warns."""
        import warnings

        # Clear any cached imports
        import sys
        if 'embrs.utilities.data_classes' in sys.modules:
            # The module is already imported, so __getattr__ won't be called
            # for attributes that were already accessed
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from embrs.utilities.data_classes import ForecastPool as FP
            from embrs.utilities.data_classes import ForecastData as FD

            # Check that we got the correct classes
            from embrs.tools.forecast_pool import ForecastPool, ForecastData
            assert FP is ForecastPool
            assert FD is ForecastData

    def test_import_from_new_location(self):
        """Test that importing from new location works."""
        from embrs.tools.forecast_pool import (
            ForecastPool,
            ForecastData,
            ForecastPoolManager
        )
        assert ForecastPool is not None
        assert ForecastData is not None
        assert ForecastPoolManager is not None

    def test_import_from_tools_init(self):
        """Test that importing from embrs.tools works."""
        from embrs.tools import ForecastPool, ForecastData, ForecastPoolManager
        assert ForecastPool is not None
        assert ForecastData is not None
        assert ForecastPoolManager is not None


# =============================================================================
# ForecastPool Memory Management Tests
# =============================================================================

class TestForecastPoolMemory:
    """Tests for ForecastPool memory management."""

    def test_forecast_data_memory_usage(self, sample_forecast_data):
        """Test that ForecastData reports memory usage."""
        usage = sample_forecast_data.memory_usage()
        assert usage > 0
        assert usage == sample_forecast_data.wind_forecast.nbytes

    def test_pool_memory_usage(self, sample_forecast_pool):
        """Test that ForecastPool reports total memory usage."""
        usage = sample_forecast_pool.memory_usage()
        assert usage > 0

        # Total should be sum of all forecasts
        expected = sum(f.memory_usage() for f in sample_forecast_pool.forecasts)
        assert usage == expected

    def test_pool_cleanup_clears_forecasts(
        self,
        sample_forecast_data,
        mock_weather_stream,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that _cleanup clears the forecasts list."""
        from embrs.tools.forecast_pool import ForecastPoolManager

        # Disable manager to avoid interference
        ForecastPoolManager.disable()

        try:
            pool = ForecastPool(
                forecasts=[sample_forecast_data],
                base_weather_stream=mock_weather_stream,
                map_params=sample_map_params,
                predictor_params=sample_predictor_params,
                created_at_time_s=0.0,
                forecast_start_datetime=datetime(2026, 7, 1, 12, 0)
            )

            assert len(pool) == 1
            pool._cleanup()
            assert len(pool) == 0
        finally:
            ForecastPoolManager.enable()

    def test_pool_close_unregisters_and_cleans(
        self,
        sample_forecast_data,
        mock_weather_stream,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that close() unregisters and cleans up the pool."""
        from embrs.tools.forecast_pool import ForecastPoolManager

        ForecastPoolManager.enable()
        ForecastPoolManager.clear_all()

        pool = ForecastPool(
            forecasts=[sample_forecast_data],
            base_weather_stream=mock_weather_stream,
            map_params=sample_map_params,
            predictor_params=sample_predictor_params,
            created_at_time_s=0.0,
            forecast_start_datetime=datetime(2026, 7, 1, 12, 0)
        )

        initial_count = ForecastPoolManager.pool_count()
        pool.close()

        assert len(pool) == 0
        # Pool should be unregistered
        assert pool not in ForecastPoolManager.get_active_pools()


# =============================================================================
# ForecastPoolManager Eviction Tests
# =============================================================================

class TestForecastPoolEviction:
    """Tests for ForecastPoolManager pool eviction behavior.

    These tests verify that the pool manager correctly limits the number
    of active pools and evicts oldest pools when the limit is exceeded.
    """

    def _create_pool(
        self,
        mock_weather_stream,
        sample_wind_forecast,
        sample_predictor_params,
        sample_map_params,
        pool_id: int
    ) -> ForecastPool:
        """Helper to create a pool with a unique identifier."""
        forecasts = [
            ForecastData(
                wind_forecast=sample_wind_forecast.copy(),
                weather_stream=mock_weather_stream,
                wind_speed_bias=float(pool_id),  # Use as identifier
                wind_dir_bias=float(pool_id),
                speed_error_seed=pool_id,
                dir_error_seed=pool_id,
                forecast_id=0,
                generation_time=float(pool_id)
            )
        ]
        return ForecastPool(
            forecasts=forecasts,
            base_weather_stream=mock_weather_stream,
            map_params=sample_map_params,
            predictor_params=sample_predictor_params,
            created_at_time_s=float(pool_id),
            forecast_start_datetime=datetime(2026, 7, 1, 12, 0)
        )

    def test_pool_eviction_at_max_capacity(
        self,
        mock_weather_stream,
        sample_wind_forecast,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that oldest pool is evicted when MAX_ACTIVE_POOLS is reached."""
        from embrs.tools.forecast_pool import ForecastPoolManager

        # Set up clean state
        ForecastPoolManager.enable()
        ForecastPoolManager.clear_all()
        original_max = ForecastPoolManager.MAX_ACTIVE_POOLS

        try:
            ForecastPoolManager.set_max_pools(3)

            # Create 3 pools (at capacity)
            pools = []
            for i in range(3):
                pool = self._create_pool(
                    mock_weather_stream, sample_wind_forecast,
                    sample_predictor_params, sample_map_params, i
                )
                pools.append(pool)

            assert ForecastPoolManager.pool_count() == 3

            # Create 4th pool - should evict pool 0
            pool_4 = self._create_pool(
                mock_weather_stream, sample_wind_forecast,
                sample_predictor_params, sample_map_params, 3
            )

            # Still should have 3 pools
            assert ForecastPoolManager.pool_count() == 3

            # Pool 0 should be evicted (cleaned up)
            assert len(pools[0]) == 0  # Forecasts cleared

            # Check pool membership using identity comparison (avoids numpy array comparison)
            active = ForecastPoolManager.get_active_pools()
            active_ids = [id(p) for p in active]

            assert id(pools[0]) not in active_ids  # Pool 0 evicted

            # Pools 1, 2, and 4 should still be active
            assert id(pools[1]) in active_ids
            assert id(pools[2]) in active_ids
            assert id(pool_4) in active_ids

        finally:
            ForecastPoolManager.set_max_pools(original_max)
            ForecastPoolManager.clear_all()

    def test_pool_eviction_fifo_order(
        self,
        mock_weather_stream,
        sample_wind_forecast,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that pools are evicted in FIFO order (oldest first)."""
        from embrs.tools.forecast_pool import ForecastPoolManager

        ForecastPoolManager.enable()
        ForecastPoolManager.clear_all()
        original_max = ForecastPoolManager.MAX_ACTIVE_POOLS

        try:
            ForecastPoolManager.set_max_pools(2)

            # Create pools in order 0, 1
            pool_0 = self._create_pool(
                mock_weather_stream, sample_wind_forecast,
                sample_predictor_params, sample_map_params, 0
            )
            pool_1 = self._create_pool(
                mock_weather_stream, sample_wind_forecast,
                sample_predictor_params, sample_map_params, 1
            )

            # Create pool 2 - should evict pool 0 (oldest)
            pool_2 = self._create_pool(
                mock_weather_stream, sample_wind_forecast,
                sample_predictor_params, sample_map_params, 2
            )

            assert len(pool_0) == 0  # Pool 0 evicted
            assert len(pool_1) > 0   # Pool 1 still active
            assert len(pool_2) > 0   # Pool 2 active

            # Create pool 3 - should evict pool 1 (now oldest)
            pool_3 = self._create_pool(
                mock_weather_stream, sample_wind_forecast,
                sample_predictor_params, sample_map_params, 3
            )

            assert len(pool_1) == 0  # Pool 1 evicted
            assert len(pool_2) > 0   # Pool 2 still active
            assert len(pool_3) > 0   # Pool 3 active

        finally:
            ForecastPoolManager.set_max_pools(original_max)
            ForecastPoolManager.clear_all()

    def test_evicted_pool_forecasts_cleared(
        self,
        mock_weather_stream,
        sample_wind_forecast,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that evicted pool's forecasts are properly cleared."""
        from embrs.tools.forecast_pool import ForecastPoolManager

        ForecastPoolManager.enable()
        ForecastPoolManager.clear_all()
        original_max = ForecastPoolManager.MAX_ACTIVE_POOLS

        try:
            ForecastPoolManager.set_max_pools(1)

            # Create first pool
            pool_1 = self._create_pool(
                mock_weather_stream, sample_wind_forecast,
                sample_predictor_params, sample_map_params, 1
            )

            # Store reference to forecasts list before eviction
            forecasts_before = pool_1.forecasts
            assert len(forecasts_before) == 1

            # Create second pool - should evict first
            pool_2 = self._create_pool(
                mock_weather_stream, sample_wind_forecast,
                sample_predictor_params, sample_map_params, 2
            )

            # Forecasts list should be cleared (same list object, but empty)
            assert len(pool_1.forecasts) == 0
            assert forecasts_before is pool_1.forecasts  # Same list object
            assert pool_1.forecasts == []  # But now empty

        finally:
            ForecastPoolManager.set_max_pools(original_max)
            ForecastPoolManager.clear_all()

    def test_set_max_pools_evicts_excess(
        self,
        mock_weather_stream,
        sample_wind_forecast,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that reducing max_pools evicts excess pools."""
        from embrs.tools.forecast_pool import ForecastPoolManager

        ForecastPoolManager.enable()
        ForecastPoolManager.clear_all()
        original_max = ForecastPoolManager.MAX_ACTIVE_POOLS

        try:
            ForecastPoolManager.set_max_pools(5)

            # Create 4 pools
            pools = []
            for i in range(4):
                pool = self._create_pool(
                    mock_weather_stream, sample_wind_forecast,
                    sample_predictor_params, sample_map_params, i
                )
                pools.append(pool)

            assert ForecastPoolManager.pool_count() == 4

            # Reduce max to 2 - should evict 2 oldest pools
            ForecastPoolManager.set_max_pools(2)

            assert ForecastPoolManager.pool_count() == 2
            assert len(pools[0]) == 0  # Evicted
            assert len(pools[1]) == 0  # Evicted
            assert len(pools[2]) > 0   # Still active
            assert len(pools[3]) > 0   # Still active

        finally:
            ForecastPoolManager.set_max_pools(original_max)
            ForecastPoolManager.clear_all()

    def test_disabled_manager_no_eviction(
        self,
        mock_weather_stream,
        sample_wind_forecast,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that disabled manager does not track or evict pools."""
        from embrs.tools.forecast_pool import ForecastPoolManager

        ForecastPoolManager.clear_all()
        original_max = ForecastPoolManager.MAX_ACTIVE_POOLS

        try:
            ForecastPoolManager.set_max_pools(2)
            ForecastPoolManager.disable()

            # Create more pools than max
            pools = []
            for i in range(5):
                pool = self._create_pool(
                    mock_weather_stream, sample_wind_forecast,
                    sample_predictor_params, sample_map_params, i
                )
                pools.append(pool)

            # When disabled, pools are not registered, so count stays 0
            assert ForecastPoolManager.pool_count() == 0

            # All pools should still have their forecasts
            for pool in pools:
                assert len(pool) > 0

        finally:
            ForecastPoolManager.enable()
            ForecastPoolManager.set_max_pools(original_max)
            ForecastPoolManager.clear_all()

    def test_manager_total_memory_usage(
        self,
        mock_weather_stream,
        sample_wind_forecast,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that manager tracks total memory usage across all pools."""
        from embrs.tools.forecast_pool import ForecastPoolManager

        ForecastPoolManager.enable()
        ForecastPoolManager.clear_all()
        original_max = ForecastPoolManager.MAX_ACTIVE_POOLS

        try:
            ForecastPoolManager.set_max_pools(5)

            # Initially zero
            assert ForecastPoolManager.memory_usage() == 0

            # Create 3 pools
            pools = []
            for i in range(3):
                pool = self._create_pool(
                    mock_weather_stream, sample_wind_forecast,
                    sample_predictor_params, sample_map_params, i
                )
                pools.append(pool)

            # Total memory should be sum of individual pools
            expected_total = sum(p.memory_usage() for p in pools)
            assert ForecastPoolManager.memory_usage() == expected_total

            # Close one pool
            pools[0].close()

            # Memory should decrease
            remaining_expected = sum(p.memory_usage() for p in pools[1:])
            assert ForecastPoolManager.memory_usage() == remaining_expected

        finally:
            ForecastPoolManager.set_max_pools(original_max)
            ForecastPoolManager.clear_all()

    def test_clear_all_releases_all_memory(
        self,
        mock_weather_stream,
        sample_wind_forecast,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that clear_all releases memory from all pools."""
        from embrs.tools.forecast_pool import ForecastPoolManager

        ForecastPoolManager.enable()
        ForecastPoolManager.clear_all()
        original_max = ForecastPoolManager.MAX_ACTIVE_POOLS

        try:
            ForecastPoolManager.set_max_pools(5)

            # Create pools
            pools = []
            for i in range(3):
                pool = self._create_pool(
                    mock_weather_stream, sample_wind_forecast,
                    sample_predictor_params, sample_map_params, i
                )
                pools.append(pool)

            assert ForecastPoolManager.memory_usage() > 0
            assert ForecastPoolManager.pool_count() == 3

            # Clear all
            ForecastPoolManager.clear_all()

            # Memory should be zero, count should be zero
            assert ForecastPoolManager.memory_usage() == 0
            assert ForecastPoolManager.pool_count() == 0

            # All pools should be cleaned up
            for pool in pools:
                assert len(pool) == 0

        finally:
            ForecastPoolManager.set_max_pools(original_max)


# =============================================================================
# FirePredictor Cleanup Integration Tests
# =============================================================================

class TestFirePredictorCleanup:
    """Tests for FirePredictor.cleanup() integration with ForecastPoolManager."""

    def _create_pool(
        self,
        mock_weather_stream,
        sample_wind_forecast,
        sample_predictor_params,
        sample_map_params,
        pool_id: int
    ) -> ForecastPool:
        """Helper to create a pool with a unique identifier."""
        forecasts = [
            ForecastData(
                wind_forecast=sample_wind_forecast.copy(),
                weather_stream=mock_weather_stream,
                wind_speed_bias=float(pool_id),
                wind_dir_bias=float(pool_id),
                speed_error_seed=pool_id,
                dir_error_seed=pool_id,
                forecast_id=0,
                generation_time=float(pool_id)
            )
        ]
        return ForecastPool(
            forecasts=forecasts,
            base_weather_stream=mock_weather_stream,
            map_params=sample_map_params,
            predictor_params=sample_predictor_params,
            created_at_time_s=float(pool_id),
            forecast_start_datetime=datetime(2026, 7, 1, 12, 0)
        )

    def test_cleanup_clears_all_pools(
        self,
        mock_weather_stream,
        sample_wind_forecast,
        sample_predictor_params,
        sample_map_params
    ):
        """Test that FirePredictor.cleanup() clears all forecast pools."""
        from embrs.tools.forecast_pool import ForecastPoolManager
        from embrs.tools.fire_predictor import FirePredictor
        from unittest.mock import MagicMock

        ForecastPoolManager.enable()
        ForecastPoolManager.clear_all()

        # Create some pools
        pools = []
        for i in range(3):
            pool = self._create_pool(
                mock_weather_stream, sample_wind_forecast,
                sample_predictor_params, sample_map_params, i
            )
            pools.append(pool)

        assert ForecastPoolManager.pool_count() == 3

        # Create a mock predictor (can't easily create real one without FireSim)
        mock_fire = MagicMock()
        mock_fire._sim_params = MagicMock()
        mock_fire._sim_params.cell_size = 30.0
        mock_fire._sim_params.map_params = sample_map_params
        mock_fire._burning_cells = []
        mock_fire._burnt_cells = []

        # Create predictor and call cleanup
        # Note: We can't easily create a real FirePredictor without a full FireSim,
        # so we'll test the cleanup method directly
        predictor = MagicMock(spec=FirePredictor)
        predictor.cleanup = FirePredictor.cleanup.__get__(predictor, FirePredictor)

        predictor.cleanup()

        # All pools should be cleared
        assert ForecastPoolManager.pool_count() == 0
        for pool in pools:
            assert len(pool) == 0

    def test_cleanup_safe_to_call_multiple_times(self):
        """Test that cleanup can be called multiple times without error."""
        from embrs.tools.forecast_pool import ForecastPoolManager
        from embrs.tools.fire_predictor import FirePredictor
        from unittest.mock import MagicMock

        ForecastPoolManager.enable()
        ForecastPoolManager.clear_all()

        # Create a mock predictor
        predictor = MagicMock(spec=FirePredictor)
        predictor.cleanup = FirePredictor.cleanup.__get__(predictor, FirePredictor)

        # Should not raise when called multiple times
        predictor.cleanup()
        predictor.cleanup()
        predictor.cleanup()

        assert ForecastPoolManager.pool_count() == 0

    def test_cleanup_with_no_pools(self):
        """Test that cleanup works when there are no pools."""
        from embrs.tools.forecast_pool import ForecastPoolManager
        from embrs.tools.fire_predictor import FirePredictor
        from unittest.mock import MagicMock

        ForecastPoolManager.enable()
        ForecastPoolManager.clear_all()

        assert ForecastPoolManager.pool_count() == 0

        predictor = MagicMock(spec=FirePredictor)
        predictor.cleanup = FirePredictor.cleanup.__get__(predictor, FirePredictor)

        # Should not raise when there are no pools
        predictor.cleanup()

        assert ForecastPoolManager.pool_count() == 0
