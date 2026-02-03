"""Unit tests for weather-related operations in BaseFireSim and WeatherManager.

These tests verify the behavior of weather management methods to ensure
the WeatherManager class matches the original BaseFireSim behavior.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from embrs.base_classes.weather_manager import WeatherManager


class MockWeatherStreamEntry:
    """Mock weather stream entry for testing."""
    def __init__(self, time_s=0, temp_f=75.0, rel_humidity_pct=30.0,
                 wind_speed_mph=10.0, wind_dir_deg=180.0, cloud_cover_pct=0.0):
        self.time_s = time_s
        self.temp_f = temp_f
        self.rel_humidity_pct = rel_humidity_pct
        self.wind_speed_mph = wind_speed_mph
        self.wind_dir_deg = wind_dir_deg
        self.cloud_cover_pct = cloud_cover_pct


class MockWeatherStream:
    """Mock weather stream for testing."""
    def __init__(self, num_entries=10, time_step_min=60):
        self.stream = [
            MockWeatherStreamEntry(time_s=i * time_step_min * 60)
            for i in range(num_entries)
        ]
        self.time_step = time_step_min  # in minutes
        self.sim_start_idx = 0
        self.live_h_mf = 0.5
        self.live_w_mf = 0.6
        self.fmc = 100


class MockWeatherManager:
    """Mock class containing weather methods extracted from BaseFireSim.

    This class replicates the exact weather logic from BaseFireSim for testing.
    """

    def __init__(self, weather_stream: MockWeatherStream, wind_res: float = 100.0,
                 size: tuple = (10000.0, 10000.0)):
        self._weather_stream = weather_stream
        self._curr_weather_idx = weather_stream.sim_start_idx
        self._last_weather_update = 0
        self.weather_t_step = weather_stream.time_step * 60  # convert to seconds
        self._wind_res = wind_res
        self._size = size
        self._curr_time_s = 0

    @property
    def curr_time_s(self):
        return self._curr_time_s

    @property
    def size(self):
        return self._size

    def _update_weather(self) -> bool:
        """Updates the current wind conditions based on the forecast.

        This method checks whether the time elapsed since the last wind update exceeds
        the wind forecast time step. If so, it updates the wind index and retrieves
        the next forecasted wind condition. If the forecast has no remaining entries,
        it raises a ValueError.

        Returns:
            bool: True if the wind conditions were updated, False otherwise.

        Raises:
            ValueError: If the wind forecast runs out of entries.
        """
        # Check if a wind forecast time step has elapsed since last update
        weather_changed = self.curr_time_s - self._last_weather_update >= self.weather_t_step

        if weather_changed:
            # Reset last wind update to current time
            self._last_weather_update = self.curr_time_s

            # Increment wind index
            self._curr_weather_idx += 1

            # Check for out of bounds index
            if self._curr_weather_idx >= len(self._weather_stream.stream):
                self._curr_weather_idx = 0
                raise ValueError("Weather forecast has no more entries!")

        return weather_changed

    def calc_wind_padding(self, forecast: np.ndarray) -> tuple:
        """Calculate padding offsets between wind forecast grid and simulation grid.

        The wind forecast grid may not align exactly with the simulation
        boundaries. This calculates the x and y offsets needed to center
        the forecast within the simulation domain.

        Args:
            forecast (np.ndarray): Wind forecast array with shape
                (time_steps, rows, cols, 2) where last dimension is (speed, direction).

        Returns:
            Tuple[float, float]: (x_padding, y_padding) in meters.
        """
        forecast_rows = forecast[0, :, :, 0].shape[0]
        forecast_cols = forecast[0, :, :, 1].shape[1]

        forecast_height = forecast_rows * self._wind_res
        forecast_width = forecast_cols * self._wind_res

        xpad = (self.size[0] - forecast_width) / 2
        ypad = (self.size[1] - forecast_height) / 2

        return xpad, ypad


class TestUpdateWeather:
    """Tests for _update_weather method."""

    def test_no_update_when_insufficient_time_elapsed(self):
        """Weather should not update when less than weather_t_step has elapsed."""
        weather_stream = MockWeatherStream(num_entries=10, time_step_min=60)
        manager = MockWeatherManager(weather_stream)

        # Set current time less than time step
        manager._curr_time_s = 30 * 60  # 30 minutes

        # Initial weather index
        initial_idx = manager._curr_weather_idx

        # Should not update
        result = manager._update_weather()

        assert result is False
        assert manager._curr_weather_idx == initial_idx

    def test_update_when_time_step_elapsed(self):
        """Weather should update when exactly weather_t_step has elapsed."""
        weather_stream = MockWeatherStream(num_entries=10, time_step_min=60)
        manager = MockWeatherManager(weather_stream)

        # Set current time to exactly one time step
        manager._curr_time_s = 60 * 60  # 60 minutes = 3600 seconds

        initial_idx = manager._curr_weather_idx

        result = manager._update_weather()

        assert result is True
        assert manager._curr_weather_idx == initial_idx + 1
        assert manager._last_weather_update == manager._curr_time_s

    def test_update_when_more_than_time_step_elapsed(self):
        """Weather should update when more than weather_t_step has elapsed."""
        weather_stream = MockWeatherStream(num_entries=10, time_step_min=60)
        manager = MockWeatherManager(weather_stream)

        # Set current time to more than one time step
        manager._curr_time_s = 90 * 60  # 90 minutes

        initial_idx = manager._curr_weather_idx

        result = manager._update_weather()

        assert result is True
        assert manager._curr_weather_idx == initial_idx + 1

    def test_multiple_sequential_updates(self):
        """Multiple weather updates should progress through stream."""
        weather_stream = MockWeatherStream(num_entries=10, time_step_min=60)
        manager = MockWeatherManager(weather_stream)

        # Simulate multiple time steps
        for i in range(1, 5):
            manager._curr_time_s = i * 60 * 60  # i hours
            result = manager._update_weather()
            assert result is True
            assert manager._curr_weather_idx == i

    def test_raises_error_when_forecast_exhausted(self):
        """Should raise ValueError when weather forecast runs out."""
        weather_stream = MockWeatherStream(num_entries=3, time_step_min=60)
        manager = MockWeatherManager(weather_stream)

        # Move to end of stream
        manager._curr_weather_idx = 2
        manager._curr_time_s = 3 * 60 * 60  # 3 hours

        with pytest.raises(ValueError, match="Weather forecast has no more entries"):
            manager._update_weather()

    def test_index_resets_on_error(self):
        """Index should reset to 0 when forecast is exhausted."""
        weather_stream = MockWeatherStream(num_entries=3, time_step_min=60)
        manager = MockWeatherManager(weather_stream)

        manager._curr_weather_idx = 2
        manager._curr_time_s = 3 * 60 * 60

        try:
            manager._update_weather()
        except ValueError:
            pass

        assert manager._curr_weather_idx == 0

    def test_last_weather_update_tracks_update_time(self):
        """_last_weather_update should track when last update occurred."""
        weather_stream = MockWeatherStream(num_entries=10, time_step_min=60)
        manager = MockWeatherManager(weather_stream)

        assert manager._last_weather_update == 0

        # First update
        manager._curr_time_s = 60 * 60
        manager._update_weather()
        assert manager._last_weather_update == 60 * 60

        # No update (not enough time)
        manager._curr_time_s = 90 * 60
        manager._update_weather()
        assert manager._last_weather_update == 60 * 60  # Unchanged

        # Second update
        manager._curr_time_s = 120 * 60
        manager._update_weather()
        assert manager._last_weather_update == 120 * 60

    def test_different_time_step_values(self):
        """Test with different weather time step values."""
        # 30 minute time step
        weather_stream = MockWeatherStream(num_entries=10, time_step_min=30)
        manager = MockWeatherManager(weather_stream)

        manager._curr_time_s = 30 * 60  # 30 minutes
        result = manager._update_weather()
        assert result is True

        # 15 minute time step
        weather_stream2 = MockWeatherStream(num_entries=10, time_step_min=15)
        manager2 = MockWeatherManager(weather_stream2)

        manager2._curr_time_s = 15 * 60  # 15 minutes
        result = manager2._update_weather()
        assert result is True


class TestCalcWindPadding:
    """Tests for calc_wind_padding method."""

    def test_centered_forecast(self):
        """Padding should be zero when forecast matches sim size exactly."""
        weather_stream = MockWeatherStream()
        wind_res = 100.0
        size = (1000.0, 1000.0)  # 1km x 1km
        manager = MockWeatherManager(weather_stream, wind_res=wind_res, size=size)

        # Forecast grid: 10x10 cells at 100m resolution = 1000m x 1000m
        forecast = np.zeros((1, 10, 10, 2))

        xpad, ypad = manager.calc_wind_padding(forecast)

        assert xpad == pytest.approx(0.0)
        assert ypad == pytest.approx(0.0)

    def test_smaller_forecast_than_sim(self):
        """Padding should be positive when forecast is smaller than sim."""
        weather_stream = MockWeatherStream()
        wind_res = 100.0
        size = (2000.0, 2000.0)  # 2km x 2km
        manager = MockWeatherManager(weather_stream, wind_res=wind_res, size=size)

        # Forecast grid: 10x10 cells at 100m resolution = 1000m x 1000m
        forecast = np.zeros((1, 10, 10, 2))

        xpad, ypad = manager.calc_wind_padding(forecast)

        # Sim is 2000m, forecast is 1000m, padding is (2000-1000)/2 = 500
        assert xpad == pytest.approx(500.0)
        assert ypad == pytest.approx(500.0)

    def test_larger_forecast_than_sim(self):
        """Padding should be negative when forecast is larger than sim."""
        weather_stream = MockWeatherStream()
        wind_res = 100.0
        size = (500.0, 500.0)  # 0.5km x 0.5km
        manager = MockWeatherManager(weather_stream, wind_res=wind_res, size=size)

        # Forecast grid: 10x10 cells at 100m resolution = 1000m x 1000m
        forecast = np.zeros((1, 10, 10, 2))

        xpad, ypad = manager.calc_wind_padding(forecast)

        # Sim is 500m, forecast is 1000m, padding is (500-1000)/2 = -250
        assert xpad == pytest.approx(-250.0)
        assert ypad == pytest.approx(-250.0)

    def test_asymmetric_forecast(self):
        """Padding should handle non-square forecast grids."""
        weather_stream = MockWeatherStream()
        wind_res = 100.0
        size = (2000.0, 3000.0)  # 2km x 3km (width x height)
        manager = MockWeatherManager(weather_stream, wind_res=wind_res, size=size)

        # Forecast grid: 15x10 cells (rows x cols) at 100m = 1500m height x 1000m width
        forecast = np.zeros((1, 15, 10, 2))

        xpad, ypad = manager.calc_wind_padding(forecast)

        # Width: sim 2000m, forecast 1000m, xpad = (2000-1000)/2 = 500
        # Height: sim 3000m, forecast 1500m, ypad = (3000-1500)/2 = 750
        assert xpad == pytest.approx(500.0)
        assert ypad == pytest.approx(750.0)

    def test_different_wind_resolution(self):
        """Padding should account for wind resolution."""
        weather_stream = MockWeatherStream()
        wind_res = 50.0  # 50m resolution
        size = (1000.0, 1000.0)
        manager = MockWeatherManager(weather_stream, wind_res=wind_res, size=size)

        # Forecast grid: 10x10 cells at 50m resolution = 500m x 500m
        forecast = np.zeros((1, 10, 10, 2))

        xpad, ypad = manager.calc_wind_padding(forecast)

        # Sim is 1000m, forecast is 500m, padding is (1000-500)/2 = 250
        assert xpad == pytest.approx(250.0)
        assert ypad == pytest.approx(250.0)

    def test_multiple_time_steps_in_forecast(self):
        """Padding calculation should work with multiple time steps."""
        weather_stream = MockWeatherStream()
        wind_res = 100.0
        size = (1500.0, 1500.0)
        manager = MockWeatherManager(weather_stream, wind_res=wind_res, size=size)

        # Forecast with 5 time steps: 5x10x10 cells
        forecast = np.zeros((5, 10, 10, 2))

        xpad, ypad = manager.calc_wind_padding(forecast)

        # Should use first time step dimensions
        # Sim 1500m, forecast 1000m, padding = (1500-1000)/2 = 250
        assert xpad == pytest.approx(250.0)
        assert ypad == pytest.approx(250.0)

    def test_single_cell_forecast(self):
        """Padding should work with single-cell forecast."""
        weather_stream = MockWeatherStream()
        wind_res = 1000.0  # 1km resolution
        size = (5000.0, 5000.0)  # 5km x 5km
        manager = MockWeatherManager(weather_stream, wind_res=wind_res, size=size)

        # Single cell forecast
        forecast = np.zeros((1, 1, 1, 2))

        xpad, ypad = manager.calc_wind_padding(forecast)

        # Sim 5000m, forecast 1000m, padding = (5000-1000)/2 = 2000
        assert xpad == pytest.approx(2000.0)
        assert ypad == pytest.approx(2000.0)


class TestWeatherStreamIntegration:
    """Integration tests for weather stream handling."""

    def test_weather_index_progression_over_simulation(self):
        """Simulate weather progression over multiple hours."""
        weather_stream = MockWeatherStream(num_entries=24, time_step_min=60)
        manager = MockWeatherManager(weather_stream)

        time_step_s = 60  # 1 minute simulation time step

        # Run for 5 hours worth of simulation time (inclusive of 5 hour mark)
        for sim_time_s in range(0, 5 * 60 * 60 + time_step_s, time_step_s):
            manager._curr_time_s = sim_time_s
            manager._update_weather()

        # After 5 hours, should have progressed through 5 weather entries
        # Updates happen at: 1hr (idx=1), 2hr (idx=2), 3hr (idx=3), 4hr (idx=4), 5hr (idx=5)
        assert manager._curr_weather_idx == 5

    def test_weather_consistency_with_irregular_updates(self):
        """Weather should update correctly with irregular time intervals.

        The update logic tracks elapsed time since last update:
        - time=0: idx=0 (no update, initial state)
        - time=1800 (30min): elapsed from 0 = 1800 < 3600, no update, idx=0
        - time=3900 (65min): elapsed from 0 = 3900 >= 3600, update! idx=1, last_update=3900
        - time=4200 (70min): elapsed from 3900 = 300 < 3600, no update, idx=1
        - time=7500 (125min): elapsed from 3900 = 3600 >= 3600, update! idx=2, last_update=7500
        - time=10800 (180min): elapsed from 7500 = 3300 < 3600, no update, idx=2
        """
        weather_stream = MockWeatherStream(num_entries=10, time_step_min=60)
        manager = MockWeatherManager(weather_stream)

        # Irregular time progression with expected indices based on update logic
        times = [0, 30*60, 65*60, 70*60, 125*60, 180*60]
        expected_idx = [0, 0, 1, 1, 2, 2]

        for time_s, expected in zip(times, expected_idx):
            manager._curr_time_s = time_s
            manager._update_weather()
            assert manager._curr_weather_idx == expected, f"At time {time_s}s, expected idx {expected}, got {manager._curr_weather_idx}"


# ============================================================================
# Tests for the actual WeatherManager class
# ============================================================================

class TestWeatherManagerInit:
    """Tests for WeatherManager initialization."""

    def test_init_with_weather_stream(self):
        """WeatherManager should initialize correctly with weather stream."""
        weather_stream = MockWeatherStream(num_entries=10, time_step_min=60)
        manager = WeatherManager(
            weather_stream=weather_stream,
            wind_res=100.0,
            sim_size=(1000.0, 1000.0)
        )

        assert manager.weather_stream is weather_stream
        assert manager.curr_weather_idx == 0
        assert manager.weather_t_step == 3600  # 60 min * 60 s/min

    def test_init_without_weather_stream(self):
        """WeatherManager should initialize correctly without weather stream."""
        manager = WeatherManager(
            weather_stream=None,
            wind_res=100.0,
            sim_size=(1000.0, 1000.0)
        )

        assert manager.weather_stream is None
        assert manager.curr_weather_idx == 0
        assert manager.weather_t_step == 3600  # default

    def test_init_with_wind_forecast(self):
        """WeatherManager should calculate padding when forecast provided."""
        weather_stream = MockWeatherStream()
        wind_forecast = np.zeros((1, 10, 10, 2))  # 10x10 grid

        manager = WeatherManager(
            weather_stream=weather_stream,
            wind_forecast=wind_forecast,
            wind_res=100.0,
            sim_size=(2000.0, 2000.0)
        )

        # Forecast is 1000m x 1000m, sim is 2000m x 2000m
        # Padding should be (2000-1000)/2 = 500
        assert manager.wind_xpad == pytest.approx(500.0)
        assert manager.wind_ypad == pytest.approx(500.0)


class TestWeatherManagerUpdateWeather:
    """Tests for WeatherManager.update_weather method."""

    def test_no_update_insufficient_time(self):
        """No update when less than time step has elapsed."""
        weather_stream = MockWeatherStream(num_entries=10, time_step_min=60)
        manager = WeatherManager(weather_stream=weather_stream)

        result = manager.update_weather(30 * 60)  # 30 minutes

        assert result is False
        assert manager.curr_weather_idx == 0

    def test_update_when_time_step_elapsed(self):
        """Update when exactly one time step has elapsed."""
        weather_stream = MockWeatherStream(num_entries=10, time_step_min=60)
        manager = WeatherManager(weather_stream=weather_stream)

        result = manager.update_weather(60 * 60)  # 60 minutes

        assert result is True
        assert manager.curr_weather_idx == 1

    def test_raises_when_forecast_exhausted(self):
        """Should raise ValueError when forecast runs out."""
        weather_stream = MockWeatherStream(num_entries=3, time_step_min=60)
        manager = WeatherManager(weather_stream=weather_stream)

        manager.curr_weather_idx = 2
        manager.last_weather_update = 2 * 60 * 60

        with pytest.raises(ValueError, match="Weather forecast has no more entries"):
            manager.update_weather(3 * 60 * 60)


class TestWeatherManagerCalcWindPadding:
    """Tests for WeatherManager.calc_wind_padding method."""

    def test_centered_forecast(self):
        """Padding should be zero when forecast matches sim size."""
        manager = WeatherManager(
            wind_res=100.0,
            sim_size=(1000.0, 1000.0)
        )

        forecast = np.zeros((1, 10, 10, 2))
        xpad, ypad = manager.calc_wind_padding(forecast)

        assert xpad == pytest.approx(0.0)
        assert ypad == pytest.approx(0.0)

    def test_smaller_forecast(self):
        """Positive padding when forecast smaller than sim."""
        manager = WeatherManager(
            wind_res=100.0,
            sim_size=(2000.0, 2000.0)
        )

        forecast = np.zeros((1, 10, 10, 2))
        xpad, ypad = manager.calc_wind_padding(forecast)

        assert xpad == pytest.approx(500.0)
        assert ypad == pytest.approx(500.0)


class TestWeatherManagerGetWindIndices:
    """Tests for WeatherManager.get_wind_indices method."""

    def test_get_wind_indices_origin(self):
        """Indices at origin should account for padding."""
        wind_forecast = np.zeros((1, 10, 10, 2))
        manager = WeatherManager(
            wind_forecast=wind_forecast,
            wind_res=100.0,
            sim_size=(2000.0, 2000.0)  # Padding = 500m
        )

        # At origin, with 500m padding, x_wind = max(0-500, 0) = 0
        row, col = manager.get_wind_indices(0, 0)

        assert row == 0
        assert col == 0

    def test_get_wind_indices_center(self):
        """Indices at center should be in middle of forecast."""
        wind_forecast = np.zeros((1, 10, 10, 2))
        manager = WeatherManager(
            wind_forecast=wind_forecast,
            wind_res=100.0,
            sim_size=(1000.0, 1000.0)  # No padding
        )

        # At center (500, 500), wind col/row = 5
        row, col = manager.get_wind_indices(500, 500)

        assert row == 5
        assert col == 5

    def test_get_wind_indices_clamped_to_bounds(self):
        """Indices should be clamped to forecast array bounds."""
        wind_forecast = np.zeros((1, 5, 5, 2))
        manager = WeatherManager(
            wind_forecast=wind_forecast,
            wind_res=100.0,
            sim_size=(500.0, 500.0)
        )

        # Request beyond bounds
        row, col = manager.get_wind_indices(1000, 1000)

        # Should be clamped to max index (4, 4)
        assert row == 4
        assert col == 4


class TestWeatherManagerGetCellWind:
    """Tests for WeatherManager.get_cell_wind method."""

    def test_get_cell_wind_returns_arrays(self):
        """Should return wind speed and direction arrays."""
        wind_forecast = np.ones((3, 5, 5, 2))
        wind_forecast[:, :, :, 0] = 10.0  # Speed
        wind_forecast[:, :, :, 1] = 180.0  # Direction

        manager = WeatherManager(
            wind_forecast=wind_forecast,
            wind_res=100.0,
            sim_size=(500.0, 500.0)
        )

        speed, direction = manager.get_cell_wind(250, 250)

        assert len(speed) == 3  # 3 time steps
        assert len(direction) == 3
        assert all(s == 10.0 for s in speed)
        assert all(d == 180.0 for d in direction)


class TestWeatherManagerBehaviorMatch:
    """Tests to verify WeatherManager matches original BaseFireSim behavior."""

    def test_update_weather_matches_mock(self):
        """WeatherManager.update_weather should match MockWeatherManager behavior."""
        weather_stream = MockWeatherStream(num_entries=10, time_step_min=60)

        mock_manager = MockWeatherManager(weather_stream)
        real_manager = WeatherManager(weather_stream=weather_stream)

        # Simulate multiple time steps
        times = [0, 30*60, 60*60, 90*60, 120*60, 150*60, 180*60]

        for time_s in times:
            mock_manager._curr_time_s = time_s
            mock_result = mock_manager._update_weather()
            real_result = real_manager.update_weather(time_s)

            assert real_result == mock_result, f"Mismatch at time {time_s}"
            assert real_manager.curr_weather_idx == mock_manager._curr_weather_idx, \
                f"Index mismatch at time {time_s}"

    def test_calc_wind_padding_matches_mock(self):
        """WeatherManager.calc_wind_padding should match MockWeatherManager behavior."""
        weather_stream = MockWeatherStream()
        wind_res = 100.0
        size = (1500.0, 2000.0)

        mock_manager = MockWeatherManager(weather_stream, wind_res=wind_res, size=size)
        real_manager = WeatherManager(
            weather_stream=weather_stream,
            wind_res=wind_res,
            sim_size=size
        )

        forecast = np.zeros((1, 10, 15, 2))  # 10 rows, 15 cols

        mock_xpad, mock_ypad = mock_manager.calc_wind_padding(forecast)
        real_xpad, real_ypad = real_manager.calc_wind_padding(forecast)

        assert real_xpad == pytest.approx(mock_xpad)
        assert real_ypad == pytest.approx(mock_ypad)
