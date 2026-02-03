"""Weather management for fire simulation.

This module provides the WeatherManager class which handles all weather-related
operations for the fire simulation, including weather stream management, wind
forecast handling, and weather update logic.

Classes:
    - WeatherManager: Manages weather data and forecasts for fire simulation.
"""

from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from embrs.models.weather import WeatherStream


class WeatherManager:
    """Manages weather data and forecasts for fire simulation.

    Handles weather stream management, wind forecast data, and weather
    update timing. Used by BaseFireSim to track and update weather conditions
    during simulation.

    Attributes:
        weather_stream (WeatherStream): The weather stream object providing
            weather data over time.
        curr_weather_idx (int): Current index in the weather stream.
        wind_forecast (np.ndarray): Wind forecast array with shape
            (time_steps, rows, cols, 2) where last dimension is (speed, direction).
        wind_res (float): Wind resolution in meters.
    """

    def __init__(self,
                 weather_stream: Optional['WeatherStream'] = None,
                 wind_forecast: Optional[np.ndarray] = None,
                 wind_res: float = 100.0,
                 sim_size: Tuple[float, float] = (0.0, 0.0)):
        """Initialize the weather manager.

        Args:
            weather_stream: WeatherStream object providing weather data.
                Can be None for prediction models that don't use weather stream.
            wind_forecast: Wind forecast array. If None, defaults to zeros.
            wind_res: Wind resolution in meters.
            sim_size: Simulation domain size (width, height) in meters.
        """
        self._weather_stream = weather_stream
        self._wind_res = wind_res
        self._sim_size = sim_size

        # Weather stream index tracking
        if weather_stream is not None:
            self._sim_start_w_idx = weather_stream.sim_start_idx
            self._curr_weather_idx = weather_stream.sim_start_idx
            self._weather_t_step = weather_stream.time_step * 60  # convert minutes to seconds
        else:
            self._sim_start_w_idx = 0
            self._curr_weather_idx = 0
            self._weather_t_step = 3600  # default 1 hour

        # Weather update timing
        self._last_weather_update = 0
        self._weather_changed = True

        # Wind forecast handling
        if wind_forecast is not None:
            self._wind_forecast = wind_forecast
            self._wind_xpad, self._wind_ypad = self.calc_wind_padding(wind_forecast)
        else:
            # Default to zeros for prediction models
            self._wind_forecast = np.zeros((1, 1, 1, 2))
            self._wind_xpad = 0.0
            self._wind_ypad = 0.0

        # Reference to logger for error messages (set by parent)
        self.logger = None

    @property
    def weather_stream(self) -> Optional['WeatherStream']:
        """The weather stream object."""
        return self._weather_stream

    @property
    def curr_weather_idx(self) -> int:
        """Current index in the weather stream."""
        return self._curr_weather_idx

    @curr_weather_idx.setter
    def curr_weather_idx(self, value: int) -> None:
        """Set the current weather index."""
        self._curr_weather_idx = value

    @property
    def sim_start_w_idx(self) -> int:
        """Weather stream index at simulation start."""
        return self._sim_start_w_idx

    @property
    def weather_t_step(self) -> float:
        """Weather time step in seconds."""
        return self._weather_t_step

    @property
    def last_weather_update(self) -> float:
        """Timestamp of last weather update in seconds."""
        return self._last_weather_update

    @last_weather_update.setter
    def last_weather_update(self, value: float) -> None:
        """Set the last weather update timestamp."""
        self._last_weather_update = value

    @property
    def weather_changed(self) -> bool:
        """Whether weather has changed since last check."""
        return self._weather_changed

    @weather_changed.setter
    def weather_changed(self, value: bool) -> None:
        """Set the weather changed flag."""
        self._weather_changed = value

    @property
    def wind_forecast(self) -> np.ndarray:
        """Wind forecast array."""
        return self._wind_forecast

    @wind_forecast.setter
    def wind_forecast(self, value: np.ndarray) -> None:
        """Set the wind forecast array and recalculate padding."""
        self._wind_forecast = value
        if value is not None:
            self._wind_xpad, self._wind_ypad = self.calc_wind_padding(value)

    @property
    def wind_res(self) -> float:
        """Wind resolution in meters."""
        return self._wind_res

    @property
    def wind_xpad(self) -> float:
        """Wind x-axis padding in meters."""
        return self._wind_xpad

    @property
    def wind_ypad(self) -> float:
        """Wind y-axis padding in meters."""
        return self._wind_ypad

    def update_weather(self, curr_time_s: float) -> bool:
        """Updates the current wind conditions based on the forecast.

        This method checks whether the time elapsed since the last wind update
        exceeds the wind forecast time step. If so, it updates the wind index
        and retrieves the next forecasted wind condition. If the forecast has
        no remaining entries, it raises a ValueError.

        Args:
            curr_time_s: Current simulation time in seconds.

        Returns:
            bool: True if the wind conditions were updated, False otherwise.

        Raises:
            ValueError: If the wind forecast runs out of entries.

        Side Effects:
            - Updates _last_weather_update to the current simulation time.
            - Increments _curr_weather_idx to the next wind forecast entry.
            - Resets _curr_weather_idx to 0 if out of bounds and raises an error.
        """
        # Check if a wind forecast time step has elapsed since last update
        weather_changed = curr_time_s - self._last_weather_update >= self._weather_t_step

        if weather_changed:
            # Reset last wind update to current time
            self._last_weather_update = curr_time_s

            # Increment wind index
            self._curr_weather_idx += 1

            # Check for out of bounds index
            if self._weather_stream is not None:
                if self._curr_weather_idx >= len(self._weather_stream.stream):
                    self._curr_weather_idx = 0
                    raise ValueError("Weather forecast has no more entries!")

        self._weather_changed = weather_changed
        return weather_changed

    def calc_wind_padding(self, forecast: np.ndarray) -> Tuple[float, float]:
        """Calculate padding offsets between wind forecast grid and simulation grid.

        The wind forecast grid may not align exactly with the simulation
        boundaries. This calculates the x and y offsets needed to center
        the forecast within the simulation domain.

        Args:
            forecast: Wind forecast array with shape
                (time_steps, rows, cols, 2) where last dimension is (speed, direction).

        Returns:
            Tuple[float, float]: (x_padding, y_padding) in meters.
        """
        forecast_rows = forecast[0, :, :, 0].shape[0]
        forecast_cols = forecast[0, :, :, 1].shape[1]

        forecast_height = forecast_rows * self._wind_res
        forecast_width = forecast_cols * self._wind_res

        xpad = (self._sim_size[0] - forecast_width) / 2
        ypad = (self._sim_size[1] - forecast_height) / 2

        return xpad, ypad

    def get_wind_indices(self, cell_x: float, cell_y: float) -> Tuple[int, int]:
        """Get wind forecast array indices for a cell position.

        Args:
            cell_x: Cell x position in meters.
            cell_y: Cell y position in meters.

        Returns:
            Tuple[int, int]: (wind_row, wind_col) indices into wind forecast array.
        """
        x_wind = max(cell_x - self._wind_xpad, 0)
        y_wind = max(cell_y - self._wind_ypad, 0)

        wind_col = int(np.floor(x_wind / self._wind_res))
        wind_row = int(np.floor(y_wind / self._wind_res))

        # Clamp to forecast bounds
        if wind_row > self._wind_forecast.shape[1] - 1:
            wind_row = self._wind_forecast.shape[1] - 1
        if wind_col > self._wind_forecast.shape[2] - 1:
            wind_col = self._wind_forecast.shape[2] - 1

        return wind_row, wind_col

    def get_cell_wind(self, cell_x: float, cell_y: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get wind speed and direction arrays for a cell position.

        Args:
            cell_x: Cell x position in meters.
            cell_y: Cell y position in meters.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (wind_speed, wind_dir) arrays
                across all forecast time steps.
        """
        wind_row, wind_col = self.get_wind_indices(cell_x, cell_y)

        wind_speed = self._wind_forecast[:, wind_row, wind_col, 0]
        wind_dir = self._wind_forecast[:, wind_row, wind_col, 1]

        return wind_speed, wind_dir
