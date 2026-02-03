"""Forecast pool management for ensemble fire predictions.

This module provides classes for managing pre-computed wind forecast pools
that can be reused across multiple ensemble fire predictions.

Classes:
    - ForecastData: Container for a single wind forecast with metadata.
    - ForecastPool: Collection of forecasts with pool size management.
    - ForecastPoolManager: Global manager for active forecast pools.

The forecast pool system allows efficient reuse of WindNinja computations
across global predictions and rollout scenarios.

Example:
    >>> from embrs.tools.forecast_pool import ForecastPool
    >>>
    >>> # Create a forecast pool from a fire predictor
    >>> pool = ForecastPool.generate(
    ...     fire=fire_sim,
    ...     predictor_params=params,
    ...     n_forecasts=30,
    ...     num_workers=4
    ... )
    >>>
    >>> # Use the pool in ensemble predictions
    >>> output = predictor.run_ensemble(
    ...     state_estimates=estimates,
    ...     forecast_pool=pool
    ... )
"""

from __future__ import annotations

import copy
import os
import uuid
import time as time_module
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import cpu_count
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from embrs.models.weather import WeatherStream
    from embrs.utilities.data_classes import MapParams, PredictorParams
    from embrs.fire_simulator.fire import FireSim


# =============================================================================
# Module-level pool management
# =============================================================================

class ForecastPoolManager:
    """Manages active forecast pools to prevent unbounded memory growth.

    This class tracks all active ForecastPool instances and enforces a
    maximum number of active pools. When a new pool is created and the
    limit is exceeded, the oldest pool is automatically cleaned up.

    Class Attributes:
        MAX_ACTIVE_POOLS (int): Maximum number of active pools (default: 3).
        _active_pools (List[ForecastPool]): Currently active pools.
        _enabled (bool): Whether pool management is enabled.

    Example:
        >>> # Check current pool count
        >>> print(f"Active pools: {ForecastPoolManager.pool_count()}")
        >>>
        >>> # Clear all pools when done
        >>> ForecastPoolManager.clear_all()
    """

    MAX_ACTIVE_POOLS: int = 3
    _active_pools: List['ForecastPool'] = []
    _enabled: bool = True

    @classmethod
    def register(cls, pool: 'ForecastPool') -> None:
        """Register a new pool and evict oldest if over limit.

        Args:
            pool: The ForecastPool to register.
        """
        if not cls._enabled:
            return

        # Evict oldest pool if we're at capacity
        while len(cls._active_pools) >= cls.MAX_ACTIVE_POOLS:
            oldest = cls._active_pools.pop(0)
            oldest._cleanup()

        cls._active_pools.append(pool)

    @classmethod
    def unregister(cls, pool: 'ForecastPool') -> None:
        """Unregister a pool without cleanup.

        Args:
            pool: The ForecastPool to unregister.
        """
        if pool in cls._active_pools:
            cls._active_pools.remove(pool)

    @classmethod
    def clear_all(cls) -> None:
        """Clear all active pools and release memory."""
        for pool in cls._active_pools:
            pool._cleanup()
        cls._active_pools.clear()

    @classmethod
    def pool_count(cls) -> int:
        """Return the number of active pools."""
        return len(cls._active_pools)

    @classmethod
    def set_max_pools(cls, n: int) -> None:
        """Set the maximum number of active pools.

        Args:
            n: Maximum number of active pools (must be >= 1).

        Raises:
            ValueError: If n < 1.
        """
        if n < 1:
            raise ValueError("MAX_ACTIVE_POOLS must be at least 1")
        cls.MAX_ACTIVE_POOLS = n

        # Evict excess pools if needed
        while len(cls._active_pools) > cls.MAX_ACTIVE_POOLS:
            oldest = cls._active_pools.pop(0)
            oldest._cleanup()

    @classmethod
    def disable(cls) -> None:
        """Disable automatic pool management."""
        cls._enabled = False

    @classmethod
    def enable(cls) -> None:
        """Enable automatic pool management."""
        cls._enabled = True

    @classmethod
    def get_active_pools(cls) -> List['ForecastPool']:
        """Return list of active pools (read-only copy)."""
        return list(cls._active_pools)

    @classmethod
    def memory_usage(cls) -> int:
        """Estimate total memory usage of all active pools in bytes."""
        total = 0
        for pool in cls._active_pools:
            total += pool.memory_usage()
        return total


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class ForecastData:
    """Container for a single wind forecast and its generating parameters.

    Stores a WindNinja output array along with the perturbation parameters
    used to generate it, enabling reproducibility and reuse of forecasts
    across ensemble predictions.

    Attributes:
        wind_forecast: WindNinja output array.
            Shape: (n_timesteps, height, width, 2) where [..., 0] = speed (m/s),
            [..., 1] = direction (degrees).
        weather_stream: The perturbed weather stream used to generate this forecast.
        wind_speed_bias: Constant wind speed bias applied (m/s).
        wind_dir_bias: Constant wind direction bias applied (degrees).
        speed_error_seed: Random seed used for AR(1) speed noise.
        dir_error_seed: Random seed used for AR(1) direction noise.
        forecast_id: Unique identifier for this forecast within the pool.
        generation_time: Unix timestamp when forecast was generated.
    """

    wind_forecast: np.ndarray
    weather_stream: 'WeatherStream'
    wind_speed_bias: float
    wind_dir_bias: float
    speed_error_seed: int
    dir_error_seed: int
    forecast_id: int
    generation_time: float

    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        return self.wind_forecast.nbytes


@dataclass
class _ForecastGenerationTask:
    """Internal task descriptor for parallel forecast generation.

    Encapsulates all data needed by a worker process to generate a single
    perturbed wind forecast via WindNinja.

    Attributes:
        forecast_id: Unique identifier for this forecast within the pool.
        perturbed_stream: Weather stream with AR(1) perturbations applied.
        map_params: Map parameters for WindNinja terrain processing.
        wind_speed_bias: Constant bias added to wind speed in m/s.
        wind_dir_bias: Constant bias added to wind direction in degrees.
        speed_seed: Random seed used for speed perturbation reproducibility.
        dir_seed: Random seed used for direction perturbation reproducibility.
    """

    forecast_id: int
    perturbed_stream: 'WeatherStream'
    map_params: 'MapParams'
    wind_speed_bias: float
    wind_dir_bias: float
    speed_seed: int
    dir_seed: int


def _generate_single_forecast(task: _ForecastGenerationTask) -> ForecastData:
    """Generate a single wind forecast via WindNinja.

    Worker function executed in a separate process via ProcessPoolExecutor.
    Creates a unique temporary directory to avoid file conflicts when multiple
    forecasts are generated in parallel.

    Args:
        task: Task descriptor with weather stream, map parameters, bias values,
            and random seeds.

    Returns:
        ForecastData: Generated wind forecast with metadata including the
            perturbed weather stream, bias values, and generation timestamp.
    """
    from embrs.models.wind_forecast import run_windninja, temp_file_path

    # Create unique temp directory for this forecast to avoid conflicts
    worker_id = f"pool_{task.forecast_id}_{uuid.uuid4().hex[:6]}"
    custom_temp = os.path.join(temp_file_path, worker_id)

    # Run WindNinja with num_workers=1 to disable internal parallelization
    # (forecast-level parallelization is already happening in generate())
    forecast_array = run_windninja(
        task.perturbed_stream,
        task.map_params,
        custom_temp,
        num_workers=1
    )

    # Apply flipud transformation (required for coordinate alignment)
    for layer in range(forecast_array.shape[0]):
        forecast_array[layer] = np.flipud(forecast_array[layer])

    return ForecastData(
        wind_forecast=forecast_array,
        weather_stream=task.perturbed_stream,
        wind_speed_bias=task.wind_speed_bias,
        wind_dir_bias=task.wind_dir_bias,
        speed_error_seed=task.speed_seed,
        dir_error_seed=task.dir_seed,
        forecast_id=task.forecast_id,
        generation_time=time_module.time()
    )


# =============================================================================
# ForecastPool class
# =============================================================================

@dataclass
class ForecastPool:
    """A collection of pre-computed wind forecasts for ensemble use.

    Provides storage and sampling methods for a pool of perturbed wind
    forecasts that can be reused across global predictions and rollouts.

    The ForecastPool class now owns the pool generation process, making
    it the central point for creating and managing forecast pools.

    Attributes:
        forecasts: List of ForecastData objects.
        base_weather_stream: Original unperturbed weather stream.
        map_params: Map parameters used for WindNinja.
        predictor_params: Predictor parameters at time of pool creation.
        created_at_time_s: Simulation time (seconds) when pool was created.
        forecast_start_datetime: Local datetime that index 0 of forecasts
            corresponds to.

    Example:
        >>> # Create a pool from fire simulation
        >>> pool = ForecastPool.generate(
        ...     fire=fire_sim,
        ...     predictor_params=params,
        ...     n_forecasts=30
        ... )
        >>>
        >>> # Sample forecast indices for ensemble
        >>> indices = pool.sample(10, seed=42)
        >>>
        >>> # Get a specific forecast
        >>> forecast = pool.get_forecast(0)
    """

    forecasts: List[ForecastData]
    base_weather_stream: 'WeatherStream'
    map_params: 'MapParams'
    predictor_params: 'PredictorParams'
    created_at_time_s: float
    forecast_start_datetime: 'datetime'

    def __post_init__(self):
        """Register this pool with the manager after creation."""
        ForecastPoolManager.register(self)

    def __len__(self) -> int:
        """Return the number of forecasts in the pool."""
        return len(self.forecasts)

    def __getitem__(self, idx: int) -> ForecastData:
        """Get a forecast by index."""
        return self.forecasts[idx]

    def sample(self, n: int, replace: bool = True, seed: int = None) -> List[int]:
        """Sample n indices from the pool.

        Args:
            n: Number of indices to sample.
            replace: If True, sample with replacement (default). If False,
                n must not exceed pool size.
            seed: Random seed for reproducibility.

        Returns:
            List of forecast indices.

        Raises:
            ValueError: If replace=False and n > pool size.
        """
        rng = np.random.default_rng(seed)
        return rng.choice(len(self.forecasts), size=n, replace=replace).tolist()

    def get_forecast(self, idx: int) -> ForecastData:
        """Get a specific forecast by index.

        Args:
            idx: Index of the forecast to retrieve.

        Returns:
            ForecastData at the specified index.
        """
        return self.forecasts[idx]

    def get_weather_scenarios(self) -> List['WeatherStream']:
        """Return all perturbed weather streams for time window calculation.

        Returns:
            List of WeatherStream objects, one per forecast in the pool.
        """
        return [f.weather_stream for f in self.forecasts]

    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        total = 0
        for forecast in self.forecasts:
            total += forecast.memory_usage()
        return total

    def _cleanup(self) -> None:
        """Release memory for this pool.

        Clears the forecasts list to allow garbage collection of the
        potentially large wind forecast arrays.
        """
        self.forecasts.clear()

    def close(self) -> None:
        """Explicitly close this pool and release memory.

        Unregisters from the manager and clears forecasts.
        """
        ForecastPoolManager.unregister(self)
        self._cleanup()

    # =========================================================================
    # Class method for pool generation
    # =========================================================================

    @classmethod
    def generate(
        cls,
        fire: 'FireSim',
        predictor_params: 'PredictorParams',
        n_forecasts: int,
        num_workers: Optional[int] = None,
        random_seed: Optional[int] = None,
        wind_speed_bias: float = 0.0,
        wind_dir_bias: float = 0.0,
        wind_uncertainty_factor: float = 0.0,
        verbose: bool = True
    ) -> 'ForecastPool':
        """Generate a pool of perturbed wind forecasts in parallel.

        Create n_forecasts independent wind forecasts, each with different
        AR(1) perturbations applied to the base weather stream. WindNinja
        is called in parallel for efficiency.

        This class method centralizes all pool generation logic, making
        ForecastPool the owner of the entire pool creation process.

        Args:
            fire: Fire simulation to generate forecasts from.
            predictor_params: Predictor parameters for time horizon and settings.
            n_forecasts: Number of forecasts to generate.
            num_workers: Number of parallel workers. Defaults to
                min(cpu_count, n_forecasts).
            random_seed: Base seed for reproducibility. If None, uses
                random seeds for each forecast.
            wind_speed_bias: Constant wind speed bias in m/s.
            wind_dir_bias: Constant wind direction bias in degrees.
            wind_uncertainty_factor: Scaling factor for AR(1) noise (0-1).
            verbose: Whether to print progress messages.

        Returns:
            ForecastPool: Container with all generated forecasts, base weather
                stream, map parameters, and creation metadata.

        Raises:
            ValueError: If fire is None or n_forecasts < 1.

        Example:
            >>> pool = ForecastPool.generate(
            ...     fire=fire_sim,
            ...     predictor_params=params,
            ...     n_forecasts=30,
            ...     random_seed=42
            ... )
        """
        if fire is None:
            raise ValueError("Cannot generate forecast pool without fire reference")
        if n_forecasts < 1:
            raise ValueError("n_forecasts must be at least 1")

        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Import tqdm only if available
        try:
            from tqdm import tqdm
            use_tqdm = verbose
        except ImportError:
            use_tqdm = False

        num_workers = num_workers or min(cpu_count(), n_forecasts)

        # Compute AR(1) parameters from predictor_params
        max_beta = predictor_params.max_beta
        base_wind_spd_std = predictor_params.base_wind_spd_std
        base_wind_dir_std = predictor_params.base_wind_dir_std

        beta = wind_uncertainty_factor * max_beta
        wnd_spd_std = base_wind_spd_std * wind_uncertainty_factor
        wnd_dir_std = base_wind_dir_std * wind_uncertainty_factor

        # Compute effective biases
        max_wind_speed_bias = predictor_params.max_wind_speed_bias
        max_wind_dir_bias = predictor_params.max_wind_dir_bias
        effective_speed_bias = wind_speed_bias * max_wind_speed_bias
        effective_dir_bias = wind_dir_bias * max_wind_dir_bias

        # Set up reproducible seeds if requested
        if random_seed is not None:
            base_rng = np.random.default_rng(random_seed)
            seeds = [
                (int(base_rng.integers(0, 2**31)), int(base_rng.integers(0, 2**31)))
                for _ in range(n_forecasts)
            ]
        else:
            seeds = [(None, None) for _ in range(n_forecasts)]

        # Get current state from fire simulation
        curr_weather_idx = fire._curr_weather_idx
        base_weather_stream = fire._weather_stream
        map_params = fire._sim_params.map_params
        time_horizon_hr = predictor_params.time_horizon_hr

        # Compute number of weather indices needed
        weather_t_step = base_weather_stream.time_step * 60  # seconds
        num_indices = int(np.ceil((time_horizon_hr * 3600) / weather_t_step))

        # Prepare tasks
        tasks = []
        for i, (speed_seed, dir_seed) in enumerate(seeds):
            # Generate perturbed weather stream
            perturbed_stream, used_speed_seed, used_dir_seed = cls._perturb_weather_stream(
                weather_stream=base_weather_stream,
                start_idx=curr_weather_idx,
                num_indices=num_indices,
                speed_seed=speed_seed,
                dir_seed=dir_seed,
                beta=beta,
                wnd_spd_std=wnd_spd_std,
                wnd_dir_std=wnd_dir_std,
                wind_speed_bias=effective_speed_bias,
                wind_dir_bias=effective_dir_bias
            )

            tasks.append(_ForecastGenerationTask(
                forecast_id=i,
                perturbed_stream=perturbed_stream,
                map_params=map_params,
                wind_speed_bias=effective_speed_bias,
                wind_dir_bias=effective_dir_bias,
                speed_seed=used_speed_seed,
                dir_seed=used_dir_seed
            ))

        # Generate forecasts in parallel
        forecasts = [None] * n_forecasts

        if verbose:
            print(f"Generating forecast pool: {n_forecasts} forecasts using {num_workers} workers...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(_generate_single_forecast, task): task.forecast_id
                for task in tasks
            }

            if use_tqdm:
                iterator = tqdm(as_completed(future_to_idx), total=n_forecasts,
                               desc="WindNinja forecasts")
            else:
                iterator = as_completed(future_to_idx)

            for future in iterator:
                idx = future_to_idx[future]
                try:
                    forecast_data = future.result()
                    forecasts[idx] = forecast_data
                except Exception as e:
                    if verbose:
                        print(f"Forecast {idx} failed: {e}")
                    raise

        # Get the datetime that index 0 of the forecasts corresponds to
        forecast_start_datetime = base_weather_stream.stream_times[curr_weather_idx]

        return cls(
            forecasts=forecasts,
            base_weather_stream=copy.deepcopy(base_weather_stream),
            map_params=copy.deepcopy(map_params),
            predictor_params=copy.deepcopy(predictor_params),
            created_at_time_s=fire._curr_time_s,
            forecast_start_datetime=forecast_start_datetime
        )

    @staticmethod
    def _perturb_weather_stream(
        weather_stream: 'WeatherStream',
        start_idx: int,
        num_indices: int,
        speed_seed: Optional[int],
        dir_seed: Optional[int],
        beta: float,
        wnd_spd_std: float,
        wnd_dir_std: float,
        wind_speed_bias: float,
        wind_dir_bias: float
    ) -> tuple:
        """Apply AR(1) perturbation to a weather stream.

        Creates a copy of the weather stream with biases and autoregressive
        noise applied to wind speed and direction.

        Args:
            weather_stream: Original weather stream to perturb.
            start_idx: Starting index in the stream.
            num_indices: Number of entries to include.
            speed_seed: Random seed for speed perturbation.
            dir_seed: Random seed for direction perturbation.
            beta: AR(1) autoregression coefficient.
            wnd_spd_std: Standard deviation for speed noise.
            wnd_dir_std: Standard deviation for direction noise.
            wind_speed_bias: Constant speed bias.
            wind_dir_bias: Constant direction bias.

        Returns:
            tuple: (perturbed_stream, speed_seed_used, dir_seed_used)
        """
        from embrs.models.weather import WeatherStream

        new_weather_stream = copy.deepcopy(weather_stream)

        end_idx = num_indices + start_idx + 1  # +1 for inclusive end

        # Generate seeds if not provided
        if speed_seed is None:
            speed_seed = np.random.randint(0, 2**31)
        if dir_seed is None:
            dir_seed = np.random.randint(0, 2**31)

        # Create separate RNGs for reproducibility
        speed_rng = np.random.default_rng(speed_seed)
        dir_rng = np.random.default_rng(dir_seed)

        speed_error = 0.0
        dir_error = 0.0
        new_stream = []

        for entry in new_weather_stream.stream[start_idx:end_idx]:
            new_entry = copy.deepcopy(entry)

            # Apply bias and accumulated error
            new_entry.wind_speed += speed_error + wind_speed_bias
            new_entry.wind_speed = max(0.0, new_entry.wind_speed)
            new_entry.wind_dir_deg += dir_error + wind_dir_bias
            new_entry.wind_dir_deg = new_entry.wind_dir_deg % 360

            new_stream.append(new_entry)

            # Update errors using AR(1) process
            speed_error = beta * speed_error + speed_rng.normal(0, wnd_spd_std)
            dir_error = beta * dir_error + dir_rng.normal(0, wnd_dir_std)

        new_weather_stream.stream = new_stream
        return new_weather_stream, speed_seed, dir_seed


# =============================================================================
# Backwards compatibility - export from this module
# =============================================================================

__all__ = [
    'ForecastData',
    'ForecastPool',
    'ForecastPoolManager',
    '_ForecastGenerationTask',
    '_generate_single_forecast',
]
