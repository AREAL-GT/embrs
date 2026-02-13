"""Fire prediction module for EMBRS.

Provides forward fire spread prediction with uncertainty modeling. Supports
single predictions, ensemble predictions with parallel execution, and
pre-computed forecast pools for efficient rollout scenarios.

Classes:
    - FirePredictor: Ensemble fire prediction with wind uncertainty.

.. autoclass:: FirePredictor
    :members:
"""

import copy
import numpy as np
import os
import uuid
import time as time_module
from multiprocessing import cpu_count
from typing import List, Optional, Tuple, TYPE_CHECKING

from embrs.base_classes.base_fire import BaseFireSim
from embrs.fire_simulator.fire import FireSim
from embrs.utilities.data_classes import (
    PredictorParams,
    PredictionOutput,
    StateEstimate,
    EnsemblePredictionOutput,
    CellStatistics,
    MapParams
)
from embrs.tools.forecast_pool import ForecastData, ForecastPool
from embrs.tools.predictor_serializer import PredictorSerializer
from embrs.utilities.fire_util import UtilFuncs, CellStates
from embrs.models.rothermel import *
from embrs.models.crown_model import *
from embrs.models.wind_forecast import run_windninja, temp_file_path

if TYPE_CHECKING:
    from embrs.models.weather import WeatherStream


class FirePredictor(BaseFireSim):
    """Fire spread predictor with uncertainty modeling.

    Extends BaseFireSim to run forward predictions from the current fire state.
    Supports wind uncertainty via AR(1) perturbations, rate of spread bias,
    and ensemble predictions with parallel execution.

    The predictor maintains a reference to the parent FireSim and synchronizes
    its state before each prediction. For ensemble predictions, the predictor
    is serialized and reconstructed in worker processes without the parent
    reference.

    Attributes:
        fire (FireSim): Reference to the parent fire simulation. None in workers.
        time_horizon_hr (float): Prediction duration in hours.
        wind_uncertainty_factor (float): Scaling factor for wind perturbation (0-1).
        wind_speed_bias (float): Constant wind speed bias in m/s.
        wind_dir_bias (float): Constant wind direction bias in degrees.
        ros_bias_factor (float): Multiplicative factor for rate of spread (0.5-1.5).
        dead_mf (float): Dead fuel moisture fraction for prediction.
        live_mf (float): Live fuel moisture fraction for prediction.
        model_spotting (bool): Whether to model ember spotting.
    """

    # =========================================================================
    # Initialization & Configuration
    # =========================================================================

    def __init__(self, params: PredictorParams, fire: FireSim) -> None:
        """Initialize fire predictor with parameters and parent simulation.

        Args:
            params (PredictorParams): Configuration for prediction behavior
                including time horizon, uncertainty factors, and fuel moisture.
            fire (FireSim): Parent fire simulation to predict from. The predictor
                synchronizes with this simulation before each prediction run.
        """
        # Live reference to the fire sim
        self.fire = fire
        self.c_size = -1
        self.start_time_s = 0

        # Store original params for serialization
        self._params = params
        self._serialization_data = None  # Will hold snapshot for pickling

        self.set_params(params)

    def set_params(self, params: PredictorParams) -> None:
        """Configure predictor parameters and optionally regenerate cell grid.

        Updates all prediction parameters from the provided PredictorParams.
        If cell_size_m has changed since the last call, regenerates the
        entire cell grid (expensive operation).

        Args:
            params (PredictorParams): New parameter values. All fields are used
                to update internal state.

        Side Effects:
            - Updates all uncertainty and bias parameters
            - May regenerate cell grid if cell_size_m changed
            - Computes nominal ignition probability for spotting
        """
        generate_cell_grid = False

        # How long the prediction will run for
        self.time_horizon_hr = params.time_horizon_hr

        # Uncertainty parameters
        self.wind_uncertainty_factor = params.wind_uncertainty_factor  # [0, 1], autoregression noise

        # Compute constant bias terms
        self.wind_speed_bias = params.wind_speed_bias * params.max_wind_speed_bias
        self.wind_dir_bias = params.wind_dir_bias * params.max_wind_dir_bias
        self.ros_bias_factor = max(min(1 + params.ros_bias, 1.5), 0.5)

        # Compute auto-regressive parameters
        self.beta = self.wind_uncertainty_factor * params.max_beta
        self.wnd_spd_std = params.base_wind_spd_std * self.wind_uncertainty_factor
        self.wnd_dir_std = params.base_wind_dir_std * self.wind_uncertainty_factor

        # If cell size has changed since last set params, regenerate cell grid
        cell_size = params.cell_size_m
        if cell_size != self.c_size:
            generate_cell_grid = True

        # Track cell size
        self.c_size = cell_size

        # Set constant fuel moistures
        self.dead_mf = params.dead_mf
        self.live_mf = params.live_mf

        # Update relevant sim_params for prediction model for initialization
        sim_params = copy.deepcopy(self.fire._sim_params)
        sim_params.cell_size = params.cell_size_m
        sim_params.t_step_s = params.time_step_s
        sim_params.duration_s = self.time_horizon_hr * 3600
        sim_params.init_mf = [params.dead_mf, params.dead_mf, params.dead_mf]
        sim_params.spot_delay_s = params.spot_delay_s
        sim_params.model_spotting = params.model_spotting

        # Set the currently burning cells as the initial ignition region
        burning_cells = [cell for cell in self.fire._burning_cells]

        # Get the merged polygon representing burning cells
        sim_params.map_params.scenario_data.initial_ign = UtilFuncs.get_cell_polygons(burning_cells)
        burnt_region = UtilFuncs.get_cell_polygons(self.fire._burnt_cells)

        # Nominal ignition probability for spotting
        self.nom_ign_prob = self._calc_nominal_prob()

        if generate_cell_grid:
            super().__init__(sim_params, burnt_region=burnt_region)
            # orig_grid/orig_dict reference the same cell objects as _cell_grid/_cell_dict.
            # _set_states() will reset all cells in-place before each prediction.
            self.orig_grid = self._cell_grid
            self.orig_dict = self._cell_dict

    def _apply_member_params(self, params: PredictorParams) -> None:
        """Apply member-specific parameters without regenerating cell grid.

        Lightweight version of set_params() that only updates uncertainty and
        bias parameters, not grid structure. Used for per-member parameter
        customization in ensemble predictions.

        Args:
            params (PredictorParams): Member-specific parameter values.

        Notes:
            cell_size_m is ignored - all members use the predictor's original
            cell size to avoid expensive grid regeneration. A warning is issued
            if cell_size_m differs from the predictor's cell size.

        Side Effects:
            Updates time_horizon_hr, bias factors, AR parameters, fuel moisture,
            and spotting parameters.
        """
        import warnings

        # Warn if cell_size_m differs from predictor's cell size
        if params.cell_size_m != self.c_size:
            warnings.warn(
                f"cell_size_m ({params.cell_size_m}) differs from predictor's "
                f"cell size ({self.c_size}). cell_size_m is ignored in per-member "
                f"params to avoid grid regeneration. Using predictor's cell size."
            )

        # Update time horizon (per-member time horizons supported)
        self.time_horizon_hr = params.time_horizon_hr

        # Update bias terms
        self.wind_speed_bias = params.wind_speed_bias * params.max_wind_speed_bias
        self.wind_dir_bias = params.wind_dir_bias * params.max_wind_dir_bias
        self.ros_bias_factor = max(min(1 + params.ros_bias, 1.5), 0.5)

        # Update auto-regressive parameters
        self.wind_uncertainty_factor = params.wind_uncertainty_factor
        self.beta = self.wind_uncertainty_factor * params.max_beta
        self.wnd_spd_std = params.base_wind_spd_std * self.wind_uncertainty_factor
        self.wnd_dir_std = params.base_wind_dir_std * self.wind_uncertainty_factor

        # Update fuel moisture
        self.dead_mf = params.dead_mf
        self.live_mf = params.live_mf

        # Update spotting parameters
        self.model_spotting = params.model_spotting
        self._spot_delay_s = params.spot_delay_s

        # Store for reference
        self._params = params

    def _perturb_weather_stream(
        self,
        weather_stream: 'WeatherStream',
        start_idx: int = 0,
        num_indices: int = None,
        speed_seed: int = None,
        dir_seed: int = None
    ) -> Tuple['WeatherStream', int, int]:
        """Apply AR(1) perturbation to a weather stream.

        Creates a copy of the weather stream with biases and autoregressive
        noise applied to wind speed and direction. Used for generating
        perturbed forecasts for ensemble predictions.

        Args:
            weather_stream (WeatherStream): Original weather stream to perturb.
            start_idx (int): Starting index in the stream. Defaults to 0.
            num_indices (int): Number of entries to include. If None, uses
                enough entries to cover time_horizon_hr.
            speed_seed (int): Random seed for speed perturbation. If None,
                generates a random seed.
            dir_seed (int): Random seed for direction perturbation. If None,
                generates a random seed.

        Returns:
            tuple: (perturbed_stream, speed_seed_used, dir_seed_used) where
                perturbed_stream is the modified WeatherStream and seeds are
                returned for reproducibility tracking.

        Notes:
            TODO:verify Consider extending to perturb temperature and relative
            humidity using AR(1) error model as well.

        Performance Note:
            Uses shallow copy of WeatherStream and creates new WeatherEntry objects
            directly to avoid expensive deepcopy operations on the entire stream.
        """
        from embrs.utilities.data_classes import WeatherEntry

        # Shallow copy preserves references to params, geo, etc. but allows
        # us to replace the stream list without modifying the original
        new_weather_stream = copy.copy(weather_stream)
        weather_t_step = weather_stream.time_step * 60  # Convert to seconds

        if num_indices is None:
            num_indices = int(np.ceil((self.time_horizon_hr * 3600) / weather_t_step))

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

        # Create new WeatherEntry objects directly instead of deepcopying
        for entry in weather_stream.stream[start_idx:end_idx]:
            # Apply bias and accumulated error to create perturbed entry
            perturbed_speed = max(0.0, entry.wind_speed + speed_error + self.wind_speed_bias)
            perturbed_dir = (entry.wind_dir_deg + dir_error + self.wind_dir_bias) % 360

            new_entry = WeatherEntry(
                wind_speed=perturbed_speed,
                wind_dir_deg=perturbed_dir,
                temp=entry.temp,
                rel_humidity=entry.rel_humidity,
                cloud_cover=entry.cloud_cover,
                rain=entry.rain,
                dni=entry.dni,
                dhi=entry.dhi,
                ghi=entry.ghi,
                solar_zenith=entry.solar_zenith,
                solar_azimuth=entry.solar_azimuth
            )
            new_stream.append(new_entry)

            # Update errors using AR(1) process
            speed_error = self.beta * speed_error + speed_rng.normal(0, self.wnd_spd_std)
            dir_error = self.beta * dir_error + dir_rng.normal(0, self.wnd_dir_std)

        new_weather_stream.stream = new_stream
        return new_weather_stream, speed_seed, dir_seed

    # =========================================================================
    # Main Public Interface
    # =========================================================================

    def run(
        self,
        fire_estimate: StateEstimate = None,
        visualize: bool = False
    ) -> PredictionOutput:
        """Run a single fire spread prediction.

        Executes forward prediction from either the current fire simulation
        state or a provided state estimate. Synchronizes with the parent
        fire simulation, generates perturbed wind forecasts, and iterates
        the fire spread model.

        Args:
            fire_estimate (StateEstimate): Optional state estimate to initialize
                from. If None, uses current fire simulation state. If provided
                with start_time_s, prediction starts from that future time.
            visualize (bool): If True, display prediction on fire visualizer.
                Defaults to False.

        Returns:
            PredictionOutput: Contains spread timeline (cell positions by time),
                flame length, fireline intensity, rate of spread, spread direction,
                crown fire status, hold probabilities, and breach status.

        Raises:
            ValueError: If fire_estimate.start_time_s is in the past or beyond
                weather forecast coverage.
        """
        # Extract custom start time if provided
        custom_start_time_s = None
        if fire_estimate is not None and fire_estimate.start_time_s is not None:
            custom_start_time_s = fire_estimate.start_time_s

        # Catch up time and weather states with the fire sim
        # (works in both main process and worker process)
        self._catch_up_with_fire(custom_start_time_s=custom_start_time_s)

        # Set fire state variables
        self._set_states(fire_estimate)

        # Initialize output containers
        self.spread = {}
        self.flame_len_m = {}
        self.fli_kw_m = {}
        self.ros_ms = {}
        self.spread_dir = {}
        self.crown_fire = {}
        self.hold_probs = {}
        self.breaches = {}
        self.active_fire_front = {}
        self.burnt_spread = {}
        self.burnt_locs = []

        # Perform the prediction
        self._prediction_loop()

        if visualize and self.fire is not None:
            self.fire.visualize_prediction(self.spread)

        output = PredictionOutput(
            spread=self.spread,
            flame_len_m=self.flame_len_m,
            fli_kw_m=self.fli_kw_m,
            ros_ms=self.ros_ms,
            spread_dir=self.spread_dir,
            crown_fire=self.crown_fire,
            hold_probs=self.hold_probs,
            breaches=self.breaches,
            active_fire_front=self.active_fire_front,
            burnt_spread=self.burnt_spread
        )

        return output

    def run_ensemble(
        self,
        state_estimates: List[StateEstimate],
        visualize: bool = False,
        num_workers: Optional[int] = None,
        random_seeds: Optional[List[int]] = None,
        return_individual: bool = False,
        predictor_params_list: Optional[List[PredictorParams]] = None,
        vary_wind_per_member: bool = False,
        forecast_pool: Optional[ForecastPool] = None,
        forecast_indices: Optional[List[int]] = None
    ) -> EnsemblePredictionOutput:
        """Run ensemble predictions using multiple initial state estimates.

        Execute predictions in parallel, each starting from a different
        StateEstimate. Results are aggregated into probabilistic burn maps
        and fire behavior statistics.

        Uses custom serialization to efficiently transfer predictor state to
        worker processes without reconstructing the full FireSim.

        Args:
            state_estimates (list[StateEstimate]): Initial fire states for
                each ensemble member. Each may include burning_polys,
                burnt_polys, and optional start_time_s.
            visualize (bool): If True, visualize aggregated burn probability
                on the fire visualizer. Defaults to False.
            num_workers (int): Number of parallel workers. Defaults to cpu_count.
            random_seeds (list[int]): Optional seeds for reproducibility,
                one per state estimate.
            return_individual (bool): If True, include individual PredictionOutput
                objects in the returned EnsemblePredictionOutput. Defaults to False.
            predictor_params_list (list[PredictorParams]): Optional per-member
                parameters. If provided, each member uses its own params.
                Automatically enables vary_wind_per_member unless using
                forecast_pool.
            vary_wind_per_member (bool): If True, each worker generates its own
                perturbed wind forecast. If False, all members share the same
                wind forecast. Ignored when forecast_pool is provided.
            forecast_pool (ForecastPool): Optional pre-computed forecasts.
                Workers use forecasts from pool instead of running WindNinja.
            forecast_indices (list[int]): Optional indices into forecast_pool,
                one per state_estimate. If None, indices are sampled randomly
                with replacement.

        Returns:
            EnsemblePredictionOutput: Aggregated ensemble results including
                burn probability maps, fire behavior statistics, and optional
                individual predictions.

        Raises:
            ValueError: If state_estimates is empty, length mismatches occur,
                or any start_time_s is invalid.
            RuntimeError: If more than 50% of ensemble members fail.
        """
        import multiprocessing as mp
        import warnings
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm

        # Validation
        if not state_estimates:
            raise ValueError("state_estimates cannot be empty")

        if random_seeds is not None and len(random_seeds) != len(state_estimates):
            raise ValueError(
                f"random_seeds length ({len(random_seeds)}) must match "
                f"state_estimates length ({len(state_estimates)})"
            )

        # Validate predictor_params_list length
        if predictor_params_list is not None:
            if len(predictor_params_list) != len(state_estimates):
                raise ValueError(
                    f"predictor_params_list length ({len(predictor_params_list)}) must match "
                    f"state_estimates length ({len(state_estimates)})"
                )
            # If params are provided and no pool, wind MUST vary
            if forecast_pool is None:
                vary_wind_per_member = True

            # Warn about cell_size_m differences
            cell_sizes = set(p.cell_size_m for p in predictor_params_list)
            if len(cell_sizes) > 1 or (len(cell_sizes) == 1 and cell_sizes.pop() != self.c_size):
                warnings.warn(
                    f"Some predictor_params have different cell_size_m values. "
                    f"cell_size_m is ignored in per-member params; all members will use "
                    f"the predictor's cell size ({self.c_size}m)."
                )

        n_ensemble = len(state_estimates)

        # Handle forecast pool
        use_pooled_forecasts = forecast_pool is not None

        if use_pooled_forecasts:
            if forecast_indices is None:
                # Sample indices with replacement (for rollouts)
                forecast_indices = forecast_pool.sample(n_ensemble, replace=True)
                print(f"Sampled forecast indices: {forecast_indices[:5]}{'...' if n_ensemble > 5 else ''}")
            else:
                # Validate explicit indices
                if len(forecast_indices) != n_ensemble:
                    raise ValueError(
                        f"forecast_indices length ({len(forecast_indices)}) must match "
                        f"state_estimates length ({n_ensemble})"
                    )

            # Validate all indices are in range
            for idx in forecast_indices:
                if idx < 0 or idx >= len(forecast_pool):
                    raise ValueError(
                        f"Invalid forecast index {idx}, pool size is {len(forecast_pool)}"
                    )

            # When using pool, wind is pre-computed (not varied per member)
            vary_wind_per_member = False

            print(f"Using forecast pool with {len(forecast_pool)} forecasts")
        else:
            forecast_indices = None

        # Validate all start times before parallel execution
        # Initialize _start_weather_idx for validation calls
        self._start_weather_idx = None
        for i, state_est in enumerate(state_estimates):
            if state_est.start_time_s is not None:
                # Use per-member time horizon if provided, otherwise use predictor's
                member_horizon = (predictor_params_list[i].time_horizon_hr
                                  if predictor_params_list else None)
                try:
                    self._validate_start_time(state_est.start_time_s, member_horizon)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid start_time_s for state_estimate[{i}]: {e}"
                    )

        num_workers = num_workers or mp.cpu_count()

        print(f"Running ensemble prediction:")
        print(f"  - {n_ensemble} ensemble members")
        print(f"  - {num_workers} parallel workers")

        # CRITICAL: Prepare for serialization
        print("Preparing predictor for serialization...")

        # Generate shared wind forecast only if not using pool and not varying per member
        if not use_pooled_forecasts and not vary_wind_per_member:
            self._predict_wind()

        # Prepare workers for serialization
        self.prepare_for_serialization(
            vary_wind=vary_wind_per_member,
            forecast_pool=forecast_pool,
            forecast_indices=forecast_indices
        )

        # Run predictions in parallel
        predictions = []
        failed_count = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all jobs
            futures = {}
            for i, state_est in enumerate(state_estimates):
                seed = random_seeds[i] if random_seeds else None
                member_params = predictor_params_list[i] if predictor_params_list else None

                # Store member index for forecast assignment in worker
                if use_pooled_forecasts:
                    self._serialization_data['member_index'] = i

                # Submit (predictor will be pickled via __getstate__)
                future = executor.submit(
                    _run_ensemble_member_worker,
                    self,
                    state_est,
                    seed,
                    member_params
                )
                futures[future] = i  # Track member index

            # Collect results with progress bar
            collected_forecast_indices = []
            for future in tqdm(as_completed(futures),
                              total=n_ensemble,
                              desc="Ensemble predictions",
                              unit="member"):
                member_idx = futures[future]
                try:
                    result = future.result()
                    if forecast_indices is not None:
                        result.forecast_index = forecast_indices[member_idx]
                        collected_forecast_indices.append(forecast_indices[member_idx])
                    predictions.append(result)
                except Exception as e:
                    failed_count += 1
                    print(f"Warning: Member {member_idx} failed: {e}")

        # Check failure rate
        if len(predictions) == 0:
            raise RuntimeError("All ensemble members failed")

        if failed_count > n_ensemble * 0.5:
            raise RuntimeError(
                f"More than 50% of ensemble members failed "
                f"({failed_count}/{n_ensemble})"
            )

        if failed_count > 0:
            print(f"Completed with {failed_count} failures, "
                  f"{len(predictions)} successful members")

        # Aggregate results
        print("Aggregating ensemble predictions...")
        ensemble_output = _aggregate_ensemble_predictions(predictions)
        ensemble_output.n_ensemble = len(predictions)

        # Stamp forecast indices on ensemble output
        if collected_forecast_indices:
            ensemble_output.forecast_indices = collected_forecast_indices

        # Optionally include individual predictions
        if return_individual:
            ensemble_output.individual_predictions = predictions

        # Optionally visualize
        if visualize:
            self._visualize_ensemble(ensemble_output)

        return ensemble_output

    def generate_forecast_pool(
        self,
        n_forecasts: int,
        num_workers: int = None,
        random_seed: int = None
    ) -> ForecastPool:
        """Generate a pool of perturbed wind forecasts in parallel.

        Create n_forecasts independent wind forecasts, each with different
        AR(1) perturbations applied to the base weather stream. WindNinja
        is called in parallel for efficiency.

        The resulting pool can be reused across multiple ensemble runs. When
        passed to run_ensemble with explicit forecast_indices, each member uses
        the specified forecast. When forecast_indices is omitted, run_ensemble
        samples from the pool with replacement.

        Args:
            n_forecasts (int): Number of forecasts to generate.
            num_workers (int): Number of parallel workers. Defaults to
                min(cpu_count, n_forecasts).
            random_seed (int): Base seed for reproducibility. If None, uses
                random seeds for each forecast.

        Returns:
            ForecastPool: Container with all generated forecasts, base weather
                stream, map parameters, and creation metadata.

        Raises:
            RuntimeError: If called without a fire reference (fire is None).
        """
        if self.fire is None:
            raise RuntimeError("Cannot generate forecast pool without fire reference")

        # Delegate to ForecastPool.generate() which owns the pool generation process
        return ForecastPool.generate(
            fire=self.fire,
            predictor_params=self._params,
            n_forecasts=n_forecasts,
            num_workers=num_workers,
            random_seed=random_seed,
            wind_speed_bias=self._params.wind_speed_bias,
            wind_dir_bias=self._params.wind_dir_bias,
            wind_uncertainty_factor=self.wind_uncertainty_factor,
            verbose=True
        )

    # =========================================================================
    # Core Prediction Logic
    # =========================================================================

    def _prediction_loop(self) -> None:
        """Execute main prediction iteration loop.

        Iterates the fire spread model until end time is reached. For each
        iteration: updates weather, computes steady-state spread rates for
        burning cells, propagates fire to neighbors, handles spotting, and
        records fire behavior metrics.

        Side Effects:
            - Populates spread, flame_len_m, fli_kw_m, ros_ms, spread_dir,
              crown_fire, hold_probs, and breaches dictionaries
            - Updates cell states (FIRE -> BURNT)
            - Sets _finished to True when complete
        """
        self._iters = 0

        while self._init_iteration():
            fires_still_burning = []

            for cell in self._burning_cells:
                if self.weather_changed or not cell.has_steady_state:
                    # Update the steady state
                    self.update_steady_state(cell)
                    cell.r_t = cell.r_ss * self.ros_bias_factor
                    cell.avg_ros = cell.r_ss * self.ros_bias_factor
                    cell.I_t = cell.I_ss * self.ros_bias_factor

                self.propagate_fire(cell)
                self.remove_neighbors(cell)

                if cell.fully_burning or len(cell.burnable_neighbors) == 0:
                    self.set_state_at_cell(cell, CellStates.BURNT)
                    self.burnt_locs.append((cell.x_pos, cell.y_pos))
                else:
                    fires_still_burning.append(cell)

                self._updated_cells[cell.id] = cell

            if self.model_spotting and self.nom_ign_prob > 0:
                self._ignite_spots()

            self.update_control_interface_elements()

            self._burning_cells = list(fires_still_burning)
            
            # Clear updated cells to prevent unbounded memory growth
            # (cells are tracked per-iteration, not across iterations)
            self._updated_cells.clear()

            self._iters += 1

        self._finished = True

    def _init_iteration(self) -> bool:
        """Initialize each prediction iteration.

        Handle first iteration setup (cell ignition) and subsequent iterations
        (fire behavior computation). Updates current time, checks weather,
        and manages ignition queues.

        Returns:
            bool: True if prediction should continue, False if end time reached.

        Behavior:
            - First iteration (iters=0): Sets cell states to FIRE, initializes
              steady-state rates with ros_bias_factor applied.
            - Subsequent iterations: Computes flame length, fireline intensity,
              determines fireline breach, and handles ember lofting.
        """
        self._curr_time_s = (self._iters * self._time_step) + self.start_time_s

        if self._iters == 0:
            self.weather_changed = True
            self._new_ignitions = []

            for cell, loc in self.starting_ignitions:
                if not cell.fuel.burnable:
                    continue

                cell.get_ign_params(loc)
                cell._set_state(CellStates.FIRE)
                surface_fire(cell)
                crown_fire(cell, self.fmc)
                cell.has_steady_state = True

                # Don't model fire acceleration in prediction model
                cell.r_t = cell.r_ss * self.ros_bias_factor
                cell.avg_ros = cell.r_ss * self.ros_bias_factor
                cell.I_t = cell.I_ss * self.ros_bias_factor

                self._updated_cells[cell.id] = cell
                self._new_ignitions.append(cell)

                for neighbor in list(cell.burnable_neighbors.keys()):
                    self._frontier.add(neighbor)

                if self.spread.get(self._curr_time_s) is None:
                    self.spread[self._curr_time_s] = [(cell.x_pos, cell.y_pos)]
                else:
                    self.spread[self._curr_time_s].append((cell.x_pos, cell.y_pos))

        else:
            for cell in self._new_ignitions:
                surface_fire(cell)
                crown_fire(cell, self.fmc)

                flame_len_ft = calc_flame_len(cell)
                flame_len_m = ft_to_m(flame_len_ft)

                self.flame_len_m[(cell.x_pos, cell.y_pos)] = flame_len_m

                if cell._break_width > 0:
                    # Determine if fire will breach fireline contained within cell
                    hold_prob = cell.calc_hold_prob(flame_len_m)
                    rand = np.random.random()
                    cell.breached = rand > hold_prob

                    self.hold_probs[(cell.x_pos, cell.y_pos)] = hold_prob
                    self.breaches[(cell.x_pos, cell.y_pos)] = cell.breached
                else:
                    cell.breached = True

                cell.has_steady_state = True

                for neighbor in list(cell.burnable_neighbors.keys()):
                    self._frontier.add(neighbor)

                if self.model_spotting:
                    if not cell.lofted and cell._crown_status != CrownStatus.NONE and self.nom_ign_prob > 0:
                        self.embers.loft(cell)

                # Don't model fire acceleration in prediction model
                cell.r_t = cell.r_ss * self.ros_bias_factor
                cell.avg_ros = cell.r_ss * self.ros_bias_factor
                cell.I_t = cell.I_ss * self.ros_bias_factor

                if self.spread.get(self._curr_time_s) is None:
                    self.spread[self._curr_time_s] = [(cell.x_pos, cell.y_pos)]
                else:
                    self.spread[self._curr_time_s].append((cell.x_pos, cell.y_pos))

                if cell._crown_status != CrownStatus.NONE:
                    self.crown_fire[(cell.x_pos, cell.y_pos)] = cell._crown_status

                self.fli_kw_m[(cell.x_pos, cell.y_pos)] = BTU_ft_min_to_kW_m(np.max(cell.I_ss))
                self.ros_ms[(cell.x_pos, cell.y_pos)] = np.max(cell.r_ss)
                self.spread_dir[(cell.x_pos, cell.y_pos)] = cell.directions[np.argmax(cell.r_ss)]

        # Store all the burnt cells at this time step
        self.burnt_spread[self._curr_time_s] = list(self.burnt_locs)

        # Store the cells actively burning at this time step
        self.active_fire_front[self._curr_time_s] = [(cell.x_pos, cell.y_pos) for cell in self._burning_cells]

        if self._curr_time_s >= self._end_time:
            return False

        # Check if weather has changed
        self.weather_changed = self._update_weather()

        # Add any new ignitions to the current set of burning cells
        self._burning_cells.extend(self._new_ignitions)
        # Reset new ignitions
        self._new_ignitions = []

        return True

    def _set_states(self, state_estimate: StateEstimate = None) -> None:
        """Initialize prediction state from fire simulation or state estimate.

        Reset the cell grid to original state and configure initial burning
        and burnt regions based on either the parent fire simulation state
        or a provided StateEstimate.

        Args:
            state_estimate (StateEstimate): Optional state estimate. If None,
                uses current fire simulation state (or serialized state in
                worker processes where fire is None).

        Side Effects:
            - Resets all cells in-place to FUEL state via reset_to_fuel()
            - Points _cell_grid and _cell_dict to orig_grid and orig_dict
            - Resets _burnt_cells, _burning_cells, _updated_cells, and
              _scheduled_spot_fires
            - Fixes weak references in cells to point to self
            - Sets starting_ignitions based on burning regions
        """
        # Reset cell grid in-place: point to the original grid objects and
        # reset all cells to initial FUEL state. This avoids expensive deepcopy
        # while providing the same semantics — each call starts from clean state.
        self._cell_grid = self.orig_grid
        self._cell_dict = self.orig_dict

        # Reset all cells to initial FUEL state and fix weak references
        for cell in self._cell_dict.values():
            cell.reset_to_fuel()
            cell.set_parent(self)

        self._burnt_cells = []
        self._burning_cells = []
        self._updated_cells = {}
        self._scheduled_spot_fires = {}

        # Sync grid manager with the cell grid
        if hasattr(self, '_grid_manager') and self._grid_manager is not None:
            self._grid_manager._cell_grid = self._cell_grid
            self._grid_manager._cell_dict = self._cell_dict

        if state_estimate is None:
            # Set the burnt cells based on fire state
            # If we're in a worker (self.fire is None), use serialized fire state
            if self.fire is not None:
                if self.fire._burnt_cells:
                    burnt_region = UtilFuncs.get_cell_polygons(self.fire._burnt_cells)
                    self._set_initial_burnt_region(burnt_region)

                # Set the burning cells based on fire state
                burning_cells = [cell for cell in self.fire._burning_cells]
                burning_region = UtilFuncs.get_cell_polygons(burning_cells)
                self._set_initial_ignition(burning_region)
            else:
                # In worker process, use serialized fire state
                if self._serialization_data and 'fire_state' in self._serialization_data:
                    fire_state = self._serialization_data['fire_state']
                    if fire_state['burnt_cell_polygons']:
                        self._set_initial_burnt_region(fire_state['burnt_cell_polygons'])
                    if fire_state['burning_cell_polygons']:
                        self._set_initial_ignition(fire_state['burning_cell_polygons'])

        else:
            # Initialize empty set of starting ignitions
            self.starting_ignitions = set()

            # Set the burnt cells based on provided estimate
            if state_estimate.burnt_polys:
                self._set_initial_burnt_region(state_estimate.burnt_polys)

            # Set the burning cells based on provided estimate
            if state_estimate.burning_polys:
                self._set_initial_ignition(state_estimate.burning_polys)

    # =========================================================================
    # Weather & Wind
    # =========================================================================

    def _catch_up_with_fire(self, custom_start_time_s: Optional[float] = None) -> None:
        """Synchronize predictor time state with parent fire simulation.

        Set prediction start time, end time, and generate wind forecast.
        In worker processes (fire is None), uses serialized state instead.

        Args:
            custom_start_time_s (float): Optional start time in seconds from
                simulation start. If provided, validates and uses this as the
                prediction start time. If None, uses current fire simulation time.

        Side Effects:
            - Sets _curr_time_s, start_time_s, _end_time
            - Sets _start_weather_idx for custom start times
            - Calls _predict_wind() unless forecast was pre-assigned from pool
        """
        # Store the start weather index for any-time predictions
        self._start_weather_idx = None

        if custom_start_time_s is not None:
            # Validate and compute weather index for custom start time
            start_weather_idx, validated_start_time = self._validate_start_time(custom_start_time_s)
            self._start_weather_idx = start_weather_idx
            self._curr_time_s = validated_start_time
            self.start_time_s = validated_start_time
            self.last_viz_update = validated_start_time
            self._end_time = (self.time_horizon_hr * 3600) + self.start_time_s

            # Only generate wind if not pre-assigned from pool
            if not getattr(self, '_wind_forecast_assigned', False):
                self._predict_wind()
            else:
                self._cap_horizon_to_pool_forecast()

        elif self.fire is not None:
            # In main process: get current state from fire
            self._curr_time_s = self.fire._curr_time_s
            self.start_time_s = self._curr_time_s
            self.last_viz_update = self._curr_time_s
            self._end_time = (self.time_horizon_hr * 3600) + self.start_time_s

            # Only generate wind if not pre-assigned from pool
            if not getattr(self, '_wind_forecast_assigned', False):
                self._predict_wind()
            else:
                self._cap_horizon_to_pool_forecast()
        else:
            # In worker process: use serialized state
            self.start_time_s = self._curr_time_s  # Already set in __setstate__
            self.last_viz_update = self._curr_time_s
            self._end_time = (self.time_horizon_hr * 3600) + self.start_time_s

            # Skip wind generation if forecast assigned from pool
            if getattr(self, '_wind_forecast_assigned', False):
                self._cap_horizon_to_pool_forecast()
                return

            # Generate perturbed wind forecast in worker if varying per member
            # or if wind forecast was not pre-computed
            if getattr(self, '_vary_wind_per_member', False) or self.wind_forecast is None:
                self._predict_wind()

    def _predict_wind(self) -> None:
        """Generate perturbed wind forecast for prediction.

        Apply bias and AR(1) autoregressive noise to create uncertain wind
        forecasts. Runs WindNinja to compute terrain-adjusted wind fields.

        In worker processes (fire is None), uses serialized weather stream.
        Uses _start_weather_idx if set (any-time mode), otherwise uses fire's
        current weather index.

        Side Effects:
            - Sets wind_forecast array with shape (n_times, rows, cols, 2)
              where last dimension is [speed_ms, direction_deg]
            - Sets wind_xpad, wind_ypad for coordinate alignment
            - Updates _weather_stream with perturbed values
            - Creates temporary directory for WindNinja output

        Performance Note:
            Uses shallow copy of WeatherStream and creates new WeatherEntry objects
            directly to avoid expensive deepcopy operations on the entire stream.
        """
        from embrs.utilities.data_classes import WeatherEntry

        # Get source weather stream (shallow copy to avoid expensive deepcopy)
        if self.fire is not None:
            source_stream = self.fire._weather_stream
            new_weather_stream = copy.copy(source_stream)
            # Use custom start weather index if set (any-time mode)
            curr_idx = (self._start_weather_idx if self._start_weather_idx is not None
                       else self.fire._curr_weather_idx)
        else:
            # In worker: use serialized weather stream
            source_stream = self._weather_stream
            new_weather_stream = copy.copy(source_stream)
            # Use custom start weather index if set (any-time mode)
            curr_idx = (self._start_weather_idx if self._start_weather_idx is not None
                       else self._curr_weather_idx)

        self.weather_t_step = source_stream.time_step * 60

        num_indices = int(np.ceil((self.time_horizon_hr * 3600) / self.weather_t_step))

        self._curr_weather_idx = 0
        if self.fire is not None:
            self._last_weather_update = self.fire._last_weather_update
        # else: _last_weather_update already set in __setstate__

        end_idx = num_indices + curr_idx

        # Clamp end_idx to stream length (defensive check)
        stream_len = len(source_stream.stream)
        if end_idx + 1 > stream_len:
            import warnings
            old_end_idx = end_idx
            end_idx = stream_len - 1
            # Update time horizon and end time to match capped range
            capped_num_indices = end_idx - curr_idx
            self.time_horizon_hr = capped_num_indices * self.weather_t_step / 3600
            self._end_time = self.start_time_s + self.time_horizon_hr * 3600
            warnings.warn(
                f"FirePredictor._predict_wind: end_idx ({old_end_idx} + 1) exceeds "
                f"stream length ({stream_len}). Capped to {end_idx}, "
                f"time_horizon_hr now {self.time_horizon_hr:.2f}."
            )

        speed_error = 0
        dir_error = 0

        new_stream = []

        # Create new WeatherEntry objects directly instead of deepcopying
        for entry in source_stream.stream[curr_idx:end_idx + 1]:
            # Apply bias and accumulated error to create perturbed entry
            perturbed_speed = max(0, entry.wind_speed + speed_error + self.wind_speed_bias)
            perturbed_dir = (entry.wind_dir_deg + dir_error + self.wind_dir_bias) % 360

            new_entry = WeatherEntry(
                wind_speed=perturbed_speed,
                wind_dir_deg=perturbed_dir,
                temp=entry.temp,
                rel_humidity=entry.rel_humidity,
                cloud_cover=entry.cloud_cover,
                rain=entry.rain,
                dni=entry.dni,
                dhi=entry.dhi,
                ghi=entry.ghi,
                solar_zenith=entry.solar_zenith,
                solar_azimuth=entry.solar_azimuth
            )
            new_stream.append(new_entry)

            speed_error = self.beta * speed_error + np.random.normal(0, self.wnd_spd_std)
            dir_error = self.beta * dir_error + np.random.normal(0, self.wnd_dir_std)

        new_weather_stream.stream = new_stream

        # Get map params from fire or from serialized sim_params
        map_params = self.fire._sim_params.map_params if self.fire is not None else self._sim_params.map_params

        # Generate unique temp directory for this worker to avoid race conditions
        # when multiple ensemble members run in parallel
        worker_id = uuid.uuid4().hex[:8]
        custom_temp = os.path.join(temp_file_path, f"worker_{worker_id}")

        self.wind_forecast = run_windninja(new_weather_stream, map_params, custom_temp)
        self.flipud_forecast = np.empty(self.wind_forecast.shape)

        for layer in range(self.wind_forecast.shape[0]):
            self.flipud_forecast[layer] = np.flipud(self.wind_forecast[layer])

        self.wind_forecast = self.flipud_forecast

        # Get mesh resolution from fire or from serialized sim_params
        sim_params = self.fire._sim_params if self.fire is not None else self._sim_params
        self._wind_res = sim_params.weather_input.mesh_resolution
        self._weather_stream = new_weather_stream

        self.wind_xpad, self.wind_ypad = self.calc_wind_padding(self.wind_forecast)

    def _set_prediction_forecast(self, cell) -> None:
        """Set wind forecast for a specific cell based on its position.

        Look up wind speed and direction from the forecast grid at the cell's
        spatial position, accounting for grid resolution and padding offsets.

        Args:
            cell (Cell): Cell to update with wind forecast data.

        Side Effects:
            Calls cell._set_wind_forecast() with interpolated wind values.
        """
        x_wind = max(cell.x_pos - self.wind_xpad, 0)
        y_wind = max(cell.y_pos - self.wind_ypad, 0)

        wind_col = int(np.floor(x_wind / self._wind_res))
        wind_row = int(np.floor(y_wind / self._wind_res))

        if wind_row > self.wind_forecast.shape[1] - 1:
            wind_row = self.wind_forecast.shape[1] - 1

        if wind_col > self.wind_forecast.shape[2] - 1:
            wind_col = self.wind_forecast.shape[2] - 1

        wind_speed = self.wind_forecast[:, wind_row, wind_col, 0]
        wind_dir = self.wind_forecast[:, wind_row, wind_col, 1]
        cell._set_wind_forecast(wind_speed, wind_dir)

    # =========================================================================
    # Spotting Model
    # =========================================================================

    def _ignite_spots(self) -> None:
        """Process ember landings and schedule or ignite spot fires.

        Compute ignition probability for each ember landing based on travel
        distance and nominal ignition probability. Schedule successful ignitions
        for future time steps and ignite any previously scheduled spot fires
        that are due.

        Side Effects:
            - Clears embers from the Perryman spotting model
            - Adds entries to _scheduled_spot_fires
            - Appends ignited cells to _new_ignitions
            - Updates cell states to FIRE for ignited spots
        """
        # Decay constant for ignition probability
        lambda_s = 0.005

        # Get all the lofted embers by the Perryman model
        spot_fires = self.embers.embers

        landings = {}
        if spot_fires:
            for spot in spot_fires:
                x = spot['x']
                y = spot['y']
                d = spot['d']

                # Get the cell the ember lands in
                landing_cell = self.get_cell_from_xy(x, y, oob_ok=True)

                if landing_cell is not None and landing_cell.fuel.burnable:
                    # Compute the probability based on how far the ember travelled
                    p_i = self.nom_ign_prob * np.exp(-lambda_s * d)

                    # Add landing probability to dict or update its probability
                    if landings.get(landing_cell.id) is None:
                        landings[landing_cell.id] = 1 - p_i
                    else:
                        landings[landing_cell.id] *= (1 - p_i)

            for cell_id in list(landings.keys()):
                # Determine if the cell will ignite
                rand = np.random.random()
                if rand < (1 - landings[cell_id]):
                    # Schedule ignition
                    ign_time = self._curr_time_s + self._spot_delay_s

                    if self._scheduled_spot_fires.get(ign_time) is None:
                        self._scheduled_spot_fires[ign_time] = [self._cell_dict[cell_id]]
                    else:
                        self._scheduled_spot_fires[ign_time].append(self._cell_dict[cell_id])

        # Clear the embers from the Perryman model
        self.embers.embers = []

        # Ignite any scheduled spot fires
        if self._scheduled_spot_fires:
            pending_times = list(self._scheduled_spot_fires.keys())

            # Check if there are any ignitions which take place this time step
            for time in pending_times:
                if time <= self.curr_time_s:
                    # Ignite the fires scheduled for this time step
                    new_spots = self._scheduled_spot_fires[time]
                    for spot in new_spots:
                        if spot.state == CellStates.FUEL and spot.fuel.burnable:
                            self._new_ignitions.append(spot)
                            spot.get_ign_params(0)
                            spot._set_state(CellStates.FIRE)
                            self._updated_cells[spot.id] = spot

                    # Delete entry from schedule if ignited
                    del self._scheduled_spot_fires[time]

                if time > self.curr_time_s:
                    break

    def _calc_nominal_prob(self) -> float:
        """Calculate nominal ignition probability for spotting.

        Compute base ignition probability using the method from "Ignition
        Probability" (Schroeder 1969) as described in the Perryman spotting
        model paper. Probability depends on dead fuel moisture content.

        Returns:
            float: Nominal ignition probability (0.0-1.0).
        """
        Q_ig = 250 + 1116 * self.dead_mf
        Q_ig_cal = BTU_lb_to_cal_g(Q_ig)

        x = (400 - Q_ig_cal) / 10
        p_i = (0.000048 * x ** 4.3) / 50

        return p_i

    def _validate_start_time(
        self,
        start_time_s: float,
        time_horizon_hr: Optional[float] = None
    ) -> Tuple[int, float]:
        """Validate start time and compute corresponding weather index.

        Check that the requested start time is not in the past and that the
        weather forecast covers the full prediction horizon from that time.

        Args:
            start_time_s (float): Desired start time in seconds from simulation
                start.
            time_horizon_hr (float): Optional time horizon in hours for validation.
                If None, uses self.time_horizon_hr.

        Returns:
            tuple: (weather_index, validated_start_time) where weather_index is
                the index into the weather stream and validated_start_time is
                the confirmed start time in seconds.

        Raises:
            ValueError: If start_time_s is in the past relative to current fire
                time, or if weather forecast doesn't cover the full prediction
                horizon.
        """
        # Use provided time horizon or fall back to instance attribute
        horizon_hr = time_horizon_hr if time_horizon_hr is not None else self.time_horizon_hr
        # Get current fire state (from fire or serialized data)
        if self.fire is not None:
            curr_time_s = self.fire._curr_time_s
            curr_weather_idx = self.fire._curr_weather_idx
            weather_stream = self.fire._weather_stream
        else:
            # In worker process: use serialized state
            fire_state = self._serialization_data['fire_state']
            curr_time_s = fire_state['curr_time_s']
            curr_weather_idx = fire_state['curr_weather_idx']
            weather_stream = self._serialization_data['weather_stream']

        # Constraint 1: start_time >= current fire time
        if start_time_s < curr_time_s:
            raise ValueError(
                f"start_time_s ({start_time_s}s) cannot be earlier than "
                f"current fire time ({curr_time_s}s). Cannot predict from the past."
            )

        # Compute weather time step in seconds
        weather_t_step = weather_stream.time_step * 60

        # Compute weather index for the start time
        time_delta_s = start_time_s - curr_time_s
        weather_offset = int(time_delta_s // weather_t_step)
        start_weather_idx = curr_weather_idx + weather_offset

        # Constraint 2: weather forecast covers full prediction horizon
        total_weather_entries = len(weather_stream.stream)
        prediction_end_time_s = start_time_s + (horizon_hr * 3600)
        prediction_end_offset = int((prediction_end_time_s - curr_time_s) // weather_t_step)
        required_weather_idx = curr_weather_idx + prediction_end_offset

        if required_weather_idx >= total_weather_entries:
            max_end_time_s = curr_time_s + (total_weather_entries - curr_weather_idx - 1) * weather_t_step
            max_start_time_s = max_end_time_s - (horizon_hr * 3600)
            raise ValueError(
                f"Weather forecast does not cover full prediction horizon. "
                f"start_time_s ({start_time_s}s) + time_horizon ({horizon_hr}hr) "
                f"requires weather data until index {required_weather_idx}, but only "
                f"{total_weather_entries} entries available. "
                f"Maximum valid start_time_s is {max_start_time_s}s."
            )

        return (start_weather_idx, start_time_s)

    def _cap_horizon_to_pool_forecast(self) -> None:
        """Cap prediction horizon to the pool forecast's time axis length.

        When using a pre-assigned forecast pool, ensure that time_horizon_hr
        does not exceed the number of time steps available in wind_forecast.
        If it does, warn and cap both time_horizon_hr and _end_time.

        Side Effects:
            - May reduce self.time_horizon_hr and self._end_time.
        """
        if self.wind_forecast is None:
            return
        n_steps = self.wind_forecast.shape[0]
        max_horizon_hr = (n_steps - 1) * self.weather_t_step / 3600
        if self.time_horizon_hr > max_horizon_hr:
            import warnings
            warnings.warn(
                f"FirePredictor._cap_horizon_to_pool_forecast: "
                f"time_horizon_hr ({self.time_horizon_hr:.2f}) exceeds pool forecast "
                f"coverage ({max_horizon_hr:.2f} hr, {n_steps} steps). "
                f"Capping to {max_horizon_hr:.2f} hr."
            )
            self.time_horizon_hr = max_horizon_hr
            self._end_time = self.start_time_s + self.time_horizon_hr * 3600

    # =========================================================================
    # Serialization (for parallel execution)
    # =========================================================================

    def prepare_for_serialization(
        self,
        vary_wind: bool = False,
        forecast_pool: Optional[ForecastPool] = None,
        forecast_indices: Optional[List[int]] = None
    ) -> None:
        """Prepare predictor for parallel execution by extracting serializable data.

        Must be called once before pickling the predictor. Capture the current
        state of the parent FireSim and store it in a serializable format.
        Call this in the main process before spawning workers.

        Args:
            vary_wind (bool): If True, workers generate their own wind forecasts.
                If False, use pre-computed shared wind forecast. Defaults to False.
            forecast_pool (ForecastPool): Optional pre-computed forecast pool.
            forecast_indices (list[int]): Indices mapping each ensemble member
                to a forecast in the pool.

        Raises:
            RuntimeError: If called without a fire reference (fire is None).

        Side Effects:
            Populates _serialization_data dict with fire state, parameters,
            weather stream, and optionally pre-computed wind forecast.
        """
        # Delegate to PredictorSerializer
        PredictorSerializer.prepare_for_serialization(
            self,
            vary_wind=vary_wind,
            forecast_pool=forecast_pool,
            forecast_indices=forecast_indices
        )

    def __getstate__(self) -> dict:
        """Serialize predictor for parallel execution.

        Return only the essential data needed to reconstruct the predictor
        in a worker process. Exclude non-serializable components like the
        parent FireSim reference, visualizer, and logger.

        Returns:
            dict: Minimal state dictionary containing serialization_data,
                orig_grid, orig_dict, and c_size.

        Raises:
            RuntimeError: If prepare_for_serialization() was not called first.
        """
        # Delegate to PredictorSerializer
        return PredictorSerializer.get_state(self)

    def __setstate__(self, state: dict) -> None:
        """Reconstruct predictor in worker process without full initialization.

        Manually restore all attributes that BaseFireSim.__init__() would set,
        but without the expensive cell creation loop. Use pre-built cell
        templates (orig_grid, orig_dict) instead of reconstructing cells from
        map data.

        Args:
            state (dict): State dictionary from __getstate__ containing
                serialization_data, orig_grid, orig_dict, and c_size.

        Side Effects:
            Restores all instance attributes needed for prediction, including
            maps, fuel models, weather stream, and optionally wind forecast.
            Sets fire to None (no parent reference in workers).
        """
        # Delegate to PredictorSerializer
        PredictorSerializer.set_state(self, state)

    # =========================================================================
    # Resource Management
    # =========================================================================

    def cleanup(self) -> None:
        """Release all predictor resources including forecast pools.

        Clears all active forecast pools managed by ForecastPoolManager
        and all prediction output data. Should be called when the predictor
        is no longer needed to free memory.

        This method is safe to call multiple times.

        Example:
            >>> predictor = FirePredictor(params, fire)
            >>> pool = predictor.generate_forecast_pool(30)
            >>> output = predictor.run_ensemble(estimates, forecast_pool=pool)
            >>> # When done with prediction
            >>> predictor.cleanup()
        """
        from embrs.tools.forecast_pool import ForecastPoolManager
        ForecastPoolManager.clear_all()
        self.clear_prediction_data()

    def clear_prediction_data(self) -> None:
        """Clear all prediction output data structures.

        Frees memory used by spread tracking, flame lengths, fire line
        intensities, and other per-timestep data accumulated during prediction.

        This is called automatically by cleanup() but can also be called
        separately to free memory while keeping the predictor usable.

        Note:
            The next call to predict() will re-initialize these data structures,
            so this is safe to call between predictions.
        """
        # Clear spread tracking dictionaries
        if hasattr(self, 'spread'):
            self.spread.clear()
        if hasattr(self, 'flame_len_m'):
            self.flame_len_m.clear()
        if hasattr(self, 'fli_kw_m'):
            self.fli_kw_m.clear()
        if hasattr(self, 'ros_ms'):
            self.ros_ms.clear()
        if hasattr(self, 'spread_dir'):
            self.spread_dir.clear()
        if hasattr(self, 'crown_fire'):
            self.crown_fire.clear()
        if hasattr(self, 'hold_probs'):
            self.hold_probs.clear()
        if hasattr(self, 'breaches'):
            self.breaches.clear()
        if hasattr(self, 'active_fire_front'):
            self.active_fire_front.clear()
        if hasattr(self, 'burnt_spread'):
            self.burnt_spread.clear()
        

        # Clear internal state tracking
        if hasattr(self, '_updated_cells'):
            self._updated_cells.clear()
        if hasattr(self, '_scheduled_spot_fires'):
            self._scheduled_spot_fires.clear()

    # =========================================================================
    # Visualization
    # =========================================================================

    def _visualize_ensemble(self, ensemble_output: EnsemblePredictionOutput) -> None:
        """Visualize ensemble prediction results on the fire visualizer.

        Args:
            ensemble_output (EnsemblePredictionOutput): Aggregated ensemble
                results containing burn probability maps.

        Side Effects:
            Calls fire.visualize_ensemble_prediction() to render burn
            probability overlay on the simulation display.
        """
        self.fire.visualize_ensemble_prediction(ensemble_output.burn_probability)


# =============================================================================
# Module-level functions for parallel execution
# =============================================================================

def _run_ensemble_member_worker(
    predictor: 'FirePredictor',
    state_estimate: StateEstimate,
    seed: Optional[int] = None,
    member_params: Optional[PredictorParams] = None
) -> PredictionOutput:
    """Execute a single ensemble member prediction in a worker process.

    Receive a deserialized FirePredictor (via __setstate__) and run a single
    prediction. The predictor has been reconstructed without the original
    FireSim reference.

    Supports any-time predictions: if state_estimate.start_time_s is set,
    the prediction starts from that future time instead of the fire's
    current time at serialization.

    Args:
        predictor (FirePredictor): Deserialized predictor instance.
        state_estimate (StateEstimate): Initial fire state for this member.
            May include start_time_s for future-time predictions.
        seed (int): Random seed for reproducibility. If None, uses default
            random state.
        member_params (PredictorParams): Optional per-member parameters.
            If provided, applies these params before running prediction.

    Returns:
        PredictionOutput: Prediction results for this ensemble member.
            Timestamps are global (from simulation start), not relative
            to start_time_s.

    Raises:
        Exception: Any errors during prediction are logged and re-raised
            for the executor to handle.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Apply member-specific params if provided
    if member_params is not None:
        predictor._apply_member_params(member_params)

    # Run prediction (visualize=False always in workers)
    # If state_estimate.start_time_s is set, run() will handle the any-time logic:
    # - _catch_up_with_fire validates and computes weather index offset
    # - _predict_wind uses the offset to slice the weather stream correctly
    try:
        output = predictor.run(fire_estimate=state_estimate, visualize=False)
        return output
    except Exception as e:
        # Log error and re-raise (executor will handle)
        import traceback
        print(f"ERROR in ensemble member: {e}")
        print(traceback.format_exc())
        raise


def _aggregate_ensemble_predictions(
    predictions: List[PredictionOutput]
) -> EnsemblePredictionOutput:
    """Aggregate multiple prediction outputs into ensemble statistics.

    Compute probabilistic burn maps, fire behavior statistics (mean, std, min,
    max), crown fire frequency, spread direction statistics using circular
    mean, and fireline breach statistics.

    Args:
        predictions (list[PredictionOutput]): Individual prediction outputs
            from ensemble members.

    Returns:
        EnsemblePredictionOutput: Aggregated ensemble results including:
            - burn_probability: Cumulative burn probability by time and location
            - flame_len_m_stats: Flame length statistics per cell
            - fli_kw_m_stats: Fireline intensity statistics per cell
            - ros_ms_stats: Rate of spread statistics per cell
            - spread_dir_stats: Circular mean direction and dispersion per cell
            - crown_fire_frequency: Fraction of members with crown fire per cell
            - hold_prob_stats: Fireline hold probability statistics
            - breach_frequency: Fraction of members breaching each fireline cell
    """
    n_ensemble = len(predictions)

    # -------------------------------------------------------------------------
    # 1. Build CUMULATIVE burn probability map
    # -------------------------------------------------------------------------
    # For each time step, track which ensemble members have burned each location
    all_time_steps = sorted(set(
        time_s for pred in predictions for time_s in pred.spread.keys()
    ))

    # Incremental O(n) approach: advance through sorted timesteps once,
    # accumulating burn counts as new burns appear at each timestep.
    burn_counts = {}  # {loc: number of ensemble members that burned it}
    member_burned = [set() for _ in range(n_ensemble)]  # per-member cumulative

    burn_probability = {}
    for time_s in all_time_steps:
        # Only process NEW burns at this timestep
        for ensemble_idx, pred in enumerate(predictions):
            if time_s in pred.spread:
                for loc in pred.spread[time_s]:
                    if loc not in member_burned[ensemble_idx]:
                        member_burned[ensemble_idx].add(loc)
                        burn_counts[loc] = burn_counts.get(loc, 0) + 1

        # Snapshot current cumulative probabilities
        burn_probability[time_s] = {
            loc: count / n_ensemble for loc, count in burn_counts.items()
        }

    # -------------------------------------------------------------------------
    # 1b. Build active fire probability and burnt probability maps
    # -------------------------------------------------------------------------
    # Collect all time keys from active_fire_front across members (union)
    all_active_times = sorted(set(
        time_s for pred in predictions for time_s in pred.active_fire_front.keys()
    ))

    active_fire_probability = {}
    for time_s in all_active_times:
        cell_counts = {}
        for pred in predictions:
            if time_s in pred.active_fire_front:
                for loc in pred.active_fire_front[time_s]:
                    cell_counts[loc] = cell_counts.get(loc, 0) + 1
        active_fire_probability[time_s] = {
            loc: count / n_ensemble for loc, count in cell_counts.items()
        }

    # Collect all time keys from burnt_spread across members (union)
    all_burnt_times = sorted(set(
        time_s for pred in predictions for time_s in pred.burnt_spread.keys()
    ))

    burnt_probability = {}
    for time_s in all_burnt_times:
        cell_counts = {}
        for pred in predictions:
            if time_s in pred.burnt_spread:
                for loc in pred.burnt_spread[time_s]:
                    cell_counts[loc] = cell_counts.get(loc, 0) + 1
        burnt_probability[time_s] = {
            loc: count / n_ensemble for loc, count in cell_counts.items()
        }

    # -------------------------------------------------------------------------
    # 2. Collect all unique cell locations that burned in any prediction
    # -------------------------------------------------------------------------
    all_burned_cells = set()
    for pred in predictions:
        all_burned_cells.update(pred.flame_len_m.keys())

    # -------------------------------------------------------------------------
    # 3. Aggregate statistics for each cell
    # -------------------------------------------------------------------------
    flame_stats = {}
    ros_stats = {}
    fli_stats = {}

    for cell_loc in all_burned_cells:
        # Collect values from all predictions where this cell burned
        flame_values = [
            p.flame_len_m.get(cell_loc) for p in predictions
            if cell_loc in p.flame_len_m
        ]
        ros_values = [
            p.ros_ms.get(cell_loc) for p in predictions
            if cell_loc in p.ros_ms
        ]
        fli_values = [
            p.fli_kw_m.get(cell_loc) for p in predictions
            if cell_loc in p.fli_kw_m
        ]

        if flame_values:
            flame_stats[cell_loc] = CellStatistics(
                mean=float(np.mean(flame_values)),
                std=float(np.std(flame_values)),
                min=float(np.min(flame_values)),
                max=float(np.max(flame_values)),
                count=len(flame_values)
            )

        if ros_values:
            ros_stats[cell_loc] = CellStatistics(
                mean=float(np.mean(ros_values)),
                std=float(np.std(ros_values)),
                min=float(np.min(ros_values)),
                max=float(np.max(ros_values)),
                count=len(ros_values)
            )

        if fli_values:
            fli_stats[cell_loc] = CellStatistics(
                mean=float(np.mean(fli_values)),
                std=float(np.std(fli_values)),
                min=float(np.min(fli_values)),
                max=float(np.max(fli_values)),
                count=len(fli_values)
            )

    # -------------------------------------------------------------------------
    # 4. Crown fire frequency
    # -------------------------------------------------------------------------
    crown_frequency = {}
    for cell_loc in all_burned_cells:
        crown_count = sum(1 for p in predictions if cell_loc in p.crown_fire)
        crown_frequency[cell_loc] = crown_count / n_ensemble

    # -------------------------------------------------------------------------
    # 5. Spread direction (using circular statistics)
    # -------------------------------------------------------------------------
    spread_dir_stats = {}
    for cell_loc in all_burned_cells:
        dirs = [
            p.spread_dir.get(cell_loc) for p in predictions
            if cell_loc in p.spread_dir
        ]
        if dirs:
            # Convert to unit vectors
            x_components = [np.cos(d) for d in dirs]
            y_components = [np.sin(d) for d in dirs]

            # Mean direction
            mean_x = np.mean(x_components)
            mean_y = np.mean(y_components)
            mean_dir = np.arctan2(mean_y, mean_x)

            # Circular standard deviation
            R = np.sqrt(mean_x**2 + mean_y**2)
            circular_std = np.sqrt(-2 * np.log(R)) if R > 0 else 0.0

            spread_dir_stats[cell_loc] = {
                'mean_dir': float(mean_dir),
                'circular_std': float(circular_std),
                'mean_x': float(mean_x),
                'mean_y': float(mean_y)
            }

    # -------------------------------------------------------------------------
    # 6. Fireline statistics
    # -------------------------------------------------------------------------
    all_fireline_cells = set()
    for pred in predictions:
        all_fireline_cells.update(pred.hold_probs.keys())

    hold_prob_stats = {}
    breach_frequency = {}

    for cell_loc in all_fireline_cells:
        hold_probs = [
            p.hold_probs.get(cell_loc) for p in predictions
            if cell_loc in p.hold_probs
        ]
        breaches = [
            p.breaches.get(cell_loc) for p in predictions
            if cell_loc in p.breaches
        ]

        if hold_probs:
            hold_prob_stats[cell_loc] = CellStatistics(
                mean=float(np.mean(hold_probs)),
                std=float(np.std(hold_probs)),
                min=float(np.min(hold_probs)),
                max=float(np.max(hold_probs)),
                count=len(hold_probs)
            )

        if breaches:
            breach_frequency[cell_loc] = sum(breaches) / len(breaches)

    return EnsemblePredictionOutput(
        n_ensemble=n_ensemble,
        burn_probability=burn_probability,
        flame_len_m_stats=flame_stats,
        fli_kw_m_stats=fli_stats,
        ros_ms_stats=ros_stats,
        spread_dir_stats=spread_dir_stats,
        crown_fire_frequency=crown_frequency,
        hold_prob_stats=hold_prob_stats,
        breach_frequency=breach_frequency,
        active_fire_probability=active_fire_probability,
        burnt_probability=burnt_probability
    )
