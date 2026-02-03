"""Serialization utilities for FirePredictor multiprocessing.

This module provides the PredictorSerializer class which handles all
serialization and deserialization logic for FirePredictor instances,
enabling efficient parallel execution of ensemble predictions.

The serializer extracts the minimal state needed to reconstruct a
predictor in worker processes without transferring the full FireSim
reference or rebuilding the cell grid from scratch.

Classes:
    - PredictorSerializer: Handles FirePredictor serialization/deserialization.

Example:
    >>> from embrs.tools.predictor_serializer import PredictorSerializer
    >>>
    >>> # Prepare for parallel execution
    >>> PredictorSerializer.prepare_for_serialization(predictor, vary_wind=False)
    >>>
    >>> # Get minimal state for pickling
    >>> state = PredictorSerializer.get_state(predictor)
    >>>
    >>> # Restore state in worker process
    >>> PredictorSerializer.set_state(predictor, state)
"""

from __future__ import annotations

import copy
from typing import Dict, Any, Optional, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from embrs.tools.fire_predictor import FirePredictor
    from embrs.tools.forecast_pool import ForecastPool
    from embrs.utilities.fire_util import UtilFuncs


class PredictorSerializer:
    """Handles serialization and deserialization for FirePredictor multiprocessing.

    This class owns the entire serialization process for FirePredictor,
    including:
    - Capturing fire simulation state for worker processes
    - Creating minimal pickle-compatible state dictionaries
    - Reconstructing predictor state without full FireSim initialization

    The serializer is designed to work with Python's pickle module and
    multiprocessing, ensuring efficient transfer of predictor state to
    worker processes for parallel ensemble predictions.

    Class Methods:
        prepare_for_serialization: Capture state from parent FireSim.
        get_state: Return minimal state dict for pickling (__getstate__).
        set_state: Restore predictor from state dict (__setstate__).

    Example:
        >>> # In main process before spawning workers
        >>> PredictorSerializer.prepare_for_serialization(
        ...     predictor,
        ...     vary_wind=False,
        ...     forecast_pool=pool
        ... )
        >>>
        >>> # Predictor can now be pickled and sent to workers
        >>> import pickle
        >>> pickled = pickle.dumps(predictor)
    """

    @staticmethod
    def prepare_for_serialization(
        predictor: 'FirePredictor',
        vary_wind: bool = False,
        forecast_pool: Optional['ForecastPool'] = None,
        forecast_indices: Optional[List[int]] = None
    ) -> None:
        """Prepare predictor for parallel execution by extracting serializable data.

        Must be called once before pickling the predictor. Captures the current
        state of the parent FireSim and stores it in a serializable format.

        Args:
            predictor: FirePredictor instance to prepare.
            vary_wind: If True, workers generate their own wind forecasts.
                If False, use pre-computed shared wind forecast.
            forecast_pool: Optional pre-computed forecast pool.
            forecast_indices: Indices mapping each ensemble member to a
                forecast in the pool.

        Raises:
            RuntimeError: If called without a fire reference (fire is None).

        Side Effects:
            Populates predictor._serialization_data dict with fire state,
            parameters, weather stream, and optionally pre-computed wind forecast.
        """
        from embrs.utilities.fire_util import UtilFuncs

        if predictor.fire is None:
            raise RuntimeError("Cannot prepare predictor without fire reference")

        fire = predictor.fire

        # Extract fire state at prediction start
        fire_state = {
            'curr_time_s': fire._curr_time_s,
            'curr_weather_idx': fire._curr_weather_idx,
            'last_weather_update': fire._last_weather_update,
            'burning_cell_polygons': UtilFuncs.get_cell_polygons(fire._burning_cells),
            'burnt_cell_polygons': (UtilFuncs.get_cell_polygons(fire._burnt_cells)
                                   if fire._burnt_cells else None),
        }

        # Deep copy simulation parameters
        sim_params_copy = copy.deepcopy(fire._sim_params)
        weather_stream_copy = copy.deepcopy(fire._weather_stream)

        # Store all serializable data
        serialization_data = {
            'sim_params': sim_params_copy,
            'predictor_params': copy.deepcopy(predictor._params),
            'fire_state': fire_state,
            'weather_stream': weather_stream_copy,
            'vary_wind_per_member': vary_wind,

            # Predictor-specific attributes
            'time_horizon_hr': predictor.time_horizon_hr,
            'wind_uncertainty_factor': predictor.wind_uncertainty_factor,
            'wind_speed_bias': predictor.wind_speed_bias,
            'wind_dir_bias': predictor.wind_dir_bias,
            'ros_bias_factor': predictor.ros_bias_factor,
            'beta': predictor.beta,
            'wnd_spd_std': predictor.wnd_spd_std,
            'wnd_dir_std': predictor.wnd_dir_std,
            'dead_mf': predictor.dead_mf,
            'live_mf': predictor.live_mf,
            'nom_ign_prob': predictor.nom_ign_prob,

            # Elevation data (always needed)
            'coarse_elevation': predictor.coarse_elevation,
        }

        # Add forecast pool information
        if forecast_pool is not None:
            serialization_data['use_pooled_forecast'] = True
            serialization_data['forecast_pool'] = forecast_pool
            serialization_data['forecast_indices'] = forecast_indices
        else:
            serialization_data['use_pooled_forecast'] = False
            serialization_data['forecast_pool'] = None
            serialization_data['forecast_indices'] = None

        # Only include pre-computed wind forecast if not varying per member
        # and not using a forecast pool
        if not vary_wind and forecast_pool is None:
            serialization_data['wind_forecast'] = predictor.wind_forecast
            serialization_data['flipud_forecast'] = predictor.flipud_forecast
            serialization_data['wind_xpad'] = predictor.wind_xpad
            serialization_data['wind_ypad'] = predictor.wind_ypad

        predictor._serialization_data = serialization_data

    @staticmethod
    def get_state(predictor: 'FirePredictor') -> Dict[str, Any]:
        """Serialize predictor for parallel execution (__getstate__).

        Return only the essential data needed to reconstruct the predictor
        in a worker process. Excludes non-serializable components like the
        parent FireSim reference, visualizer, and logger.

        Args:
            predictor: FirePredictor instance to serialize.

        Returns:
            Minimal state dictionary containing serialization_data,
            orig_grid, orig_dict, and c_size.

        Raises:
            RuntimeError: If prepare_for_serialization() was not called first.
        """
        if predictor._serialization_data is None:
            raise RuntimeError(
                "Must call prepare_for_serialization() before pickling. "
                "This ensures all necessary state is captured from the parent FireSim."
            )

        # Return minimal state
        state = {
            'serialization_data': predictor._serialization_data,
            'orig_grid': predictor.orig_grid,  # Template cells (pre-built)
            'orig_dict': predictor.orig_dict,  # Template cells (pre-built)
            'c_size': predictor.c_size,
        }

        return state

    @staticmethod
    def set_state(predictor: 'FirePredictor', state: Dict[str, Any]) -> None:
        """Reconstruct predictor in worker process without full initialization.

        Manually restores all attributes that BaseFireSim.__init__() would set,
        but without the expensive cell creation loop. Uses pre-built cell
        templates (orig_grid, orig_dict) instead of reconstructing cells from
        map data.

        Args:
            predictor: FirePredictor instance to restore (typically empty/new).
            state: State dictionary from get_state() containing serialization_data,
                orig_grid, orig_dict, and c_size.

        Side Effects:
            Restores all instance attributes needed for prediction, including
            maps, fuel models, weather stream, and optionally wind forecast.
            Sets fire to None (no parent reference in workers).
        """
        from embrs.models.perryman_spot import PerrymanSpotting
        from embrs.models.fuel_models import Anderson13, ScottBurgan40
        from embrs.base_classes.grid_manager import GridManager
        from embrs.base_classes.weather_manager import WeatherManager

        # Extract serialization data
        data = state['serialization_data']
        sim_params = data['sim_params']

        # =====================================================================
        # Phase 1: Restore FirePredictor-specific attributes
        # =====================================================================
        predictor.fire = None  # No parent fire in worker
        predictor.c_size = state['c_size']
        predictor._params = data['predictor_params']
        predictor._serialization_data = data

        predictor.time_horizon_hr = data['time_horizon_hr']
        predictor.wind_uncertainty_factor = data['wind_uncertainty_factor']
        predictor.wind_speed_bias = data['wind_speed_bias']
        predictor.wind_dir_bias = data['wind_dir_bias']
        predictor.ros_bias_factor = data['ros_bias_factor']
        predictor.beta = data['beta']
        predictor.wnd_spd_std = data['wnd_spd_std']
        predictor.wnd_dir_std = data['wnd_dir_std']
        predictor.dead_mf = data['dead_mf']
        predictor.live_mf = data['live_mf']
        predictor.nom_ign_prob = data['nom_ign_prob']
        predictor._start_weather_idx = None  # Will be set by _catch_up_with_fire if needed

        # =====================================================================
        # Phase 2: Restore BaseFireSim attributes (manually, without __init__)
        # =====================================================================

        # From BaseFireSim.__init__ lines 45-88
        predictor.display_frequency = 300
        predictor._sim_params = sim_params
        predictor.sim_start_w_idx = 0
        predictor._curr_weather_idx = 0
        predictor._last_weather_update = data['fire_state']['last_weather_update']
        predictor.weather_changed = True
        predictor._curr_time_s = data['fire_state']['curr_time_s']
        predictor._iters = 0
        predictor.logger = None  # No logger in worker
        predictor._visualizer = None  # No visualizer in worker
        predictor._finished = False

        # Empty containers (will be populated by _set_states)
        predictor._updated_cells = {}
        predictor._cell_dict = {}
        predictor._long_term_retardants = set()
        predictor._active_water_drops = []
        predictor._burning_cells = []
        predictor._new_ignitions = []
        predictor._burnt_cells = set()
        predictor._frontier = set()
        predictor._fire_break_cells = []
        predictor._active_firelines = {}
        predictor._new_fire_break_cache = []
        predictor.starting_ignitions = set()
        predictor._urban_cells = []
        predictor._scheduled_spot_fires = {}

        # From _parse_sim_params (lines 252-353)
        map_params = sim_params.map_params
        predictor._cell_size = predictor._params.cell_size_m
        predictor._sim_duration = sim_params.duration_s
        predictor._time_step = predictor._params.time_step_s
        predictor._init_mf = predictor._params.dead_mf
        predictor._fuel_moisture_map = getattr(sim_params, 'fuel_moisture_map', {})
        predictor._fms_has_live = getattr(sim_params, 'fms_has_live', False)
        predictor._init_live_h_mf = getattr(sim_params, 'live_h_mf', 0.0)
        predictor._init_live_w_mf = getattr(sim_params, 'live_w_mf', 0.0)
        predictor._size = map_params.size()
        predictor._shape = map_params.shape(predictor._cell_size)
        predictor._roads = map_params.roads
        predictor.coarse_elevation = data['coarse_elevation']

        # Fuel class selection
        fbfm_type = map_params.fbfm_type
        if fbfm_type == "Anderson":
            predictor.FuelClass = Anderson13
        elif fbfm_type == "ScottBurgan":
            predictor.FuelClass = ScottBurgan40
        else:
            raise ValueError(f"FBFM Type {fbfm_type} not supported")

        # Map data (from lcp_data, but already in sim_params)
        lcp_data = map_params.lcp_data
        predictor._elevation_map = np.flipud(lcp_data.elevation_map)
        predictor._slope_map = np.flipud(lcp_data.slope_map)
        predictor._aspect_map = np.flipud(lcp_data.aspect_map)
        predictor._fuel_map = np.flipud(lcp_data.fuel_map)
        predictor._cc_map = np.flipud(lcp_data.canopy_cover_map)
        predictor._ch_map = np.flipud(lcp_data.canopy_height_map)
        predictor._cbh_map = np.flipud(lcp_data.canopy_base_height_map)
        predictor._cbd_map = np.flipud(lcp_data.canopy_bulk_density_map)
        predictor._data_res = lcp_data.resolution

        # Scenario data
        scenario = map_params.scenario_data
        predictor._fire_breaks = list(zip(scenario.fire_breaks, scenario.break_widths, scenario.break_ids))
        predictor.fire_break_dict = {
            id: (fire_break, break_width)
            for fire_break, break_width, id in predictor._fire_breaks
        }
        predictor._initial_ignition = scenario.initial_ign

        # Datetime and orientation
        predictor._start_datetime = sim_params.weather_input.start_datetime
        predictor._north_dir_deg = map_params.geo_info.north_angle_deg

        # Initialize grid manager for geometry operations
        predictor._grid_manager = GridManager(
            num_rows=predictor._shape[0],
            num_cols=predictor._shape[1],
            cell_size=predictor._cell_size
        )

        # Check for pooled forecast mode
        predictor._use_pooled_forecast = data.get('use_pooled_forecast', False)
        predictor._wind_forecast_assigned = False

        predictor._wind_res = sim_params.weather_input.mesh_resolution

        # Initialize weather manager for wind padding calculations
        predictor._weather_manager = WeatherManager(
            weather_stream=None,  # Will be set later
            wind_forecast=None,   # Will be set later
            wind_res=predictor._wind_res,
            sim_size=predictor._size
        )

        if predictor._use_pooled_forecast:
            PredictorSerializer._restore_pooled_forecast(predictor, data)
        else:
            PredictorSerializer._restore_standard_forecast(predictor, data)

        # Spotting parameters
        predictor.model_spotting = predictor._params.model_spotting
        predictor._spot_ign_prob = 0.0
        if predictor.model_spotting:
            predictor._canopy_species = sim_params.canopy_species
            predictor._dbh_cm = sim_params.dbh_cm
            predictor._spot_ign_prob = sim_params.spot_ign_prob
            predictor._min_spot_distance = sim_params.min_spot_dist
            predictor._spot_delay_s = predictor._params.spot_delay_s

        # Moisture (prediction model specific)
        predictor.fmc = 100  # Prediction model default

        # =====================================================================
        # Phase 3: Restore cell templates (CRITICAL - uses pre-built cells)
        # =====================================================================

        # Use the serialized templates instead of reconstructing
        predictor.orig_grid = state['orig_grid']
        predictor.orig_dict = state['orig_dict']

        # Initialize cell_grid to the template shape
        predictor._cell_grid = np.empty(predictor._shape, dtype=object)
        predictor._grid_width = predictor._cell_grid.shape[1] - 1
        predictor._grid_height = predictor._cell_grid.shape[0] - 1

        # Fix weak references in cells (point to self instead of original fire)
        for cell in predictor.orig_dict.values():
            cell.set_parent(predictor)

        # Sync grid manager's internal state with restored cells
        predictor._grid_manager._cell_grid = predictor.orig_grid
        predictor._grid_manager._cell_dict = predictor.orig_dict

        # =====================================================================
        # Phase 4: Rebuild lightweight components
        # =====================================================================

        size = map_params.size()

        # Rebuild spotting model (PerrymanSpotting for prediction)
        if predictor.model_spotting:
            predictor.embers = PerrymanSpotting(predictor._spot_delay_s, size)

    @staticmethod
    def _restore_pooled_forecast(predictor: 'FirePredictor', data: Dict[str, Any]) -> None:
        """Restore wind forecast from a pre-computed forecast pool.

        Args:
            predictor: FirePredictor instance to update.
            data: Serialization data dictionary.
        """
        forecast_pool = data['forecast_pool']
        forecast_indices = data['forecast_indices']
        member_idx = data.get('member_index', 0)

        # Get the assigned forecast for this member
        forecast_idx = forecast_indices[member_idx]
        assigned_forecast = forecast_pool.get_forecast(forecast_idx)

        # Set wind forecast from pool
        predictor.wind_forecast = assigned_forecast.wind_forecast
        predictor.flipud_forecast = assigned_forecast.wind_forecast  # Already flipped in pool generation
        predictor._weather_stream = assigned_forecast.weather_stream
        predictor.weather_t_step = predictor._weather_stream.time_step * 60

        # Update weather manager with actual wind forecast
        predictor._weather_manager._wind_forecast = predictor.wind_forecast
        predictor._weather_manager._weather_stream = predictor._weather_stream

        # Calculate padding using weather manager
        predictor.wind_xpad, predictor.wind_ypad = predictor._weather_manager.calc_wind_padding(predictor.wind_forecast)
        predictor._weather_manager._wind_xpad = predictor.wind_xpad
        predictor._weather_manager._wind_ypad = predictor.wind_ypad

        # Mark that wind is already set
        predictor._wind_forecast_assigned = True
        predictor._vary_wind_per_member = False

    @staticmethod
    def _restore_standard_forecast(predictor: 'FirePredictor', data: Dict[str, Any]) -> None:
        """Restore wind forecast from serialization data or mark for generation.

        Args:
            predictor: FirePredictor instance to update.
            data: Serialization data dictionary.
        """
        # Check if wind should be computed per-member or restored from serialization
        predictor._vary_wind_per_member = data.get('vary_wind_per_member', False)

        if not predictor._vary_wind_per_member:
            # Use pre-computed wind (shared across all members)
            predictor.wind_forecast = data['wind_forecast']
            predictor.flipud_forecast = data['flipud_forecast']
            predictor.wind_xpad = data['wind_xpad']
            predictor.wind_ypad = data['wind_ypad']

            # Update weather manager with pre-computed wind
            predictor._weather_manager._wind_forecast = predictor.wind_forecast
            predictor._weather_manager._wind_xpad = predictor.wind_xpad
            predictor._weather_manager._wind_ypad = predictor.wind_ypad
        else:
            # Wind will be computed later by _predict_wind() in _catch_up_with_fire
            predictor.wind_forecast = None
            predictor.flipud_forecast = None
            predictor.wind_xpad = None
            predictor.wind_ypad = None

        # Weather stream from serialized data
        predictor._weather_stream = data['weather_stream']
        predictor.weather_t_step = predictor._weather_stream.time_step * 60
        predictor._weather_manager._weather_stream = predictor._weather_stream


# =============================================================================
# Module exports
# =============================================================================

__all__ = ['PredictorSerializer']
