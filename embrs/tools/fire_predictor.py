from embrs.base_classes.base_fire import BaseFireSim
from embrs.fire_simulator.fire import FireSim
from embrs.utilities.data_classes import PredictorParams, PredictionOutput, StateEstimate
from embrs.utilities.fire_util import UtilFuncs, CellStates
from embrs.models.rothermel import *
from embrs.models.crown_model import *
from embrs.models.wind_forecast import run_windninja

import copy
import numpy as np
from typing import List, Optional

class FirePredictor(BaseFireSim):
    def __init__(self, params: PredictorParams, fire: FireSim):

        # Live reference to the fire sim
        self.fire = fire
        self.c_size = -1

        # Store original params for serialization
        self._params = params
        self._serialization_data = None  # Will hold snapshot for pickling

        self.set_params(params)

    def set_params(self, params: PredictorParams):

        generate_cell_grid = False

        # How long the prediction will run for
        self.time_horizon_hr = params.time_horizon_hr

        # Uncertainty parameters
        self.wind_uncertainty_factor = params.wind_uncertainty_factor # [0, 1], autoregression noise

        # Compute constant bias terms
        self.wind_speed_bias = params.wind_speed_bias * params.max_wind_speed_bias
        self.wind_dir_bias   = params.wind_dir_bias * params.max_wind_dir_bias
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
            self.orig_grid = copy.deepcopy(self._cell_grid)
            self.orig_dict = copy.deepcopy(self._cell_dict)

    def run(self, fire_estimate: StateEstimate=None, visualize=False) -> PredictionOutput:
        # Catch up time and weather states with the fire sim
        # (works in both main process and worker process)
        self._catch_up_with_fire()

        # Set fire state variables
        self._set_states(fire_estimate)

        self.spread = {}
        self.flame_len_m = {}
        self.fli_kw_m = {}
        self.ros_ms = {}
        self.spread_dir = {}
        self.crown_fire = {}
        self.hold_probs = {}
        self.breaches = {}

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
            breaches=self.breaches
        )

        return output
    
    def _set_states(self, state_estimate: StateEstimate = None):
        # Reset all data structures to the original
        self._cell_grid = copy.deepcopy(self.orig_grid)
        self._cell_dict = copy.deepcopy(self.orig_dict)
        self._burnt_cells = []
        self._burning_cells = []
        self._updated_cells = {}
        self._scheduled_spot_fires = {}


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

    def _set_prediction_forecast(self, cell: Cell):
        x_wind = max(cell.x_pos - self.wind_xpad, 0)
        y_wind = max(cell.y_pos - self.wind_ypad, 0)

        wind_col = int(np.floor(x_wind/self._wind_res))
        wind_row = int(np.floor(y_wind/self._wind_res))

        if wind_row > self.wind_forecast.shape[1] - 1:
            wind_row = self.wind_forecast.shape[1] - 1

        if wind_col > self.wind_forecast.shape[2] - 1:
            wind_col = self.wind_forecast.shape[2] - 1

        wind_speed = self.wind_forecast[:, wind_row, wind_col, 0]
        wind_dir = self.wind_forecast[:, wind_row, wind_col, 1]
        cell._set_wind_forecast(wind_speed, wind_dir)

    def _init_iteration(self):

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

        if self._curr_time_s >= self._end_time:
            return False

        # Check if weather has changed
        self.weather_changed = self._update_weather()

        # Add any new ignitions to the current set of burning cells
        self._burning_cells.extend(self._new_ignitions)
        # Reset new ignitions
        self._new_ignitions = []

        return True

    def _prediction_loop(self):

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

                if cell.fully_burning:
                    self.set_state_at_cell(cell, CellStates.BURNT)
                else:
                    fires_still_burning.append(cell)

                self._updated_cells[cell.id] = cell

            if self.model_spotting and self.nom_ign_prob > 0:
                self._ignite_spots()

            self.update_control_interface_elements()

            self._burning_cells = list(fires_still_burning)

            self._iters += 1

        self._finished = True

    def _ignite_spots(self):
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
    
    def _calc_nominal_prob(self):
        # Calculate P(I) in Perryman paper
        # Method from "Ignition Probability" (Schroeder 1969)
        Q_ig = 250 + 1116 * self.dead_mf
        Q_ig_cal = BTU_lb_to_cal_g(Q_ig)

        x = (400 - Q_ig_cal) / 10
        p_i = (0.000048 * x ** 4.3)/50

        return p_i

    def _catch_up_with_fire(self):
        """
        Synchronize predictor time state with parent fire simulation.

        In worker processes (self.fire is None), uses serialized state instead.
        """
        if self.fire is not None:
            # In main process: get current state from fire
            self._curr_time_s = self.fire._curr_time_s
            self.start_time_s = self._curr_time_s
            self.last_viz_update = self._curr_time_s
            self._end_time = (self.time_horizon_hr * 3600) + self.start_time_s
        else:
            # In worker process: use serialized state
            self.start_time_s = self._curr_time_s  # Already set in __setstate__
            self.last_viz_update = self._curr_time_s
            self._end_time = (self.time_horizon_hr * 3600) + self.start_time_s

        # Create a perturbed wind forecast (works in both main and worker)
        self._predict_wind()

    def _predict_wind(self):
        """
        Generate perturbed wind forecast for prediction.

        In worker processes (self.fire is None), uses serialized weather stream.
        """
        if self.fire is not None:
            new_weather_stream = copy.deepcopy(self.fire._weather_stream)
            curr_idx = self.fire._curr_weather_idx
        else:
            # In worker: use serialized weather stream
            new_weather_stream = copy.deepcopy(self._weather_stream)
            curr_idx = self._curr_weather_idx

        self.weather_t_step = new_weather_stream.time_step * 60

        num_indices = int(np.ceil((self.time_horizon_hr * 3600) / self.weather_t_step))

        self._curr_weather_idx = 0
        if self.fire is not None:
            self._last_weather_update = self.fire._last_weather_update
        # else: _last_weather_update already set in __setstate__

        end_idx = num_indices + curr_idx

        speed_error = 0
        dir_error = 0

        new_stream = []

        for entry in new_weather_stream.stream[curr_idx:end_idx + 1]:
            
            new_entry = copy.deepcopy(entry)

            new_entry.wind_speed += speed_error + self.wind_speed_bias
            new_entry.wind_speed = np.max([0, new_entry.wind_speed])
            new_entry.wind_dir_deg += dir_error + self.wind_dir_bias
            new_entry.wind_dir_deg = new_entry.wind_dir_deg % 360

            new_stream.append(new_entry)

            speed_error = self.beta * speed_error + np.random.normal(0, self.wnd_spd_std)
            dir_error = self.beta * dir_error + np.random.normal(0, self.wnd_dir_std)

        new_weather_stream.stream = new_stream

        # Get map params from fire or from serialized sim_params
        map_params = self.fire._sim_params.map_params if self.fire is not None else self._sim_params.map_params

        self.wind_forecast = run_windninja(new_weather_stream, map_params)
        self.flipud_forecast = np.empty(self.wind_forecast.shape)

        for layer in range(self.wind_forecast.shape[0]):
            self.flipud_forecast[layer] = np.flipud(self.wind_forecast[layer])

        self.wind_forecast = self.flipud_forecast

        # Get mesh resolution from fire or from serialized sim_params
        sim_params = self.fire._sim_params if self.fire is not None else self._sim_params
        self._wind_res = sim_params.weather_input.mesh_resolution
        self._weather_stream = new_weather_stream

        self.wind_xpad, self.wind_ypad = self.calc_wind_padding(self.wind_forecast)

    def prepare_for_serialization(self):
        """
        Prepare predictor for parallel execution by extracting serializable data.

        Must be called once before pickling the predictor. Captures the current
        state of the parent FireSim and stores it in a serializable format.

        This method should be called in the main process before spawning workers.

        Raises:
            RuntimeError: If called without a fire reference
        """
        if self.fire is None:
            raise RuntimeError("Cannot prepare predictor without fire reference")

        # Extract fire state at prediction start
        fire_state = {
            'curr_time_s': self.fire._curr_time_s,
            'curr_weather_idx': self.fire._curr_weather_idx,
            'last_weather_update': self.fire._last_weather_update,
            'burning_cell_polygons': UtilFuncs.get_cell_polygons(self.fire._burning_cells),
            'burnt_cell_polygons': (UtilFuncs.get_cell_polygons(self.fire._burnt_cells)
                                   if self.fire._burnt_cells else None),
        }

        # Deep copy simulation parameters
        sim_params_copy = copy.deepcopy(self.fire._sim_params)
        weather_stream_copy = copy.deepcopy(self.fire._weather_stream)

        # Store all serializable data
        self._serialization_data = {
            'sim_params': sim_params_copy,
            'predictor_params': copy.deepcopy(self._params),
            'fire_state': fire_state,
            'weather_stream': weather_stream_copy,

            # Predictor-specific attributes
            'time_horizon_hr': self.time_horizon_hr,
            'wind_uncertainty_factor': self.wind_uncertainty_factor,
            'wind_speed_bias': self.wind_speed_bias,
            'wind_dir_bias': self.wind_dir_bias,
            'ros_bias_factor': self.ros_bias_factor,
            'beta': self.beta,
            'wnd_spd_std': self.wnd_spd_std,
            'wnd_dir_std': self.wnd_dir_std,
            'dead_mf': self.dead_mf,
            'live_mf': self.live_mf,
            'nom_ign_prob': self.nom_ign_prob,

            # Wind and elevation data
            'wind_forecast': self.wind_forecast,
            'flipud_forecast': self.flipud_forecast,
            'wind_xpad': self.wind_xpad,
            'wind_ypad': self.wind_ypad,
            'coarse_elevation': self.coarse_elevation,
        }

    def __getstate__(self):
        """
        Serialize predictor for parallel execution.

        Returns only the essential data needed to reconstruct the predictor
        in a worker process. Excludes non-serializable components like the
        parent FireSim reference, visualizer, and logger.

        Returns:
            dict: Minimal state dictionary for serialization

        Raises:
            RuntimeError: If prepare_for_serialization() was not called first
        """
        if self._serialization_data is None:
            raise RuntimeError(
                "Must call prepare_for_serialization() before pickling. "
                "This ensures all necessary state is captured from the parent FireSim."
            )

        # Return minimal state
        state = {
            'serialization_data': self._serialization_data,
            'orig_grid': self.orig_grid,  # Template cells (pre-built)
            'orig_dict': self.orig_dict,  # Template cells (pre-built)
            'c_size': self.c_size,
        }

        return state

    def __setstate__(self, state):
        """
        Reconstruct predictor in worker process WITHOUT calling BaseFireSim.__init__.

        This method manually restores all attributes that BaseFireSim.__init__()
        would have set, but WITHOUT the expensive cell creation loop.

        The key optimization: use pre-built cell templates (orig_grid, orig_dict)
        instead of reconstructing cells from map data.

        Args:
            state (dict): State dictionary from __getstate__
        """
        import numpy as np
        from embrs.models.perryman_spot import PerrymanSpotting
        from embrs.models.fuel_models import Anderson13, ScottBurgan40

        # Extract serialization data
        data = state['serialization_data']
        sim_params = data['sim_params']

        # =====================================================================
        # Phase 1: Restore FirePredictor-specific attributes
        # =====================================================================
        self.fire = None  # No parent fire in worker
        self.c_size = state['c_size']
        self._params = data['predictor_params']
        self._serialization_data = data

        self.time_horizon_hr = data['time_horizon_hr']
        self.wind_uncertainty_factor = data['wind_uncertainty_factor']
        self.wind_speed_bias = data['wind_speed_bias']
        self.wind_dir_bias = data['wind_dir_bias']
        self.ros_bias_factor = data['ros_bias_factor']
        self.beta = data['beta']
        self.wnd_spd_std = data['wnd_spd_std']
        self.wnd_dir_std = data['wnd_dir_std']
        self.dead_mf = data['dead_mf']
        self.live_mf = data['live_mf']
        self.nom_ign_prob = data['nom_ign_prob']

        # =====================================================================
        # Phase 2: Restore BaseFireSim attributes (manually, without __init__)
        # =====================================================================

        # From BaseFireSim.__init__ lines 45-88
        self.display_frequency = 300
        self._sim_params = sim_params
        self.burnout_thresh = 0.01
        self.sim_start_w_idx = 0
        self._curr_weather_idx = data['fire_state']['curr_weather_idx']
        self._last_weather_update = data['fire_state']['last_weather_update']
        self.weather_changed = True
        self._curr_time_s = data['fire_state']['curr_time_s']
        self._iters = 0
        self.logger = None  # No logger in worker
        self._visualizer = None  # No visualizer in worker
        self._finished = False

        # Empty containers (will be populated by _set_states)
        self._updated_cells = {}
        self._cell_dict = {}
        self._long_term_retardants = set()
        self._active_water_drops = []
        self._burning_cells = []
        self._new_ignitions = []
        self._burnt_cells = set()
        self._frontier = set()
        self._fire_break_cells = []
        self._active_firelines = {}
        self._new_fire_break_cache = []
        self.starting_ignitions = set()
        self._urban_cells = []
        self._scheduled_spot_fires = {}

        # From _parse_sim_params (lines 252-353)
        map_params = sim_params.map_params
        self._cell_size = sim_params.cell_size
        self._sim_duration = sim_params.duration_s
        self._time_step = sim_params.t_step_s
        self._init_mf = sim_params.init_mf
        self._fuel_moisture_map = getattr(sim_params, 'fuel_moisture_map', {})
        self._fms_has_live = getattr(sim_params, 'fms_has_live', False)
        self._init_live_h_mf = getattr(sim_params, 'live_h_mf', 0.0)
        self._init_live_w_mf = getattr(sim_params, 'live_w_mf', 0.0)
        self._size = map_params.size()
        self._shape = map_params.shape(self._cell_size)
        self._roads = map_params.roads
        self.coarse_elevation = data['coarse_elevation']

        # Fuel class selection
        fbfm_type = map_params.fbfm_type
        if fbfm_type == "Anderson":
            self.FuelClass = Anderson13
        elif fbfm_type == "ScottBurgan":
            self.FuelClass = ScottBurgan40
        else:
            raise ValueError(f"FBFM Type {fbfm_type} not supported")

        # Map data (from lcp_data, but already in sim_params)
        lcp_data = map_params.lcp_data
        self._elevation_map = np.flipud(lcp_data.elevation_map)
        self._slope_map = np.flipud(lcp_data.slope_map)
        self._aspect_map = np.flipud(lcp_data.aspect_map)
        self._fuel_map = np.flipud(lcp_data.fuel_map)
        self._cc_map = np.flipud(lcp_data.canopy_cover_map)
        self._ch_map = np.flipud(lcp_data.canopy_height_map)
        self._cbh_map = np.flipud(lcp_data.canopy_base_height_map)
        self._cbd_map = np.flipud(lcp_data.canopy_bulk_density_map)
        self._data_res = lcp_data.resolution

        # Scenario data
        scenario = map_params.scenario_data
        self._fire_breaks = list(zip(scenario.fire_breaks, scenario.break_widths, scenario.break_ids))
        self.fire_break_dict = {
            id: (fire_break, break_width)
            for fire_break, break_width, id in self._fire_breaks
        }
        self._initial_ignition = scenario.initial_ign

        # Datetime and orientation
        self._start_datetime = sim_params.weather_input.start_datetime
        self._north_dir_deg = map_params.geo_info.north_angle_deg

        # Wind forecast (already computed, just restore)
        self.wind_forecast = data['wind_forecast']
        self.flipud_forecast = data['flipud_forecast']
        self._wind_res = sim_params.weather_input.mesh_resolution
        self.wind_xpad = data['wind_xpad']
        self.wind_ypad = data['wind_ypad']

        # Weather stream
        self._weather_stream = data['weather_stream']
        self.weather_t_step = self._weather_stream.time_step * 60

        # Spotting parameters
        self.model_spotting = sim_params.model_spotting
        self._spot_ign_prob = 0.0
        if self.model_spotting:
            self._canopy_species = sim_params.canopy_species
            self._dbh_cm = sim_params.dbh_cm
            self._spot_ign_prob = sim_params.spot_ign_prob
            self._min_spot_distance = sim_params.min_spot_dist
            self._spot_delay_s = sim_params.spot_delay_s

        # Moisture (prediction model specific)
        self.fmc = 100  # Prediction model default

        # =================================================================ƒ===
        # Phase 3: Restore cell templates (CRITICAL - uses pre-built cells)
        # =====================================================================

        # Use the serialized templates instead of reconstructing
        self.orig_grid = state['orig_grid']
        self.orig_dict = state['orig_dict']

        # Initialize cell_grid to the template shape
        self._cell_grid = np.empty(self._shape, dtype=object)
        self._grid_width = self._cell_grid.shape[1] - 1
        self._grid_height = self._cell_grid.shape[0] - 1

        # Fix weak references in cells (point to self instead of original fire)
        for cell in self.orig_dict.values():
            cell.set_parent(self)

        # =====================================================================
        # Phase 4: Rebuild lightweight components
        # =====================================================================

        size = map_params.size()

        # Rebuild spotting model (PerrymanSpotting for prediction)
        if self.model_spotting:
            self.embers = PerrymanSpotting(self._spot_delay_s, size)

        # Note: _set_states() will be called by run() to deep copy the cells
        # and set up the initial burning/burnt regions

    def run_ensemble(
        self,
        state_estimates: List[StateEstimate],
        visualize: bool = False,
        num_workers: Optional[int] = None,
        random_seeds: Optional[List[int]] = None,
        return_individual: bool = False
    ):
        """
        Run ensemble predictions using multiple initial state estimates.

        Executes predictions in parallel, each starting from a different
        StateEstimate. Results are aggregated into probabilistic predictions.

        This method uses custom serialization to avoid reconstructing FireSim
        for each ensemble member, providing ~25% speedup over naive approaches.

        Args:
            state_estimates: List of StateEstimate objects representing
                            different possible initial fire states
            visualize: If True, visualize aggregated burn probability
            num_workers: Number of parallel workers (default: cpu_count)
            random_seeds: Optional list of random seeds for reproducibility
            return_individual: If True, include individual predictions in output

        Returns:
            EnsemblePredictionOutput with aggregated predictions

        Raises:
            ValueError: If state_estimates is empty or seeds length mismatch
            RuntimeError: If more than 50% of members fail
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        from typing import List, Optional

        # Validation
        if not state_estimates:
            raise ValueError("state_estimates cannot be empty")

        if random_seeds is not None and len(random_seeds) != len(state_estimates):
            raise ValueError(
                f"random_seeds length ({len(random_seeds)}) must match "
                f"state_estimates length ({len(state_estimates)})"
            )

        n_ensemble = len(state_estimates)
        num_workers = num_workers or mp.cpu_count()

        print(f"Running ensemble prediction:")
        print(f"  - {n_ensemble} ensemble members")
        print(f"  - {num_workers} parallel workers")

        # CRITICAL: Prepare for serialization
        print("Preparing predictor for serialization...")
        self.prepare_for_serialization()

        # Run predictions in parallel
        predictions = []
        failed_count = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all jobs
            futures = {}
            for i, state_est in enumerate(state_estimates):
                seed = random_seeds[i] if random_seeds else None

                # Submit (predictor will be pickled via __getstate__)
                future = executor.submit(
                    _run_ensemble_member_worker,
                    self,
                    state_est,
                    seed
                )
                futures[future] = i  # Track member index

            # Collect results with progress bar
            for future in tqdm(as_completed(futures),
                              total=n_ensemble,
                              desc="Ensemble predictions",
                              unit="member"):
                member_idx = futures[future]
                try:
                    result = future.result()
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

        # Optionally include individual predictions
        if return_individual:
            ensemble_output.individual_predictions = predictions

        # Optionally visualize
        if visualize:
            print("Visualization not yet implemented")
            # self._visualize_ensemble(ensemble_output)

        return ensemble_output


# ============================================================================
# Module-level functions for parallel execution
# ============================================================================

def _run_ensemble_member_worker(
    predictor: 'FirePredictor',
    state_estimate: StateEstimate,
    seed: Optional[int] = None
) -> PredictionOutput:
    """
    Worker function for parallel ensemble prediction.

    Receives a deserialized FirePredictor (via __setstate__) and runs
    a single prediction. The predictor has been reconstructed in this
    worker process without the original FireSim reference.

    Args:
        predictor: Deserialized FirePredictor instance
        state_estimate: Initial fire state for this ensemble member
        seed: Random seed for reproducibility

    Returns:
        PredictionOutput for this ensemble member

    Raises:
        Exception: Any errors during prediction (will be caught by executor)
    """
    import numpy as np

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Run prediction (visualize=False always in workers)
    try:
        output = predictor.run(fire_estimate=state_estimate, visualize=False)
        return output
    except Exception as e:
        # Log error and re-raise (executor will handle)
        import traceback
        print(f"ERROR in ensemble member: {e}")
        print(traceback.format_exc())
        raise


def _aggregate_ensemble_predictions(predictions: List[PredictionOutput]):
    """
    Aggregate multiple prediction outputs into ensemble statistics.

    Args:
        predictions: List of PredictionOutput from ensemble members

    Returns:
        EnsemblePredictionOutput with probabilistic predictions
    """
    from embrs.utilities.data_classes import EnsemblePredictionOutput, CellStatistics

    n_ensemble = len(predictions)

    # 1. Build burn probability map
    burn_counts = {}  # {time_s: {(x,y): count}}
    for pred in predictions:
        for time_s, locations in pred.spread.items():
            if time_s not in burn_counts:
                burn_counts[time_s] = {}
            for loc in locations:
                burn_counts[time_s][loc] = burn_counts[time_s].get(loc, 0) + 1

    burn_probability = {
        time_s: {loc: count / n_ensemble for loc, count in counts.items()}
        for time_s, counts in burn_counts.items()
    }

    # 2. Collect all unique cell locations that burned in any prediction
    all_burned_cells = set()
    for pred in predictions:
        all_burned_cells.update(pred.flame_len_m.keys())

    # 3. Aggregate statistics for each cell
    flame_stats = {}
    ros_stats = {}
    fli_stats = {}

    for cell_loc in all_burned_cells:
        # Collect values from all predictions where this cell burned
        flame_values = [p.flame_len_m.get(cell_loc) for p in predictions
                       if cell_loc in p.flame_len_m]
        ros_values = [p.ros_ms.get(cell_loc) for p in predictions
                     if cell_loc in p.ros_ms]
        fli_values = [p.fli_kw_m.get(cell_loc) for p in predictions
                     if cell_loc in p.fli_kw_m]

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

    # 4. Crown fire frequency
    crown_frequency = {}
    for cell_loc in all_burned_cells:
        crown_count = sum(1 for p in predictions if cell_loc in p.crown_fire)
        crown_frequency[cell_loc] = crown_count / n_ensemble

    # 5. Spread direction (using circular statistics)
    spread_dir_stats = {}
    for cell_loc in all_burned_cells:
        dirs = [p.spread_dir.get(cell_loc) for p in predictions
                if cell_loc in p.spread_dir]
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

    # 6. Fireline statistics
    all_fireline_cells = set()
    for pred in predictions:
        all_fireline_cells.update(pred.hold_probs.keys())

    hold_prob_stats = {}
    breach_frequency = {}

    for cell_loc in all_fireline_cells:
        hold_probs = [p.hold_probs.get(cell_loc) for p in predictions
                     if cell_loc in p.hold_probs]
        breaches = [p.breaches.get(cell_loc) for p in predictions
                   if cell_loc in p.breaches]

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
        breach_frequency=breach_frequency
    )
