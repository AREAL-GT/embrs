from embrs.base_classes.base_fire import BaseFireSim
from embrs.fire_simulator.fire import FireSim
from embrs.utilities.data_classes import PredictorParams
from embrs.utilities.fire_util import UtilFuncs, CellStates
from embrs.models.rothermel import *
from embrs.models.crown_model import *
from embrs.models.wind_forecast import run_windninja

import copy
import numpy as np

class FirePredictor(BaseFireSim):
    def __init__(self, params: PredictorParams, fire: FireSim):

        # Live reference to the fire sim
        self.fire = fire
        self.c_size = -1

        self.set_params(params)
        self.nom_ign_prob = self._calc_nominal_prob()

    def set_params(self, params: PredictorParams):

        generate_cell_grid = False

        # How long the prediction will run for
        self.time_horizon_hr = params.time_horizon_hr

        # Uncertainty parameters
        self.wind_bias_factor = params.wind_bias_factor # [-1 , 1], systematic error
        self.wind_uncertainty_factor = params.wind_uncertainty_factor # [0, 1], autoregression noise

        # Compute constant bias terms
        self.wind_speed_bias = self.wind_bias_factor * params.max_wind_speed_bias
        self.wind_dir_bias   = self.wind_bias_factor * params.max_wind_dir_bias

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
        sim_params.init_mf = params.dead_mf
        sim_params.spot_delay_s = params.spot_delay_s
        sim_params.model_spotting = params.model_spotting

        # Set the currently burning cells as the initial ignition region
        burning_cells = [cell for cell, _ in self.fire._burning_cells]

        # Get the merged polygon representing burning cells
        sim_params.map_params.scenario_data.initial_ign = UtilFuncs.get_cell_polygons(burning_cells)
        burnt_region = UtilFuncs.get_cell_polygons(self.fire._burnt_cells)

        if generate_cell_grid:
            super().__init__(sim_params, burnt_region=burnt_region)
            self.orig_grid = copy.deepcopy(self._cell_grid)
            self.orig_dict = copy.deepcopy(self._cell_dict)

    def run(self, visualize = False):
        # Catch states of predictor cells up with the fire sim
        self._catch_up_with_fire()

        self.output = {}

        # Perform the prediction
        self._prediction_loop()

        if visualize:
            self.fire.visualize_prediction(self.output)

        return self.output
    
    def _set_prediction_forecast(self, cell):
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
            self._new_ignitions = self.starting_ignitions

            for cell, loc in self._new_ignitions:
                self._set_prediction_forecast(cell)
                cell.directions, cell.distances, end_pts = UtilFuncs.get_ign_parameters(loc, self.cell_size)
                cell.end_pts = copy.deepcopy(end_pts)
                cell._set_state(CellStates.FIRE)

                surface_fire(cell)
                crown_fire(cell, self.fmc)
                cell.has_steady_state = True

                # Don't model fire acceleration in prediction model
                cell.r_t = cell.r_ss
                cell.I_t = cell.I_ss

                self._updated_cells[cell.id] = cell


        else:
            for cell, loc in self._new_ignitions:
                self._set_prediction_forecast(cell)
                surface_fire(cell)
                crown_fire(cell, self.fmc)

                if cell._break_width > 0:
                    # Determine if fire will breach fireline contained within cell
                    flame_len_ft = calc_flame_len(np.max(cell.I_ss))
                    flame_len_m = ft_to_m(flame_len_ft)
                    hold_prob = cell.calc_hold_prob(flame_len_m)
                    rand = np.random.random()
                    cell.breached = rand > hold_prob

                else:
                    cell.breached = True

                cell.has_steady_state = True

                if self.model_spotting:
                    if not cell.lofted and cell._crown_status != CrownStatus.NONE and self._spot_ign_prob > 0:
                        self.embers.loft(cell)

                # Don't model fire acceleration in prediction model
                cell.r_t = cell.r_ss
                cell.I_t = cell.I_ss

                if self.output.get(self._curr_time_s) is None:
                    self.output[self._curr_time_s] = [(cell.x_pos, cell.y_pos)]

                else:
                    self.output[self._curr_time_s].append((cell.x_pos, cell.y_pos))

        
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
            for cell, loc in self._burning_cells:

                if self.weather_changed or not cell.has_steady_state:
                    cell._update_weather(self._curr_weather_idx, self._weather_stream, True)

                    self.update_steady_state(cell)

                    cell.r_t = cell.r_ss
                    cell.I_t = cell.I_ss

                self.propagate_fire(cell)

                self.remove_neighbors(cell)

                if cell.fully_burning:
                    self.set_state_at_cell(cell, CellStates.BURNT)

                self.updated_cells[cell.id] = cell

            if self.model_spotting and self._spot_ign_prob > 0:
                self._ignite_spots()

            self.update_control_interface_elements()
            
            self._iters += 1

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
                        self._new_ignitions.append((spot, 0))
                        spot.directions, spot.distances, end_pts = UtilFuncs.get_ign_parameters(0, spot.cell_size)
                        spot.end_pts = copy.deepcopy(end_pts)
                        spot._set_state(CellStates.FIRE)
                        self.updated_cells[spot.id] = spot

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
        # Reset all data structures to the original
        self._cell_grid = copy.deepcopy(self.orig_grid)
        self._cell_dict = copy.deepcopy(self.orig_dict)
        self._burnt_cells = []
        self._burning_cells = []
        self._updated_cells = {}
        self._scheduled_spot_fires = {}

        # Set current time to fire sim time
        self._curr_time_s = self.fire._curr_time_s
        self.start_time_s = self._curr_time_s
        self.last_viz_update = self._curr_time_s
        self._end_time = (self.time_horizon_hr * 3600) + self.start_time_s

        # Set the burnt cells based on fire state
        if self.fire._burnt_cells:
            burnt_region = UtilFuncs.get_cell_polygons(self.fire._burnt_cells)
            self._set_state_from_geometries(burnt_region, CellStates.BURNT)

        # Set the burning cells based on fire state
        burning_cells = [cell for cell, _ in self.fire._burning_cells]
        burning_region = UtilFuncs.get_cell_polygons(burning_cells)
        self._set_state_from_geometries(burning_region, CellStates.FIRE)

        # Create a erroneous wind forecast 
        self._predict_wind()

    def _predict_wind(self):
        new_weather_stream = copy.deepcopy(self.fire._weather_stream)
        self.weather_t_step = new_weather_stream.time_step * 60

        num_indices = int(np.ceil((self.time_horizon_hr * 3600) / self.weather_t_step))
        curr_idx = self.fire._curr_weather_idx

        self._curr_weather_idx = 0
        self._last_weather_update = self.fire._last_weather_update

        end_idx = num_indices + curr_idx

        speed_error = 0
        dir_error = 0

        new_stream = []

        for entry in new_weather_stream.stream[curr_idx:end_idx + 1]:
            
            new_entry = copy.deepcopy(entry)

            new_entry.wind_speed += speed_error + self.wind_speed_bias
            new_entry.wind_dir_deg += dir_error + self.wind_dir_bias
            new_entry.wind_dir_deg = new_entry.wind_dir_deg % 360

            new_stream.append(new_entry)

            speed_error = self.beta * speed_error + np.random.normal(0, self.wnd_spd_std)
            dir_error = self.beta * dir_error + np.random.normal(0, self.wnd_dir_std)

        new_weather_stream.stream = new_stream
        
        self.wind_forecast = run_windninja(new_weather_stream, self.fire._sim_params.map_params)
        self.flipud_forecast = np.empty(self.wind_forecast.shape)

        for layer in range(self.wind_forecast.shape[0]):
            self.flipud_forecast[layer] = np.flipud(self.wind_forecast[layer])
            
        self.wind_forecast = self.flipud_forecast
        self._wind_res = self.fire._sim_params.weather_input.mesh_resolution
        self._weather_stream = new_weather_stream

        self.wind_xpad, self.wind_ypad = self.calc_wind_padding(self.wind_forecast)
