
from embrs.base_classes.base_fire import BaseFireSim
from embrs.fire_simulator.fire import FireSim
from embrs.fire_simulator.cell import Cell
from embrs.fire_simulator.visualizer import Visualizer
from embrs.utilities.data_classes import PredictorParams, CellData
from embrs.utilities.fire_util import UtilFuncs, CellStates
from embrs.utilities.fuel_models import Anderson13
from embrs.utilities.rothermel import *
from embrs.utilities.crown_model import *
from embrs.utilities.wind_forecast import run_windninja

import copy
import numpy as np

class FirePredictor(BaseFireSim):
    def __init__(self, params: PredictorParams, fire: FireSim):

        # Live reference to the fire sim
        self.fire = fire
        self.c_size = -1

        self.beta_s = 0.5
        self.beta_d = 0.5

        self.wnd_spd_uncertainty = 2
        self.wnd_dir_uncertainty = 5

        self.set_params(params)

    def set_params(self, params: PredictorParams):

        generate_cell_grid = False

        # How long the prediction will run for
        self.time_horizon_hr = params.time_horizon_hr

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

        # Set the currently burning cells as the initial ignition region
        burning_cells = [cell for cell, _ in self.fire._burning_cells]

        # Get the merged polygon representing burning cells
        sim_params.map_params.scenario_data.initial_ign = UtilFuncs.get_cell_polygons(burning_cells)
        burnt_region = UtilFuncs.get_cell_polygons(self.fire._burnt_cells)

        if generate_cell_grid:
            super().__init__(sim_params, burnt_region=burnt_region)

    def run(self, visualize = False):
        # Catch states of predcictor cells up with the fire sim
        self._catch_up_with_fire()

        self.output = {}

        viz = None
        if visualize:
            viz = Visualizer(self)
        
        # Perform the prediction
        self._prediction_loop(viz)

        return self.output

    def _init_iteration(self):

        self._curr_time_s = (self._iters * self._time_step) + self.start_time_s

        if self._iters == 0:

            self.weather_changed = True
            self._new_ignitions = self.starting_ignitions

            for cell, loc in self._new_ignitions:
                cell.directions, cell.distances, cell.end_pts = UtilFuncs.get_ign_parameters(loc, self.cell_size)
                cell._set_state(CellStates.FIRE)

                r_list, I_list = calc_propagation_in_cell(cell) # r in m/s, I in BTU/ft/min
                cell.r_ss = r_list
                cell.I_ss = I_list
                cell.has_steady_state = True

                # Don't model fire acceleration in prediction model
                cell.r_t = r_list
                cell.I_t = I_list

                self._updated_cells[cell.id] = cell

                wind_col = int(np.floor(cell.x_pos/self._wind_res))
                wind_row = int(np.floor(cell.y_pos/self._wind_res))

                if wind_row > self.wind_forecast.shape[1] - 1:
                    wind_row = self.wind_forecast.shape[1] - 1

                if wind_col > self.wind_forecast.shape[2] - 1:
                    wind_col = self.wind_forecast.shape[2] - 1

                wind_speed = self.wind_forecast[:, wind_row, wind_col, 0]
                wind_dir = self.wind_forecast[:, wind_row, wind_col, 1]
                cell._set_wind_forecast(wind_speed, wind_dir)

        else:
            for cell, loc in self._new_ignitions:
                r_list, I_list = calc_propagation_in_cell(cell) # r in m/s, I in BTU/ft/min

                cell.r_ss = r_list
                cell.I_ss = I_list

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
                    if cell._crown_status != CrownStatus.NONE and self._spot_ign_prob > 0:
                        pass # TODO: implement simplified probabilistic spotting model

                # Don't model fire acceleration in prediction model
                cell.r_t = cell.r_ss
                cell.I_t = cell.I_ss

                if self.output.get(self._curr_time_s) is None:
                    self.output[self._curr_time_s] = [(cell.x_pos, cell.y_pos)]

                else:
                    self.output[self._curr_time_s].append((cell.x_pos, cell.y_pos))

        # Check if weather has changed
        self.weather_changed = self._update_weather()

        # Add any new ignitions to the current set of burning cells
        self._burning_cells.extend(self._new_ignitions)
        # Reset new ignitions
        self._new_ignitions = []

    def _prediction_loop(self, viz):

        self._iters = 0

        end_time = (self.time_horizon_hr * 60) + self.start_time_s

        while self.curr_time_s < end_time:
            self._init_iteration()

            for cell, loc in self._burning_cells:

                if self.weather_changed or not cell.has_steady_state:
                    
                    # TODO: Need weather for prediction model
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
                pass # TODO: Add this model

            self._iters += 1

            if viz is not None:
                time_since_last_update = self._curr_time_s - self.last_viz_update

                if time_since_last_update >= 300:
                    viz.update_grid(self)
                    self.last_viz_update = self._curr_time_s

    def _catch_up_with_fire(self):
        # Set current time to fire sim time
        self._curr_time_s = self.fire._curr_time_s

        self.start_time_s = self._curr_time_s
        self.last_viz_update = self._curr_time_s

        # Set the burnt cells based on fire state
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

        for entry in new_weather_stream.stream[curr_idx:end_idx]:
            
            new_entry = copy.deepcopy(entry)

            new_entry.wind_speed += speed_error
            new_entry.wind_dir_deg += dir_error
            new_entry.wind_dir_deg = new_entry.wind_dir_deg % 360

            new_stream.append(new_entry)

            speed_error = self.beta_s * speed_error + np.random.normal(0, self.wnd_spd_uncertainty)
            dir_error = self.beta_d * dir_error + np.random.normal(0, self.wnd_dir_uncertainty)


        new_weather_stream.stream = new_stream
        
        self.wind_forecast = run_windninja(new_weather_stream, self.fire._sim_params.map_params)
        self.flipud_forecast = np.empty(self.wind_forecast.shape)

        for layer in range(self.wind_forecast.shape[0]):
            self.flipud_forecast[layer] = np.flipud(self.wind_forecast[layer])
            
        self.wind_forecast = self.flipud_forecast
        self._wind_res = self.fire._sim_params.weather_input.mesh_resolution
        self._weather_stream = new_weather_stream

