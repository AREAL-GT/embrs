"""Base class implementation of fire simulation model.

TODO: fill in 

.. autoclass:: BaseFireSim
    :members:
"""

from shapely.geometry import Point, Polygon, LineString
from typing import Tuple, Union, List
from tqdm import tqdm
import numpy as np
import pickle
import copy
import os

from embrs.utilities.fire_util import HexGridMath as hex, UtilFuncs, HexGridMath
from embrs.models.wind_forecast import run_windninja, create_uniform_wind
from embrs.utilities.logger_schemas import ActionsEntry, PredictionEntry
from embrs.models.fuel_models import Anderson13, ScottBurgan40
from embrs.utilities.fire_util import CellStates, CrownStatus
from embrs.utilities.data_classes import SimParams, CellData
from embrs.models.perryman_spot import PerrymanSpotting
from embrs.base_classes.agent_base import AgentBase
from embrs.models.crown_model import crown_fire
from embrs.models.weather import WeatherStream
from embrs.fire_simulator.cell import Cell
from embrs.models.embers import Embers
from embrs.models.burnup import Burnup
from embrs.models.rothermel import *

class BaseFireSim:
    """_summary_
    """
    def __init__(self, sim_params: SimParams, burnt_region: list = None):
        # Check if current instance is a prediction model
        prediction = self.is_prediction()
        
        if prediction:
            print("Initializing prediction model backing array...")

        else:
            print("Initializing fire sim backing array...")

        # Constant parameters
        self.display_frequency = 300
        self._sim_params = sim_params
        self.burnout_thresh = 0.01

        # Store sim input values in class variables
        self._parse_sim_params(sim_params)
        
        # Variables to keep track of sim progress
        self._curr_time_s = 0
        self._iters = 0

        # Variables to keep track of current wind conditions
        self._curr_weather_idx = 0
        self._last_weather_update = 0
        self.weather_changed = True

        # Variables to store logger and visualizer object
        self.logger = None
        self._visualizer = None

        # Track whether sim is finished or not
        self._finished = False

        # Containers for keeping track of updates to cells 
        self._updated_cells = {}

        # Containers for cells
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

        # Crown fire containers
        self._scheduled_spot_fires = {}

        # Set up backing array
        self._cell_grid = np.empty(self._shape, dtype=Cell)
        self._grid_width = self._cell_grid.shape[1] - 1
        self._grid_height = self._cell_grid.shape[0] - 1

        if not prediction: # Regular FireSim
            if self._fms_has_live:
                live_h_mf = self._init_live_h_mf
                live_w_mf = self._init_live_w_mf
            else:
                # Set live moisture and foliar moisture to weather stream values
                live_h_mf = self._weather_stream.live_h_mf
                live_w_mf = self._weather_stream.live_w_mf
            self.fmc = self._weather_stream.fmc

        else:
            # Prediction model has live_mf as an attribute
            live_h_mf = self.live_mf
            live_w_mf = self.live_mf
            self.fmc = 100

        if self.model_spotting:
            # Limits to pass into spotting models
            limits = (self.x_lim, self.y_lim)
            if not prediction:
                # Spot fire modelling class
                self.embers = Embers(self._spot_ign_prob, self._canopy_species, self._dbh_cm, self._min_spot_distance, limits, self.get_cell_from_xy)

            else:
                self.embers = PerrymanSpotting(self._spot_delay_s, limits)

        # Load Duff loading lookup table from LANDFIRE FCCS
        base_dir = os.path.dirname(__file__)
        duff_path = os.path.join(base_dir, '..', 'utilities', 'duff_loading.pkl')
        duff_path = os.path.normpath(duff_path)

        with open(duff_path, "rb") as file:
            duff_lookup = pickle.load(file)

        # Populate cell_grid with cells
        id = 0
        total_cells = self._shape[0] * self._shape[1]
        with tqdm(total=total_cells, desc="Initializing cells") as pbar:
            for i in range(self._shape[1]):
                for j in range(self._shape[0]):
                    # Initialize cell object
                    new_cell = Cell(id, i, j, self._cell_size)

                    # Set cell's parent reference to the current sim instance
                    new_cell.set_parent(self)

                    # Initialize cell data class
                    cell_data = CellData()

                    cell_x, cell_y = new_cell.x_pos, new_cell.y_pos

                    # Get row and col of data arrays corresponding to cell
                    data_col = int(np.floor(cell_x/self._data_res))
                    data_row = int(np.floor(cell_y/self._data_res))

                    # Ensure data row and col in bounds
                    data_col = min(data_col, sim_params.map_params.lcp_data.cols-1)
                    data_row = min(data_row, sim_params.map_params.lcp_data.rows-1)

                    # Get fuel type
                    fuel_key = self._fuel_map[data_row, data_col]

                    # Set initial moisture values (default)
                    cell_data.init_dead_mf = self._init_mf
                    cell_data.live_h_mf = live_h_mf
                    cell_data.live_w_mf = live_w_mf
                    if self._fuel_moisture_map.get(fuel_key) is not None:
                        mf_vals = self._fuel_moisture_map[fuel_key]
                        cell_data.init_dead_mf = mf_vals[:3]
                        if self._fms_has_live and len(mf_vals) >= 5:
                            cell_data.live_h_mf = mf_vals[3]
                            cell_data.live_w_mf = mf_vals[4]

                    fuel = self.FuelClass(fuel_key, cell_data.live_h_mf)
                    cell_data.fuel_type = fuel

                    # Get cell elevation from elevation map
                    cell_data.elevation = self._elevation_map[data_row, data_col]
                    self.coarse_elevation[j, i] = cell_data.elevation

                    # Get cell aspect from aspect map
                    cell_data.aspect = self._aspect_map[data_row, data_col]

                    # Get cell slope from slope map
                    cell_data.slope_deg = self._slope_map[data_row, data_col]

                    # Get canopy cover from canopy cover map
                    cell_data.canopy_cover = self._cc_map[data_row, data_col]

                    # Get canopy height from canopy height map
                    cell_data.canopy_height = self._ch_map[data_row, data_col]

                    # Get canopy base height from cbh map
                    cell_data.canopy_base_height = self._cbh_map[data_row, data_col]

                    # Get canopy bulk density from cbd map
                    cell_data.canopy_bulk_density = self._cbd_map[data_row, data_col]

                    # Get duff fuel loading from fccs map
                    fccs_id = int(self._fccs_map[data_row, data_col])
                    if not prediction and duff_lookup.get(fccs_id) is not None:
                        cell_data.wdf = duff_lookup[fccs_id] # tons/acre
                    else:
                        # Default to 0 duff loading
                        cell_data.wdf = 0

                    # Get data for cell
                    new_cell._set_cell_data(cell_data)

                    # If the fuel type is urban add it to urban cell list
                    if fuel_key == 91:
                        self._urban_cells.append(new_cell)

                    # Set wind forecast in cell
                    x_wind = max(cell_x - self.wind_xpad, 0)
                    y_wind = max(cell_y - self.wind_ypad, 0)

                    wind_col = int(np.floor(x_wind/self._wind_res))
                    wind_row = int(np.floor(y_wind/self._wind_res))

                    # Account for WindNinja differences in mesh_resolution
                    if wind_row > self.wind_forecast.shape[1] - 1:
                        wind_row = self.wind_forecast.shape[1] - 1

                    if wind_col > self.wind_forecast.shape[2] - 1:
                        wind_col = self.wind_forecast.shape[2] - 1
                    
                    # Collect wind speeds and directions for cell
                    wind_speed = self.wind_forecast[:, wind_row, wind_col, 0]
                    wind_dir = self.wind_forecast[:, wind_row, wind_col, 1]
                    new_cell._set_wind_forecast(wind_speed, wind_dir)

                    # Add cell to the backing array
                    self._cell_grid[j,i] = new_cell
                    self._cell_dict[id] = new_cell

                    # Increment id counter
                    id +=1
                    pbar.update(1)

        # Populate neighbors field for each cell with pointers to each of its neighbors
        self._add_cell_neighbors()

        # Set initial ignitions
        self._set_initial_ignition(self.initial_ignition)
        
        # Set burnt cells
        if burnt_region is not None:
            self._set_initial_burnt_region(burnt_region)
        
        # Overwrite urban cells to their neighbors (road modelling handles fire spread through roads)
        for cell in self._urban_cells:
            self._overwrite_urban_fuel(cell)

        # Apply fire breaks
        self._set_firebreaks()
        
        # Apply Roads 
        self._set_roads()
        
        print("Base initialization complete...")

    def _parse_sim_params(self, sim_params: SimParams):
        """_summary_

        Args:
            sim_params (SimParams): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        # Load general sim params
        self._cell_size = sim_params.cell_size
        self._sim_duration = sim_params.duration_s
        self._time_step = sim_params.t_step_s
        self._init_mf = sim_params.init_mf
        self._fuel_moisture_map = getattr(sim_params, 'fuel_moisture_map', {})
        self._fms_has_live = getattr(sim_params, 'fms_has_live', False)
        self._init_live_h_mf = getattr(sim_params, 'live_h_mf', 0.0)
        self._init_live_w_mf = getattr(sim_params, 'live_w_mf', 0.0)

        # Load map params
        map_params = sim_params.map_params
        self._size = map_params.size()
        self._shape = map_params.shape(self._cell_size)
        self._roads = map_params.roads
        self.coarse_elevation = np.empty(self._shape)

        fbfm_type = map_params.fbfm_type
        if fbfm_type == "Anderson":
            self.FuelClass = Anderson13

        elif fbfm_type == "ScottBurgan":
            self.FuelClass = ScottBurgan40

        else:
            raise ValueError(f"FBFM Type {fbfm_type} not supported")

        # Load DataProductParams for each data product
        lcp_data = map_params.lcp_data

        # Get map for each data product
        self._elevation_map = np.flipud(lcp_data.elevation_map)
        self._slope_map = np.flipud(lcp_data.slope_map)
        self._aspect_map = np.flipud(lcp_data.aspect_map)
        self._fuel_map = np.flipud(lcp_data.fuel_map)
        self._cc_map = np.flipud(lcp_data.canopy_cover_map)
        self._ch_map = np.flipud(lcp_data.canopy_height_map)
        self._cbh_map = np.flipud(lcp_data.canopy_base_height_map)
        self._cbd_map = np.flipud(lcp_data.canopy_bulk_density_map)
        self._fccs_map = np.flipud(lcp_data.fccs_map)

        # Get resolution for data products
        self._data_res = lcp_data.resolution

        # Load scenario specific data
        scenario = map_params.scenario_data
        self._fire_breaks = list(zip(scenario.fire_breaks, scenario.break_widths, scenario.break_ids))
        self.fire_break_dict = {id: (fire_break, break_width) 
                                for fire_break, break_width, id in list(zip(scenario.fire_breaks, scenario.break_widths, scenario.break_ids))}
        self._initial_ignition = scenario.initial_ign

        # Grab starting datetime
        self._start_datetime = sim_params.weather_input.start_datetime

        # Handle regular map
        if not map_params.uniform_map:
            self._uniform_map = False
            self._north_dir_deg = map_params.geo_info.north_angle_deg

            # If loading from lcp file change aspect to uphill direction
            self._aspect_map = (180 + self._aspect_map) % 360 

            if self.is_prediction():
                # Set prediction wind forecast to zeros initially
                self.wind_forecast = np.zeros((1, 1, 1, 2))
                self.flipud_forecast = self.wind_forecast
                self._wind_res = 10e10

                self.wind_xpad = 0
                self.wind_ypad = 0

            # Generate a weather stream
            else:
                self._weather_stream = WeatherStream(
                    sim_params.weather_input, sim_params.map_params.geo_info, use_gsi=not self._fms_has_live
                )
                self.weather_t_step = self._weather_stream.time_step * 60 # convert to seconds
                
                # Get wind data
                self._wind_res = sim_params.weather_input.mesh_resolution
                self.wind_forecast = run_windninja(self._weather_stream, sim_params.map_params)

                self.wind_xpad, self.wind_ypad = self.calc_wind_padding(self.wind_forecast)

                self.flipud_forecast = np.empty(self.wind_forecast.shape)

            # Iterate over each layer (time step or vertical level, depending on the dataset structure)
            for layer in range(self.wind_forecast.shape[0]):
                self.flipud_forecast[layer] = np.flipud(self.wind_forecast[layer])
            
            self.wind_forecast = self.flipud_forecast

        # Handle uniform map
        else:
            self._uniform_map = True

            # Set north to straight up
            self._north_dir_deg = 0
            
            # Check that OpenMeteo option is not selected (if it is throw an error)
            if sim_params.weather_input.input_type == "OpenMeteo":
                raise ValueError(f"Error: If using a uniform map, OpenMeteo can not be used. Must specify a weather file")

            # Create weather stream (just consisting of wind)
            self._weather_stream = WeatherStream(
                sim_params.weather_input, sim_params.map_params.geo_info, use_gsi=not self._fms_has_live
            )
            self.weather_t_step = self._weather_stream.time_step * 60 # convert to seconds

            # Create a uniform wind forecast
            self._wind_res = 10e10
            self.wind_forecast = create_uniform_wind(self._weather_stream)

            self.wind_xpad, self.wind_ypad = self.calc_wind_padding(self.wind_forecast)

        self.model_spotting = sim_params.model_spotting
        self._spot_ign_prob = 0.0

        if self.model_spotting:
            self._canopy_species = sim_params.canopy_species
            self._dbh_cm = sim_params.dbh_cm
            self._spot_ign_prob = sim_params.spot_ign_prob
            self._min_spot_distance = sim_params.min_spot_dist
            self._spot_delay_s = sim_params.spot_delay_s

    def _add_cell_neighbors(self):
        """_summary_
        """
        for j in range(self._shape[1]):
            for i in range(self._shape[0]):
                cell = self._cell_grid[i][j]

                neighbors = {}
                if cell.row % 2 == 0:
                    neighborhood = hex.even_neighborhood
                else:
                    neighborhood = hex.odd_neighborhood

                for dx, dy in neighborhood:
                    row_n = int(cell.row + dy)
                    col_n = int(cell.col + dx)

                    if self._grid_height >= row_n >= 0 and self._grid_width >= col_n >= 0:
                        neighbor_id = self._cell_grid[row_n, col_n].id
                        neighbors[neighbor_id] = (dx, dy)

                cell._neighbors = neighbors
                cell._burnable_neighbors = dict(neighbors)

    def remove_neighbors(self, cell: Cell):
        """_summary_

        Args:
            cell (Cell): _description_
        """
        # Remove any neighbors which are no longer burnable
        neighbors_to_rem = []
        for n_id in cell.burnable_neighbors:
            neighbor = self._cell_dict[n_id]

            if neighbor.state != CellStates.FUEL:
                neighbors_to_rem.append(n_id)

        if neighbors_to_rem:
            for n_id in neighbors_to_rem:
                del cell.burnable_neighbors[n_id]

    def update_steady_state(self, cell: Cell):
        """_summary_

        Args:
            cell (Cell): _description_
        """
        # Checks if fire in cell meets threshold for crown fire, calls calc_propagation_in_cell using the crown ROS if active crown fire
        crown_fire(cell, self.fmc)

        if cell._crown_status != CrownStatus.ACTIVE:
            # Update values for cells that are not active crown fires
            surface_fire(cell)

        cell.has_steady_state = True

        if self.model_spotting:
            if self.is_firesim() and cell._crown_status != CrownStatus.NONE and self._spot_ign_prob > 0:
                if not cell.lofted:
                    self.embers.loft(cell, self.curr_time_m)

            elif self.is_prediction() and cell._crown_status != CrownStatus.NONE and self._spot_ign_prob > 0:
                if not cell.lofted:
                    self.embers.loft(cell)

    def propagate_fire(self, cell: Cell):
        """_summary_

        Args:
            cell (Cell): _description_
        """
        if np.all(cell.r_t == 0) and np.all(cell.r_ss == 0) and self._iters != 0:
            cell.fully_burning = True
            return

        # Update extent of fire spread along each direction
        cell.fire_spread = cell.fire_spread + (cell.avg_ros * self._time_step)

        # Compute intersections between fire spread and distances to neighbors
        intersections = np.where(cell.fire_spread > cell.distances)[0]

        for idx in sorted(intersections, reverse=True):
            if idx not in cell.intersections:
                # Check if ignition signal should be sent to each intersecting neighbor
                if cell.breached: # Check if the cell can spread fire (breached only false if there is a fire break and the probability test failed)
                    self.ignite_neighbors(cell, cell.r_t[idx], cell.directions[idx], cell.end_pts[idx])

        # Add new intersections to tracked intersections
        cell.intersections.update(intersections)

        if len(cell.burnable_neighbors) == 0 or len(cell.intersections) == len(cell.directions):
            cell.fully_burning = True

    def propagate_fire_parallel(self, cell: Cell):
        """Compute fire spread without igniting neighbors.

        This helper mirrors :meth:`propagate_fire` but instead of igniting
        neighbors immediately it returns a list of ignition parameters.  This
        allows fire spread computations to be performed in parallel while the
        side effects of igniting neighbors are applied sequentially to maintain
        deterministic behaviour.

        Args:
            cell (Cell): The cell being processed.

        Returns:
            list: A list of tuples ``(r_gamma, gamma, end_pts)`` for each
            newly intersected neighbour that should be ignited.
        """

        if np.all(cell.r_t == 0) and np.all(cell.r_ss == 0) and self._iters != 0:
            cell.fully_burning = True
            return []

        # Update extent of fire spread along each direction
        cell.fire_spread = cell.fire_spread + (cell.avg_ros * self._time_step)

        # Compute intersections between fire spread and distances to neighbors
        intersections = np.where(cell.fire_spread > cell.distances)[0]

        ignition_params = []
        for idx in sorted(intersections, reverse=True):
            if idx not in cell.intersections and cell.breached:
                ignition_params.append(
                    (cell.r_t[idx], cell.directions[idx], cell.end_pts[idx])
                )

        # Add new intersections to tracked intersections
        cell.intersections.update(intersections)

        if len(cell.burnable_neighbors) == 0 or len(cell.intersections) == len(cell.directions):
            cell.fully_burning = True

        return ignition_params

    def ignite_neighbors(self, cell: Cell, r_gamma: float, gamma: float, end_point: list):
        """_summary_

        Args:
            cell (Cell): _description_
            r_gamma (float): _description_
            end_point (list): _description_
        """
        # Loop through end points
        for pt in end_point:

            # Get the location of the potential ignition on the neighbor
            n_loc = pt[0]

            # Get the Cell object of the neighbor
            neighbor = self.get_neighbor_from_end_point(cell, pt)

            if neighbor:
                # Check that neighbor state is burnable
                if neighbor.state == CellStates.FUEL and neighbor.fuel.burnable:
                    # Make ignition calculation
                    if self.is_firesim():
                        neighbor._update_moisture(self._curr_weather_idx, self._weather_stream)

                    # Check that ignition ros is greater than no wind no slope ros
                    if neighbor._retardant_factor > 0:
                        self._new_ignitions.append(neighbor)
                        neighbor.get_ign_params(n_loc)
                        neighbor._set_state(CellStates.FIRE)

                        if cell._crown_status == CrownStatus.ACTIVE and neighbor.has_canopy:
                            neighbor._crown_status = CrownStatus.ACTIVE

                        self.set_surface_accel_constant(neighbor)
                        surface_fire(neighbor)

                        r_eff = self.calc_ignition_ros(cell, neighbor, r_gamma)

                        neighbor.r_t, _ = calc_vals_for_all_directions(neighbor, r_eff, -999, neighbor.alpha, neighbor.e)

                        self._updated_cells[neighbor.id] = neighbor

                        if neighbor.id in self._frontier:
                            self._frontier.remove(neighbor.id)

                    else:
                        if neighbor.id not in self._frontier:
                            self._frontier.add(neighbor.id)

    def get_neighbor_from_end_point(self, cell: Cell, end_point: Tuple[int, str]) -> Cell:
        """_summary_

        Args:
            cell (Cell): _description_
            end_point (Tuple[int, str]): _description_

        Returns:
            Cell: _description_
        """
        # Get the letter representing the neighbor location relative to cell
        neighbor_letter = end_point[1]

        # Get neighbor based on neighbor_letter
        if cell._row % 2 == 0:
            diff_to_letter_map = HexGridMath.even_neighbor_letters
            
        else:
            diff_to_letter_map = HexGridMath.odd_neighbor_letters

        # Get the row and col difference between cell and neighbor
        dx, dy = diff_to_letter_map[neighbor_letter]

        row_n = int(cell.row + dy)
        col_n = int(cell.col + dx)

        if self._grid_height >= row_n >=0 and self._grid_width >= col_n >= 0:
            # Retrieve neighbor from cell grid
            neighbor = self._cell_grid[row_n, col_n]

            # If neighbor in cell's neighbors return it
            if neighbor.id in cell.burnable_neighbors:
                return neighbor

        return None
    
    def calc_ignition_ros(self, cell: Cell, neighbor: Cell, r_gamma: float) -> float:
        """_summary_

        Args:
            cell (Cell): _description_
            neighbor (Cell): _description_
            r_gamma (float): _description_

        Returns:
            float: _description_
        """
        # Get the rate of spread in ft/min
        r_ft_min = m_s_to_ft_min(r_gamma)

        # Get the heat source in the direction of question by eliminating denominator
        heat_source = r_ft_min * calc_heat_sink(cell.fuel, cell.fmois)

        # Get the heat sink using the neighbors fuel and moisture content
        heat_sink = calc_heat_sink(neighbor.fuel, neighbor.fmois)
        
        # Calculate a ignition rate of spread
        r_ign = heat_source / heat_sink

        return r_ign

    def propagate_embers(self):
        """_summary_
        """
        spot_fires = self.embers.flight(self.curr_time_s + (self.time_step/60))

        if spot_fires:
            # Schedule spot fires using the ignition delay
            for spot in spot_fires:
                ign_time = self.curr_time_s + self._spot_delay_s

                if self._scheduled_spot_fires.get(ign_time) is None:
                    self._scheduled_spot_fires[ign_time] = [spot]
                else:
                    self._scheduled_spot_fires[ign_time].append(spot)

        if self._scheduled_spot_fires:
            # Ignite spots that have been scheduled previously
            pending_times = list(self._scheduled_spot_fires.keys())

            for time in pending_times:
                if time <= self.curr_time_s:
                    # All cells with this time should be ignited
                    new_spots = self._scheduled_spot_fires[time]
                    
                    for spot in new_spots:
                        self._new_ignitions.append(spot)
                        self._updated_cells[spot.id] = spot

                    del self._scheduled_spot_fires[time]

                if time > self.curr_time_s:
                    break

    def update_control_interface_elements(self):
        """_summary_
        """
        if self._long_term_retardants:
            self.update_long_term_retardants()

        if self._active_firelines:
            self._update_active_firelines()

        if self._active_water_drops:
            self._update_active_water_drops()
    
    def _update_active_water_drops(self):
        """_summary_
        """
        for cell in self._active_water_drops.copy():
            if self.is_firesim():
                cell._update_moisture(self._curr_weather_idx, self._weather_stream)
            dead_mf, _ = get_characteristic_moistures(cell.fuel, cell.fmois)

            if dead_mf < cell.fuel.dead_mx * 0.5:
                self._active_water_drops.remove(cell)

    def update_long_term_retardants(self):
        """_summary_
        """
        for cell in self._long_term_retardants.copy():
            if cell.retardant_expiration_s <= self._curr_time_s:
                cell._retardant = False
                cell._retardant_factor = 1.0
                cell.retardant_expiration_s = -1.0

                self._updated_cells[cell.id] = cell

                self._long_term_retardants.remove(cell)

    def generate_burn_history_entry(self, cell, fuel_loads) -> List[float]:
        """_summary_

        Args:
            cell (_type_): _description_
            fuel_loads (_type_): _description_

        Returns:
            List[float]: _description_
        """
        entry = np.zeros_like(cell.fuel.w_0)
        j = 0
        for i in range(len(cell.fuel.w_0)):
            if i in cell.fuel.rel_indices:
                entry[i] = fuel_loads[j]
                j += 1

        return entry

    def compute_burn_histories(self, new_ignitions):
        """_summary_

        Args:
            new_ignitions (_type_): _description_
        """
        curr_weather = self._weather_stream.stream[self._curr_weather_idx]
        wind_speed = curr_weather.wind_speed

        if self._uniform_map:
            t_ambF = 75
        else:
            t_ambF = curr_weather.temp

        dt = self._time_step

        for cell in new_ignitions:
            # Reset cell burn history
            cell.burn_history = []

            # Get cell duff loading (tons/acre)
            wdf = cell.wdf

            I_r = cell.reaction_intensity  # BTU/ft2/min

            # Calculate wind speed at midflame height
            u = wind_speed * cell.wind_adj_factor
            
            # Get fuel bed depth
            depth = cell.fuel.fuel_depth_ft

            # Calculate duff moisture content
            if 2 in cell.fuel.rel_indices:
                mx = cell.fmois[2]
            elif 1 in cell.fuel.rel_indices:
                mx = cell.fmois[1]
            else:
                mx = cell.fmois[0]

            dfm = -0.347 + 6.42 * mx
            dfm = max(dfm, 0.10)

            # Calculate Residence time using FARSITE equation
            fli = np.max(cell.I_ss) # BTU/ft/min
            ros = m_s_to_ft_min(np.max(cell.r_ss)) #ft/min
            
            t_r = (fli*60) / (ros * I_r) # residence time in seconds
            
            # Clip to allowable values in FOFEM
            t_r = np.min([t_r, 120])
            t_r = np.max([t_r, 10])

            burn_mgr = Burnup(cell)
            burn_mgr.set_fire_data(3000, I_r, t_r, u, depth, t_ambF, dt, wdf, dfm)

            burn_mgr.arrays()
            now = 1
            d_time = burn_mgr.ti
            burn_mgr.duff_burn()

            if not (burn_mgr.start(d_time, now)):
                # Burnup does not predict ignition
                # Set to amount consumed in flaming front
                # This is how farsite does it
                fuel_loads = burn_mgr.get_flaming_front_consumption()
                entry = self.generate_burn_history_entry(cell, fuel_loads)
                cell.burn_history = [entry]

                continue

            burn_mgr.fi = burn_mgr.fire_intensity()

            if d_time > burn_mgr.tdf:
                burn_mgr.dfi = 0

            while now <= burn_mgr.ntimes:
                burn_mgr.step(burn_mgr.dt, d_time, burn_mgr.dfi)
                now += 1

                d_time += burn_mgr.dt
                if d_time > burn_mgr.tdf:
                    burn_mgr.dfi = 0

                burn_mgr.fi = burn_mgr.fire_intensity()
                
                if burn_mgr.fi <= burn_mgr.fi_min:
                    break

                fuel_loads = burn_mgr.get_updated_fuel_loading()
                entry = self.generate_burn_history_entry(cell, fuel_loads)
                cell.burn_history.append(entry)

            if len(cell.burn_history) == 0:
                # Intensity was not high enough to ignite
                # Set to amount consumed in flaming front
                # This is how farsite does it
                fuel_loads = burn_mgr.get_flaming_front_consumption()
                entry = self.generate_burn_history_entry(cell, fuel_loads)
                cell.burn_history = [entry]

    def calc_wind_padding(self, forecast: np.ndarray) -> Tuple[float, float]:
        """_summary_

        Args:
            forecast (np.ndarray): _description_

        Returns:
            Tuple[float, float]: _description_
        """
        forecast_rows = forecast[0, :, :, 0].shape[0]
        forecast_cols = forecast[0, :, :, 1].shape[1]

        forecast_height = forecast_rows * self._wind_res
        forecast_width = forecast_cols * self._wind_res

        xpad = (self.size[0] - forecast_width)/2
        ypad = (self.size[1] - forecast_height)/2
        
        return xpad, ypad

    def _set_roads(self):
        """_summary_
        """
        if self.roads is not None:
            for road, _, road_width in self.roads:
                for road_x, road_y in zip(road[0], road[1]):
                    road_cell = self.get_cell_from_xy(road_x, road_y, oob_ok = True)

                    if road_cell is not None:
                        if road_width > self._cell_size:
                            # Set to urban fuel type
                            road_cell._set_fuel_type(self.FuelClass(91))
                        else:
                            if road_cell._fuel.model_num == 91:
                                self._overwrite_urban_fuel(road_cell)

                        road_cell._break_width += road_width
                        
                        if road_cell.state == CellStates.FIRE:
                            road_cell._set_state(CellStates.FUEL)

    def _overwrite_urban_fuel(self, cell: Cell):
        """_summary_

        Args:
            cell (Cell): _description_
        """
        fuel_types = []
        for id in cell.neighbors.keys():
            neighbor = self._cell_dict[id]
            fuel_num = neighbor._fuel.model_num
            if neighbor._fuel.burnable:
                fuel_types.append(fuel_num)

        if fuel_types:
            counts = np.bincount(fuel_types)
            new_fuel_num = np.argmax(counts)

            cell._set_fuel_type(self.FuelClass(new_fuel_num))

    def _set_firebreaks(self):
        """_summary_
        """
        for line, break_width, _ in self.fire_breaks:
            self._apply_firebreak(line, break_width)

    def _apply_firebreak(self, line: LineString, break_width: float):
        """_summary_

        Args:
            line (LineString): _description_
            break_width (float): _description_
        """

        cells = self.get_cells_at_geometry(line)

        if not cells:
            return
        
        for cell in cells:
            if cell not in self._fire_break_cells:
                self._fire_break_cells.append(cell)
            
            cell._break_width += break_width
            if cell._break_width > self._cell_size:
                cell._set_fuel_type(self.FuelClass(91))


    def _set_initial_ignition(self, geometries: list):
        """_summary_

        Args:
            geometries (list): _description_
        """

        all_cells = []
        for geom in geometries:
            cells = self.get_cells_at_geometry(geom)
            all_cells.extend(cells)

        for cell in all_cells:
            self.starting_ignitions.add((cell, 0))

    def _set_initial_burnt_region(self, geometries: list):
        """_summary_

        Args:
            geometries (list): _description_
        """

        all_cells = []
        for geom in geometries:
            cells = self.get_cells_at_geometry(geom)
            all_cells.extend(cells)

        for cell in all_cells:
            cell._set_state(CellStates.BURNT)


    @property
    def frontier(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """
        frontier_copy = set(self._frontier)

        # Loop over frontier to remove any cells completely surround by fire
        for c in frontier_copy:
            remove = True
            for neighbor_id in self.cell_dict[c].burnable_neighbors:
                neighbor = self.cell_dict[neighbor_id]
                if neighbor.state == CellStates.FUEL:
                    remove = False
                    break

            if remove:
                self._frontier.remove(c)

        return self._frontier

    def get_avg_fire_coord(self) -> Tuple[float, float]:
        """Get the average position of all the cells on fire.

        If there is more than one independent fire this will include the points from both.

        :return: average position of all the cells on fire in the form (x_avg, y_avg)
        :rtype: Tuple[float, float]
        """

        x_coords = np.array([cell.x_pos for cell in self._burning_cells])
        y_coords = np.array([cell.y_pos for cell in self._burning_cells])

        return np.mean(x_coords), np.mean(y_coords)

    def hex_round(self, q, r):
        """Rounds floating point hex coordinates to their nearest integer hex coordinates.

        Args:
            q (float): q coordinate in hex coordinate system
            r (float): r coordinate in hex coordinate system

        Returns:
            tuple: (q, r) integer coordinates of the nearest hex cell
        """
        s = -q - r
        q_r = round(q)
        r_r = round(r)
        s_r = round(s)
        q_diff = abs(q_r - q)
        r_diff = abs(r_r - r)
        s_diff = abs(s_r - s)

        if q_diff > r_diff and q_diff > s_diff:
            q_r = -r_r - s_r
        elif r_diff > s_diff:
            r_r = -q_r - s_r
        else:
            s_r = -q_r - r_r

        return (int(q_r), int(r_r))

    def get_cell_from_xy(self, x_m: float, y_m: float, oob_ok = False) -> Cell:
        """Returns the cell in the sim that contains the point (x_m, y_m) in the cartesian
        plane.
        
        (0,0) is considered the lower left corner of the sim window, x increases to the
        right, y increases up.

        Args:
            x_m (float): x position of the desired point in units of meters
            y_m (float): y position of the desired point in units of meters
            oob_ok (bool, optional): whether out of bounds input is ok, if set to `True` out of bounds input
                                   will return None. Defaults to False.

        Raises:
            ValueError: oob_ok is `False` and (x_m, y_m) is out of the sim bounds

        Returns:
            Cell: Cell at the requested point, returns `None` if the point is out of bounds and oob_ok is `True`
        """
        try:
            if x_m < 0 or y_m < 0:
                if not oob_ok:
                    raise IndexError("x and y coordinates must be positive")

                else:
                    return None

            q = (np.sqrt(3)/3 * x_m - 1/3 * y_m) / self._cell_size
            r = (2/3 * y_m) / self._cell_size

            q, r = self.hex_round(q, r)

            row = r
            col = q + row//2

            # Check if the estimated cell contains the point
            estimated_cell = self._cell_grid[row, col]

            return estimated_cell

        except IndexError:
            if not oob_ok:
                msg = f'Point ({x_m}, {y_m}) is outside the grid.'
                self.logger.log_message(f"Following error occurred in 'FireSim.get_cell_from_xy()': {msg}")
                raise ValueError(msg)

            return None

    def get_cell_from_indices(self, row: int, col: int) -> Cell:
        """Returns the cell in the sim at the indices [row, col] in the cell_grid.
        
        Columns increase left to right in the sim visualization window, rows increase bottom to
        top.

        Args:
            row (int): row index of the desired cell
            col (int): col index of the desired cell

        Raises:
            TypeError: if row or col is not of type int
            ValueError: if row or col is out of the array bounds

        Returns:
            Cell: Cell instance at the indices [row, col] in the cell_grid
        """
        if not isinstance(row, int) or not isinstance(col, int):
            msg = (f"Row and column must be integer index values. "
                f"Input was {type(row)}, {type(col)}")

            if self.logger:
                self.logger.log_message(f"Following erorr occurred in 'FireSim.get_cell_from_indices(): "
                                        f"{msg} Program terminated.")
            raise TypeError(msg)

        if col < 0 or row < 0 or row >= self._grid_height or col >= self._grid_width:
            msg = (f"Out of bounds error. {row}, {col} "
                f"are out of bounds for grid of size "
                f"{self._grid_height}, {self._grid_width}")

            if self.logger:
                self.logger.log_message(f"Following erorr occurred in 'FireSim.get_cell_from_indices(): "
                                        f"{msg} Program terminated.")
            raise ValueError(msg)

        return self._cell_grid[row, col]

    # Functions for setting state of cells
    def set_state_at_xy(self, x_m: float, y_m: float, state: CellStates):
        """Set the state of the cell at the point (x_m, y_m) in the Cartesian plane.

        Args:
            x_m (float): x position of the desired point in meters
            y_m (float): y position of the desired point in meters
            state (CellStates): desired state to set the cell to (CellStates.FIRE,
                               CellStates.FUEL, or CellStates.BURNT)
        """
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        self.set_state_at_cell(cell, state)

    def set_state_at_indices(self, row: int, col: int, state: CellStates):
        """Set the state of the cell at the indices [row, col] in the cell_grid.

        Args:
            row (int): row index of the desired cell
            col (int): col index of the desired cell
            state (CellStates): desired state to set the cell to (CellStates.FIRE,
                               CellStates.FUEL, or CellStates.BURNT)
        """
        cell = self.get_cell_from_indices(row, col)
        self.set_state_at_cell(cell, state)

    def set_state_at_cell(self, cell: Cell, state: CellStates):
        """Set the state of the specified cell

        Args:
            cell (Cell): Cell object whose state is to be changed
            state (CellStates): desired state to set the cell to (CellStates.FIRE,
                               CellStates.FUEL, or CellStates.BURNT)

        Raises:
            TypeError: if 'cell' is not of type Cell
            ValueError: if 'cell' is not a valid Cell in the current fire Sim
            TypeError: if 'state' is not a valid CellStates value
        """
        if not isinstance(cell, Cell):
            msg = f"'cell' must be of type 'Cell' not {type(cell)}"

            if self.logger:
                self.logger.log_message(f"Following erorr occurred in 'FireSim.set_state_at_cell(): "
                                        f"{msg} Program terminated.")
            
            raise TypeError(msg)

        if cell.id not in self._cell_dict:
            msg = f"{cell} is not a valid cell in the current fire Sim"

            if self.logger:
                self.logger.log_message(f"Following erorr occurred in 'FireSim.set_state_at_cell(): "
                                        f"{msg} Program terminated.")
            
            raise ValueError(msg)

        if not isinstance(state, int) or 0 > state > 2:
            msg = (
                f"{state} is not a valid cell state. Must be of type CellStates. "
                f"Valid states: fireUtil.CellStates.BURNT, fireUtil.CellStates.FUEL, "
                f"fireUtil.CellStates.FIRE or 0, 1, 2"
            )

            if self.logger:
                self.logger.log_message(f"Following erorr occurred in 'FireSim.set_state_at_cell(): "
                                        f"{msg} Program terminated.")
            
            raise TypeError(msg)

        # Set new state
        cell._set_state(state)

    def set_ignition_at_xy(self, x_m: float, y_m: float):
        """_summary_

        Args:
            x_m (float): _description_
            y_m (float): _description_
        """

        # Get cell from specified x, y location
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            # Set ignition at cell
            self.set_ignition_at_cell(cell)

    def set_ignition_at_indices(self, row: int, col: int):
        """_summary_

        Args:
            row (int): _description_
            col (int): _description_
        """
        # Get cell from specified indices
        cell = self.get_cell_from_indices(row, col)

        # Set ignition at cell
        self.set_ignition_at_cell(cell)

    def set_ignition_at_cell(self, cell: Cell):
        """_summary_

        Args:
            cell (Cell): _description_
        """

        # Set ignition at cell
        cell.get_ign_params(0)
        self.set_state_at_cell(cell, CellStates.FIRE)
        self._new_ignitions.append(cell)

    def add_retardant_at_xy(self, x_m: float, y_m: float, duration_hr: float, effectiveness: float):
        """_summary_

        Args:
            x_m (float): _description_
            y_m (float): _description_
            duration_hr (float): _description_
            effectiveness (float): _description_
        """
        # Get cell from specified x, y location
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            # Apply retardant at cell
            self.add_retardant_at_cell(cell, duration_hr, effectiveness)

    def add_retardant_at_indices(self, row: int, col: int, duration_hr: float, effectiveness: float):
        """_summary_

        Args:
            row (int): _description_
            col (int): _description_
            duration_hr (float): _description_
            effectiveness (float): _description_
        """
        # Get cell from specified indices
        cell = self.get_cell_from_indices(row, col)
        
        # Apply retardant at cell 
        self.add_retardant_at_cell(cell, duration_hr, effectiveness)

    def add_retardant_at_cell(self, cell: Cell, duration_hr: float, effectiveness: float):
        """_summary_

        Args:
            cell (Cell): _description_
            duration_hr (float): _description_
            effectiveness (float): _description_
        """
        # Ensure that effectiveness is between 0 and 1
        effectiveness = min(max(effectiveness, 0), 1)

        if cell.fuel.burnable:
            # Apply long-term retardant at specified cell
            cell.add_retardant(duration_hr, effectiveness)
            self._long_term_retardants.add(cell)
            self._updated_cells[cell.id] = cell

    def water_drop_at_xy_as_rain(self, x_m: float, y_m: float, water_depth_cm: float):
        """_summary_

        Args:
            x_m (float): _description_
            y_m (float): _description_
            water_depth_cm (float): _description_
        """
        # Get cell from specified x, y location
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            # Apply water drop at cell
            self.water_drop_at_cell_as_rain(cell, water_depth_cm)

    def water_drop_at_indices_as_rain(self, row: int, col: int, water_depth_cm: float):
        """_summary_

        Args:
            row (int): _description_
            col (int): _description_
            water_depth_cm (float): _description_
        """
        # Get cell from specified indices
        cell = self.get_cell_from_indices(row, col)

        # Apply water drop as rain at cell
        self.water_drop_at_cell_as_rain(cell, water_depth_cm)

    def water_drop_at_cell_as_rain(self, cell: Cell, water_depth_cm: float):
        """_summary_

        Args:
            cell (Cell): _description_
            water_depth_cm (float): _description_

        Raises:
            ValueError: _description_
        """
        if water_depth_cm < 0:
            raise ValueError(f"Water depth must be >=0, {water_depth_cm} passed in")

        if cell.fuel.burnable:
            # Apply water drop as rain at specified cell
            cell.water_drop_as_rain(water_depth_cm)
            self._active_water_drops.append(cell)
            self._updated_cells[cell.id] = cell

    def water_drop_at_xy_as_moisture_bump(self, x_m: float, y_m: float, moisture_inc: float):
        """_summary_

        Args:
            x_m (float): _description_
            y_m (float): _description_
            moisture_inc (float): _description_
        """
        # Get cell from specified x, y location
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            # Apply water drop at cell
            self.water_drop_at_cell_as_moisture_bump(cell, moisture_inc)

    def water_drop_at_indices_as_moisture_bump(self, row: int, col: int, moisture_inc: float):
        """_summary_

        Args:
            row (int): _description_
            col (int): _description_
            moisture_inc (float): _description_
        """
        # Get cell from specified indices
        cell = self.get_cell_from_indices(row, col)

        # Apply water drop at cell
        self.water_drop_at_cell_as_moisture_bump(cell, moisture_inc)

    def water_drop_at_cell_as_moisture_bump(self, cell: Cell, moisture_inc: float):
        """_summary_

        Args:
            cell (Cell): _description_
            moisture_inc (float): _description_

        Raises:
            ValueError: _description_
        """

        if moisture_inc < 0:
            raise ValueError(f"Moisture increase must be >0, {moisture_inc} passed in")

        if cell.fuel.burnable:
            # Apply water drop as moisture bump
            cell.water_drop_as_moisture_bump(moisture_inc)
            self._active_water_drops.append(cell)
            self._updated_cells[cell.id] = cell

    def construct_fireline(self, line: LineString, width_m: float, construction_rate: float = None, id: str = None) -> str:
        """_summary_

        Args:
            line (LineString): _description_
            width_m (float): _description_
            construction_rate (float, optional): _description_. Defaults to None.
        """

        if construction_rate is None:
            # Add fire break instantly
            self._apply_firebreak(line, width_m)

            if id is None:
                id = str(len(self.fire_breaks) + 1)
                
            self._fire_breaks.append((line, width_m, id))
            self.fire_break_dict[id] = (line, width_m)
            
            # Add to a cache for visualization and logging
            cache_entry = {
                "id": id,
                "line": line,
                "width": width_m,
                "time": self.curr_time_s,
                "logged": False,
                "visualized": False
            }

            self._new_fire_break_cache.append(cache_entry)

        else:
            if id is None:
                id = str(len(self.fire_breaks) + len(self._active_firelines) + 1)

            # Create an active fireline to be updated over time
            self._active_firelines[id] = {
                "line": line,
                "width": width_m,
                "rate": construction_rate,
                "progress": 0.0,
                "partial_line": LineString([]),
                "cells": set()
            }

        return id

    def stop_fireline_construction(self, fireline_id: str):

        if self._active_firelines.get(fireline_id) is not None:
            fireline = self._active_firelines[fireline_id]
            partial_line = fireline["partial_line"]
            self._fire_breaks.append((partial_line, fireline["width"], fireline_id))
            self.fire_break_dict[fireline_id] = (partial_line, fireline["width"])
            del self._active_firelines[fireline_id]

        # TODO: do we want to throw an error here if an invalid id is passed in

    def _update_active_firelines(self) -> None:
        """_summary_
        """

        step_size = self._cell_size / 4.0
        firelines_to_remove = []

        for id in list(self._active_firelines.keys()):
            fireline = self._active_firelines[id]

            full_line = fireline["line"]
            length = full_line.length
            
            # Get the progress from previous update
            prev_progress = fireline["progress"]

            # Update progress based on line construction rate
            fireline["progress"] += fireline["rate"] * self._time_step

            # Cap progress at full length
            fireline["progress"] = min(fireline["progress"], length)

            # Interpolate new points between prev_progress and current progress
            num_steps = int(fireline["progress"] / step_size)
            prev_steps = int(prev_progress / step_size)

            for i in range(prev_steps, num_steps):
                # Get the interpolated point
                point = fireline["line"].interpolate(i * step_size)

                # Find the cell containing the new point
                cell = self.get_cell_from_xy(point.x, point.y, oob_ok=True)

                if cell is not None:
                    # Add cell to fire break cell container
                    if cell not in self._fire_break_cells:
                        self._fire_break_cells.append(cell)

                    # Add cell to the lines container
                    if cell not in fireline["cells"]:
                        cell._break_width += fireline["width"]
                        fireline["cells"].add(cell)

                        # Increment the cell's break width only once per line
                        if cell._break_width > self._cell_size:
                            cell._set_fuel_type(self.FuelClass(91))
                    
            # If line has met its full length we can add it to the permanent fire lines 
            # and remove it from active
            if fireline["progress"] == length:
                fireline["partial_line"] = full_line
                self._fire_breaks.append((full_line, fireline["width"], id))
                self.fire_break_dict[id] = (full_line, fireline["width"])
                firelines_to_remove.append(id)

            else:
                # Store the truncated line based on progress
                fireline["partial_line"] = self.truncate_linestring(fireline["line"], fireline["progress"])

        # Remove the lines marked for removal from active firelines
        for fireline_id in firelines_to_remove:
            del self._active_firelines[fireline_id]

    def truncate_linestring(self, line: LineString, length: float) -> LineString:
        """_summary_

        Args:
            line (LineString): _description_
            length (float): _description_

        Returns:
            LineString: _description_
        """
        if length <= 0:
            return LineString([line.coords[0]])
        if length >= line.length:
            return line

        coords = list(line.coords)
        accumulated = 0.0
        new_coords = [coords[0]]

        for i in range(1, len(coords)):
            seg = LineString([coords[i - 1], coords[i]])
            seg_len = seg.length
            if accumulated + seg_len >= length:
                ratio = (length - accumulated) / seg_len
                x = coords[i - 1][0] + ratio * (coords[i][0] - coords[i - 1][0])
                y = coords[i - 1][1] + ratio * (coords[i][1] - coords[i - 1][1])
                new_coords.append((x, y))
                break
            else:
                new_coords.append(coords[i])
                accumulated += seg_len

        return LineString(new_coords)

    def get_cells_at_geometry(self, geom: Union[Polygon, LineString, Point]) -> List[Cell]:
        """Get all cells that intersect with the given geometry.

        Args:
            geometry (Union[Polygon, LineString, Point]): The geometry to check for cell intersections.

        Returns:
            List[Cell]: A list of Cell objects that intersect with the geometry.
        """
        cells = set()
        if isinstance(geom, Polygon):
                minx, miny, maxx, maxy = geom.bounds
                # Get row and col indices for bounding box
                min_row = int(miny // (self.cell_size * 1.5))
                max_row = int(maxy // (self.cell_size * 1.5))
                min_col = int(minx // (self.cell_size * np.sqrt(3)))
                max_col = int(maxx // (self.cell_size * np.sqrt(3)))
                
                for row in range(min_row - 1, max_row + 2):
                    for col in range(min_col - 1, max_col + 2):
                        # Check that row and col are in bounds
                        if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
                            cell = self._cell_grid[row, col]

                            # See if cell is within polygon
                            if geom.intersection(cell.polygon).area > 1e-6:                                
                                cells.add(cell)

        elif isinstance(geom, LineString):
                length = geom.length

                step_size = self._cell_size / 4.0
                num_steps = int(length/step_size) + 1

                for i in range(num_steps):
                    point = geom.interpolate(i * step_size)
                    cell = self.get_cell_from_xy(point.x, point.y, oob_ok = True)

                    if cell is not None:
                        cells.add(cell)
                        
        elif isinstance(geom, Point):
                x, y = geom.x, geom.y
                cell = self.get_cell_from_xy(x, y, oob_ok=True)

                if cell is not None:
                    cells.add(cell)

        else:
            raise ValueError(f"Unknown geometry type: {type(geom)}")

        return list(cells)

    def set_surface_accel_constant(self, cell: Cell):
        """Sets the surface acceleration constant for a burning cell based on the state of its neighbors.

        If a cell has any burning neighbors it is modelled as a line fire.

        Args:
            cell (Cell): Cell object to set the surface acceleration constant for
        """
        # Only set for non-active crown fires, active crown fire acceleration handled in crown model
        if cell._crown_status != CrownStatus.ACTIVE: 
            for n_id in cell.neighbors.keys():
                neighbor = self._cell_dict[n_id]
                if neighbor.state == CellStates.FIRE:
                    # Model as a line fire
                    cell.a_a = 0.3 / 60 # convert to 1 /sec
                    return
            
            # Model as a point fire
            cell.a_a = 0.115 / 60 # convert to 1 / sec

    def get_action_entries(self, logger: bool = False) -> List[ActionsEntry]:
        """_summary_

        Args:
            logger (bool, optional): _description_. Defaults to False.

        Returns:
            List[ActionsEntry]: _description_
        """

        entries = []
        if self._long_term_retardants:
            lt_xs = []
            lt_ys = []
            e_vals = []

            # Collect relevant values for long term retardants
            for cell in self._long_term_retardants:
                lt_xs.append(cell.x_pos)
                lt_ys.append(cell.y_pos)
                e_vals.append(cell._retardant_factor)

            # Create single action entry
            entries.append(ActionsEntry(
                timestamp=self.curr_time_s,
                action_type="long_term_retardant",
                x_coords= lt_xs,
                y_coords= lt_ys,
                effectiveness= e_vals
            ))

        if self._active_firelines:
            for id in list(self._active_firelines.keys()):
                fireline = self._active_firelines[id]
                # For each fireline, collect relevant information on its current state
                # Create an entry for each line
                entries.append(ActionsEntry(
                    timestamp=self.curr_time_s,
                    action_type="fireline_construction",
                    x_coords= [coord[0] for coord in fireline["partial_line"].coords],
                    y_coords= [coord[1] for coord in fireline["partial_line"].coords],
                    width=fireline["width"]
                ))

        if self._active_water_drops:
            w_xs = []
            w_ys = []
            e_vals = []

            # Collect relevant values for water drops
            for cell in self._active_water_drops.copy():
                # Get the fraction of extinction moisture at the treated cell
                dead_mf, _ = get_characteristic_moistures(cell.fuel, cell.fmois)
                frac = dead_mf/cell.fuel.dead_mx
                
                # Only include cells at more than 50% of moisture extinction
                if frac > 0.5 and cell.state == CellStates.FUEL:
                    w_xs.append(cell.x_pos)
                    w_ys.append(cell.y_pos)
                    e_vals.append(dead_mf/cell.fuel.dead_mx)
            
            # Create single entry of water drops
            entries.append(ActionsEntry(
                timestamp=self.curr_time_s,
                action_type="short_term_suppressant",
                x_coords=w_xs,
                y_coords=w_ys,
                effectiveness=e_vals
            ))

        # Check if there are any instanteously constructed firelines
        if self._new_fire_break_cache:
            to_remove = []
            for entry in self._new_fire_break_cache:
                # Create an entry for each fire break
                entries.append(ActionsEntry(
                    timestamp=entry["time"],
                    action_type="fireline_construction",
                    x_coords= [coord[0] for coord in entry["line"].coords],
                    y_coords= [coord[1] for coord in entry["line"].coords],
                    width=entry["width"]
                ))

                # Only remove it from cache if it has been registered by both logger 
                # and the visualizer
                if logger:
                    entry["logged"] = True

                    # Check if visualizer is being used
                    if not self._visualizer:
                        entry["visualized"] = True

                else:
                    entry["visualized"] = True

                    # Check if logger is being used
                    if not self.logger:
                        entry["logged"] = True

                if entry["logged"] and entry["visualized"]:
                    to_remove.append(entry)

            for entry in to_remove:
                self._new_fire_break_cache.remove(entry)

        return entries
    
    def get_prediction_entry(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return PredictionEntry(self._curr_time_s, self.curr_prediction)

    def is_firesim(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        return self.__class__.__name__ == "FireSim"
    
    def is_prediction(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        return self.__class__.__name__ == "FirePredictor"

    def add_agent(self, agent: AgentBase):
        """Add agent to sim's list of registered agent.
        
        Enables sim to log agent data along with sim data so that it is included in visualizations.

        :param agent: agent to be added to the sim's list
        :type agent: :class:`~base_classes.agent_base.AgentBase`
        :raises TypeError: if agent is not an instance of :class:`~base_classes.agent_base.AgentBase`
        """
        if isinstance(agent, AgentBase):
            self._agent_list.append(agent)
            self._agents_added = True
            if self.logger:
                self.logger.log_message(f"Agent with id {agent.id} added to agent list.")
        else:
            msg = "'agent' must be an instance of 'AgentBase' or a subclass"

            if self.logger:
                self.logger.log_message(f"Following erorr occurred in 'FireSim.add_agent(): "
                                        f"{msg} Program terminated.")
            raise TypeError(msg)

    @property
    def cell_grid(self) -> np.ndarray:
        """2D array of all the cells in the sim at the current instant.
        """
        return self._cell_grid

    @property
    def cell_dict(self) -> dict:
        """Dictionary mapping cell IDs to their respective :class:`~fire_simulator.cell.Cell` instances.
        """
        return self._cell_dict

    @property
    def iters(self) -> int:
        """Number of iterations run so far by the sim
        """
        return self._iters

    @property
    def curr_time_s(self) -> int:
        """Current sim time in seconds
        """
        return self._curr_time_s

    @property
    def curr_time_m(self) -> float:
        """Current sim time in minutes
        """
        return self.curr_time_s/60

    @property
    def curr_time_h(self) -> float:
        """Current sim time in hours
        """
        return self.curr_time_m/60

    @property
    def time_step(self) -> int:
        """Time-step of the sim. Number of seconds per iteration
        """
        return self._time_step

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the sim's backing array in (rows, cols)
        """
        return self._shape

    @property
    def size(self) -> Tuple[float, float]:
        """Size of the sim region (width_m, height_m)
        """
        return self._size

    @property
    def x_lim(self) -> float:
        """Max x coordinate in the sim's map in meters
        """
        return self._size[0]
    
    @property
    def y_lim(self) -> float:
        """Max y coordinate in the sim's map in meters
        """
        return self._size[1]

    @property
    def cell_size(self) -> float:
        """Size of each cell in the simulation.
        
        Measured as the distance in meters between two parallel sides of the regular hexagon cells.
        """
        return self._cell_size

    @property
    def sim_duration(self) -> float:
        """Duration of time (in seconds) the simulation should run for, the sim will
        run for this duration unless the fire is extinguished before the duration has passed.
        """
        return self._sim_duration


    @property
    def roads(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """
        return self._roads

    @property
    def fire_break_cells(self) -> list:
        """List of :class:`~fire_simulator.cell.Cell` objects that fall along fire breaks
        """
        return self._fire_break_cells

    @property
    def fire_breaks(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """
        return self._fire_breaks

    @property
    def finished(self) -> bool:
        """`True` if the simulation is finished running. `False` otherwise
        """
        return self._finished

    @property
    def initial_ignition(self) -> list:
        """List of shapely polygons that were initially ignited at the start of the sim
        """
        return self._initial_ignition

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

        Side Effects:
            - Updates _last_wind_update to the current simulation time.
            - Increments _curr_weather_idx to the next wind forecast entry.
            - Resets _curr_weather_idx to 0 if out of bounds and raises an error.
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