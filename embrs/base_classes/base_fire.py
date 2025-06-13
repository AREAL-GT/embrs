"""Base class implementation of fire simulation model.

Contains code to initilaize fire simulation. Hosts getters and setters for cell state properties within a simulation.
Also contains various properties on the overall state of the fire.

.. autoclass:: BaseFireSim
    :members:
"""

from typing import Tuple
from shapely.geometry import Point, Polygon, LineString
import numpy as np
from tqdm import tqdm
import pickle
import os

from embrs.models.embers import Embers
from embrs.utilities.fire_util import CellStates, CrownStatus
from embrs.utilities.fire_util import HexGridMath as hex, UtilFuncs, HexGridMath
from embrs.utilities.data_classes import SimParams, CellData
from embrs.fire_simulator.cell import Cell
from embrs.models.crown_model import crown_fire
from embrs.models.rothermel import *
from embrs.models.fuel_models import Anderson13, ScottBurgan40
from embrs.base_classes.agent_base import AgentBase
from embrs.models.weather import WeatherStream
from embrs.models.burnup import Burnup
from embrs.models.wind_forecast import run_windninja, create_uniform_wind # TODO: Should wind_forecast beceom a class? Should it be a member of WeatherStream?
from embrs.models.perryman_spot import PerrymanSpotting


class BaseFireSim:
    def __init__(self, sim_params: SimParams, burnt_region: list = None):
        """Base fire class, takes in a sim input object to initialize a fire simulation object.

        Args:
            sim_params (SimParams): Contains all the data necessary for initializing a sim
            burnt_region (list, optional): List of regions that are already burnt. Defaults to None.

        Attributes Initialized:
            - **Core Parameters:**
                - display_frequency (int): How often to display updates
                - _sim_params (SimParams): Original simulation parameters
                - burnout_thresh (float): Fuel fraction threshold for considering a cell burnt

            - **Progress Tracking:**
                - _curr_time_s (int): Current simulation time in seconds
                - _iters (int): Number of iterations completed

            - **Weather Management:**
                - _curr_weather_idx (int): Current index in weather forecast
                - _last_weather_update (float): Time of last weather update
                - weather_changed (bool): Whether weather has changed this iteration

            - **Cell Management:**
                - _cell_grid (ndarray): 2D array of Cell objects
                - _cell_dict (dict): Dictionary mapping cell IDs to Cell objects
                - _updated_cells (dict): Cells modified in current iteration
                - _burning_cells (list): Currently burning cells
                - _new_ignitions (list): New ignitions for next iteration
                - _burnt_cells (list): Cells that have fully burnt
                - _soaked (list): Cells that have been suppressed
                - _frontier (set): Cells at the fire front
                - starting_ignitions (list): Initial ignition points
        """
        
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

        # Variable to store logger object
        self.logger = None

        # Containers for keeping track of updates to cells 
        self._updated_cells = {}

        # Containers for cells
        self._cell_dict = {}
        self._soaked = []
        self._long_term_retardants = set()
        self._burning_cells = []
        self._new_ignitions = []
        self._burnt_cells = set()
        self._frontier = set()
        self._fire_break_cells = []
        self.starting_ignitions = set()
        self._urban_cells = []

        # Crown fire containers
        self._scheduled_spot_fires = {}

        # Set up backing array
        self._cell_grid = np.empty(self._shape, dtype=Cell)
        self._grid_width = self._cell_grid.shape[1] - 1
        self._grid_height = self._cell_grid.shape[0] - 1

        if not prediction:
            live_h_mf = self._weather_stream.live_h_mf
            live_w_mf = self._weather_stream.live_w_mf
            self.fmc = self._weather_stream.fmc

        else:
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

                    # Initialize cell data class
                    cell_data = CellData()

                    # Set initial moisture values
                    cell_data.init_dead_mf = self._init_mf
                    cell_data.live_h_mf = live_h_mf
                    cell_data.live_w_mf = live_w_mf

                    cell_x, cell_y = new_cell.x_pos, new_cell.y_pos

                    # Get row and col of data arrays corresponding to cell
                    data_col = int(np.floor(cell_x/self._data_res))
                    data_row = int(np.floor(cell_y/self._data_res))
                    
                    data_col = min(data_col, sim_params.map_params.lcp_data.cols-1)
                    data_row = min(data_row, sim_params.map_params.lcp_data.rows-1)

                    # Get fuel type
                    fuel_key = self._fuel_map[data_row, data_col]
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
                        # TODO: Figure out why this is sometimes getting called
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

                    wind_speed = self.wind_forecast[:, wind_row, wind_col, 0]
                    wind_dir = self.wind_forecast[:, wind_row, wind_col, 1]
                    new_cell._set_wind_forecast(wind_speed, wind_dir)

                    # Add cell to the backing array
                    self._cell_grid[j,i] = new_cell
                    self._cell_dict[id] = new_cell

                    new_cell.set_parent(self)
                    id +=1
                    pbar.update(1)

        # Populate neighbors field for each cell with pointers to each of its neighbors
        self._add_cell_neighbors()

        # Set initial ignitions
        self._set_state_from_geometries(self.initial_ignition, CellStates.FIRE)
        
        # Set burnt cells
        if burnt_region is not None:
            self._set_state_from_geometries(burnt_region, CellStates.BURNT)
        
        # Overwrite urban cells to their neighbors (road modelling handles fire spread through roads)
        for cell in self._urban_cells:
            self._overwrite_urban_fuel(cell)

        # Apply fire breaks
        self._set_firebreaks()
        
        # Apply Roads 
        self._set_roads()
        
        print("Base initialization complete...")

    def _parse_sim_params(self, sim_params: SimParams):
        """Parses and initializes simulation input parameters.

        This method extracts relevant data from the provided `SimInput` object
        and assigns it to internal attributes for use in the wildfire simulation.

        Args:
            sim_input (SimInput): An object containing all necessary input data 
                                for the simulation, including terrain, fuel, 
                                wind conditions, and initial ignition points.

        Attributes Set:
            display_frequency (float): Frequency (in seconds) at which the 
                                    simulation state is displayed.
            _size (tuple): The size of the simulation grid.
            _shape (tuple): The shape of the simulation grid.
            _cell_size (float): The size of each grid cell in the simulation.
            _sim_duration (float): Total duration of the simulation in seconds.
            _time_step (float): Time step interval for simulation updates.
            _roads (array-like): Representation of roads in the simulation.
            coarse_elevation (ndarray): Placeholder for processed elevation data.
            _fire_breaks (array-like): Representation of fire breaks.
            _elevation_map (ndarray): Elevation map flipped vertically for processing.
            base_slope (ndarray): Base slope map without flipping.
            _slope_map (ndarray): Slope map flipped vertically.
            _aspect_map (ndarray): Aspect map flipped and adjusted to match compass angles.
            _fuel_map (ndarray): Fuel type distribution map flipped vertically.
            _initial_ignition (array-like): Initial ignition points in the simulation.
            _elevation_res (float): Resolution of the elevation data.
            _aspect_res (float): Resolution of the aspect data.
            _slope_res (float): Resolution of the slope data.
            _fuel_res (float): Resolution of the fuel data.
            _wind_res (float): Resolution of the wind data.
            wind_forecast (ndarray): Wind data map (may need flipping).
            wind_forecast_t_step (float): Wind data time step in seconds.
        """
        # Load general sim params
        self._cell_size = sim_params.cell_size
        self._sim_duration = sim_params.duration_s
        self._time_step = sim_params.t_step_s
        self._init_mf = sim_params.init_mf

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
        self._fire_breaks = list(zip(scenario.fire_breaks, scenario.break_widths))
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
                self._weather_stream = WeatherStream(sim_params.weather_input, sim_params.map_params.geo_info)
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
            self._weather_stream = WeatherStream(sim_params.weather_input, sim_params.map_params.geo_info)
            self.weather_t_step = self._weather_stream.time_step * 60 # convert to seconds

            # Create a uniform wind forecast
            self._wind_res = 10e10
            self.wind_forecast = create_uniform_wind(self._weather_stream)

        self.model_spotting = sim_params.model_spotting

        if self.model_spotting:
            self._canopy_species = sim_params.canopy_species
            self._dbh_cm = sim_params.dbh_cm
            self._spot_ign_prob = sim_params.spot_ign_prob
            self._min_spot_distance = sim_params.min_spot_dist
            self._spot_delay_s = sim_params.spot_delay_s


    def _add_cell_neighbors(self):
        """Populate the "neighbors" property of each cell in the simulation with each cell's
        neighbors
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
        # Remove any neighbors which are no longer burnable
        neighbors_to_rem = []
        for n_id in cell.burnable_neighbors:
            neighbor = self._cell_dict[n_id]

            if neighbor.state != CellStates.FUEL:
                neighbors_to_rem.append(n_id)

        if neighbors_to_rem:
            for n_id in neighbors_to_rem:
                del cell.burnable_neighbors[n_id]

        if len(cell.burnable_neighbors) == 0:
            # Set cell to fully burning when no burnable neighbors remain
            cell.fully_burning = True        

    def update_steady_state(self, cell: Cell):
        """Updates the steady state fire spread values for a cell.

        Calculates and sets the steady state rate of spread (ROS) and fireline intensity values 
        for a given cell. This includes checking for crown fire conditions and updating the cell's 
        spread parameters accordingly.

        Args:
            cell (Cell): The cell object to update steady state values for.

        Notes:
            - For non-crown fires, calculates surface fire ROS and intensity using Rothermel's model
            - For active crown fires, uses crown fire spread calculations from crown_model.py
            - Updates cell attributes including:
                - r_ss: Steady state ROS values
                - I_ss: Steady state intensity values 

                - has_steady_state: Set to True after calculation
        """

        # Checks if fire in cell meets threshold for crown fire, calls calc_propagation_in_cell using the crown ROS if active crown fire
        crown_fire(cell, self.fmc)

        if cell._crown_status != CrownStatus.ACTIVE:
            # TODO: can we make this "surface_fire()" and set cell values in the function to make everything a bit clearer
            # Update values for cells that are not active crown fires
            r_list, I_list = calc_propagation_in_cell(cell) # r in m/s, I in BTU/ft/min
            cell.r_ss = r_list
            cell.I_ss = I_list

        cell.has_steady_state = True

        if self.model_spotting:
            if self.is_firesim() and cell._crown_status != CrownStatus.NONE and self._spot_ign_prob > 0:
                if not cell.lofted:
                    self.embers.loft(cell, self.curr_time_m)

            elif self.is_prediction() and cell._crown_status != CrownStatus.NONE and self._spot_ign_prob > 0:
                if not cell.lofted:
                    self.embers.loft(cell)

    def propagate_fire(self, cell: Cell):
        if np.all(cell.r_t == 0) and np.all(cell.r_ss == 0) and self._iters != 0:
            cell.fully_burning = True

        # Update extent of fire spread along each direction
        cell.fire_spread = cell.fire_spread + (cell.r_t * self._time_step)

        # TODO: is there a way to prevent distances that are done from being computed?
        intersections = np.where(cell.fire_spread > cell.distances)[0]

        # Check where fire spread has reached edge of cell
        if len(intersections) >= int(len(cell.distances)):
            # Set cell to fully burning when all edges reached
            cell.fully_burning = True

        if cell.breached: # Check if the cell can spread fire (breached only false if there is a fire break and the probability test failed)
            for idx in intersections:
                # Check if ignition signal should be sent to each intersecting neighbor
                self.ignite_neighbors(cell, cell.r_t[idx], cell.end_pts[idx])

    def ignite_neighbors(self, cell: Cell, r_gamma: float, end_point: list) -> list:
        """Attempts to ignite neighboring cells based on fire spread conditions.

        This method evaluates fire spread from a burning cell to its neighbors. 
        If a neighboring cell meets ignition criteria, it is transitioned to the `FIRE` state 
        and its fire spread parameters are updated.

        Args:
            cell (Cell): The currently burning cell attempting to ignite its neighbors.
            r_gamma (float): The rate of spread within the burning cell along the ignition direction.
            end_point (list): A list of tuples representing fire spread endpoints, where each tuple 
                            contains:
                            - An integer indicating the ignition location along the neighboring cell.
                            - A letter (A-F) indicating which neighbor the fire is spreading to.

        Returns:
            list: A list of successfully ignited neighboring `Cell` objects.

        Behavior:
            - Iterates through `end_point` to identify potential ignition locations.
            - Calls `get_neighbor_from_end_point()` to retrieve the corresponding neighboring cell.
            - Checks if the neighbor is in a burnable state (`CellStates.FUEL` and has a burnable fuel type).
            - Computes the **ignition rate of spread** (`r_ign`) using `calc_ignition_ros()`.
            - If `r_ign > 1e-3`, the neighbor is ignited:
                - Adds the cell to `_new_ignitions`.
                - Initializes fire spread parameters (`directions`, `distances`, `end_pts`).
                - Updates wind conditions using `_update_wind()`.
                - Computes in-cell fire propagation using `calc_propagation_in_cell()`.
                - Logs the update to `_updated_cells`

        Notes:
            - The ignition threshold (`1e-3`) is a placeholder; consider using mass-loss calculations 
            or setting `R_min` dynamically.
            - If a neighboring cell is **not** ignitable but exists in `cell.burnable_neighbors`, 
            it is removed from that list.
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
                    if self.is_firesim(): # TODO: better way to handle with wanting to update moisture, but wind only populated for cells that are ignited for prediction model?
                        neighbor._update_weather(self._curr_weather_idx, self._weather_stream, self._uniform_map)
                    r_ign = self.calc_ignition_ros(cell, neighbor, r_gamma) # ft/min
                    r_0, _ = calc_r_0(neighbor.fuel, neighbor.fmois) # ft/min

                    if neighbor._retardant:
                        r_ign *= neighbor._retardant_factor
                        r_0 *= neighbor._retardant_factor

                    # Check that ignition ros is greater than no wind no slope ros
                    if 0 < r_0 < r_ign:
                        self._new_ignitions.append((neighbor, n_loc))
                        neighbor.directions, neighbor.distances, neighbor.end_pts = UtilFuncs.get_ign_parameters(n_loc, self.cell_size)
                        neighbor._set_state(CellStates.FIRE)

                        if cell._crown_status == CrownStatus.ACTIVE and neighbor.has_canopy:
                            neighbor._crown_status = CrownStatus.ACTIVE

                        neighbor.r_prev_list, _ = calc_propagation_in_cell(neighbor, r_ign) # r in m/s, I in BTU/ft/min
                        
                        self._updated_cells[neighbor.id] = neighbor

    def calc_ignition_ros(self, cell: Cell, neighbor: Cell, r_gamma: float) -> float:
        """Calculates the rate of spread (ROS) required for ignition between a burning cell 
        and an unburnt neighboring cell.

        This method determines the ignition ROS by comparing the heat source of the burning 
        cell to the heat sink of the unburned neighbor. The calculation follows:

            r_ign = heat_source_of_burning_cell / heat_sink_of_unburned_neighbor

        where:
            - `heat_source_of_burning_cell` is calculated as:
                
                heat_source_of_burning_cell = r_gamma * heat_sink_of_burning_cell

            This accounts for the energy available for fire spread along the ignition direction.
            - `r_gamma` represents the rate of spread within the burning cell in the ignition direction.
            - `heat_sink_of_burning_cell` and `heat_sink_of_unburned_neighbor` are computed 
            using the **Rothermel fire spread model**, which accounts for fuel properties and moisture content.

        Args:
            cell (Cell): The burning cell acting as the heat source.
            neighbor (Cell): The adjacent unburned cell receiving heat (potential ignition target).
            r_gamma (float): The rate of spread within the burning cell along the igniting direction.

        Returns:
            float: The calculated ignition rate of spread (ROS), representing the minimum 
                fire spread rate required for ignition of the neighboring cell.

        Notes:
            - The `calc_heat_sink` function is used to compute both heat source and sink values.
            - This method assumes that `r_gamma` is precomputed and valid.
            - The accuracy of this calculation depends on correct fuel moisture modeling.
            - Currently, fuel moisture content updates are not implemented.
        """


        # Get the rate of spread in ft/s
        r_ft_s = m_s_to_ft_min(r_gamma)

        # Get the heat source in the direction of question by eliminating denominator
        heat_source = r_ft_s * calc_heat_sink(cell.fuel, cell.fmois) # TODO: make sure this computation is valid (I think it is)

        # Get the heat sink using the neighbors fuel and moisture content
        heat_sink = calc_heat_sink(neighbor.fuel, neighbor.fmois)
        
        # Calculate a ignition rate of spread
        r_ign = heat_source / heat_sink

        return r_ign
    
    def get_neighbor_from_end_point(self, cell: Cell, end_point: Tuple[int, str]) -> Cell:
        """Retrieves the neighboring cell corresponding to a fire spread endpoint.

            This method identifies which neighboring cell is adjacent to a given fire spread 
            endpoint within the burning cell. The endpoint location is represented as a tuple:

                (position_index, neighbor_letter)

            where:
                - `position_index` (int) is a number from `1-12` indicating the fire spread endpoint 
                on the neighboring cell.
                - `neighbor_letter` (str) is a letter from `A-F` indicating which of the six 
                neighboring cells the endpoint borders.

            The mapping of these conventions is defined in **HexGridMath** (see `utilities.fire_util`).

            Args:
                cell (Cell): The burning cell from which the fire spreads.
                end_point (Tuple[int, str]): A tuple representing the endpoint of the fire spread direction.

            Returns:
                Optional[Cell]: The neighboring cell that the endpoint borders if it exists and is burnable, 
                                otherwise `None`.

            Notes:
                - Even-row and odd-row hexagonal grids use different neighbor mappings (handled via `HexGridMath`).
                - The method ensures that the retrieved neighbor exists within the simulation grid bounds.
                - Only neighbors listed in `cell.burnable_neighbors` are considered valid.
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
    
    def propagate_embers(self):
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
                        self.updated_cells[spot[0].id] = spot[0]

                    del self._scheduled_spot_fires[time]

                if time > self.curr_time_s:
                    break
    
    def update_long_term_retardants(self):
        for cell in self._long_term_retardants.copy():
            if cell.retardant_expiration_s <= self._curr_time_s:
                cell._retardant = False
                cell._retardant_factor = 1.0
                cell.retardant_expiration_s = -1.0

                self._updated_cells[cell.id] = cell

                self._long_term_retardants.remove(cell)

    def generate_burn_history_entry(self, cell, fuel_loads):
        # TODO: this assumes that any live fuel will be totally consumed
        # TODO: verify this assumption

        entry = [0] * len(cell.fuel.w_0)
        j = 0
        for i in range(len(cell.fuel.w_0)):
            if i in cell.fuel.burnup_indices:
                entry[i] = fuel_loads[j]
                j += 1

        return entry

    def compute_burn_histories(self, new_ignitions):
        # TODO: This assumes weather will be static across burn history
        curr_weather = self._weather_stream.stream[self._curr_weather_idx]

        wind_speed = curr_weather.wind_speed

        if self._uniform_map:
            # TODO: should uniform maps use temperature too?
            t_ambF = 75
        else:
            t_ambF = curr_weather.temp

        dt = self._time_step

        for cell, _ in new_ignitions:
            # Reset cell burn history
            cell.burn_history = []

            # Get cell duff loading (tons/acre)
            wdf = cell.wdf

            I_r = cell.reaction_intensity  # BTU/ft2/min

            # TODO: should we add wind speed to the burnup model?
            u = 0   #wind_speed * cell.wind_adj_factor
            
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


    def calc_wind_padding(self, forecast: np.ndarray):

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
        """Updates the simulation grid to incorporate firebreaks.

        Firebreaks are linear barriers designed to slow or stop the spread of fire.
        This method iterates over all firebreaks, interpolates points along their 
        geometry, and updates the corresponding grid cells by reducing their fuel content.

        The function:
        - Extracts the geometry and fuel reduction value of each firebreak.
        - Interpolates points along the firebreak line at fixed intervals.
        - Identifies the corresponding grid cell for each interpolated point.
        - Updates the fuel content of the affected cells.
        - Adds affected cells to `_fire_break_cells` to track their locations.

        Notes:
            - The `step_size` parameter controls the granularity of firebreak interpolation.
            - Fuel content is normalized by dividing `fuel_value` by 100.

        Side Effects:
            - Modifies the fuel content of grid cells along firebreaks.
            - Updates the `_fire_break_cells` list with affected cells.
        """
        for line, break_width in self.fire_breaks:
            length = line.length

            step_size = 0.5
            num_steps = int(length/step_size) + 1

            for i in range(num_steps):
                point = line.interpolate(i * step_size)

                cell = self.get_cell_from_xy(point.x, point.y, oob_ok = True)

                if cell is not None:
                    if cell not in self._fire_break_cells:
                        self._fire_break_cells.append(cell)

                    if break_width > self._cell_size:
                        cell._set_fuel_type(self.FuelClass(91))

                    cell._break_width = break_width

    def _set_state_from_geometries(self, geometries: list, state: CellStates):
        """Updates the state of grid cells within given polygons.

        This method iterates over a list of polygons and updates the state of 
        grid cells that fall within each polygon's boundaries. It is primarily 
        used for marking cells as burnt or ignited.

        Args:
            polygons (list): A list of polygon geometries (Shapely Polygon objects).
            state (CellStates): The state to assign to the affected grid cells 
                (e.g., `CellStates.BURNT` or `CellStates.FIRE`).

        Behavior:
            - Determines the bounding box of each polygon to limit the search space.
            - Iterates over grid cells within the bounding box and checks if 
            they fall inside the polygon.
            - If `state == CellStates.BURNT`, directly sets the cell's state.
            - If `state == CellStates.FIRE` and the cell has burnable fuel, 
            adds it to `starting_ignitions`.

        Notes:
            - Uses hexagonal grid indexing, adjusting row and column calculations
            based on the hex cell structure.
            - Ensures that all row and column indices remain within bounds.

        Side Effects:
            - Modifies `_cell_grid` by updating cell states.
            - Appends new ignitions to `starting_ignitions` when applicable.
        """

        for geom in geometries:

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
                                if state == CellStates.BURNT:
                                    cell._set_state(state)
                                
                                elif state == CellStates.FIRE and cell._fuel.burnable:
                                    self.starting_ignitions.add((cell, 0))
            
            elif isinstance(geom, LineString):
                length = geom.length

                step_size = 0.5
                num_steps = int(length/step_size) + 1

                for i in range(num_steps):
                    point = geom.interpolate(i * step_size)

                    cell = self.get_cell_from_xy(point.x, point.y, oob_ok = True)

                    if cell is not None:
                        
                        if state == CellStates.BURNT:
                            cell._set_state(state)
                        
                        elif state == CellStates.FIRE and cell._fuel.burnable and (cell, 0) not in self.starting_ignitions:
                            self.starting_ignitions.add((cell, 0))

            elif isinstance(geom, Point):
                x, y = geom.x, geom.y

                cell = self.get_cell_from_xy(x, y, oob_ok=True)

                if cell is not None:
                    if state == CellStates.BURNT:
                        cell._set_state(state)
                    
                    elif state == CellStates.FIRE and cell._fuel.burnable and (cell, 0) not in self.starting_ignitions:
                        self.starting_ignitions.add((cell, 0))

            else:
                raise ValueError(f"Unknown geometry type: {type(geom)}")


    # TODO: need to re-implement this functionality
    @property
    def frontier(self) -> list:
        """List of cells on the frontier of the fire.
        
        Cells that are in the :py:attr:`CellStates.FUEL` state and neighboring at least one 
        cell in the :py:attr:`CellStates.FIRE` state. Excludes any cells surrounded completely by
        :py:attr:`CellStates.FIRE`.
        """

        front = []
        frontier_copy = set(self._frontier)

        for c in frontier_copy:
            remove = True
            for neighbor_id, _ in c.neighbors:
                neighbor = self.cell_dict[neighbor_id]
                if neighbor.state == CellStates.FUEL:
                    remove = False
                    break

            if remove:
                self._frontier.remove(c)
            else:
                front.append(c)

        return front

    def get_avg_fire_coord(self) -> Tuple[float, float]:
        """Get the average position of all the cells on fire.

        If there is more than one independent fire this will include the points from both.

        :return: average position of all the cells on fire in the form (x_avg, y_avg)
        :rtype: Tuple[float, float]
        """

        x_coords = np.array([cell.x_pos for cell, _ in self._burning_cells])
        y_coords = np.array([cell.y_pos for cell, _ in self._burning_cells])

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

    # TODO: should we log messages when get_cell_from_xy fails here?
    # Functions for setting wild fires
    def set_ignition_at_xy(self, x_m: float, y_m: float):
        """Set a wild fire in the cell at position (x_m, y_m) in the Cartesian plane.

        Args:
            x_m (float): x position of the desired wildfire ignition point in meters
            y_m (float): y position of the desired wildfire ignition point in meters
        """
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            self.set_ignition_at_cell(cell)

    def set_ignition_at_indices(self, row: int, col: int):
        """Set a wild fire in the cell at indices [row, col] in the cell_grid

        Args:
            row (int): row index of the desired wildfire ignition cell
            col (int): col index of the desired wildfire ignition cell
        """
        cell = self.get_cell_from_indices(row, col)
        self.set_ignition_at_cell(cell)

    def set_ignition_at_cell(self, cell: Cell):
        """Set a wild fire at a specific cell

        Args:
            cell (Cell): Cell object to set a wildfire in
        """
        self.set_state_at_cell(cell, CellStates.FIRE)

    # Functions for suppression
    def add_retardant_at_xy(self, x_m: float, y_m: float, duration_hr: float, effectiveness: float):
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            self.add_retardant_at_cell(cell, duration_hr, effectiveness)

    def add_retardant_at_indices(self, row: int, col: int, duration_hr: float, effectiveness: float):
        cell = self.get_cell_from_indices(row, col)
        self.add_retardant_at_cell(cell, duration_hr, effectiveness)

    def add_retardant_at_cell(self, cell: Cell, duration_hr: float, effectiveness: float):

        # Ensure that effectiveness is between 0 and 1
        effectiveness = min(max(effectiveness, 0), 1)

        cell.add_retardant(duration_hr, effectiveness)
        self._long_term_retardants.add(cell)

        self._updated_cells[cell.id] = cell


    # Short-term supression functions
    def water_drop_at_xy_as_rain(self, x_m: float, y_m: float, water_depth_cm: float):
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            self.water_drop_at_cell_as_rain(cell, water_depth_cm)

    def water_drop_at_indices_as_rain(self, row: int, col: int, water_depth_cm: float):
        cell = self.get_cell_from_indices(row, col)
        self.water_drop_at_cell_as_rain(cell, water_depth_cm)

    def water_drop_at_cell_as_rain(self, cell: Cell, water_depth_cm: float):
        
        if water_depth_cm < 0:
            raise ValueError(f"Water depth must be >=0, {water_depth_cm} passed in")

        cell.water_drop_as_rain(water_depth_cm)

    def water_drop_at_xy_as_moisture_bump(self, x_m: float, y_m: float, moisture_inc: float):
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            self.water_drop_at_cell_as_moisture_bump(cell, moisture_inc)

    def water_drop_at_indices_as_moisture_bump(self, row: int, col: int, moisture_inc: float):
        cell = self.get_cell_from_indices(row, col)
        self.water_drop_at_cell_as_moisture_bump(cell, moisture_inc)

    def water_drop_at_cell_as_moisture_bump(self, cell: Cell, moisture_inc: float):

        if moisture_inc < 0:
            raise ValueError(f"Moisture increase must be >0, {moisture_inc} passed in")

        cell.water_drop_as_moisture_bump(moisture_inc)
        

    # Functions for fireline construction

    # TODO: Add functions that allow user to input a lineString object defining a fireline along with its width
        # TODO: Take in construction rate and form the fireline using that rate over time
            # TODO: Have user specify which end it should be constructed from
            # TODO: if possible allow for line to be constructed from both ends



    # TODO: Write function that gets cells along lineString or any shapely geometry
        # TODO: This will be useful for users to use as a way to get cells to set states or interact with


    def set_surface_accel_constant(self, cell: Cell):
        """Sets the surface acceleration constant for a cell based on the state of its neighbors.

        Args:
            cell (Cell): Cell object to set the surface acceleration constant for
        """
        if cell._crown_status != CrownStatus.ACTIVE: # Only set for non-active crown fires, active crown fire acceleration handled in crown model

            for n_id in cell.neighbors.keys():
                neighbor = self._cell_dict[n_id]
                if neighbor.state == CellStates.FIRE:
                    # Model as a line fire
                    cell.a_a = 0.3
                    return
            
            # Model as a point fire
            cell.a_a = 0.115

    def is_firesim(self) -> bool:
        return self.__class__.__name__ == "FireSim"
    
    def is_prediction(self) -> bool:
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
    def grid_width(self) -> int:
        """Width of the sim's backing array or the number of columns in the array
        """
        return self._grid_width

    @property
    def grid_height(self) -> int:
        """Height of the sim's backing array or the number of rows in the array
        """
        return self._grid_height

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
        return self.grid_width * np.sqrt(3) * self.cell_size

    @property
    def y_lim(self) -> float:
        """Max y coordinate in the sim's map in meters
        """
        return self.grid_height * 1.5 * self.cell_size

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
    def updated_cells(self) -> dict:
        """Dictionary containing cells updated since last time real-time visualization was updated. Dict keys
        are the ids of the :class:`~fire_simulator.cell.Cell` objects.
        """
        return self._updated_cells

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
        """List of dictionaries representing fire-breaks.
        
        Each dictionary has:
        
        - a "geometry"  key with a :py:attr:`shapely.LineString`
        - a "fuel_value" key with a float value which represents the amount of fuel modeled along the :py:attr:`LineString`.
        """
        return self._fire_breaks

    @property
    def finished(self) -> bool:
        """`True` if the simulation is finished running. `False` otherwise
        """
        return self._finished

    @property
    def elevation_map(self) -> np.ndarray:
        """2D array that represents the elevation in meters at each point in space
        """
        return self._elevation_map

    @property
    def fuel_map(self) -> np.ndarray:
        """2D array that represents the spatial distribution of fuel types in the sim.

        Each element is one of the `13 Anderson FBFMs <https://www.fs.usda.gov/rm/pubs_int/int_gtr122.pdf>`_.
        """
        return self._fuel_map

    @property
    def elevation_res(self) -> float:
        """Resolution of the elevation map in meters
        """
        return self._elevation_res

    @property
    def fuel_res(self) -> float:
        """Resolution of the fuel map in meters
        """
        return self._fuel_res

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