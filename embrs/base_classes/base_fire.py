"""Base class implementation of fire simulation model.

Contains code to initilaize fire simulation. Hosts getters and setters for cell state properties within a simulation.
Also contains various properties on the overall state of the fire.

.. autoclass:: BaseFireSim
    :members:
"""

from typing import Tuple
from shapely.geometry import Point
import numpy as np
from tqdm import tqdm
import pickle

from embrs.utilities.fire_util import CellStates
from embrs.utilities.fire_util import RoadConstants as rc
from embrs.utilities.fire_util import HexGridMath as hex
from embrs.utilities.data_classes import SimParams
from embrs.fire_simulator.cell import Cell
from embrs.utilities.fuel_models import Anderson13
from embrs.base_classes.agent_base import AgentBase
from embrs.utilities.weather import WeatherStream
from embrs.utilities.wind_forecast import run_windninja # TODO: Should wind_forecast beceom a class? Should it be a member of WeatherStream?

class BaseFireSim:
    def __init__(self, sim_params: SimParams):
        """Base fire class, takes in a sim input object to initialize a fire simulation object.

        Args:
            sim_input (SimInput): Contains all the data necessary for initializing a sim,
                                  see SimInput documentation for more info
        """

        # Constant parameters
        self.display_frequency = 300

        # Store sim input values in class variables
        self._parse_sim_params(sim_params)
        
        # Variables to keep track of sim progress
        self._curr_time_s = 0
        self._iters = 0
        
        # Variable to store logger object
        self.logger = None

        # Containers for cells
        self._burning_cells = []
        self._cell_dict = {}
        self._updated_cells = {}
        self._soaked = []
        self._frontier = set()
        self._fire_break_cells = []
        self.starting_ignitions = []

        # List to store agents in the sim
        self._agent_list = []

        # Boolean indicating if wind has changed since last iteration        
        self.wind_changed = True

        # Boolean indicating if sim is finished
        self._finished = False

        # Set up backing array
        self._cell_grid = np.empty(self._shape, dtype=Cell)
        self._grid_width = self._cell_grid.shape[1] - 1
        self._grid_height = self._cell_grid.shape[0] - 1

        live_h_mf = self._weather_stream.live_h_mf
        live_w_mf = self._weather_stream.live_w_mf

        # Load Duff loading lookup table from LANDFIRE FCCS
        with open("embrs/utilities/duff_loading.pkl", "rb") as file:
            duff_lookup = pickle.load(file)

        # Populate cell_grid with cells
        id = 0
        for i in tqdm(range(self._shape[1]), desc="Initializing cells"):
            for j in range(self._shape[0]):
                # Initialize cell object
                new_cell = Cell(id, i, j, self._cell_size)
                cell_x, cell_y = new_cell.x_pos, new_cell.y_pos

                # Get row and col of data arrays corresponding to cell
                data_col = int(np.floor(cell_x/self._data_res))
                data_row = int(np.floor(cell_y/self._data_res))
                
                # Get fuel type
                fuel_key = self._fuel_map[data_row, data_col]
                fuel = Anderson13(fuel_key)

                # Get cell elevation from elevation map
                elev = self._elevation_map[data_row, data_col]
                self.coarse_elevation[j, i] = elev

                # Get cell aspect from aspect map
                asp = self._aspect_map[data_row, data_col]

                # Get cell slope from slope map
                slp = self._slope_map[data_row, data_col]

                # Get canopy cover from canopy cover map
                cc = self._cc_map[data_row, data_col]

                # Get canopy height from canopy height map
                ch = self._ch_map[data_row, data_col]

                # Get canopy base height from cbh map
                cbh = self._cbh_map[data_row, data_col]

                # Get canopy bulk density from cbd map
                cbd = self._cbd_map[data_row, data_col]

                # Get duff fuel loading from fccs map
                fccs_id = int(self._fccs_map[data_row, data_col])
                if duff_lookup.get(fccs_id) is not None:
                    wdf = duff_lookup[fccs_id] # tons/acre
                    wdf /= 4.46 # convert to kg/m2

                # Get data for cell
                new_cell._set_cell_data(fuel, elev, asp, slp, cc, ch, wdf, self._init_mf, live_h_mf, live_w_mf)

                # Set wind forecast in cell
                wind_col = int(np.floor(cell_x/self._wind_res))
                wind_row = int(np.floor(cell_y/self._wind_res))

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
                id +=1

        # Populate neighbors field for each cell with pointers to each of its neighbors
        self._add_cell_neighbors()

        # Set initial ignitions
        self._set_state_in_polygons(self.initial_ignition, CellStates.FIRE)

        # Set burnt cells
        # if not self._burnt_cells is None:
        #     self._set_state_in_polygons(self.burnt_cells, CellStates.BURNT)

        # Apply fire breaks
        self._set_firebreaks()
        
        # Apply Roads 
        self._set_roads()
        
        print("Initialization complete...")

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
        self._north_dir_deg = map_params.geo_info.north_angle_deg
        self.coarse_elevation = np.empty(self._shape)

        # Load DataProductParams for each data product
        lcp_data = map_params.lcp_data

        # Get map for each data product
        self._elevation_map = np.flipud(lcp_data.elevation_map)
        self._slope_map = np.flipud(lcp_data.slope_map)
        self._aspect_map = np.flipud(lcp_data.aspect_map)
        self._aspect_map = (180 + self._aspect_map) % 360 
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
        self._fire_breaks = zip(scenario.fire_breaks, scenario.fuel_vals)
        self._initial_ignition = scenario.initial_ign

        # Grab starting datetime
        self._start_datetime = sim_params.weather_input.start_datetime

        # Generate a weather stream
        self._weather_stream = WeatherStream(sim_params.weather_input, sim_params.map_params.geo_info, input_type=sim_params.weather_input.input_type)
        self.weather_t_step = self._weather_stream.time_step * 60 # convert to seconds
        
        # Get wind data
        self._wind_res = sim_params.weather_input.mesh_resolution
        self.wind_forecast = run_windninja(self._weather_stream, sim_params.map_params)
        self.flipud_forecast = np.empty(self.wind_forecast.shape)

        # Iterate over each layer (time step or vertical level, depending on the dataset structure)
        for layer in range(self.wind_forecast.shape[0]):
            self.flipud_forecast[layer] = np.flipud(self.wind_forecast[layer])
        
        self.wind_forecast = self.flipud_forecast

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

    def _set_roads(self):
        """Updates the simulation grid to incorporate roads.

        This method processes the road data and modifies grid cells accordingly. 
        Roads are defined as a list of coordinate points with an associated type. 
        If a road cell is currently burning (`CellStates.FIRE`), it is reset to 
        `CellStates.FUEL`. The fuel content is updated based on road type.

        Side Effects:
            - Modifies the state of grid cells corresponding to roads.
            - Updates fuel content for road cells.

        Raises:
            - None explicitly, but depends on `get_cell_from_xy`.

        """
        #TODO: need to figure out how we are going to model roads going forward
        if self.roads is not None:
            for road, road_type in self.roads:
                for road_x, road_y in zip(road[0], road[1]):
                    road_cell = self.get_cell_from_xy(road_x, road_y, oob_ok = True)

                    if road_cell is not None:
                        if road_cell.state == CellStates.FIRE:
                            road_cell._set_state(CellStates.FUEL)
                        road_cell._set_fuel_content(rc.road_fuel_vals[road_type])

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

        for line, fuel_val in self.fire_breaks:
            length = line.length

            step_size = 0.5
            num_steps = int(length/step_size) + 1

            for i in range(num_steps):
                point = line.interpolate(i * step_size)

                cell = self.get_cell_from_xy(point.x, point.y, oob_ok = True)

                if cell is not None:
                    cell._set_fuel_content(fuel_val/100)
                    if cell not in self._fire_break_cells:
                        self._fire_break_cells.append(cell)
    
    def _set_state_in_polygons(self, polygons: list, state: CellStates):
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
        for polygon in polygons:
            minx, miny, maxx, maxy = polygon.bounds
            # Get row and col indices for bounding box
            min_row = int(miny // (self.cell_size * 1.5))
            max_row = int(maxy // (self.cell_size * 1.5))
            min_col = int(minx // (self.cell_size * np.sqrt(3)))
            max_col = int(maxx // (self.cell_size * np.sqrt(3)))
            
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):

                    # Check that row and col are in bounds
                    if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
                        cell = self._cell_grid[row, col]

                        # See if cell is within polygon
                        if polygon.contains(Point(cell.x_pos, cell.y_pos)):
                            
                            if state == CellStates.BURNT:
                                cell._set_state(state)
                            
                            elif state == CellStates.FIRE and cell._fuel.burnable:
                                self.starting_ignitions.append((cell, 0))

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

        x_coords = np.array([cell.x_pos for cell in self._burning_cells])
        y_coords = np.array([cell.y_pos for cell in self._burning_cells])

        return np.mean(x_coords), np.mean(y_coords)

    def hex_round(self, q, r):
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

        :param x_m: x position of the desired point in units of meters
        :type x_m: float
        :param y_m: y position of the desired point in units of meters
        :type y_m: float
        :param oob_ok: whether out of bounds input is ok, if set to `True` out of bounds input
                       will return None, defaults to `False`
        :type oob_ok: bool, optional
        :raises ValueError: oob_ok is `False` and (x_m, y_m) is out of the sim bounds
        :return: :class:`~fire_simulator.cell.Cell` at the requested point, returns `None` if the
                 point is out of bounds and oob_ok is `True`
        :rtype: :class:`~fire_simulator.cell.Cell`
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
        """Returns the cell in the sim at the indices [row, col] in 
           :py:attr:`~fire_simulator.fire.FireSim.cell_grid`.
        
        Columns increase left to right in the sim visualization window, rows increase bottom to
        top.

        :param row: row index of the desired cell
        :type row: int
        :param col: col index of the desired cell
        :type col: int
        :raises TypeError: if row or col is not of type int
        :raises ValueError: if row or col is out of the array bounds
        :return: :class:`~fire_simulator.cell.Cell` instance at the indices [row, col] in the 
                 :py:attr:`~fire_simulator.fire.FireSim.cell_grid`.
        :rtype: :class:`~fire_simulator.cell.Cell`
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

        :param x_m: x position of the desired point in meters
        :type x_m: float
        :param y_m: y position of the desired point in meters
        :type y_m: float
        :param state: desired state to set the cell to (:py:attr:`CellStates.FIRE`,
                      :py:attr:`CellStates.FUEL`, or :py:attr:`CellStates.BURNT`)
        :type state: :class:`~utilities.fire_util.CellStates`
        """
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        self.set_state_at_cell(cell, state)

    def set_state_at_indices(self, row: int, col: int, state: CellStates):
        """Set the state of the cell at the indices [row, col] in 
        :py:attr:`~fire_simulator.fire.FireSim.cell_grid`.

        Columns increase left to right in the sim window, rows increase bottom to top.

        :param row: row index of the desired cell
        :type row: int
        :param col: col index of the desired cell
        :type col: int
        :param state: desired state to set the cell to (:py:attr:`CellStates.FIRE`,
                      :py:attr:`CellStates.FUEL`, or :py:attr:`CellStates.BURNT`) if set to
                      :py:attr:`CellStates.FIRE`
        :type state: :class:`~utilities.fire_util.CellStates`
        """
        cell = self.get_cell_from_indices(row, col)
        self.set_state_at_cell(cell, state)

    def set_state_at_cell(self, cell: Cell, state: CellStates):
        """Set the state of the specified cell

        :param cell: :class:`~fire_simulator.cell.Cell` object whose state is to be changed
        :type cell: :class:`~fire_simulator.cell.Cell`
        :param state: desired state to set the cell to (:py:attr:`CellStates.FIRE`,
                      :py:attr:`CellStates.FUEL`, or :py:attr:`CellStates.BURNT`) if set to
                      :py:attr:`CellStates.FIRE`
        :type state: :class:`~utilities.fire_util.CellStates`
        :raises TypeError: if 'cell' is not of type :class:`~fire_simulator.cell.Cell`
        :raises ValueError: if 'cell' is not a valid :class:`~fire_simulator.cell.Cell` in the 
                            current fire Sim
        :raises TypeError: if 'state' is not a valid :class:`~utilities.fire_util.CellStates` value
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

    # Functions for setting wild fires
    def set_ignition_at_xy(self, x_m: float, y_m: float):
        """Set a wild fire in the cell at position (x_m, y_m) in the Cartesian plane.

        :param x_m: x position of the desired wildfire ignition point in meters
        :type x_m: float
        :param y_m: y position of the desired wildfire ignition point in meters
        :type y_m: float
        """
        cell = self.get_cell_from_xy(x_m, y_m)
        self.set_wild_fire_at_cell(cell)

    def set_ignition_at_indices(self, row: int, col: int):
        """Set a wild fire in the cell at indices [row, col] in 
        :py:attr:`~fire_simulator.fire.FireSim.cell_grid`

        :param row: row index of the desired wildfire ignition cell
        :type row: int
        :param col: col index of the desired wildfire ignition cell
        :type col: int
        """
        cell = self.get_cell_from_indices(row, col)
        self.set_wild_fire_at_cell(cell)

    def set_ignition_at_cell(self, cell: Cell):
        """Set a wild fire at a specific cell

        :param cell: :class:`~fire_simulator.cell.Cell` object to set a wildfire in
        :type cell: :class:`~fire_simulator.cell.Cell`
        """
        self.set_state_at_cell(cell, CellStates.FIRE)

    # Functions for setting fuel content
    def set_fuel_content_at_xy(self, x_m: float, y_m: float, fuel_content: float):
        """Set the fraction of fuel remaining at a point (x_m, y_m) in the Cartesian plane between
        0 and 1.

        :param x_m: x position in meters of the point where fuel content should be changed 
        :type x_m: float
        :param y_m: y position in meters of the point where fuel content should be changed
        :type y_m: float
        :param fuel_content: desired fuel content at point (x_m, y_m) between 0 and 1. 
        :type fuel_content: float
        """
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok = True)
        self.set_fuel_content_at_cell(cell, fuel_content)

    def set_fuel_content_at_indices(self, row: int, col: int, fuel_content: float):
        """Set the fraction of fuel remanining in the cell at indices [row, col] in 
        :py:attr:`~fire_simulator.fire.FireSim.cell_grid` between 0 and 1.

        :param row: row index of the cell where fuel content should be changed
        :type row: int
        :param col: col index of the cell where fuel content should be changed
        :type col: int
        :param fuel_content: desired fuel content at indices [row, col} between 0 and 1.
        :type fuel_content: float
        """
        cell = self.get_cell_from_indices(row, col)
        self.set_fuel_content_at_cell(cell, fuel_content)

    def set_fuel_content_at_cell(self, cell: Cell, fuel_content: float):
        """Set the fraction of fuel remaining in a cell between 0 and 1

        :param cell: :class:`~fire_simulator.cell.Cell` object to set fuel content in
        :type cell: :class:`~fire_simulator.cell.Cell`
        :param fuel_content: desired fuel content at cell between 0 and 1.
        :type fuel_content: float
        :raises TypeError: if 'cell' is not of type :class:`~fire_simulator.cell.Cell`
        :raises ValueError: if 'cell' is not a valid :class:`~fire_simulator.cell.Cell` in the
                            current sim
        :raises ValueError: if 'fuel_content' is not between 0 and 1
        """
        if not isinstance(cell, Cell):
            msg = f"'cell' must be of type Cell not {type(cell)}"

            if self.logger:
                self.logger.log_message(f"Following erorr occurred in 'FireSim.set_fuel_content_at_cell(): "
                                        f"{msg} Program terminated.")
            
            raise TypeError(msg)

        if cell.id not in self._cell_dict:
            msg = f"{cell} is not a valid cell in the current fire Sim"

            if self.logger:
                self.logger.log_message(f"Following erorr occurred in 'FireSim.set_fuel_content_at_cell(): "
                                        f"{msg} Program terminated.")
            
            raise ValueError(msg)

        if fuel_content < 0 or fuel_content > 1:
            msg = (f"'fuel_content' must be a float between 0 and 1. "
                f"{fuel_content} was provided as input")

            if self.logger:
                self.logger.log_message(f"Following erorr occurred in 'FireSim.set_fuel_content_at_cell(): "
                                        f"{msg} Program terminated.")
            
            raise ValueError(msg)

        cell._set_fuel_content(fuel_content)

        # Add cell to update dictionary
        self._updated_cells[cell.id] = cell

    # Functions for setting fuel moisture
    def set_fuel_moisture_at_xy(self, x_m: float, y_m: float, fuel_moisture: float):
        """Set the fuel moisture at the point (x_m, y_m) in the Cartesian plane.

        :param x_m: x position in meters of the point where fuel moisture is set
        :type x_m: float
        :param y_m: y position in meters of the point where fuel moisture is set
        :type y_m: float
        :param fuel_moisture: desired fuel moisture at point (x_m, y_m), between 0 and 1.
        :type fuel_moisture: float
        """

        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        self.set_fuel_moisture_at_cell(cell, fuel_moisture)

    def set_fuel_moisture_at_indices(self, row: int, col: int, fuel_moisture: float):
        """Set the fuel moisture at the cell at indices [row, col] in the sim's backing array.

        :param row: row index of the cell where fuel moisture is set
        :type row: int
        :param col: col index of the cell where fuel moisture is set
        :type col: int
        :param fuel_moisture: desired fuel moisture at indices [row, col], between 0 and 1.
        :type fuel_moisture: float
        """
        cell = self.get_cell_from_indices(row, col)
        self.set_fuel_moisture_at_cell(cell, fuel_moisture)

    def set_fuel_moisture_at_cell(self, cell: Cell, fuel_moisture: float):
        """Set the fuel mositure at a cell

        :param cell: cell where fuel moisture is set
        :type cell: :class:`~fire_simulator.cell.Cell`
        :param fuel_moisture: desired fuel mositure at cell, between 0 and 1.
        :type fuel_moisture: float
        :raises TypeError: if 'cell' is not of type :class:`~fire_simulator.cell.Cell`
        :raises ValueError: if 'cell' is not a valid :class:`~fire_simulator.cell.Cell` in the
                            current sim
        :raises ValueError: if 'fuel_moisture' is not between 0 and 1
        """
        if not isinstance(cell, Cell):
            msg = f"'cell' must be of type Cell not {type(cell)}"

            if self.logger:
                self.logger.log_message(f"Following erorr occurred in 'FireSim.set_fuel_moisture_at_cell(): "
                                        f"{msg} Program terminated.")
            
            raise TypeError(msg)

        if cell.id not in self._cell_dict:
            msg = f"{cell} is not a valid cell in the current fire Sim"

            if self.logger:
                self.logger.log_message(f"Following erorr occurred in 'FireSim.set_fuel_moisture_at_cell(): "
                                        f"{msg} Program terminated.")
            
            raise ValueError(msg)

        if fuel_moisture < 0 or fuel_moisture > 1:
            msg = (f"'fuel_moisture' must be a float between 0 and 1. "
                f"{fuel_moisture} was provided as input")
            
            if self.logger:
                self.logger.log_message(f"Following erorr occurred in 'FireSim.set_fuel_moisture_at_cell(): "
                                        f"{msg} Program terminated.")
            
            raise ValueError(msg)

        # Add cell to update dictionary
        self._updated_cells[cell.id] = cell
        self._soaked.append(cell.to_log_format())

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
        """List of points that define the roads for the simulation.
        
        Format for each element in list: ((x,y), fuel_content). 
        
        - (x,y) is the spatial position in the sim measured in meters
            
        - fuel_content is the amount of fuel modeled at that point (between 0 and 1)
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