"""Representation of the discrete cells that make up the fire simulation.

This module defines the `Cell` class, which represents the fundamental units of 
the wildfire simulation grid. Each `Cell` object stores terrain properties, fire spread 
parameters, and state transitions necessary for modeling fire behavior.

Classes:
    - Cell: A hexagonal simulation unit with fire propagation attributes.

.. autoclass:: Cell
    :members:
"""
import numpy as np
from shapely.geometry import Polygon

from embrs.utilities.fire_util import CellStates
from embrs.utilities.fuel_models import Fuel, Anderson13

class Cell:
    """Represents a hexagonal simulation cell in the wildfire model.

    Each cell maintains its physical properties (elevation, slope, aspect), 
    fuel characteristics, fire state, and interactions with neighboring cells. 
    Cells are structured in a **point-up** hexagonal grid to model fire spread dynamics.

    Attributes:
        id (int): Unique identifier for the cell.
        col (int): Column index of the cell in the simulation grid.
        row (int): Row index of the cell in the simulation grid.
        cell_size (float): Edge length of the hexagonal cell (meters).
        cell_area (float): Area of the hexagonal cell (square meters).
        x_pos (float): X-coordinate of the cell in the simulation space (meters).
        y_pos (float): Y-coordinate of the cell in the simulation space (meters).
        z (float): Elevation of the cell (meters).
        aspect (float): Upslope direction in degrees (0° = North, 90° = East, etc.).
        slope_deg (float): Slope angle of the terrain at the cell (degrees).
        fuel_type (Fuel): Fuel classification based on the 13 Anderson FBFMs.
        fuel_content (float): Remaining fuel fraction (0.0 to 1.0).
        state (CellStates): Current fire state (FUEL, FIRE, BURNT).
        neighbors (dict): Dictionary of adjacent cell neighbors.
        burnable_neighbors (dict): Subset of `neighbors` that are in a burnable state.
        dead_m (float): Dead fuel moisture fraction (0.0 to 1.0).
        wind_forecast (list): Forecasted wind conditions [(speed, direction)].
        curr_wind (tuple): Current wind conditions (speed, direction).
    """

    def __init__(self, id: int, col: int, row: int, cell_size: float, z = 0.0, aspect = 0.0, slope_deg = 0.0, fuel_type=Anderson13(1)):
        """Initializes a simulation cell with terrain, fire properties, and fuel characteristics.

            The cell is positioned within a **point-up hexagonal grid** and stores relevant 
            environmental data for fire propagation modeling.

            Args:
                id (int): Unique identifier for the cell.
                col (int): Column index within the simulation grid.
                row (int): Row index within the simulation grid.
                cell_size (float): Edge length of the hexagonal cell (meters).
                z (float, optional): Elevation of the cell (meters). Defaults to 0.0.
                aspect (float, optional): Upslope direction in degrees (0° = North). Defaults to 0.0.
                slope_deg (float, optional): Terrain slope angle in degrees. Defaults to 0.0.
                fuel_type (Fuel, optional): Fuel classification based on the 13 Anderson FBFMs. Defaults to `Fuel(1)`.

            Attributes Initialized:
                - `_col`, `_row`: Stores the grid position.
                - `_z`: Elevation of the cell.
                - `aspect`, `slope_deg`: Terrain properties.
                - `_cell_size`: Edge length of the hexagonal cell.
                - `_cell_area`: Computed area of the hexagonal cell.
                - `_x_pos`, `_y_pos`: Computed global coordinates.
                - `_fuel_type`: Assigned fuel model.
                - `_state`: Set to `CellStates.FUEL` if burnable, otherwise `CellStates.BURNT`.
                - `_neighbors`, `_burnable_neighbors`: Dictionaries for tracking adjacency.
                - `_dead_m`: Default dead fuel moisture content (0.08).
                - `polygon`: Shapely polygon representation of the hexagonal cell.
                - `distances`, `directions`, `end_pts`: Fire spread direction variables.
                - `r_t`, `fire_spread`, `r_prev_list`, `t_elapsed_min`: Fire dynamics variables.
                - `r_ss`, `I_ss`: Steady-state rate of spread and fireline intensity.
                - `has_steady_state`: Tracks whether steady-state ROS has been reached.
                - `wind_forecast`, `curr_wind`: Wind condition storage.
                - `a_a`: Fire acceleration constant (default `0.115`).

            """
        self.id = id

        # Set cell indices
        self._col = col
        self._row = row

        # z is the elevation of cell in m
        self._z = z

        # Upslope direction in degrees - 0 degrees = North
        self.aspect = aspect

        # Slope angle in degrees
        self.slope_deg = slope_deg

        self._cell_size = cell_size # defined as the edge length of hexagon
        self._cell_area = self.calc_cell_area()

        # x_pos, y_pos are the global position of cell in m
        if row % 2 == 0:
            self._x_pos = col * cell_size * np.sqrt(3)
        else:
            self._x_pos = (col + 0.5) * cell_size * np.sqrt(3)

        self._y_pos = row * cell_size * 1.5

        # Set Fuel type
        self._fuel_type = fuel_type

        # Set state for non burnable types to BURNT
        if self._fuel_type.burnable:
            self._state = CellStates.FUEL
        
        else:
            self._state = CellStates.BURNT

        # Dictionaries to store neighbors
        self._neighbors = {}
        self._burnable_neighbors = {}

        # dead fuel moisture at this cell, value based on Anderson fuel model paper
        self._dead_m = 0.08

        # Get shapely polygon representation of cell
        self.polygon = self.to_polygon()

        # Variables that define spread directions within cell
        self.distances = None
        self.directions = None
        self.end_pts = None

        # Variables that keep track of elliptical spread within cell
        self.r_t = None
        self.fire_spread = np.array([])
        self.r_prev_list = np.array([])
        self.t_elapsed_min = 0
        self.r_ss = np.array([])
        self.I_ss = np.array([])
        
        # Boolean defining the cell already has a steady-state ROS
        self.has_steady_state = False

        # Wind forecast and current wind within cell
        self.wind_forecast = []
        self.curr_wind = (0,0)

        # Constant defining fire acceleration characteristics
        self.a_a = 0.115 # TODO: find fuel type dependent values for this

    def set_real_time_vals(self):
        """Updates real-time fire spread parameters using fire acceleration.

        This method applies fire acceleration dynamics similar to FARSITE. It updates the 
        current rate of spread (`self.r_t`) in each direction by considering the prescribed 
        steady-state rate of spread (`self.r_ss`) and previous rates (`self.r_prev_list`). 
        Additionally, it recalculates fireline intensity (`self.I_t`) based on the new `r_t` value.

        Behavior:
            - If the maximum steady-state ROS (`self.r_ss`) is less than the maximum previous 
            ROS (`self.r_prev_list`), the cell undergoes **instantaneous deceleration**, 
            setting `self.r_t` and `self.I_t` directly to their steady-state values.
            - Otherwise, `self.r_t` is updated using an exponential function that accounts 
            for fire acceleration over elapsed time.
            - Fireline intensity (`self.I_t`) is adjusted proportionally based on the updated 
            rate of spread.

        Notes:
            - The acceleration factor (`self.a_a`) and elapsed time (`self.t_elapsed_min`) 
            influence how quickly the fire reaches steady-state ROS.
            - A small epsilon (`1e-7`) is added to prevent division by zero.

        TODO:
            - Validate correctness of the fire acceleration implementation.

        Side Effects:
            - Updates `self.r_t` (current rate of spread).
            - Updates `self.I_t` (current fireline intensity).
        """

        # TODO: validate that this works correctly

        if np.max(self.r_ss) < np.max(self.r_prev_list):
            # Allow for instant deceleration as in FARSITE
            self.r_t = self.r_ss
            self.I_t = self.I_ss

        else:
            self.r_t = self.r_ss - (self.r_ss - self.r_prev_list) * np.exp(-self.a_a * self.t_elapsed_min)
            self.I_t = (self.r_t / (self.r_ss+1e-7)) * self.I_ss

    def _set_wind_forecast(self, wind_speed: np.ndarray, wind_dir: np.ndarray):
        """Stores the local forecasted wind speed and direction for the cell.

        This method takes in forecasted wind speed and direction values for the specific 
        cell and stores them as a list of tuples (`self.wind_forecast`), where each tuple 
        contains wind speed (m/s) and direction (degrees). The initial wind conditions 
        are set to the first forecasted value.

        Args:
            wind_speed (np.ndarray): An array of forecasted wind speeds (in meters per second).
            wind_dir (np.ndarray): An array of corresponding wind directions (in degrees).

        Behavior:
            - Combines `wind_speed` and `wind_dir` into a list of tuples.
            - Stores the result in `self.wind_forecast`.
            - Initializes `self.curr_wind` to the first forecasted wind value.

        Side Effects:
            - Updates `self.wind_forecast` with the forecasted wind data.
            - Sets `self.curr_wind` to the first forecasted wind tuple.

        Notes:
            - Wind direction is assumed to be in degrees, following cartesian convention.
        """
        self.wind_forecast = [(speed, dir) for speed, dir in zip(wind_speed, wind_dir)]
        self.curr_wind = self.wind_forecast[0] # Note: (m/s, degrees)

    def _update_wind(self, idx: int):
        """Updates the current wind conditions based on the forecast index.

        This method selects the wind speed and direction from the stored wind forecast 
        (`self.wind_forecast`) at the specified index and updates the current wind conditions 
        (`self.curr_wind`).

        Args:
            idx (int): The index of the wind forecast array to use for updating the current wind.

        Side Effects:
            - Updates `self.curr_wind` to the wind conditions at the given forecast index.

        Notes:
            - Assumes `idx` is within the valid range of `self.wind_forecast`.
            - Wind conditions are stored as tuples `(wind_speed, wind_direction)`, where 
            wind speed is in meters per second (m/s) and wind direction is in degrees.
        """
        self.curr_wind = self.wind_forecast[idx]

    def _set_elev(self, elevation: float):
        """Sets the elevation of the cell.

        Args:
            elevation (float): Elevation of the terrain at the cell location, in meters.

        Side Effects:
            - Updates the `_z` attribute with the new elevation value.
        """
        self._z = elevation

    def _set_slope(self, slope: float):
        """Sets the slope of the cell.

        The slope represents the steepness of the terrain at the cell's location.

        Args:
            slope (float): The slope of the terrain in degrees, where 0° represents 
                        flat terrain and higher values indicate steeper inclines.

        Side Effects:
            - Updates the `slope_deg` attribute with the new slope value.
        """
        self.slope_deg = slope

    def _set_aspect(self, aspect: float):
        """Sets the aspect (upslope direction) of the cell.

        Aspect is the compass direction that the upslope direction faces, measured in degrees:
        - 0° = North
        - 90° = East
        - 180° = South
        - 270° = West

        Args:
            aspect (float): The aspect of the terrain in degrees.

        Side Effects:
            - Updates the `aspect` attribute with the new aspect value.
        """
        self.aspect = aspect

    def _set_fuel_content(self, fuel_content: float):
        """Sets the remaining fuel content in the cell.

        Fuel content represents the proportion of available fuel, where:
        - `1.0` indicates a fully fueled cell.
        - `0.0` indicates a completely burned-out cell.

        Args:
            fuel_content (float): The amount of fuel remaining in the cell, ranging from 0 to 1.

        Side Effects:
            - Updates the `_fuel_content` attribute with the specified value.
        """
        self._fuel_content = fuel_content

    def _set_fuel_type(self, fuel_type: Fuel):
        """Sets the fuel type of the cell based on the selected Fuel Model.

        The fuel type determines the fire spread characteristics of the cell.

        Args:
            fuel_type (Fuel): The fuel classification, based on the Anderson
                                    or Scott Burgan standard fire behavior fuel models.

        Side Effects:
            - Updates the `_fuel_type` attribute with the specified fuel model.
        """
        self._fuel_type = fuel_type

    def _set_state(self, state: CellStates):
        """Sets the state of the cell.

        The state represents the fire condition of the cell and must be one of:
        - `CellStates.FUEL`: The cell contains unburned fuel.
        - `CellStates.FIRE`: The cell is actively burning.
        - `CellStates.BURNT`: The cell has already burned.

        Args:
            state (CellStates): The new state of the cell.

        Side Effects:
            - Updates the `_state` attribute with the given state.
            - If the state is `CellStates.FIRE`, initializes fire spread-related attributes:
                - `fire_spread`: Tracks fire spread rates in different directions.
                - `r_prev_list`: Stores previous rates of spread.
                - `r_ss`: Stores steady-state rates of spread.
                - `I_ss`: Stores steady-state fireline intensity values.
        """
        self._state = state

        if self._state == CellStates.FIRE:
            self.fire_spread = np.zeros(len(self.directions))
            self.r_prev_list = np.zeros(len(self.directions))
            self.r_ss = np.zeros(len(self.directions))
            self.I_ss = np.zeros(len(self.directions))

    def _set_dead_m(self, dead_m: float):
        """Sets the dead fuel moisture content at the cell.

        Dead fuel moisture affects fire propagation, with higher moisture levels slowing 
        the rate of spread. Moisture content is expressed as a fraction between 0 and 1.

        Args:
            dead_m (float): The dead fuel moisture content, where:
                - `0.0` represents completely dry fuel.
                - `1.0` represents fully saturated fuel.

        Side Effects:
            - Updates the `_dead_m` attribute with the specified moisture value.
        """
        self._dead_m = dead_m

    def __str__(self):
        """Returns a formatted string representation of the cell.

        The string includes the cell's ID, coordinates, elevation, fuel type, and state.

        Returns:
            str: A formatted string representing the cell.
        """
        return (f"(id: {self.id}, {self.x_pos}, {self.y_pos}, {self.z}, "
                f"type: {self.fuel_type.name}, "
                f"state: {self.state}")

    def calc_cell_area(self):
        """Calculates the area of the hexagonal cell in square meters.

        The formula for the area of a regular hexagon is:

            Area = (3 * sqrt(3) / 2) * side_length²

        Returns:
            float: The area of the hexagonal cell in square meters.
        """
        area_m2 = (3 * np.sqrt(3) * self.cell_size ** 2) / 2
        return area_m2

    def to_polygon(self):
        """Generates a Shapely polygon representation of the hexagonal cell.

        The polygon is created in a point-up orientation using the center (`x_pos`, `y_pos`) 
        and the hexagon's side length.

        Returns:
            Polygon: A Shapely polygon representing the hexagonal cell.
        """
        l = self.cell_size
        x, y = self.x_pos, self.y_pos

        # Define the vertices for the hexagon in point-up orientation
        hex_coords = [
            (x, y + l),
            (x + (np.sqrt(3) / 2) * l, y + l / 2),
            (x + (np.sqrt(3) / 2) * l, y - l / 2),
            (x, y - l),
            (x - (np.sqrt(3) / 2) * l, y - l / 2),
            (x - (np.sqrt(3) / 2) * l, y + l / 2),
            (x, y + l)  # Close the polygon
        ]

        return Polygon(hex_coords)

    def to_log_format(self):
        """Returns a dictionary of cell data at the current simulation state.

        This dictionary captures changes in the cell's state over time, making it useful 
        for visualization and post-processing.

        Returns:
            dict: A dictionary with the following keys:
                - `"id"` (int): The unique ID of the cell.
                - `"state"` (CellStates): The current state of the cell.
        """
        cell_data = {
            "id": self.id,
            "state": self._state,
        }

        return cell_data

    def get_spread_directions(self, ignited_pos):
        return self.DIRECTION_MAP.get(ignited_pos)

    # ------ Compare operators overloads ------ #
    def __lt__(self, other) -> bool:
        """Compares two cells based on their unique ID.

        This method allows for sorting and comparison of cells using the `<` (less than) operator.

        Args:
            other (Cell): Another cell to compare against.

        Returns:
            bool: `True` if this cell's ID is less than the other cell's ID, `False` otherwise.

        Raises:
            TypeError: If `other` is not an instance of `Cell`.
        """
        if not isinstance(other, type(self)):
            raise TypeError("Comparison must be between two Cell instances.")
        return self.id < other.id

    def __gt__(self, other) -> bool:
        """Compares two cells based on their unique ID.

        This method allows for sorting and comparison of cells using the `>` (greater than) operator.

        Args:
            other (Cell): Another cell to compare against.

        Returns:
            bool: `True` if this cell's ID is greater than the other cell's ID, `False` otherwise.

        Raises:
            TypeError: If `other` is not an instance of `Cell`.
        """
        if not isinstance(other, type(self)):
            raise TypeError("Comparison must be between two Cell instances.")
        return self.id > other.id

    @property
    def col(self) -> int:
        """Column index of the cell within the :py:attr:`~fire_simulator.fire.FireSim.cell_grid`
        """
        return self._col

    @property
    def row(self) -> int:
        """Row index of the cell within the :py:attr:`~fire_simulator.fire.FireSim.cell_grid`
        """
        return self._row

    @property
    def cell_size(self) -> float:
        """Size of the cell in meters.
        
        Measured as the side length of the hexagon.
        """
        return self._cell_size

    @property
    def cell_area(self) -> float:
        """Area of the cell, measured in meters squared.
        """
        return self._cell_area

    @property
    def x_pos(self) -> float:
        """x position of the cell within the sim, measured in meters.
        
        x values increase left to right in the sim visualization window
        """
        return self._x_pos

    @property
    def y_pos(self) -> float:
        """y position of the cell within the sim, measured in meters.
        
        y values increase bottom to top in the sim visualization window
        """
        return self._y_pos

    @property
    def z(self) -> float:
        """Elevation of the cell measured in meters
        """
        return self._z

    @property
    def fuel_type(self) -> Fuel:
        """Type of fuel present at the cell.
        
        Can be any of the 13 Anderson FBFMs
        """
        return self._fuel_type

    @property
    def fuel_content(self) -> float:
        """Fraction of fuel remaining at the cell, between 0 and 1
        """
        return self._fuel_content

    @property
    def state(self) -> CellStates:
        """Current state of the cell.
        
        Can be :py:attr:`CellStates.FUEL`, :py:attr:`CellStates.BURNT`, or :py:attr:`CellStates.FIRE`.
        """
        return self._state

    @property
    def neighbors(self) -> dict:
        """List of cells that are adjacent to the cell.
        
        Each list element is in the form (id, (dx, dy))
        
        - "id" is the id of the neighboring cell
        - "(dx, dy)" is the difference between the column and row respectively of the cell and its neighbor
        """
        return self._neighbors

    @property
    def burnable_neighbors(self) -> dict:
        """Set of cells adjacent to the cell which are in a burnable state.

        Each element is in the form (id, (dx, dy))
        
        - "id" is the id of the neighboring cell
        - "(dx, dy)" is the difference between the column and row respectively of the cell and its neighbor
        """
        return self._burnable_neighbors

    @property
    def dead_m(self) -> float:
        """Dead fuel moisture of the cell, as a fraction (0 to 1)
        """
        return self._dead_m