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
from embrs.utilities.data_classes import CellData
from embrs.utilities.fuel_models import Fuel, Anderson13
from embrs.utilities.dead_fuel_moisture import DeadFuelMoisture
from embrs.utilities.weather import WeatherStream, apply_site_specific_correction, calc_local_solar_radiation

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
        wind_forecast (list): Forecasted wind conditions [(speed, direction)].
        curr_wind (tuple): Current wind conditions (speed, direction).
    """

    def __init__(self, id: int, col: int, row: int, cell_size: float):
        """_summary_

        Args:
            id (int): _description_
            col (int): _description_
            row (int): _description_
            cell_size (float): _description_
        """
        self.id = id

        # Set cell indices
        self._col = col
        self._row = row

        # x_pos, y_pos are the global position of cell in m
        if row % 2 == 0:
            self._x_pos = col * cell_size * np.sqrt(3)
        else:
            self._x_pos = (col + 0.5) * cell_size * np.sqrt(3)

        self._y_pos = row * cell_size * 1.5

        self._cell_size = cell_size # defined as the edge length of hexagon
        self._cell_area = self.calc_cell_area()

    def _set_cell_data(self, cell_data: CellData):
        """_summary_

        Args:
            cell_data (CellData): _description_
        """

        # Set Fuel type
        self._fuel = cell_data.fuel_type
        
        # z is the elevation of cell in m
        self._elevation_m = cell_data.elevation

        # Upslope direction in degrees - 0 degrees = North
        self.aspect = cell_data.aspect

        # Slope angle in degrees
        self.slope_deg = cell_data.slope_deg

        # Canopy cover as a percentage
        self.canopy_cover = cell_data.canopy_cover

        # Canopy height in meters
        self.canopy_height = cell_data.canopy_height

        # Canopy base height in meters
        self.canopy_base_height = cell_data.canopy_base_height
        
        # Canopy bulk density in kg/m^3
        self.canopy_bulk_density = cell_data.canopy_bulk_density

        # TODO: See if we can just check one of these values instead of all 3
        self.has_canopy = self.canopy_height > 0 or self.canopy_cover > 0 or self.canopy_bulk_density > 0

        # Duff loading (tons/acre)
        self.wdf = cell_data.wdf

        # Wind adjustment factor based on sheltering condition
        self._set_wind_adj_factor()

        # Variables to keep track of burning fuel content
        self._fuel_content = 1
        self.fully_burning = False

        self.init_dead_mf = cell_data.init_dead_mf
        self.init_live_h_mf = cell_data.live_h_mf
        self.init_live_w_mf = cell_data.live_w_mf

        self.reaction_intensity = 0

        if self.fuel.burnable:
            self.set_arrays()

        # Fuel loading for each class over time starting from end of flame residence time
        self.burn_history = []

        self.dynamic_fuel_load = []
        self.burn_idx = -1

        # Set state for non burnable types to BURNT
        if self._fuel.burnable:
            self._state = CellStates.FUEL
        
        else:
            self._state = CellStates.BURNT

        # Dictionaries to store neighbors
        self._neighbors = {}
        self._burnable_neighbors = {}

        self.moist_update = -1

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
        self.forecast_wind_speeds = []
        self.forecast_wind_dirs = []
        self.curr_wind = (0,0)

        # Constant defining fire acceleration characteristics
        self.a_a = 0.115 # TODO: find fuel type dependent values for this

    def set_arrays(self):
        """_summary_
        """
        indices = self._fuel.rel_indices

        self.wdry = self._fuel.w_n[indices]
        self.sigma = self._fuel.s[indices]

        self.dfms = []
        fmois = []

        if 0 in indices:
            self.dfm1 = DeadFuelMoisture.createDeadFuelMoisture1()
            self.dfms.append(self.dfm1)
            fmois.append(self.init_dead_mf)
        if 1 in indices:
            self.dfm10 = DeadFuelMoisture.createDeadFuelMoisture10()
            self.dfms.append(self.dfm10)
            fmois.append(self.init_dead_mf)
        if 2 in indices:
            self.dfm100 = DeadFuelMoisture.createDeadFuelMoisture100()
            self.dfms.append(self.dfm100)
            fmois.append(self.init_dead_mf)
        if 3 in indices:
            fmois.append(self.init_live_h_mf)
        if 4 in indices:
            fmois.append(self.init_live_w_mf)

        self.fmois = np.array(fmois)

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
        self.forecast_wind_speeds = wind_speed
        self.forecast_wind_dirs = wind_dir

        self.curr_wind = (self.forecast_wind_speeds[0], self.forecast_wind_dirs[0]) # Note: (m/s, degrees)

    def _step_moisture(self, weather_stream: WeatherStream, idx: int):

        elev_ref = weather_stream.ref_elev

        curr_weather = weather_stream.stream[idx]

        bp0 = 0.0218
        update_interval_hr = 1

        t_f_celsius, h_f_frac = apply_site_specific_correction(self, elev_ref, curr_weather)
        solar_radiation = calc_local_solar_radiation(self, curr_weather)

        for dfm in self.dfms:
            if not dfm.initialized():
                dfm.initializeEnvironment(
                    t_f_celsius, # Intial ambient air temeperature
                    h_f_frac, # Initial ambient air rel. humidity (g/g)
                    solar_radiation, # Initial solar radiation (W/m^2)
                    0, # Initial cumulative rainfall (cm)
                    t_f_celsius, # Initial stick temperature (degrees C)
                    h_f_frac, # Intial stick surface relative humidity (g/g)
                    self.init_dead_mf, # Initial stick fuel moisture fraction (g/g)
                    bp0) # Initial stick barometric pressure (cal/cm^3)

            dfm.update_internal(
                update_interval_hr, # Elapsed time since the previous observation (hours)
                t_f_celsius, # Current observation's ambient air temperature (degrees C)
                h_f_frac, # Current observation's ambient air relative humidity (g/g)
                solar_radiation, # Current observation's solar radiation (W/m^2)
                curr_weather.rain, # Current observation's total cumulative rainfall (cm)
                bp0) # Current observation's stick barometric pressure (cal/cm^3)

    def _update_moisture(self, idx: float, weather_stream: WeatherStream):
        if self.moist_update == idx:
            return
        
        else:
            # TODO: need to check that these indices are right
            for i in range(self.moist_update + 1, idx + 1): 
                self._step_moisture(weather_stream, i)

        self.moist_update = idx
        self.fmois[0:len(self.dfms)] = [dfm.meanWtdMoisture() for dfm in self.dfms]

    def _update_weather(self, idx: int, weather_stream: WeatherStream):
        # Update moisture content based on weather stream
        self._update_moisture(idx, weather_stream)

        # Update wind to next value in forecast
        self.curr_wind = (self.forecast_wind_speeds[idx], self.forecast_wind_dirs[idx])

    def _set_elev(self, elevation: float):
        """Sets the elevation of the cell.

        Args:
            elevation (float): Elevation of the terrain at the cell location, in meters.

        Side Effects:
            - Updates the `elevation_m` attribute with the new elevation value.
        """
        self.elevation_m = elevation

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


    def _set_canopy_cover(self, canopy_cover: float):
        """Sets the canopy cover (as a percentage) of the cell

        Args:
            canopy_cover (float): canopy cover as a percentage
        """
        self.canopy_cover = canopy_cover

    def _set_canopy_height(self, canopy_height: float):
        """Sets the average top of canopy height within the cell in meters

        Args:
            canopy_height (float): average top of canopy height within the cell in meters
        """
        self.canopy_height = canopy_height

    def _set_canopy_base_height(self, canopy_base_height: float):
        self.canopy_base_height = canopy_base_height

    
    def _set_canopy_bulk_density(self, canopy_bulk_density: float):
        self.canopy_bulk_density = canopy_bulk_density

    def _set_wind_adj_factor(self):
        """Sets the wind adjustment factor (WAF) for the cell based on the fuel type and canopy characteristics.
        The wind adjustment factor is calculated using equations adapted from Albini and Baughman (1979).
        It adjusts the wind speed to account for the effects of vegetation and canopy cover on fire spread.
        If the fuel type is not burnable, the wind adjustment factor is set to 1. Otherwise, the factor is 
        calculated based on the canopy cover and height.
        For canopy cover less than or equal to 5%, an unsheltered WAF equation is used:
            WAF = 1.83 / log((20 + 0.36 * H) / (0.13 * H))
        where H is the fuel depth in feet.
        For canopy cover greater than 5%, a sheltered WAF equation is used:
            WAF = 0.555 / (sqrt(f * 3.28 * H) * log((20 + 1.18 * H) / (0.43 * H)))
        where H is the canopy height in feet and f is the crown fill portion.
        Attributes:
            wind_adj_factor (float): The calculated wind adjustment factor.
        """
        
        if not self.fuel.burnable:
            self.wind_adj_factor = 1

        else:
            # Calcuate crown fill portion
            f = (self.canopy_cover / 100) * (np.pi / 12)

            if f <= 0.05:
                # Use un-sheltered WAF equation
                H = self.fuel.fuel_depth_ft
                self.wind_adj_factor = 1.83 / np.log((20 + 0.36*H)/(0.13*H))

            else:
                # Use sheltered WAF equation
                H = self.canopy_height
                self.wind_adj_factor = 0.555/(np.sqrt(f*3.28*H) * np.log((20 + 1.18*H)/(0.43*H)))

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
            - Updates the `_fuel` attribute with the specified fuel model.
        """
        self._fuel = fuel_type

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

            self.fuel_at_ignition = self._fuel_content

    def __str__(self):
        """Returns a formatted string representation of the cell.

        The string includes the cell's ID, coordinates, elevation, fuel type, and state.

        Returns:
            str: A formatted string representing the cell.
        """
        return (f"(id: {self.id}, {self.x_pos}, {self.y_pos}, {self.elevation_m}, "
                f"type: {self.fuel.name}, "
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
    def elevation_m(self) -> float:
        """Elevation of the cell measured in meters
        """
        return self._elevation_m

    @property
    def fuel(self) -> Fuel:
        """Type of fuel present at the cell.
        
        Can be any of the 13 Anderson FBFMs
        """
        return self._fuel

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