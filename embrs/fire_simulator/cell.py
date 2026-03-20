"""Representation of the discrete cells that make up the fire simulation.

This module defines the `Cell` class, which represents the fundamental units of 
the wildfire simulation grid. Each `Cell` object stores terrain properties, fire spread 
parameters, and state transitions necessary for modeling fire behavior.

Classes:
    - Cell: A hexagonal simulation unit with fire propagation attributes.

.. autoclass:: Cell
    :members:
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Polygon
import weakref

if TYPE_CHECKING:
    from embrs.base_classes.base_fire import BaseFireSim

from embrs.utilities.fire_util import CellStates, CrownStatus, UtilFuncs
from embrs.utilities.data_classes import CellData
from embrs.models.fuel_models import Fuel
from embrs.models.dead_fuel_moisture import DeadFuelMoisture
from embrs.models.weather import WeatherStream, apply_site_specific_correction, calc_local_solar_radiation
from embrs.utilities.logger_schemas import CellLogEntry

# Pre-allocated read-only arrays for reset_to_fuel. These are shared across all
# cells and never modified — each cell replaces them with new arrays when it
# starts burning (via get_ign_params / surface_fire).
_RESET_ARRAY_ZERO = np.array([0], dtype=np.float64)
_RESET_ARRAY_EMPTY = np.array([], dtype=np.float64)
_RESET_IXN_EMPTY = np.zeros(0, dtype=np.bool_)


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
        elevation_m (float): Elevation of the cell (meters).
        aspect (float): Upslope direction in degrees (0° = North, 90° = East, etc.).
        slope_deg (float): Slope angle of the terrain at the cell (degrees).
        fuel (Fuel): Fire Behavior Fuel Model (FBFM) for the cell, from either Anderson 13 or Scott-Burgan 40.
        state (CellStates): Current fire state (FUEL, FIRE, BURNT).
        neighbors (dict): Dictionary of adjacent cell neighbors.
        burnable_neighbors (dict): Subset of `neighbors` that are in a burnable state.
        forecast_wind_speeds (list): Forecasted wind speeds in m/s.
        forecast_wind_dirs (list): Forecasted wind directions in degrees (cartesian).
    """

    def __init__(self, id: int, col: int, row: int, cell_size: float):
        """Initialize a hexagonal cell with position and geometry.

        Creates a cell at the specified grid position and calculates its spatial
        coordinates based on the hexagonal grid layout. The cell is initialized
        with default values for fire state and fuel properties.

        Args:
            id (int): Unique identifier for this cell.
            col (int): Column index in the simulation grid.
            row (int): Row index in the simulation grid.
            cell_size (float): Edge length of the hexagon in meters.

        Notes:
            - Spatial position is calculated using point-up hexagon geometry.
            - For even rows: x = col * cell_size * sqrt(3)
            - For odd rows: x = (col + 0.5) * cell_size * sqrt(3)
            - y = row * cell_size * 1.5
        """
        self.id = id

        # Deterministic hash based on cell ID (not memory address) so that
        # set iteration order is reproducible across process invocations.
        self._hash = hash(id)

        # Set cell indices
        self._col = col
        self._row = row

        # x_pos, y_pos are the global position of cell in m
        _sqrt3 = 1.7320508075688772  # math.sqrt(3)
        if row % 2 == 0:
            self._x_pos = col * cell_size * _sqrt3
        else:
            self._x_pos = (col + 0.5) * cell_size * _sqrt3

        self._y_pos = row * cell_size * 1.5

        self._cell_size = cell_size # defined as the edge length of hexagon
        self._cell_area = self.calc_cell_area()

        # Flag to track if cell has been treated with long-term fire retardant
        self._retardant = False
        self._retardant_factor = 1.0 # Factor multiplied by rate of spread (0-1)
        self.retardant_expiration_s = -1.0 # Time at which long-term retardant in cell expires

        # Track the total amount of water dropped (if modelled as rain)
        self.local_rain = 0.0

        # Van Wagner energy-balance water suppression (Van Wagner & Taylor 2022)
        self.water_applied_kJ = 0.0   # Cumulative cooling energy from water drops
        self._vw_efficiency = 2.5     # Application efficiency multiplier (Table 4)

        # Width in meters of any fuel discontinuity within cell (road or firebreak)
        self._break_width = 0 

        # Variable to track if fuel discontinuity within cell can be breached
        self.breached = True

        # Track if firebrands have been lofted from cell
        self.lofted = False

        # Source cell coordinates if ignited by ember spotting (set by Embers.flight)
        self.spot_source_xy = None

        # Weak reference to parent BaseFire object
        self._parent = None

        self._arrival_time = -999

    def __hash__(self):
        return self._hash

    def set_parent(self, parent: BaseFireSim) -> None:
        """Sets the parent BaseFire object for this cell.

        Args:
            parent (BaseFireSim): The BaseFire object that owns this cell.
        """
        self._parent = weakref.ref(parent)

    def reset_to_fuel(self):
        """Reset cell to initial FUEL state, preserving terrain/fuel/geometry data.

        Resets all mutable fire-state attributes to their initial values as set
        by ``__init__`` and ``_set_cell_data``. Immutable properties such as
        position, fuel model, elevation, slope, aspect, canopy attributes,
        polygon, wind adjustment factor, and neighbor topology are preserved.

        This is used by FirePredictor to efficiently restore cells to a clean
        state between predictions, avoiding expensive deepcopy operations.

        Side Effects:
            - Resets fire state to CellStates.FUEL
            - Clears all spread tracking arrays
            - Resets suppression effects (retardant, rain, firebreaks)
            - Resets fuel moisture to initial values
            - Restores full neighbor set to _burnable_neighbors
        """
        # Fire state
        self._state = CellStates.FUEL
        self.fully_burning = False
        self._crown_status = CrownStatus.NONE
        self.cfb = 0
        self.reaction_intensity = 0

        # Spread arrays — shared read-only constants (replaced when cell ignites)
        self.distances = None
        self.directions = None
        self.end_pts = None
        self._ign_n_loc = None
        self.r_h_ss = None
        self.I_h_ss = None
        self.r_t = _RESET_ARRAY_ZERO
        self.fire_spread = _RESET_ARRAY_EMPTY
        self.avg_ros = _RESET_ARRAY_EMPTY
        self.r_ss = _RESET_ARRAY_EMPTY
        self.I_ss = _RESET_ARRAY_ZERO
        self.I_t = _RESET_ARRAY_ZERO
        self.intersections = _RESET_IXN_EMPTY
        self.e = 0
        self.alpha = None
        self.has_steady_state = False

        # Suppression effects — back to __init__ defaults
        self._retardant = False
        self._retardant_factor = 1.0
        self.retardant_expiration_s = -1.0
        self.local_rain = 0.0
        self.water_applied_kJ = 0.0
        self._vw_efficiency = 2.5
        self._break_width = 0
        self.breached = True
        self.lofted = False
        self.spot_source_xy = None
        self._arrival_time = -999

        # Wind forecast
        self.forecast_wind_speeds = []
        self.forecast_wind_dirs = []

        # Moisture — reset to initial values (use cached template if available)
        self.moist_update_time_s = 0
        if self._fuel.burnable:
            if self.dfms:
                for dfm in self.dfms:
                    dfm.initializeStick()
            # Reset moisture array from cached initial values
            self.fmois = self._init_fmois.copy()

        # Restore full neighbor set (remove_neighbors modifies _burnable_neighbors during sim)
        self._burnable_neighbors = dict(self._neighbors)

    def _set_cell_data(self, cell_data: CellData):
        """Configure cell properties from terrain and fuel data.

        Initializes the cell's physical properties, fuel characteristics, and
        fire spread parameters from a CellData object. This method is called
        during simulation setup after the cell geometry is established.

        Args:
            cell_data (CellData): Container with fuel type, elevation, slope,
                aspect, canopy properties, and initial moisture fractions.

        Side Effects:
            - Sets terrain attributes (elevation, slope, aspect).
            - Sets canopy attributes and calculates wind adjustment factor.
            - Initializes fuel moisture arrays for dead and live fuels.
            - Creates the Shapely polygon representation.
            - Sets cell state to CellStates.FUEL.
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

        # Check if cell has a canopy
        self.has_canopy = self.canopy_height > 0 and self.canopy_cover > 0

        # Wind adjustment factor based on sheltering condition
        self._set_wind_adj_factor()

        # Set to true when fire has spread across entire cell
        self.fully_burning = False

        self.init_dead_mf = cell_data.init_dead_mf
        self.init_live_h_mf = cell_data.live_h_mf
        self.init_live_w_mf = cell_data.live_w_mf

        self.reaction_intensity = 0

        if self.fuel.burnable:
            self.set_arrays()

        # Default state is fuel
        self._state = CellStates.FUEL

        # Crown fire attribute
        self._crown_status = CrownStatus.NONE

        # Crown fraction burned
        self.cfb = 0

        # Dictionaries to store neighbors
        self._neighbors = {}
        self._burnable_neighbors = {}

        # Last time the moisture was updated in cell
        self.moist_update_time_s = 0

        # Polygon is set externally via batch creation for performance
        self.polygon = None

        # Variables that define spread directions within cell
        self.distances = None
        self.directions = None
        self.end_pts = None
        self._ign_n_loc = None  # Set when cell ignites via get_ign_params()

        # Heading rate of spread and fireline intensity
        self.r_h_ss = None
        self.I_h_ss = None

        # Variables that keep track of elliptical spread within cell
        self.r_t = np.array([0])
        self.fire_spread = np.array([])
        self.avg_ros = np.array([]) # Average rate of spread for current time step
        self.r_ss = np.array([])
        self.I_ss = np.array([0])
        self.I_t = np.array([0])
        self.intersections = np.zeros(0, dtype=np.bool_)
        self.e = 0
        self.alpha = None

        # Boolean defining the cell already has a steady-state ROS
        self.has_steady_state = False

        # Wind forecast and current wind within cell
        self.forecast_wind_speeds = []
        self.forecast_wind_dirs = []

        # Constant defining fire acceleration characteristics
        self.a_a = 0.3 / 60

    def set_arrays(self):
        """Initialize fuel moisture tracking arrays for this cell.

        DFM objects are created lazily on first moisture update to avoid
        allocating ~12 numpy arrays per object for cells that never burn.

        Side Effects:
            - Sets self.wdry and self.sigma from fuel model properties.
            - Initializes self.fmois array with initial moisture fractions.
            - Sets self._dfms_needed tuple for lazy DFM creation.
        """
        indices = self._fuel.rel_indices

        self.wdry = self._fuel.w_n[self._fuel.rel_indices]
        self.sigma = self._fuel.s[self._fuel.rel_indices]

        # Defer DFM creation — store which classes are needed
        self.dfms = []
        self._dfms_needed = (0 in indices, 1 in indices, 2 in indices)

        fmois = np.zeros(6)

        if 0 in indices:
            fmois[0] = self.init_dead_mf[0]
        if 1 in indices:
            fmois[1] = self.init_dead_mf[1]
        if 2 in indices:
            fmois[2] = self.init_dead_mf[2]
        if 3 in indices:
            fmois[3] = self.init_dead_mf[0]
        if 4 in indices:
            fmois[4] = self.init_live_h_mf
        if 5 in indices:
            fmois[5] = self.init_live_w_mf

        self.fmois = np.array(fmois)
        # Cache initial fmois for efficient reset_to_fuel
        self._init_fmois = self.fmois.copy()

    def _ensure_dfms(self):
        """Create DeadFuelMoisture objects on demand (lazy initialization).

        Called before first moisture update. Uses template cloning for fast
        object creation.
        """
        if self.dfms:
            return  # Already initialized

        need_1hr, need_10hr, need_100hr = self._dfms_needed
        if need_1hr:
            self.dfm1 = DeadFuelMoisture.createDeadFuelMoisture1()
            self.dfms.append(self.dfm1)
        if need_10hr:
            self.dfm10 = DeadFuelMoisture.createDeadFuelMoisture10()
            self.dfms.append(self.dfm10)
        if need_100hr:
            self.dfm100 = DeadFuelMoisture.createDeadFuelMoisture100()
            self.dfms.append(self.dfm100)

    def project_distances_to_surf(self, distances: np.ndarray):
        """Project horizontal distances onto the sloped terrain surface.

        Adjusts the flat-ground distances to each cell edge by accounting for
        the slope and aspect of the terrain. This ensures fire spread distances
        are measured along the actual terrain surface.

        Args:
            distances (np.ndarray): Horizontal distances to cell edges in meters.

        Side Effects:
            - Sets self.distances to the slope-adjusted distances in meters.
        """
        slope_rad = np.deg2rad(self.slope_deg)
        aspect = (self.aspect + 180) % 360
        deltas = np.deg2rad(aspect - np.array(self.directions))
        proj = np.sqrt(np.cos(deltas) ** 2 * np.cos(slope_rad) ** 2 + np.sin(deltas) ** 2)
        self.distances = distances / proj

    def get_ign_params(self, n_loc: int):
        """Calculate fire spread directions and distances from an ignition location.

        Computes the radial spread directions from the specified ignition point
        within the cell to each edge or vertex. Initializes arrays for tracking
        rate of spread and fireline intensity in each direction.

        Args:
            n_loc (int): Ignition location index within the cell.
                0=center, 1-6=vertices, 7-12=edge midpoints.

        Side Effects:
            - Sets self.directions: array of compass directions in degrees.
            - Sets self.distances: slope-adjusted distances to cell boundaries.
            - Sets self.end_pts: coordinates of cell boundary points.
            - Initializes self.avg_ros, self.I_t, self.r_t to zero arrays.
        """
        self._ign_n_loc = n_loc
        self.directions, distances, self.end_pts = UtilFuncs.get_ign_parameters(n_loc, self.cell_size)
        self.project_distances_to_surf(distances)
        self.avg_ros = np.zeros_like(self.directions)
        self.I_t = np.zeros_like(self.directions)
        self.r_t = np.zeros_like(self.directions)

    def _set_wind_forecast(self, wind_speed: np.ndarray, wind_dir: np.ndarray):
        """Store the local forecasted wind speed and direction for the cell.

        Args:
            wind_speed (np.ndarray): Array of forecasted wind speeds in m/s.
            wind_dir (np.ndarray): Array of wind directions in degrees, using
                cartesian convention (0° = blowing toward North/+y,
                90° = blowing toward East/+x).

        Side Effects:
            - Sets self.forecast_wind_speeds with the speed array.
            - Sets self.forecast_wind_dirs with the direction array.
        """ 
        self.forecast_wind_speeds = wind_speed
        self.forecast_wind_dirs = wind_dir

    def _step_moisture(self, weather_stream: WeatherStream, idx: int, update_interval_hr: float = 1):
        """Advance dead fuel moisture calculations by one time interval.

        Updates the moisture content of each dead fuel size class using the
        Nelson dead fuel moisture model. Applies site-specific corrections
        for elevation and calculates local solar radiation.

        Args:
            weather_stream (WeatherStream): Weather data source with temperature,
                humidity, and solar radiation.
            idx (int): Index into the weather stream for current conditions.
            update_interval_hr (float): Time step for moisture update in hours.

        Side Effects:
            - Updates internal state of each DeadFuelMoisture object in self.dfms.
        """
        # Lazy DFM creation — only allocate when first needed
        if not self.dfms:
            self._ensure_dfms()

        elev_ref = weather_stream.ref_elev

        curr_weather = weather_stream.stream[idx]

        bp0 = 0.0218

        t_f_celsius, h_f_frac = apply_site_specific_correction(self, elev_ref, curr_weather)
        solar_radiation = calc_local_solar_radiation(self, curr_weather)

        for i, dfm in enumerate(self.dfms):
            if not dfm.initialized():
                dfm.initializeEnvironment(
                    t_f_celsius, # Intial ambient air temeperature
                    h_f_frac, # Initial ambient air rel. humidity (g/g)
                    solar_radiation, # Initial solar radiation (W/m^2)
                    0, # Initial cumulative rainfall (cm)
                    t_f_celsius, # Initial stick temperature (degrees C)
                    h_f_frac, # Intial stick surface relative humidity (g/g)
                    self.init_dead_mf[0], # Initial stick fuel moisture fraction (g/g)
                    bp0) # Initial stick barometric pressure (cal/cm^3)

            dfm.update_internal(
                update_interval_hr, # Elapsed time since the previous observation (hours)
                t_f_celsius, # Current observation's ambient air temperature (degrees C)
                h_f_frac, # Current observation's ambient air relative humidity (g/g)
                solar_radiation, # Current observation's solar radiation (W/m^2)
                curr_weather.rain + self.local_rain, # Current observation's total cumulative rainfall (cm)
                bp0) # Current observation's stick barometric pressure (cal/cm^3)

    def _update_moisture(self, idx: float, weather_stream: WeatherStream):
        """Update fuel moisture content to the current weather interval.

        Advances the moisture model to the midpoint of the specified weather
        interval and updates the moisture content array. For dynamic fuel models,
        also updates dead herbaceous moisture.

        Args:
            idx (float): Weather interval index (0-based).
            weather_stream (WeatherStream): Weather data source.

        Side Effects:
            - Updates self.fmois array with current moisture fractions.
            - May update dead herbaceous moisture for dynamic fuel models.
        """
        # Make target time the midpoint of the weather interval
        target_time_s = (idx + 0.5) * self._parent().weather_t_step  # Convert index to seconds

        self._catch_up_moisture_to_curr(target_time_s, weather_stream)

        # Update moisture content for each dead fuel moisture class
        self.fmois[0:len(self.dfms)] = [dfm.meanWtdMoisture() for dfm in self.dfms]

        if self.fuel.dynamic:
            # Set dead herbaceous moisture to 1-hr value
            self.fmois[3] = self.fmois[0]

    def _catch_up_moisture_to_curr(self, target_time_s: float, weather_stream: WeatherStream):
        """Advance moisture calculations from last update time to target time.

        Steps through weather intervals between the last moisture update and
        the target time, calling _step_moisture for each interval. Handles
        partial intervals at the start and end.

        Args:
            target_time_s (float): Target simulation time in seconds.
            weather_stream (WeatherStream): Weather data source.

        Side Effects:
            - Updates self.moist_update_time_s to target_time_s.
            - Updates internal state of DeadFuelMoisture objects.
        """
        if self.moist_update_time_s >= target_time_s:
            return

        curr = self.moist_update_time_s
        weather_t_step = self._parent().weather_t_step # seconds

        # Align to next weather boundary
        interval_end_s = ((curr // weather_t_step) + 1) * weather_t_step
        if interval_end_s > target_time_s:
            interval_end_s = target_time_s
        
        if curr < interval_end_s:
            idx = int(curr // weather_t_step)
            dt_hr = (interval_end_s - curr) / 3600  # Convert seconds to hours
            self._step_moisture(weather_stream, idx, update_interval_hr=dt_hr)
            curr = interval_end_s

        # Take full interval steps until we reach the target time
        while curr + weather_t_step <= target_time_s:
            idx = int(curr // weather_t_step)
            self._step_moisture(weather_stream, idx, update_interval_hr=weather_t_step / 3600)
            curr += weather_t_step

        if curr < target_time_s:
            idx = int(curr // weather_t_step)
            dt_hr = (target_time_s - curr) / 3600  # Convert seconds to hours
            self._step_moisture(weather_stream, idx, update_interval_hr=dt_hr)
            curr = target_time_s

        self.moist_update_time_s = curr

    def curr_wind(self) -> tuple:
        """Get the current wind speed and direction at this cell.

        Returns the wind conditions for the current weather interval from the
        cell's local wind forecast. For prediction runs, may trigger forecast
        updates if needed.

        Returns:
            tuple: (wind_speed, wind_direction) where speed is in m/s and
                direction is in degrees using cartesian convention
                (0° = blowing toward North/+y, 90° = blowing toward East/+x).
        """
        parent = self._parent()
        w_idx = parent._curr_weather_idx - parent.sim_start_w_idx

        if parent.is_prediction() and len(self.forecast_wind_speeds) <= w_idx:
            parent._set_prediction_forecast(self)

        # Defensive clamp: prevent index-out-of-bounds on forecast arrays
        if w_idx >= len(self.forecast_wind_speeds):
            import warnings
            warnings.warn(
                f"Cell.curr_wind: w_idx ({w_idx}) >= forecast length "
                f"({len(self.forecast_wind_speeds)}) for cell {self.id}. "
                f"Clamping to last available index."
            )
            w_idx = len(self.forecast_wind_speeds) - 1

        curr_wind = (self.forecast_wind_speeds[w_idx], self.forecast_wind_dirs[w_idx])

        return curr_wind

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
        """Set the canopy base height for the cell.

        Args:
            canopy_base_height (float): Height from ground to bottom of canopy
                in meters.

        Side Effects:
            - Updates the canopy_base_height attribute.
        """
        self.canopy_base_height = canopy_base_height

    
    def _set_canopy_bulk_density(self, canopy_bulk_density: float):
        """Set the canopy bulk density for the cell.

        Args:
            canopy_bulk_density (float): Mass of canopy fuel per unit volume
                in kg/m^3.

        Side Effects:
            - Updates the canopy_bulk_density attribute.
        """
        self.canopy_bulk_density = canopy_bulk_density

    def _set_wind_adj_factor(self):
        """Calculate and set the wind adjustment factor for this cell.

        Computes the wind adjustment factor (WAF) based on fuel type and canopy
        characteristics using equations from Albini and Baughman (1979).

        For non-burnable fuels, WAF is set to 1. For burnable fuels:
        - Unsheltered (canopy cover <= 5%): uses fuel depth
        - Sheltered (canopy cover > 5%): uses canopy height and crown fill

        Side Effects:
            - Sets self.wind_adj_factor.
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

        if self._fuel.burnable:
            self.set_arrays()

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
                - `fire_spread`: Tracks fire spread extent in different directions.
                - `r_ss`: Stores steady-state rates of spread.
                - `I_ss`: Stores steady-state fireline intensity values.
        """
        self._state = state

        if self._state == CellStates.FIRE:
            n_dirs = len(self.directions)
            self.fire_spread = np.zeros(n_dirs)
            self.r_ss = np.zeros(n_dirs)
            self.I_ss = np.zeros(n_dirs)
            self.intersections = np.zeros(n_dirs, dtype=np.bool_)

            self.r_h_ss = 0
            self.I_h_ss = 0

    def add_retardant(self, duration_hr: float, effectiveness: float):
        """Apply long-term fire retardant to this cell.

        Marks the cell as treated with retardant, which reduces the rate of
        spread by the effectiveness factor until the retardant expires.

        Args:
            duration_hr (float): Duration of retardant effectiveness in hours.
            effectiveness (float): Reduction factor for rate of spread (0.0-1.0).
                A value of 0.5 reduces ROS by 50%.

        Raises:
            ValueError: If effectiveness is not in range [0, 1].

        Side Effects:
            - Sets self._retardant to True.
            - Sets self._retardant_factor to (1 - effectiveness).
            - Sets self.retardant_expiration_s to expiration time.
        """
        if effectiveness < 0 or effectiveness > 1:
            raise ValueError(f"Retardant effectiveness must be between 0 and 1 ({effectiveness} passed in)")

        self._retardant = True
        self._retardant_factor = 1 - effectiveness

        self.retardant_expiration_s = self._parent().curr_time_s + (duration_hr * 3600)

    def water_drop_as_rain(self, water_depth_cm: float, duration_s: float = 30):
        """Apply a water drop modeled as equivalent rainfall.

        Simulates water delivery by treating the water as cumulative
        rainfall input to the fuel moisture model. Updates moisture state
        immediately.

        Args:
            water_depth_cm (float): Equivalent water depth in centimeters.
            duration_s (float): Duration of the water application in seconds.

        Side Effects:
            - Updates self.local_rain with accumulated water depth.
            - Advances moisture model through the application period.
            - Updates self.fmois with new moisture fractions.

        Notes:
            - No effect on non-burnable fuel types.
        """
        if not self.fuel.burnable:
            return

        parent = self._parent()
        now_idx = parent._curr_weather_idx
        now_s = parent.curr_time_s
        weather_stream = parent._weather_stream

        # 1. Step moisture from last update to the current time
        self._catch_up_moisture_to_curr(now_s, weather_stream)

        # 2. Increment local rain
        self.local_rain += water_depth_cm

        # 3. Step moisture from current time to end of local rain interval
        interval_hr = duration_s / 3600 # Convert seconds to hours
        self._step_moisture(weather_stream, now_idx, update_interval_hr=interval_hr)

        # 4. Store the time so that next update just updates from current time over a weather interval
        self.moist_update_time_s = now_s + duration_s

         # Update moisture content for each dead fuel moisture class
        self.fmois[0:len(self.dfms)] = [dfm.meanWtdMoisture() for dfm in self.dfms]

        if self.fuel.dynamic:
            # Set dead herbaceous moisture to 1-hr value
            self.fmois[3] = self.fmois[0]

        self.has_steady_state = False

    def water_drop_as_moisture_bump(self, moisture_bump: float):
        """Apply a water drop as a direct fuel moisture increase.

        Simulates water delivery by directly increasing the outer node moisture
        of each dead fuel class, then advances the moisture model briefly to
        allow diffusion.

        Args:
            moisture_bump (float): Moisture fraction to add to fuel surface.

        Side Effects:
            - Increases outer node moisture on each DeadFuelMoisture object.
            - Advances moisture model by 30 seconds.
            - Updates self.fmois with new moisture fractions.

        Notes:
            - No effect on non-burnable fuel types.
        """
        if not self.fuel.burnable:
            return

        if not self.dfms:
            self._ensure_dfms()

        # Catch cell up to current time
        parent = self._parent()
        now_idx = parent._curr_weather_idx
        now_s = parent.curr_time_s
        weather_stream = parent._weather_stream
        self._catch_up_moisture_to_curr(now_s, weather_stream)

        for dfm in self.dfms:
            # Add moisture bump to outer node of each dead fuel moisture class
            dfm.m_w[0] += moisture_bump
            dfm.m_w[0] = max(dfm.m_w[0], dfm.m_wmx)

        self._step_moisture(weather_stream, now_idx, update_interval_hr=30/3600)

        # Update moisture content for each dead fuel moisture class
        self.fmois[0:len(self.dfms)] = [dfm.meanWtdMoisture() for dfm in self.dfms]

        if self.fuel.dynamic:
            # Set dead herbaceous moisture to 1-hr value
            self.fmois[3] = self.fmois[0]

        self.has_steady_state = False

    def water_drop_vw(self, volume_L: float, efficiency: float = 2.5,
                       T_a: float = 20.0):
        """Apply water using Van Wagner (2022) energy-balance model.

        Converts water volume to cooling energy (Eq. 1b) and accumulates in
        water_applied_kJ. Moisture injection is applied during iterate() by
        apply_vw_suppression().

        Args:
            volume_L: Water volume in liters (1 L = 1 kg).
            efficiency: Application efficiency multiplier (Table 4, Van Wagner
                2022). Typical range 2.0–4.0. Default 2.5.
            T_a: Ambient air temperature in °C. Default 20.

        Raises:
            ValueError: If volume_L < 0.
        """
        if volume_L < 0:
            raise ValueError(f"Water volume must be >= 0, got {volume_L}")

        if not self.fuel.burnable:
            return

        from embrs.models.van_wagner_water import volume_L_to_energy_kJ

        self._vw_efficiency = efficiency
        energy_kJ = volume_L_to_energy_kJ(volume_L, T_a)
        self.water_applied_kJ += energy_kJ
        self.has_steady_state = False

    def apply_vw_suppression(self):
        """Apply moisture injection from accumulated Van Wagner water energy.

        Called by iterate() for cells with water_applied_kJ > 0. Computes
        suppression ratio from current fire state, then injects moisture
        toward dead_mx proportionally.

        Uses:
            - I_ss (BTU/ft/min) converted to kW/m
            - fuel.w_n_dead + w_n_live (lb/ft²) converted to kg/m²
            - fire_area_m2 property
            - self._vw_efficiency (stored from water_drop_vw call)
            - heat_to_extinguish_kJ() (Eq. 7b + 10b + Table 4)
            - compute_suppression_ratio()
            - compute_moisture_injection()

        Side Effects:
            - Modifies self.fmois
            - Sets self.has_steady_state = False
        """
        from embrs.models.van_wagner_water import (
            heat_to_extinguish_kJ, compute_suppression_ratio,
            compute_moisture_injection
        )
        from embrs.utilities.unit_conversions import BTU_ft_min_to_kW_m, Lbsft2_to_KiSq

        # Convert fireline intensity from BTU/(ft·min) to kW/m
        # I_t is a numpy array; use head-fire value (max)
        I_btu = max(self.I_t) if len(self.I_t) > 0 else 0.0
        I_kW_m = BTU_ft_min_to_kW_m(I_btu)

        # Dead fuel loading in kg/m² (w_n_dead is in lb/ft²)
        W_1_kg_m2 = Lbsft2_to_KiSq(self.fuel.w_n_dead)

        # Current fire area
        area_m2 = self.fire_area_m2

        # Compute energy needed to extinguish
        heat_needed = heat_to_extinguish_kJ(
            I_kW_m, W_1_kg_m2, area_m2,
            efficiency=self._vw_efficiency
        )

        # Suppression ratio
        ratio = compute_suppression_ratio(self.water_applied_kJ, heat_needed)

        # Inject moisture into dead fuel classes toward extinction
        self.fmois = compute_moisture_injection(
            self.fmois, self.fuel.dead_mx, ratio
        )

        self.has_steady_state = False

    def __str__(self) -> str:
        """Returns a formatted string representation of the cell.

        The string includes the cell's ID, coordinates, elevation, fuel type, and state.

        Returns:
            str: A formatted string representing the cell.
        """
        return (f"(id: {self.id}, {self.x_pos}, {self.y_pos}, {self.elevation_m}, "
                f"type: {self.fuel.name}, "
                f"state: {self.state}")
    
    def calc_hold_prob(self, flame_len_m: float) -> float:
        """Calculate the probability that a fuel break will stop fire spread.

        Uses the Mees et al. (1993) model to estimate the probability that a
        fuel discontinuity (road, firebreak) will prevent fire from crossing.

        Args:
            flame_len_m (float): Flame length at the fire front in meters.

        Returns:
            float: Probability that the fuel break holds (0.0-1.0).
                Returns 0 if no fuel break is present in this cell.

        Notes:
            - Based on Mees, et al. (1993).
        """
        # Mees et. al 1993
        if self._break_width == 0:
            return 0

        x = self._break_width
        m = flame_len_m
        T = 1

        if m == 0:
            return 1

        if m <= 0.61:
            h = 0
        elif 0.61 < m < 2.44:
            h = (m - 0.61)/m
        else:
            h = 0.75

        if x < (h*m):
            prob = 0
        else:
            prob = 1 - np.exp(((x - h*m) * np.log(0.15))/(T*m - h*m))

        return prob

    def calc_cell_area(self) -> float:
        """Calculates the area of the hexagonal cell in square meters.

        The formula for the area of a regular hexagon is:

            Area = (3 * sqrt(3) / 2) * side_length²

        Returns:
            float: The area of the hexagonal cell in square meters.
        """
        area_m2 = (3 * 1.7320508075688772 * self.cell_size ** 2) / 2  # 3*sqrt(3)/2
        return area_m2

    def to_polygon(self) -> Polygon:
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
    
    def to_log_entry(self, time: float) -> CellLogEntry:
        """Create a log entry capturing the cell's current state.

        Generates a structured record of the cell's fire behavior, fuel moisture,
        wind conditions, and other properties for logging and playback.

        Args:
            time (float): Current simulation time in seconds.

        Returns:
            CellLogEntry: Dataclass containing cell state for logging.
        """

        if self.fuel.burnable:
            w_n_dead = self.fuel.w_n_dead
            w_n_dead_start = self.fuel.w_n_dead_nominal
            w_n_live = self.fuel.w_n_live
            dfm_1hr = self.fmois[0]
            dfm_10hr = self.fmois[1]
            dfm_100hr = self.fmois[2]
        else:
            w_n_dead = 0
            w_n_dead_start = 0
            w_n_live = 0
            dfm_1hr = 0
            dfm_10hr = 0
            dfm_100hr = 0

        if len(self.r_t) > 0:
            r_t = np.max(self.r_t)
            I_ss = np.max(self.I_ss)
        else:
            r_t = 0
            I_ss = 0

        wind_speed, wind_dir = self.curr_wind()

        entry = CellLogEntry(
            timestamp=time,
            id=self.id,
            x=self.x_pos,
            y=self.y_pos,
            fuel=self.fuel.model_num,
            state=self.state,
            crown_state=self._crown_status,
            w_n_dead=w_n_dead,
            w_n_dead_start=w_n_dead_start,
            w_n_live=w_n_live,
            dfm_1hr=dfm_1hr,
            dfm_10hr=dfm_10hr,
            dfm_100hr=dfm_100hr,
            ros=r_t,
            I_ss=I_ss,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            retardant=self._retardant,
            arrival_time=self._arrival_time
        )

        return entry

    def iter_neighbor_cells(self) -> Iterator[Cell]:
        """Iterate over neighboring Cell objects.

        Yields each adjacent cell by looking up neighbor IDs in the parent
        simulation's cell_dict.

        Yields:
            Cell: Each neighboring cell object.

        Notes:
            - Returns immediately if parent reference is None.
        """
        parent = self._parent()

        if parent is None:
            return
        
        cell_dict = parent.cell_dict
        for nid in self._neighbors.keys():
            yield cell_dict[nid]

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

    def __getstate__(self) -> dict:
        """Prepare cell state for pickling.

        Excludes the weak reference to parent which cannot be pickled.

        Returns:
            dict: Cell state dictionary with _parent set to None.
        """
        state = self.__dict__.copy()
        # Remove weak reference - will be restored later
        state['_parent'] = None
        return state

    def __setstate__(self, state):
        """Restore cell state after unpickling.

        Args:
            state (dict): Cell state dictionary from __getstate__.
        """
        self.__dict__.update(state)
        # Parent will be set later via cell.set_parent(predictor)

    @property
    def col(self) -> int:
        """Column index of the cell in the simulation grid."""
        return self._col

    @property
    def row(self) -> int:
        """Row index of the cell in the simulation grid."""
        return self._row

    @property
    def cell_size(self) -> float:
        """Size of the cell in meters.
        
        Measured as the side length of the hexagon.
        """
        return self._cell_size

    @property
    def cell_area(self) -> float:
        """Area of the cell in square meters."""
        return self._cell_area

    @property
    def fire_area_m2(self) -> float:
        """Fire area within cell via trapezoidal polar integration.

        Computes A = (1/2) Σ Δθ_i · (r_i² + r_{i+1}²) / 2 over the
        discretized spread directions.  For center ignition (n_loc=0) the
        sum includes a closing segment from the last direction back to the
        first.  No area_fraction is needed — the angular range of the
        directions already encodes whether the fire covers the full cell,
        a half-cell, or a 60° sector.

        Clamped to cell_area as an upper bound.

        Returns 0.0 for non-burning cells, cells with no spread data, or
        cells where _ign_n_loc has not yet been set.
        """
        if self._state != CellStates.FIRE or len(self.fire_spread) == 0:
            return 0.0
        if self._ign_n_loc is None:
            return 0.0

        fs = self.fire_spread
        dirs = self.directions
        n = len(fs)

        area = 0.0
        for i in range(n - 1):
            dth = (dirs[i + 1] - dirs[i]) % 360
            area += dth * (fs[i] * fs[i] + fs[i + 1] * fs[i + 1])

        if self._ign_n_loc == 0:
            dth = (dirs[0] - dirs[-1] + 360) % 360
            area += dth * (fs[-1] * fs[-1] + fs[0] * fs[0])

        # 0.25 = (1/2 from polar integral) × (1/2 from trapezoidal average)
        # π/180 converts the degree-valued Δθ to radians
        area *= 0.25 * (np.pi / 180)

        return min(area, self._cell_area)

    @property
    def x_pos(self) -> float:
        """X-coordinate of the cell center in meters.

        Increases left to right in the visualization.
        """
        return self._x_pos

    @property
    def y_pos(self) -> float:
        """Y-coordinate of the cell center in meters.

        Increases bottom to top in the visualization.
        """
        return self._y_pos

    @property
    def elevation_m(self) -> float:
        """Elevation of the cell in meters."""
        return self._elevation_m

    @property
    def fuel(self) -> Fuel:
        """Fuel model for this cell.

        Can be any Anderson or Scott-Burgan fuel model.
        """
        return self._fuel

    @property
    def state(self) -> CellStates:
        """Current fire state of the cell (FUEL, FIRE, or BURNT)."""
        return self._state

    @property
    def neighbors(self) -> dict:
        """Dictionary of adjacent cells.

        Keys are neighbor cell IDs, values are (dx, dy) tuples indicating
        the column and row offset from this cell to the neighbor.
        """
        return self._neighbors

    @property
    def burnable_neighbors(self) -> dict:
        """Dictionary of adjacent cells that are in a burnable state.

        Same format as neighbors: keys are cell IDs, values are (dx, dy) offsets.
        """
        return self._burnable_neighbors