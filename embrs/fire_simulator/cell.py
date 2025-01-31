"""Representation of the discrete cells that make up the fire sim.

.. autoclass:: Cell
    :members:

"""
import numpy as np
from shapely.geometry import Polygon

from embrs.utilities.fire_util import CellStates, FireTypes
from embrs.utilities.fire_util import HexGridMath as hex
from embrs.utilities.fuel_models import Anderson13

import copy

class Cell:
    """Class representation of the discrete cells that make up the fire sim. This holds the state
    of each cell as well as the relevant fire spread parameters. Cells are regular hexagons in the
    "point-up" orientation.

    :param id: unique id of the cell within the sim
    :type id: int
    :param col: column index of the cell within the :py:attr:`~fire_simulator.fire.FireSim.cell_grid`
    :type col: int
    :param row: row index of the cell within the :py:attr:`~fire_simulator.fire.FireSim.cell_grid`
    :type row: int
    :param cell_size: size of the cell, edge length of the hexagons, units: meters
    :type cell_size: float
    :param z: elevation of the cell measured in meters, defaults to 0.0
    :type z: float, optional
    :param fuel_type: type of fuel present at the cell, can be any of the 13 Anderson FBFMs,
                      defaults to Fuel(1)
    :type fuel_type: :class:`~fire_simulator.fuel.Fuel`, optional
    """

    def __init__(self, id: int, col: int, row: int, cell_size: float, z = 0.0, aspect = 0.0, slope_deg = 0.0, fuel_type=Anderson13(1)):
        """Constructor method to initialize a cell instance. Calculates x,y position in Cartesian
        plane.
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

        # Variable that tracks which type of fire a burning cell is
        self._fire_type = -1

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

        # Store initial state for potential reset
        self.initial_state = None

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

    def get_copy_of_state(self):
        """_summary_

        Returns:
            Cell: _description_
        """
        return copy.deepcopy(self)

    def reset(self) -> CellStates:
        """_summary_

        Raises:
            ValueError: _description_

        Returns:
            CellStates: _description_
        """

        if self.initial_state:
            return self.initial_state
        else:
            raise ValueError("Initial state not stored.")

    def set_real_time_vals(self):
        """_summary_
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
        """_summary_

        Args:
            wind_speed (np.ndarray): _description_
            wind_dir (np.ndarray): _description_
        """
        self.wind_forecast = [(speed, dir) for speed, dir in zip(wind_speed, wind_dir)]
        self.curr_wind = self.wind_forecast[0] # Note: (m/s, degrees)

    def _update_wind(self, idx: int):
        """_summary_

        Args:
            idx (int): _description_
        """
        self.curr_wind = self.wind_forecast[idx]

    def _set_elev(self, z: float):
        """Set the elevation of a cell

        :param z: elevation in meters
        :type z: float
        """

        self._z = z

    def _set_slope(self, slope: float):
        """_summary_

        Args:
            slope (float): _description_
        """
        self.slope_deg = slope

    def _set_aspect(self, aspect: float):
        """_summary_

        Args:
            aspect (float): _description_
        """
        self.aspect = aspect

    def _set_fuel_content(self, fuel_content: float):
        """Set the fuel content remaining in a cell.

        :param fuel_content: amount of fuel remaining in a cell, ranging from 0 to 1
        :type fuel_content: float
        """
        self._fuel_content = fuel_content

    def _set_fuel_type(self, fuel_type: Anderson13):
        """Set the fuel type of a cell.

        :param fuel_type: Fuel type, one of the 13 Anderson FBFMs
        :type fuel_type: :class:`~fire_simulator.fuel.Fuel`
        """
        self._fuel_type = fuel_type

    def _set_state(self, state: CellStates):
        """Set the state of the cell to :py:attr:`CellStates.FUEL`, :py:attr:`CellStates.FIRE`,
        or :py:attr:`CellStates.BURNT`

        :param state: state of the cell
        :type state: :class:`~utilities.fire_util.CellStates`
        """
        self._state = state

        if self._state == CellStates.FIRE:
            self.fire_spread = np.zeros(len(self.directions))
            self.r_prev_list = np.zeros(len(self.directions))
            self.r_ss = np.zeros(len(self.directions))
            self.I_ss = np.zeros(len(self.directions))
            
    def _set_fire_type(self, fire_type: FireTypes):
        """Set the type of fire at a cell

        :param fire_type: Fire type to set, either :py:attr:`FireTypes.WILD` or :py:attr:`FireTypes.PRESCRIBED`
        :type fire_type: :class:`~utilities.fire_util.FireTypes`
        """
        self._fire_type = fire_type

    def _set_dead_m(self, dead_m: float):
        """Set the dead fuel moisture at a cell.
        
        Higher fuel moisture will slow the propagation of fire

        :param dead_m: dead fuel moisture as a fraction (0 to 1)
        :type dead_m: float
        """
        self._dead_m = dead_m

    def __str__(self):
        """To string method for cells.

        :return: formatted string representing the cell
        :rtype: str
        """
        return (f"(id: {self.id}, {self.x_pos}, {self.y_pos}, {self.z}, "
                f"type: {self.fuel_type.name}, "
                f"state: {self.state}")

    def calc_cell_area(self):
        """Calculate the area of a cell in meters squared

        :return: area of a cell given in meters squared
        :rtype: float
        """
        area_m2 = (3 * np.sqrt(3) * self.cell_size ** 2) / 2
        return area_m2

    def to_polygon(self):
        """Generate a Shapely polygon representation of the hexagonal cell.

        :return: A Shapely polygon representing the hexagonal cell
        :rtype: shapely.geometry.Polygon
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
        """Returns a dictionary of cell data at the current instant.
        Contains all data necessary to capture changes to the state of a cell over time. 
        Used to create logs of the sim for visualization or post-processing use.

        :return: **Dictionary with the following fields:**

                - "id": (int) representing the id of the cell object
                - "state": (:class:`~utilities.fire_util.CellStates`) representing the current state of the cell
                - "fuel_content": (float) representing the amount of fuel remaining in the cell
        
                **If the cell's state is on fire the dictionary will also include:**
                    
                - "fire_type": (:class:`~utilities.fire_util.FireTypes`) representing type of fire in the cell

        :rtype: dict
        """
        cell_data = {
            "id": self.id,
            "state": self._state,
        }

        if self.state == CellStates.FIRE:
            cell_data['fire_type'] = self._fire_type

        return cell_data

    def get_spread_directions(self, ignited_pos):
        return self.DIRECTION_MAP.get(ignited_pos)

    # ------ Compare operators overloads ------ #
    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
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
    def fuel_type(self) -> Anderson13:
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
    def fire_type(self) -> FireTypes:
        """Current type of fire burning at the cell.
        
        Either :py:attr:`FireTypes.PRESCRIBED` or :py:attr:`FireTypes.WILD`, `-1` if not currently in state :py:attr:`CellStates.FIRE`.
        """
        return self._fire_type

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