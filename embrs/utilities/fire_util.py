"""Various sets of constants and helper functions useful throughout the codebase

.. autoclass:: UtilFuncs
    :members:

.. autoclass:: CellStates
    :members:

.. autoclass:: FuelConstants
    :members:

.. autoclass:: RoadConstants
    :members:

.. autoclass:: HexGridMath
    :members:

"""

from typing import Tuple
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from functools import lru_cache

cell_type = np.dtype([
    ('id', np.int32),
    ('state', np.int32),
    ('fuelType', np.int32),
    ('fuelContent', np.float32),
    ('moisture', np.float32),
    ('position', [('x', np.float32), ('y', np.float32), ('z', np.float32)]),
    ('indices', [('i', np.int32), ('j', np.int32)]),
    ('changed', np.bool_)
])

action_type = np.dtype([
    ('type', np.int32),
    ('pos', [('x', np.float32), ('y', np.float32)]),
    ('time', np.float32),
    ('value', np.float32)
])

class SpreadDecomp:
    """_summary_ TODO: need to carefully think about how to best document this
    """

    self_loc_to_neighbor_loc_mapping = {
        1: [(7, 'A')],
        2: [(6, 'A'), (10, 'B')],
        3: [(9, 'B')],
        4: [(8, 'B'), (12, 'C')],
        5: [(11, 'C')],
        6: [(10, 'C'), (2, 'D')],
        7: [(1, 'D')],
        8: [(12, 'D'), (4, 'E')],
        9: [(3, 'E')],
        10: [(2, 'E'), (6, 'F')],
        11: [(5, 'F')],
        12: [(4, 'F'), (8, 'A')]
    }

class CrownStatus:
    # Crown statuses
    NONE, PASSIVE, ACTIVE = 0, 1, 2

class CanopySpecies:
    species_names = {
        0: "Engelmann spruce",
        1: "Douglas fir",
        2: "Western hemlock",
        3: "Ponderosa pine",
        4: "White pine",
        5: "Grand fir",
        6: "Longleaf pine",
        7: "Pond pine",
        8: "Loblolly pine"
    }

    species_ids = {
        "Engelmann spruce": 0,
        "Douglas fir": 1,
        "Western hemlock": 2,
        "Ponderosa pine": 3,
        "White pine": 4,
        "Grand fir": 5,
        "Longleaf pine": 6,
        "Pond pine": 7,
        "Loblolly pine": 8
    }

    # Row of matrix corresponds to the species id
    properties = np.array([
        [15.7, 0.451, 12.6, 0.256],
        [15.7, 0.451, 10.7, 0.278],
        [15.7, 0.451, 6.3, 0.249],
        [12.9, 0.453, 12.6, 0.256],
        [12.9, 0.453, 10.7, 0.278],
        [16.5, 0.515, 10.7, 0.278],
        [2.71, 1.0, 11.9, 0.389],
        [2.71, 1.0, 7.91, 0.344],
        [2.71, 1.0, 13.5, 0.544]
    ])

class CellStates:
    """Enumeration of the possible cell states.

    Attributes:
        - **BURNT** (int): Represents a cell that has been burnt and has no fuel remaining.
        - **FUEL** (int): Represents a cell that still contains fuel and is not on fire.
        - **FIRE** (int): Represents a cell that is currently on fire.
    """
    # Cell States:
    BURNT, FUEL, FIRE = 0, 1, 2

# TODO: all this should be with the fuel model class
class FuelConstants:
    """Various values and dictionaries pertaining to modelling of fuel types.

    Attributes:
        - **burnout_thresh** (float): fuel content which dictates what is considered to be a burned out cell.
        - **fbfm_13_keys** (list): list of ints corresponding to each of Anderson's 13 FBFMs.
        - **fuel_names** (dict): dictionary where keys are ints for each FBFM and values are the names of each fuel type.
        - **fuel_type_revers_lookup** (dict): dictionary where keys are the fuel type names and values are the ints for each FBFM.
        - **dead_fuel_moisture_ext_table** (dict): dictionary of each fuel type's dead fuel moisture of extinction.
        - **fuel_color_mapping** (dict): dictionary mapping each fuel type to the display color for visualizations.

    """
    burnout_thresh = 0.01

    # Valid keys for the FBFM 13 fuel model
    fbfm_13_keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 91, 92, 93, 98, 99]

    # Dictionary of fuel number to name
    fuel_names = {1: "Short grass", 2: "Timber grass", 3: "Tall grass", 4: "Chaparral",
                5: "Brush", 6: "Hardwood slash", 7: "Southern rough", 8: "Closed timber litter",
                9: "Hardwood litter", 10: "Timber litter", 11: "Light logging slash",
                12: "Medium logging slash", 13: "Heavy logging slash", 91: 'Urban', 92: 'Snow/ice',
                93: 'Agriculture', 98: 'Water', 99: 'Barren'}

    fuel_type_reverse_lookup = {"Short grass": 1, "Timber grass": 2, "Tall grass": 3, "Chaparral": 4,
                "Brush": 5, "Hardwood slash": 6, "Southern rough": 7 , "Closed timber litter": 8,
                "Hardwood litter": 9, "Timber litter": 10, "Light logging slash": 11,
                "Medium logging slash": 12, "Heavy logging slash": 13, 'Urban': 91, 'Snow/ice': 92,
                'Agriculture': 93, 'Water': 98, 'Barren': 99}

    # based on the values in the Anderson Fuel model
    dead_fuel_moisture_ext_table = {1: 0.12, 2: 0.15, 3: 0.25, 4: 0.20, 5: 0.20, 6: 0.25, 7: 0.40,
                                    8: 0.30, 9: 0.25, 10: 0.25, 11: 0.15, 12: 0.20, 13: 0.25,
                                    91: 0.1, 92: 1, 93: 1, 98: 1, 99: 1}

    # Color mapping for each fuel type
    fuel_color_mapping = {1: 'xkcd:pale green', 2:'xkcd:lime', 3: 'xkcd:bright green',
                        4: 'xkcd:teal', 5: 'xkcd:bluish green', 6: 'xkcd:greenish teal',
                        7: 'xkcd:light blue green', 8: 'xkcd:pale olive' , 9: 'xkcd:olive',
                        10: 'xkcd:light forest green', 11: 'xkcd:bright olive',
                        12: 'xkcd:tree green', 13: 'xkcd:avocado green', 91: 'xkcd:ugly purple',
                        92: 'xkcd:pale cyan' , 93: "xkcd:perrywinkle", 98: 'xkcd:water blue',
                        99: 'xkcd:black', -100: 'xkcd:red'}


class RoadConstants:
    # Road types
    major_road_types = ['motorway', 'trunk' , 'primary', 'secondary',
                        'tertiary', 'residential']

    # Standard lane widths for each road type
    lane_widths_m = {
        'motorway': 3.66,
        'big_motorway': 3.66,
        'trunk': 3.66,
        'primary': 3.66,
        'secondary': 3.05,
        'tertiary': 3.05,
        'residential': 3.05
    }

    # Standard shoulder width for each road type (total shoulder width)
    shoulder_widths_m = {
        'motorway': 4.27,
        'big_motorway': 6.10, # For motorways with more than 2 lanes
        'trunk': 4.27,
        'primary': 4.27,
        'secondary': 1.83,
        'tertiary': 1.83,
        'residential': 1.83
    }

    # Standard number of lanes for each road type
    default_lanes = 2
    
    road_color_mapping = {
        'motorway': '#4B0082',  # Indigo
        'big_motorway': '#4B0082',
        'trunk': '#800080',    # Purple
        'primary': '#9400D3',  # DarkViolet
        'secondary': '#9932CC',  # DarkOrchid
        'tertiary': '#BA55D3',  # MediumOrchid
        'residential': '#EE82EE',  # Violet
    }

class HexGridMath:
    """Data structures to help with handling cell neighbors in a hexagonal grid.

    Attributes:
        - **even_neighborhood** (list): list of relative indices of a cell's neighbors, for even rows.
        - **even_neighbor_letters** (dict):
        - **odd_neighborhood** (list): list of relative indices of a cell's neighbors, for odd rows.
        - **even_neighbor_letters** (dict):

    """
    even_neighborhood = [(-1,1), (0, 1), (1,0), (0, -1), (-1, -1), (-1,0)]
    even_neighbor_letters = {'F': (-1, 1),
                             'A':(0, 1),
                             'B':(1, 0),
                             'C': (0, -1),
                             'D': (-1, -1),
                             'E': (-1, 0)}

    odd_neighborhood = [(1,0), (1,1), (0,1), (-1,0), (0,-1), (1, -1)]
    odd_neighbor_letters = {'B': (1, 0),
                            'A': (1, 1),
                            'F': (0, 1),
                            'E': (-1, 0),
                            'D': (0, -1),
                            'C': (1, -1)}

class UtilFuncs:
    """Various utility functions that are useful across numerous files.
    """
    def get_indices_from_xy(x_m: float, y_m: float, cell_size: float, grid_width: int,
                            grid_height: int) -> Tuple[int, int]:
        """Get the row and column indices in a backing array of a cell containing the point
        (x_m, y_m). 
        
        Does not require a :class:`~fire_simulator.fire.FireSim` object, uses 'cell_size' and the size of the array
        to calculate indices.

        :param x_m: x position in meters where indices should be found
        :type x_m: float
        :param y_m: y position in meters where indices should be found
        :type y_m: float
        :param cell_size: cell size in meters, measured as the distance across two parallel sides
                          of a regular hexagon
        :type cell_size: float
        :param grid_width: number of columns in the backing array of interest
        :type grid_width: int
        :param grid_height: number of rows in the backing array of interest
        :type grid_height: int
        :raises ValueError: if x or y inputs are out of bounds for the array constructed by
                            'cell_size', 'grid_width', and 'grid_height'
        :return: tuple containing [row, col] indices at the point (x_m, y_m)
        :rtype: Tuple[int, int]
        """
        row = int(y_m // (cell_size * 1.5))

        if row % 2 == 0:
            col = int(x_m // (cell_size * np.sqrt(3))) + 1
        else:
            col = int((x_m // (cell_size * np.sqrt(3))) - 0.5) + 1

        if col < 0 or row < 0 or row >= grid_height or col >= grid_width:
            msg = (f'Point ({x_m}, {y_m}) is outside the grid. '
                f'Column: {col}, Row: {row}, '
                f'simSize: ({grid_height} , {grid_width})')
            raise ValueError(msg)

        return row, col

    def get_time_str(time_s: int, show_sec = False) -> str:
        """Returns a formatted time string in m-h-s format from the time in seconds.
        
        Useful for generating readable display of the time.

        :param time_s: time value in seconds
        :type time_s: int
        :param show_sec: set to `True` if seconds should be displayed, `False` if not, defaults to `False`
        :type show_sec: bool, optional
        :return: formatted time string in h-m-s format
        :rtype: str
        """
        hours = int(time_s // 3600)
        minutes = int((time_s % 3600) // 60)

        if show_sec:
            seconds = int((time_s % 3600) % 60)

            if hours > 0:
                result = f"{hours} h {minutes} min {seconds} s"
            elif minutes > 0:
                result = f"{minutes} min {seconds} s"
            else:
                result = f"{seconds} s"
            return result

        if hours > 0:
            result = f"{hours} h {minutes} min"
        else:
            result = f"{minutes} min"
        return result

    def get_dominant_fuel_type(fuel_map: np.ndarray) -> int:
        """Finds the most commonly occurring fuel type within a fuel map.

        :param fuel_map: Fuel map for a region
        :type fuel_map: np.ndarray
        :return: Integer representation of the dominant fuel type. 
            See :py:attr:`~utilities.fire_util.FuelConstants.fuel_names`
        :rtype: int
        """
        counts = np.bincount(fuel_map.ravel())

        return np.argmax(counts)

    def get_cell_polygons(cells: list) -> list:
        """Converts a list of cell objects into the minimum number of :py:attr:`shapely.Polygon`
        required to describe all of them

        :param cells: list of :class:`~fire_simulator.cell.Cell` objects to be converted
        :type cells: list
        :return: list of :py:attr:`shapely.Polygon` representing the cells
        :rtype: list
        """

        if not cells:
            return None

        polygons = [cell.polygon for cell in cells]

        merged_polygon = unary_union(polygons)

        if isinstance(merged_polygon, MultiPolygon):
            return list(merged_polygon.geoms)

        return [merged_polygon]

    @staticmethod
    def hexagon_vertices(x: float, y: float, s: float) -> list:
        """Calculates the locations of each of a hexagons vertices with center (x,y) and side
        length s.

        :param x: x location of hexagon center
        :type x: float
        :param y: y location of hexagon center
        :type y: float
        :param s: length of hexagon sides
        :type s: float
        :return: list of (x,y) points representing the hexagon's vertices
        :rtype: list
        """
        vertices = [
            (x, y + s),
            (x + s * np.sqrt(3) / 2, y + s / 2),
            (x + s * np.sqrt(3) / 2, y - s / 2),
            (x, y - s),
            (x - s * np.sqrt(3) / 2, y - s / 2),
            (x - s * np.sqrt(3) / 2, y + s / 2)
        ]
        return vertices
    

    def get_dist(edge_loc: int, idx_diff: int, cell_size: float) -> float:
        """_summary_ # TODO: need to carefully think about how to document this

        Args:
            edge_loc (int): _description_
            idx_diff (int): _description_
            cell_size (float): _description_

        Returns:
            float: _description_
        """

        # Keys equal to difference between indices
        odd_loc_distance_dict = {
            1: cell_size / 2,
            2: (np.sqrt(3)/2) * cell_size, # Law of sines
            3: (np.sqrt(7)/2) * cell_size, # Law of cosines
            4: (3 * cell_size) / 2,
            5: (np.sqrt(13) * cell_size) / 2, # Law of cosines
            6: np.sqrt(3) * cell_size
        }

        even_loc_distance_dict = {
            2: cell_size,
            3: (np.sqrt(7)/2) * cell_size,
            4: np.sqrt(3) * cell_size,
            5: (np.sqrt(13) * cell_size) / 2,
            6: 2 * cell_size,
        }

        # Handle case where ignition starts at center
        if edge_loc == 0:
            if idx_diff % 2 == 0:
                return cell_size
            else:
                return (np.sqrt(3) * cell_size)/2

        elif edge_loc %  2 == 0:
            return even_loc_distance_dict[idx_diff]

        else:
            return odd_loc_distance_dict[idx_diff]

    @lru_cache
    def get_ign_parameters(edge_loc: int, cell_size: float) -> Tuple[np.ndarray, np.ndarray, list]:
        """_summary_# TODO: need to think carefully about how to document this

        Args:
            edge_loc (int): _description_
            cell_size (float): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray, list]: _description_
        """

        if edge_loc == 0:
            # Ignition is at the center of cell
            start_angle = 30
            end_angle = 360

            directions = np.linspace(start_angle, end_angle, 12)

            start_end_point = 1
            end_end_point = 12

        elif edge_loc % 2 == 0:
            # Ignition is at a corner point
            start_angle = (30 * edge_loc + 120) % 360
            end_angle = (start_angle + 120)

            directions = np.linspace(start_angle, end_angle, 9)

            start_end_point = (edge_loc + 2) % 12 or 12
            end_end_point = (start_end_point + 8) % 12 or 12

        else:
            # Ignition is along an edge
            start_angle = (30 * edge_loc + 90) % 360
            end_angle = (start_angle + 180)

            directions = np.linspace(start_angle, end_angle, 11)

            start_end_point = (edge_loc + 1) % 12 or 12
            end_end_point = (12 + (edge_loc - 1)) % 12 or 12

        directions = np.array([direction % 360 for direction in directions])

        if end_end_point < start_end_point:
            self_end_points = np.concatenate([
                np.arange(start_end_point, 13),
                np.arange(1, end_end_point + 1)
            ])
        
        else:
            self_end_points = np.arange(start_end_point, end_end_point + 1)

        end_points = []
        distances = []

        for end_point in self_end_points:

            idx_diff = np.abs(end_point - edge_loc)

            if idx_diff > 6:
                idx_diff = 12 - idx_diff


            dist = UtilFuncs.get_dist(edge_loc, idx_diff, cell_size)

            distances.append(dist)

            neighbor_locs = SpreadDecomp.self_loc_to_neighbor_loc_mapping[end_point]
            end_points.append(neighbor_locs)

        return np.array(directions), np.array(distances), end_points
