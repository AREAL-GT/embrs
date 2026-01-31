"""Core constants and utility functions for fire simulation.

This module provides essential data structures, constants, and helper functions
used throughout the EMBRS codebase, including cell states, hexagonal grid math,
road constants, and various utility functions.

Classes:
    - CellStates: Enumeration of cell states (BURNT, FUEL, FIRE).
    - CrownStatus: Crown fire status constants.
    - CanopySpecies: Canopy species definitions and properties.
    - RoadConstants: Road type definitions and standard dimensions.
    - HexGridMath: Hexagonal grid neighbor calculations.
    - SpreadDecomp: Fire spread direction decomposition mappings.
    - UtilFuncs: General utility functions.

.. autoclass:: CellStates
    :members:

.. autoclass:: CrownStatus
    :members:

.. autoclass:: CanopySpecies
    :members:

.. autoclass:: RoadConstants
    :members:

.. autoclass:: HexGridMath
    :members:

.. autoclass:: SpreadDecomp
    :members:

.. autoclass:: UtilFuncs
    :members:
"""

from typing import Tuple, List, Optional, TYPE_CHECKING
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from functools import lru_cache

if TYPE_CHECKING:
    from embrs.fire_simulator.cell import Cell

class SpreadDecomp:
    """Mapping for fire spread direction decomposition across cell boundaries.

    Maps spread endpoint locations on a cell's boundary to corresponding
    entry points on neighboring cells. Used for tracking fire propagation
    between adjacent hexagonal cells.

    Attributes:
        self_loc_to_neighbor_loc_mapping (dict): Maps edge location indices (1-12)
            to list of tuples (neighbor_edge_loc, neighbor_letter) indicating
            where fire enters the adjacent cell.
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
    """Enumeration of crown fire status values.

    Attributes:
        NONE (int): No crown fire activity (value: 0).
        PASSIVE (int): Passive crown fire (value: 1).
        ACTIVE (int): Active crown fire (value: 2).
    """
    NONE, PASSIVE, ACTIVE = 0, 1, 2


class CanopySpecies:
    """Canopy species definitions and properties for spotting calculations.

    Contains species identification mappings and physical properties used
    in firebrand lofting and spotting distance calculations.

    Attributes:
        species_names (dict): Maps species ID (int) to species name (str).
        species_ids (dict): Maps species name (str) to species ID (int).
        properties (np.ndarray): Physical properties matrix where each row
            corresponds to a species ID. Columns are species-specific parameters
            for spotting calculations. TODO:verify column definitions and units. (find the source and cite it here)
    """

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
    """Enumeration of possible cell states.

    Attributes:
        BURNT (int): Cell has been burnt, no fuel remaining (value: 0).
        FUEL (int): Cell contains fuel and is not on fire (value: 1).
        FIRE (int): Cell is currently burning (value: 2).
    """

    BURNT, FUEL, FIRE = 0, 1, 2

class RoadConstants:
    """Constants for road types imported from OpenStreetMap.

    Defines standard road classifications, lane widths, shoulder widths,
    and visualization colors for roads used as fuel breaks in simulation.

    Attributes:
        major_road_types (list): List of supported OSM road type strings.
        lane_widths_m (dict): Lane width in meters for each road type.
        shoulder_widths_m (dict): Total shoulder width in meters for each road type.
        default_lanes (int): Default number of lanes (2).
        road_color_mapping (dict): Hex color codes for visualization by road type.
        TODO:verify the source for the lane widths
    """

    major_road_types = ['motorway', 'trunk', 'primary', 'secondary',
                        'tertiary', 'residential']

    lane_widths_m = {
        'motorway': 3.66,
        'big_motorway': 3.66,
        'trunk': 3.66,
        'primary': 3.66,
        'secondary': 3.05,
        'tertiary': 3.05,
        'residential': 3.05
    }

    shoulder_widths_m = {
        'motorway': 4.27,
        'big_motorway': 6.10,
        'trunk': 4.27,
        'primary': 4.27,
        'secondary': 1.83,
        'tertiary': 1.83,
        'residential': 1.83
    }

    default_lanes = 2

    road_color_mapping = {
        'motorway': '#4B0082',
        'big_motorway': '#4B0082',
        'trunk': '#800080',
        'primary': '#9400D3',
        'secondary': '#9932CC',
        'tertiary': '#BA55D3',
        'residential': '#EE82EE',
    }

class HexGridMath:
    """Data structures for hexagonal grid neighbor calculations.

    Provides mappings for finding neighbors of cells in a point-up hexagonal
    grid. Even and odd rows have different neighbor offsets due to the
    staggered grid layout.

    Neighbor letters (A-F) identify the six directions around a hexagon,
    starting from the upper-right and proceeding clockwise.

    Attributes:
        even_neighborhood (list): Relative (row, col) offsets for neighbors
            of cells in even-numbered rows.
        even_neighbor_letters (dict): Maps letter (A-F) to (row, col) offset
            for even rows.
        even_neighbor_rev_letters (dict): Maps (row, col) offset to letter
            for even rows.
        odd_neighborhood (list): Relative (row, col) offsets for neighbors
            of cells in odd-numbered rows.
        odd_neighbor_letters (dict): Maps letter (A-F) to (row, col) offset
            for odd rows.
        odd_neighbor_rev_letters (dict): Maps (row, col) offset to letter
            for odd rows.
    """

    even_neighborhood = [(-1, 1), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 0)]
    even_neighbor_letters = {'F': (-1, 1),
                             'A': (0, 1),
                             'B': (1, 0),
                             'C': (0, -1),
                             'D': (-1, -1),
                             'E': (-1, 0)}

    even_neighbor_rev_letters = {(-1, 1): 'F',
                                (0, 1): 'A',
                                (1, 0): 'B',
                                (0, -1): 'C',
                                (-1, -1): 'D',
                                (-1, 0): 'E'}

    odd_neighborhood = [(1, 0), (1, 1), (0, 1), (-1, 0), (0, -1), (1, -1)]
    odd_neighbor_letters = {'B': (1, 0),
                            'A': (1, 1),
                            'F': (0, 1),
                            'E': (-1, 0),
                            'D': (0, -1),
                            'C': (1, -1)}

    odd_neighbor_rev_letters = {(1, 0): 'B',
                                (1, 1): 'A',
                                (0, 1): 'F',
                                (-1, 0): 'E',
                                (0, -1): 'D',
                                (1, -1): 'C'}

class UtilFuncs:
    """Collection of utility functions used across the codebase."""

    def get_indices_from_xy(x_m: float, y_m: float, cell_size: float, grid_width: int,
                            grid_height: int) -> Tuple[int, int]:
        """Get grid indices for a point in spatial coordinates.

        Calculates the (row, col) indices in the cell_grid array for the cell
        containing the point (x_m, y_m). Does not require a FireSim object.

        Args:
            x_m (float): X position in meters.
            y_m (float): Y position in meters.
            cell_size (float): Cell size in meters (hexagon side length).
            grid_width (int): Number of columns in the grid.
            grid_height (int): Number of rows in the grid.

        Returns:
            Tuple[int, int]: (row, col) indices of the cell containing the point.

        Raises:
            ValueError: If the point is outside the grid bounds.
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

    def get_time_str(time_s: int, show_sec: bool = False) -> str:
        """Format a time value in seconds as a human-readable string.

        Args:
            time_s (int): Time value in seconds.
            show_sec (bool): If True, include seconds in output. Defaults to False.

        Returns:
            str: Formatted time string (e.g., "2 h 30 min" or "2 h 30 min 15 s").
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
        """Find the most common fuel type in a fuel map.

        Args:
            fuel_map (np.ndarray): 2D array of fuel type IDs.

        Returns:
            int: Fuel type ID that occurs most frequently.
        """
        counts = np.bincount(fuel_map.ravel())

        return np.argmax(counts)

    def get_cell_polygons(cells: List["Cell"]) -> Optional[List[Polygon]]:
        """Merge cell polygons into minimal covering polygons.

        Args:
            cells (List[Cell]): List of Cell objects to convert.

        Returns:
            Optional[List[Polygon]]: List of shapely Polygon objects representing
                the merged cells, or None if cells is empty.
        """
        if not cells:
            return None

        polygons = [cell.polygon for cell in cells]

        merged_polygon = unary_union(polygons)

        if isinstance(merged_polygon, MultiPolygon):
            return list(merged_polygon.geoms)

        return [merged_polygon]

    @staticmethod
    def hexagon_vertices(x: float, y: float, s: float) -> List[Tuple[float, float]]:
        """Calculate vertex positions for a point-up hexagon.

        Args:
            x (float): X coordinate of hexagon center in meters.
            y (float): Y coordinate of hexagon center in meters.
            s (float): Side length of hexagon in meters.

        Returns:
            List[Tuple[float, float]]: Six (x, y) vertex coordinates, starting
                from the top vertex and proceeding clockwise.
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
        """Calculate distance from an edge location to an endpoint on the cell boundary.

        Used internally for fire spread calculations to determine the distance
        fire must travel from its current position to reach a cell boundary point.

        Args:
            edge_loc (int): Starting edge location index (0 for center, 1-12 for
                boundary positions where odd=edge midpoints, even=corners).
            idx_diff (int): Absolute difference between edge_loc and target endpoint
                index (range 1-6 due to hexagon symmetry).
            cell_size (float): Hexagon side length in meters.

        Returns:
            float: Distance in meters from edge_loc to the target endpoint.
        """
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
    def get_ign_parameters(edge_loc: int, cell_size: float) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """Compute fire spread parameters from an ignition point within a cell.

        Calculates the spread directions, distances to cell boundary endpoints,
        and neighbor cell entry points for fire propagating from a given ignition
        location. Results are cached for performance.

        Args:
            edge_loc (int): Ignition location index. 0 for cell center, 1-12 for
                boundary positions (odd indices are edge midpoints, even indices
                are corner vertices).
            cell_size (float): Hexagon side length in meters.

        Returns:
            Tuple containing:
                - np.ndarray: Spread direction angles in degrees (0-360).
                - np.ndarray: Distances to each boundary endpoint in meters.
                - tuple: Nested tuples of (neighbor_edge_loc, neighbor_letter)
                    pairs indicating where fire enters adjacent cells.
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

            directions = [
                start_angle,
                start_angle + 19.107,
                start_angle + 30,
                start_angle + 46.102,
                start_angle + 60,
                end_angle - 46.102,
                end_angle - 30,
                end_angle - 19.107,
                end_angle 
            ]

            start_end_point = (edge_loc + 2) % 12 or 12
            end_end_point = (start_end_point + 8) % 12 or 12

        else:
            # Ignition is along an edge
            start_angle = (30 * edge_loc + 90) % 360
            end_angle = (start_angle + 180)

            directions = [
                start_angle,
                start_angle + 30,
                start_angle + 40.893,
                start_angle + 60,
                start_angle + 73.898,
                start_angle + 90,
                end_angle - 73.898,
                end_angle - 60,
                end_angle - 40.893,
                end_angle - 30,
                end_angle 
            ]

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

        end_points = tuple(tuple(neighbor_locs) for neighbor_locs in end_points)

        return np.array(directions), np.array(distances), end_points
