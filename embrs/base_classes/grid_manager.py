"""Grid management for hexagonal fire simulation.

This module provides the GridManager class which handles all grid-related
operations for the fire simulation, including cell storage, coordinate
conversion, neighbor calculations, and geometry operations.

Classes:
    - GridManager: Manages the hexagonal cell grid for fire simulation.
"""

from typing import Optional, List, Tuple, Union, Dict, Callable, TYPE_CHECKING
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree

from embrs.utilities.fire_util import HexGridMath

if TYPE_CHECKING:
    from embrs.fire_simulator.cell import Cell


class GridManager:
    """Manages the hexagonal cell grid for fire simulation.

    Handles grid initialization, cell storage, coordinate conversion between
    Cartesian and grid indices, neighbor calculations, and geometry-based
    cell lookups.

    Attributes:
        cell_grid (np.ndarray): 2D array of Cell objects.
        cell_dict (Dict[int, Cell]): Dictionary mapping cell IDs to Cell objects.
        shape (Tuple[int, int]): Grid dimensions (num_rows, num_cols).
        cell_size (float): Edge length of hexagonal cells in meters.
    """

    def __init__(self,
                 num_rows: int,
                 num_cols: int,
                 cell_size: float):
        """Initialize the grid manager.

        Creates the backing array for the hexagonal cell grid but does not
        populate cells. Use init_grid() to populate with Cell objects.

        Args:
            num_rows: Number of rows in the grid.
            num_cols: Number of columns in the grid.
            cell_size: Edge length of hexagonal cells in meters.
        """
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._cell_size = cell_size

        self._shape = (num_rows, num_cols)
        self._cell_grid = np.empty(self._shape, dtype=object)
        self._grid_width = num_cols - 1
        self._grid_height = num_rows - 1

        self._cell_dict: Dict[int, 'Cell'] = {}

        # Reference to logger for error messages (set by parent)
        self.logger = None

        # Spatial index (built lazily or after init_grid)
        self._strtree = None
        self._strtree_cells = None

    @property
    def cell_grid(self) -> np.ndarray:
        """2D array of Cell objects."""
        return self._cell_grid

    @property
    def cell_dict(self) -> Dict[int, 'Cell']:
        """Dictionary mapping cell IDs to Cell objects."""
        return self._cell_dict

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid dimensions (num_rows, num_cols)."""
        return self._shape

    @property
    def cell_size(self) -> float:
        """Edge length of hexagonal cells in meters."""
        return self._cell_size

    @property
    def num_rows(self) -> int:
        """Number of rows in the grid."""
        return self._num_rows

    @property
    def num_cols(self) -> int:
        """Number of columns in the grid."""
        return self._num_cols

    def set_cell(self, row: int, col: int, cell: 'Cell') -> None:
        """Place a cell in the grid at the specified position.

        Args:
            row: Row index.
            col: Column index.
            cell: Cell object to place.
        """
        self._cell_grid[row, col] = cell
        self._cell_dict[cell.id] = cell

    def init_grid(self,
                  cell_factory: Callable[[int, int, int], 'Cell'],
                  progress_callback: Optional[Callable[[int], None]] = None) -> None:
        """Initialize the grid by creating cells using the provided factory.

        Iterates through all grid positions and calls the cell factory to
        create each cell. The factory is responsible for creating fully
        initialized Cell objects with terrain data, fuel types, etc.

        Args:
            cell_factory: Callable that takes (cell_id, col, row) and returns
                a fully initialized Cell object. The factory should handle:
                - Creating the Cell with correct position
                - Setting terrain data (elevation, slope, aspect, etc.)
                - Setting fuel type and moisture values
                - Setting wind forecast data
                - Any other cell initialization
            progress_callback: Optional callable that takes the number of cells
                processed (1 per call) for progress tracking. Can be used with
                tqdm or other progress indicators.

        Example:
            def my_cell_factory(cell_id, col, row):
                cell = Cell(cell_id, col, row, cell_size)
                cell.set_parent(sim)
                # ... initialize cell data ...
                return cell

            grid_manager.init_grid(my_cell_factory, pbar.update)
        """
        cell_id = 0
        for col in range(self._num_cols):
            for row in range(self._num_rows):
                cell = cell_factory(cell_id, col, row)
                self.set_cell(row, col, cell)
                cell_id += 1

                if progress_callback is not None:
                    progress_callback(1)

        self._build_spatial_index()

    def _build_spatial_index(self) -> None:
        """Build R-tree spatial index over all cell polygons.

        Creates a Shapely STRtree from all cell polygons for efficient
        spatial queries. Must be called after all cells have been created
        and have valid polygon attributes.

        Side Effects:
            Sets self._strtree and self._strtree_cells.
        """
        cells = []
        polygons = []
        for row in range(self._shape[0]):
            for col in range(self._shape[1]):
                cell = self._cell_grid[row, col]
                cells.append(cell)
                polygons.append(cell.polygon)
        self._strtree_cells = cells
        self._strtree = STRtree(polygons)

    def hex_round(self, q: float, r: float) -> Tuple[int, int]:
        """Round floating point hex coordinates to their nearest integer hex coordinates.

        Uses cube coordinate rounding to find the nearest valid hexagonal cell.
        The algorithm ensures the cube coordinate constraint (q + r + s = 0)
        is maintained.

        Args:
            q: q coordinate in hex coordinate system.
            r: r coordinate in hex coordinate system.

        Returns:
            Tuple of (q, r) integer coordinates of the nearest hex cell.
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

    def get_cell_from_xy(self, x_m: float, y_m: float, oob_ok: bool = False) -> Optional['Cell']:
        """Return the cell containing the point (x_m, y_m) in Cartesian coordinates.

        Converts Cartesian coordinates to hexagonal grid indices and returns
        the cell at that position.

        Args:
            x_m: x position in meters. (0,0) is lower-left corner.
            y_m: y position in meters. y increases upward.
            oob_ok: If True, return None for out-of-bounds coordinates.
                   If False, raise ValueError.

        Returns:
            Cell at the requested point, or None if out of bounds and oob_ok=True.

        Raises:
            ValueError: If coordinates are out of bounds and oob_ok=False.
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

            estimated_cell = self._cell_grid[row, col]
            return estimated_cell

        except IndexError:
            if not oob_ok:
                msg = f'Point ({x_m}, {y_m}) is outside the grid.'
                if self.logger:
                    self.logger.log_message(f"Following error occurred in 'GridManager.get_cell_from_xy()': {msg}")
                raise ValueError(msg)

            return None

    def get_cell_from_indices(self, row: int, col: int) -> 'Cell':
        """Return the cell at grid indices [row, col].

        Columns increase left to right, rows increase bottom to top.

        Args:
            row: Row index of the desired cell.
            col: Column index of the desired cell.

        Returns:
            Cell at the specified indices.

        Raises:
            TypeError: If row or col is not an integer.
            ValueError: If row or col is out of bounds.
        """
        if not isinstance(row, int) or not isinstance(col, int):
            msg = (f"Row and column must be integer index values. "
                f"Input was {type(row)}, {type(col)}")

            if self.logger:
                self.logger.log_message(f"Following error occurred in 'GridManager.get_cell_from_indices(): "
                                        f"{msg} Program terminated.")
            raise TypeError(msg)

        if col < 0 or row < 0 or row >= self._grid_height or col >= self._grid_width:
            msg = (f"Out of bounds error. {row}, {col} "
                f"are out of bounds for grid of size "
                f"{self._grid_height}, {self._grid_width}")

            if self.logger:
                self.logger.log_message(f"Following error occurred in 'GridManager.get_cell_from_indices(): "
                                        f"{msg} Program terminated.")
            raise ValueError(msg)

        return self._cell_grid[row, col]

    def get_cells_at_geometry(self, geom: Union[Polygon, LineString, Point]) -> List['Cell']:
        """Get all cells that intersect with the given geometry.

        Supports Point, LineString, and Polygon geometries from Shapely.

        Args:
            geom: Shapely geometry to check for cell intersections.

        Returns:
            List of Cell objects that intersect with the geometry.

        Raises:
            ValueError: If geometry type is not supported.
        """
        cells = set()

        if isinstance(geom, Polygon):
            # Lazily build spatial index if not yet constructed
            if self._strtree is None:
                self._build_spatial_index()

            # STRtree query returns indices of candidate cells
            indices = self._strtree.query(geom, predicate='intersects')
            # Post-filter to exclude edge-only touches (zero-area intersection)
            for i in indices:
                cell = self._strtree_cells[i]
                if geom.intersection(cell.polygon).area > 1e-6:
                    cells.add(cell)

        elif isinstance(geom, LineString):
            length = geom.length
            step_size = self._cell_size / 4.0
            num_steps = int(length/step_size) + 1

            for i in range(num_steps):
                point = geom.interpolate(i * step_size)
                cell = self.get_cell_from_xy(point.x, point.y, oob_ok=True)
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

    def add_cell_neighbors(self) -> None:
        """Populate neighbor references for all cells in the grid.

        For each cell, determines its neighbors based on hexagonal grid
        geometry (even/odd row offset pattern) and stores neighbor IDs
        with their relative positions.
        """
        for j in range(self._shape[1]):
            for i in range(self._shape[0]):
                cell = self._cell_grid[i][j]

                neighbors = {}
                if cell.row % 2 == 0:
                    neighborhood = HexGridMath.even_neighborhood
                else:
                    neighborhood = HexGridMath.odd_neighborhood

                for dx, dy in neighborhood:
                    row_n = int(cell.row + dy)
                    col_n = int(cell.col + dx)

                    if self._grid_height >= row_n >= 0 and self._grid_width >= col_n >= 0:
                        neighbor_id = self._cell_grid[row_n, col_n].id
                        neighbors[neighbor_id] = (dx, dy)

                cell._neighbors = neighbors
                cell._burnable_neighbors = dict(neighbors)

    def get_cells_in_radius(self, center_x: float, center_y: float,
                            radius: float) -> List['Cell']:
        """Get all cells within a given radius of a center point.

        Args:
            center_x: x coordinate of center point in meters.
            center_y: y coordinate of center point in meters.
            radius: Radius in meters.

        Returns:
            List of Cell objects within the specified radius.
        """
        cells = []

        # Calculate bounding box in grid coordinates
        min_row = max(0, int((center_y - radius) // (self._cell_size * 1.5)))
        max_row = min(self._shape[0] - 1, int((center_y + radius) // (self._cell_size * 1.5)) + 1)
        min_col = max(0, int((center_x - radius) // (self._cell_size * np.sqrt(3))))
        max_col = min(self._shape[1] - 1, int((center_x + radius) // (self._cell_size * np.sqrt(3))) + 1)

        radius_sq = radius * radius

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cell = self._cell_grid[row, col]
                dx = cell.x_pos - center_x
                dy = cell.y_pos - center_y

                if dx*dx + dy*dy <= radius_sq:
                    cells.append(cell)

        return cells

    def compute_all_cell_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-compute world coordinates for all cell centers.

        Uses vectorized operations to compute x,y positions for all cells
        in the grid based on hexagonal geometry.

        The formula matches Cell.__init__:
        - Even row: x = col * cell_size * sqrt(3)
        - Odd row: x = (col + 0.5) * cell_size * sqrt(3)
        - y = row * cell_size * 1.5

        Returns:
            Tuple of (all_x, all_y) where each is a 2D numpy array with
            shape (num_rows, num_cols) containing the cell center coordinates.
        """
        # Create meshgrid of row and column indices
        rows, cols = np.meshgrid(
            np.arange(self._num_rows),
            np.arange(self._num_cols),
            indexing='ij'
        )

        # Compute cell centers using hexagonal grid geometry
        # Matching Cell.__init__ formula exactly
        hex_width = np.sqrt(3) * self._cell_size

        # x position: col * hex_width for even rows, (col + 0.5) * hex_width for odd rows
        all_x = (cols + 0.5 * (rows % 2)) * hex_width

        # y position: row * cell_size * 1.5
        all_y = rows * self._cell_size * 1.5

        return all_x, all_y

    def compute_data_indices(self, all_x: np.ndarray, all_y: np.ndarray,
                             data_res: float, data_rows: int, data_cols: int
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert cell positions to terrain data array indices.

        Vectorized computation of which terrain data pixels correspond to
        each cell center.

        Args:
            all_x: 2D array of cell x coordinates.
            all_y: 2D array of cell y coordinates.
            data_res: Resolution of terrain data in meters per pixel.
            data_rows: Number of rows in terrain data arrays.
            data_cols: Number of columns in terrain data arrays.

        Returns:
            Tuple of (data_row_indices, data_col_indices) as 2D integer arrays.
        """
        # Convert world coordinates to data array indices
        data_col_indices = np.floor(all_x / data_res).astype(np.int32)
        data_row_indices = np.floor(all_y / data_res).astype(np.int32)

        # Clip to valid range
        np.clip(data_col_indices, 0, data_cols - 1, out=data_col_indices)
        np.clip(data_row_indices, 0, data_rows - 1, out=data_row_indices)

        return data_row_indices, data_col_indices
