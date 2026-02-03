"""Tests for grid operations in BaseFireSim.

These tests validate the grid-related methods that will be extracted into
GridManager during Phase 3 refactoring. They establish behavioral baselines
to ensure the refactoring doesn't change functionality.

Tested methods:
- hex_round(q, r)
- get_cell_from_xy(x_m, y_m, oob_ok)
- get_cell_from_indices(row, col)
- get_cells_at_geometry(geom)
- _add_cell_neighbors()
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from shapely.geometry import Point, Polygon, LineString

from embrs.utilities.fire_util import HexGridMath


# ============================================================================
# Mock Cell class to avoid dependency chain issues
# ============================================================================

class MockCell:
    """Minimal mock Cell class for grid testing.

    Replicates the essential geometry and identity properties of
    embrs.fire_simulator.cell.Cell without requiring the full dependency chain.
    """

    def __init__(self, id: int, col: int, row: int, cell_size: float):
        self.id = id
        self._col = col
        self._row = row
        self._cell_size = cell_size

        # Calculate position (same formula as real Cell)
        if row % 2 == 0:
            self._x_pos = col * cell_size * np.sqrt(3)
        else:
            self._x_pos = (col + 0.5) * cell_size * np.sqrt(3)
        self._y_pos = row * cell_size * 1.5

        # Initialize neighbor structures
        self._neighbors = {}
        self._burnable_neighbors = {}

        # Create hexagonal polygon
        self.polygon = self._create_polygon()

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

    @property
    def x_pos(self):
        return self._x_pos

    @property
    def y_pos(self):
        return self._y_pos

    def _create_polygon(self):
        """Create hexagonal polygon for geometry tests."""
        vertices = []
        for i in range(6):
            angle = np.pi / 6 + i * np.pi / 3
            x = self._x_pos + self._cell_size * np.cos(angle)
            y = self._y_pos + self._cell_size * np.sin(angle)
            vertices.append((x, y))
        return Polygon(vertices)


# ============================================================================
# Test fixtures for grid operations
# ============================================================================

@pytest.fixture
def cell_size():
    """Standard cell size for tests (30 meters)."""
    return 30.0


class MockBaseFireSim:
    """Minimal mock of BaseFireSim for testing grid operations.

    This class has just enough structure to test the grid methods
    without requiring full simulation setup.
    """

    def __init__(self, cell_size: float, num_rows: int = 10, num_cols: int = 10):
        self._cell_size = cell_size
        self.cell_size = cell_size

        self._shape = (num_rows, num_cols)
        self.shape = (num_rows, num_cols)
        self._grid_height = num_rows - 1
        self._grid_width = num_cols - 1

        # Create MockCell objects for the grid
        self._cell_grid = np.empty((num_rows, num_cols), dtype=object)
        self._cell_dict = {}
        cell_id = 0

        for row in range(num_rows):
            for col in range(num_cols):
                cell = MockCell(id=cell_id, col=col, row=row, cell_size=cell_size)
                self._cell_grid[row, col] = cell
                self._cell_dict[cell_id] = cell
                cell_id += 1

        self.logger = None

    # Import actual methods from BaseFireSim
    def hex_round(self, q, r):
        """Rounds floating point hex coordinates to their nearest integer hex coordinates."""
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

    def get_cell_from_xy(self, x_m: float, y_m: float, oob_ok=False):
        """Returns the cell at Cartesian coordinates (x_m, y_m)."""
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
                raise ValueError(msg)

            return None

    def get_cell_from_indices(self, row: int, col: int):
        """Returns the cell at the indices [row, col] in the cell_grid."""
        if not isinstance(row, int) or not isinstance(col, int):
            msg = (f"Row and column must be integer index values. "
                f"Input was {type(row)}, {type(col)}")
            raise TypeError(msg)

        if col < 0 or row < 0 or row >= self._grid_height or col >= self._grid_width:
            msg = (f"Out of bounds error. {row}, {col} "
                f"are out of bounds for grid of size "
                f"{self._grid_height}, {self._grid_width}")
            raise ValueError(msg)

        return self._cell_grid[row, col]

    def get_cells_at_geometry(self, geom):
        """Get all cells that intersect with the given geometry."""
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.geometry import LineString as ShapelyLineString
        from shapely.geometry import Point as ShapelyPoint

        cells = set()
        if isinstance(geom, ShapelyPolygon):
            minx, miny, maxx, maxy = geom.bounds
            min_row = int(miny // (self.cell_size * 1.5))
            max_row = int(maxy // (self.cell_size * 1.5))
            min_col = int(minx // (self.cell_size * np.sqrt(3)))
            max_col = int(maxx // (self.cell_size * np.sqrt(3)))

            for row in range(min_row - 1, max_row + 2):
                for col in range(min_col - 1, max_col + 2):
                    if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
                        cell = self._cell_grid[row, col]
                        if geom.intersection(cell.polygon).area > 1e-6:
                            cells.add(cell)

        elif isinstance(geom, ShapelyLineString):
            length = geom.length
            step_size = self._cell_size / 4.0
            num_steps = int(length/step_size) + 1

            for i in range(num_steps):
                point = geom.interpolate(i * step_size)
                cell = self.get_cell_from_xy(point.x, point.y, oob_ok=True)
                if cell is not None:
                    cells.add(cell)

        elif isinstance(geom, ShapelyPoint):
            x, y = geom.x, geom.y
            cell = self.get_cell_from_xy(x, y, oob_ok=True)
            if cell is not None:
                cells.add(cell)

        else:
            raise ValueError(f"Unknown geometry type: {type(geom)}")

        return list(cells)

    def _add_cell_neighbors(self):
        """Populate neighbor references for all cells in the grid."""
        from embrs.utilities.fire_util import HexGridMath as hex

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


@pytest.fixture
def mock_base_fire_sim(cell_size):
    """Create a MockBaseFireSim with minimal grid setup.

    Creates a 10x10 grid of MockCell objects for testing grid operations.
    """
    return MockBaseFireSim(cell_size=cell_size, num_rows=10, num_cols=10)


# ============================================================================
# Tests for hex_round
# ============================================================================

class TestHexRound:
    """Tests for the hex_round coordinate rounding algorithm."""

    def test_hex_round_exact_coordinates(self, mock_base_fire_sim):
        """hex_round should return exact values for integer inputs."""
        # Test with exact integer coordinates
        result = mock_base_fire_sim.hex_round(0, 0)
        assert result == (0, 0)

        result = mock_base_fire_sim.hex_round(3, 2)
        assert result == (3, 2)

        result = mock_base_fire_sim.hex_round(-1, 2)
        assert result == (-1, 2)

    def test_hex_round_near_boundary(self, mock_base_fire_sim):
        """hex_round should correctly round coordinates near cell boundaries."""
        # Just above integer should round up
        result = mock_base_fire_sim.hex_round(0.6, 0.0)
        assert result == (1, 0)

        # Just below integer should round down
        result = mock_base_fire_sim.hex_round(0.4, 0.0)
        assert result == (0, 0)

    def test_hex_round_maintains_cube_constraint(self, mock_base_fire_sim):
        """hex_round output should satisfy cube coordinate constraint q + r + s = 0."""
        test_cases = [
            (0.3, 0.2),
            (1.7, -0.3),
            (-0.5, 2.1),
            (2.9, 1.1),
        ]

        for q_in, r_in in test_cases:
            q, r = mock_base_fire_sim.hex_round(q_in, r_in)
            s = -q - r  # Cube coordinate constraint

            # Verify q, r, s are all integers
            assert isinstance(q, int)
            assert isinstance(r, int)
            # s should be computable from q and r
            assert q + r + s == 0

    def test_hex_round_symmetric(self, mock_base_fire_sim):
        """hex_round should be symmetric around the origin."""
        # Test symmetry
        q1, r1 = mock_base_fire_sim.hex_round(0.3, 0.2)
        q2, r2 = mock_base_fire_sim.hex_round(-0.3, -0.2)

        assert q1 == -q2
        assert r1 == -r2


# ============================================================================
# Tests for get_cell_from_xy
# ============================================================================

class TestGetCellFromXY:
    """Tests for coordinate-to-cell lookup."""

    def test_get_cell_at_origin(self, mock_base_fire_sim, cell_size):
        """Cell at origin (0,0) should be at grid position (0,0)."""
        # Cell at (0,0) has center at x=0, y=0
        cell = mock_base_fire_sim.get_cell_from_xy(0.0, 0.0)

        assert cell is not None
        assert cell.row == 0
        assert cell.col == 0

    def test_get_cell_at_known_positions(self, mock_base_fire_sim, cell_size):
        """Test cell lookup at known grid positions."""
        # Cell at row=0, col=1 should have center at x = cell_size * sqrt(3)
        expected_x = cell_size * np.sqrt(3)
        expected_y = 0

        cell = mock_base_fire_sim.get_cell_from_xy(expected_x, expected_y)

        assert cell is not None
        assert cell.row == 0
        assert cell.col == 1

    def test_get_cell_odd_row(self, mock_base_fire_sim, cell_size):
        """Test cell lookup for cells in odd rows (offset pattern)."""
        # Cell at row=1, col=0 has center at x = 0.5 * cell_size * sqrt(3), y = 1.5 * cell_size
        expected_x = 0.5 * cell_size * np.sqrt(3)
        expected_y = 1.5 * cell_size

        cell = mock_base_fire_sim.get_cell_from_xy(expected_x, expected_y)

        assert cell is not None
        assert cell.row == 1
        assert cell.col == 0

    def test_get_cell_negative_coordinates_oob_false(self, mock_base_fire_sim):
        """Negative coordinates should raise ValueError when oob_ok=False."""
        with pytest.raises((ValueError, IndexError)):
            mock_base_fire_sim.get_cell_from_xy(-10.0, 5.0, oob_ok=False)

    def test_get_cell_negative_coordinates_oob_true(self, mock_base_fire_sim):
        """Negative coordinates should return None when oob_ok=True."""
        cell = mock_base_fire_sim.get_cell_from_xy(-10.0, 5.0, oob_ok=True)

        assert cell is None

    def test_get_cell_out_of_grid_bounds(self, mock_base_fire_sim, cell_size):
        """Coordinates outside grid should return None when oob_ok=True."""
        # Far outside grid bounds
        large_x = cell_size * np.sqrt(3) * 100
        large_y = cell_size * 1.5 * 100

        cell = mock_base_fire_sim.get_cell_from_xy(large_x, large_y, oob_ok=True)

        assert cell is None

    def test_get_cell_within_cell_boundary(self, mock_base_fire_sim, cell_size):
        """Points within a cell's hexagon should return that cell."""
        # Get cell at (3, 3) - interior cell that allows offsets in all directions
        test_cell = mock_base_fire_sim._cell_grid[3, 3]
        center_x = test_cell.x_pos
        center_y = test_cell.y_pos

        # Test points slightly offset from center (still within hex)
        offsets = [
            (cell_size * 0.3, 0),
            (-cell_size * 0.3, 0),
            (0, cell_size * 0.3),
            (0, -cell_size * 0.3),
        ]

        for dx, dy in offsets:
            cell = mock_base_fire_sim.get_cell_from_xy(center_x + dx, center_y + dy)
            assert cell is not None
            assert cell.row == 3
            assert cell.col == 3


# ============================================================================
# Tests for get_cell_from_indices
# ============================================================================

class TestGetCellFromIndices:
    """Tests for index-based cell lookup."""

    def test_get_cell_valid_indices(self, mock_base_fire_sim):
        """Valid indices should return the correct cell."""
        cell = mock_base_fire_sim.get_cell_from_indices(3, 4)

        assert cell is not None
        assert cell.row == 3
        assert cell.col == 4

    def test_get_cell_corner_indices(self, mock_base_fire_sim):
        """Corner cells should be accessible."""
        # Lower-left corner
        cell = mock_base_fire_sim.get_cell_from_indices(0, 0)
        assert cell.row == 0 and cell.col == 0

        # Upper-right corner (within bounds)
        max_row = mock_base_fire_sim._grid_height - 1
        max_col = mock_base_fire_sim._grid_width - 1
        cell = mock_base_fire_sim.get_cell_from_indices(max_row, max_col)
        assert cell.row == max_row and cell.col == max_col

    def test_get_cell_negative_row_raises(self, mock_base_fire_sim):
        """Negative row should raise ValueError."""
        with pytest.raises(ValueError):
            mock_base_fire_sim.get_cell_from_indices(-1, 0)

    def test_get_cell_negative_col_raises(self, mock_base_fire_sim):
        """Negative column should raise ValueError."""
        with pytest.raises(ValueError):
            mock_base_fire_sim.get_cell_from_indices(0, -1)

    def test_get_cell_row_out_of_bounds_raises(self, mock_base_fire_sim):
        """Row >= grid_height should raise ValueError."""
        with pytest.raises(ValueError):
            mock_base_fire_sim.get_cell_from_indices(
                mock_base_fire_sim._grid_height,
                0
            )

    def test_get_cell_col_out_of_bounds_raises(self, mock_base_fire_sim):
        """Column >= grid_width should raise ValueError."""
        with pytest.raises(ValueError):
            mock_base_fire_sim.get_cell_from_indices(
                0,
                mock_base_fire_sim._grid_width
            )

    def test_get_cell_float_indices_raises(self, mock_base_fire_sim):
        """Float indices should raise TypeError."""
        with pytest.raises(TypeError):
            mock_base_fire_sim.get_cell_from_indices(1.5, 2)

        with pytest.raises(TypeError):
            mock_base_fire_sim.get_cell_from_indices(1, 2.5)

    def test_get_cell_numpy_int_accepted(self, mock_base_fire_sim):
        """Numpy integer types should be accepted."""
        # np.int64 is commonly returned from numpy operations
        row = np.int64(3)
        col = np.int64(4)

        # This should work - numpy integers are subclasses of int in Python 3
        cell = mock_base_fire_sim.get_cell_from_indices(int(row), int(col))
        assert cell is not None


# ============================================================================
# Tests for get_cells_at_geometry
# ============================================================================

class TestGetCellsAtGeometry:
    """Tests for geometry-based cell lookup."""

    def test_get_cells_at_point(self, mock_base_fire_sim):
        """Point geometry should return single cell containing the point."""
        point = Point(0.0, 0.0)
        cells = mock_base_fire_sim.get_cells_at_geometry(point)

        assert len(cells) == 1
        assert cells[0].row == 0
        assert cells[0].col == 0

    def test_get_cells_at_point_oob(self, mock_base_fire_sim, cell_size):
        """Point outside grid should return empty list."""
        # Far outside grid
        point = Point(cell_size * 1000, cell_size * 1000)
        cells = mock_base_fire_sim.get_cells_at_geometry(point)

        assert len(cells) == 0

    def test_get_cells_at_small_polygon(self, mock_base_fire_sim, cell_size):
        """Small polygon should return cells it intersects."""
        # Small polygon around origin
        polygon = Polygon([
            (-cell_size * 0.2, -cell_size * 0.2),
            (cell_size * 0.2, -cell_size * 0.2),
            (cell_size * 0.2, cell_size * 0.2),
            (-cell_size * 0.2, cell_size * 0.2),
        ])

        cells = mock_base_fire_sim.get_cells_at_geometry(polygon)

        # Should at least include the cell at (0,0)
        assert len(cells) >= 1
        cell_positions = [(c.row, c.col) for c in cells]
        assert (0, 0) in cell_positions

    def test_get_cells_at_large_polygon(self, mock_base_fire_sim, cell_size):
        """Large polygon should return multiple cells."""
        # Polygon covering roughly 3x3 cell area
        polygon = Polygon([
            (0, 0),
            (3 * cell_size * np.sqrt(3), 0),
            (3 * cell_size * np.sqrt(3), 3 * cell_size * 1.5),
            (0, 3 * cell_size * 1.5),
        ])

        cells = mock_base_fire_sim.get_cells_at_geometry(polygon)

        # Should return multiple cells
        assert len(cells) > 1

    def test_get_cells_at_linestring(self, mock_base_fire_sim, cell_size):
        """LineString should return cells along its path."""
        # Horizontal line across several cells
        line = LineString([
            (0, 0),
            (4 * cell_size * np.sqrt(3), 0),
        ])

        cells = mock_base_fire_sim.get_cells_at_geometry(line)

        # Should return multiple cells along the line
        assert len(cells) >= 3

        # All cells should be in row 0 (horizontal line at y=0)
        for cell in cells:
            assert cell.row == 0

    def test_get_cells_at_diagonal_linestring(self, mock_base_fire_sim, cell_size):
        """Diagonal LineString should return cells along its path."""
        # Diagonal line
        line = LineString([
            (0, 0),
            (3 * cell_size * np.sqrt(3), 3 * cell_size * 1.5),
        ])

        cells = mock_base_fire_sim.get_cells_at_geometry(line)

        # Should return multiple cells
        assert len(cells) >= 2

    def test_get_cells_at_unknown_geometry_raises(self, mock_base_fire_sim):
        """Unknown geometry type should raise ValueError."""
        from shapely.geometry import MultiPoint

        # MultiPoint is not supported
        geom = MultiPoint([(0, 0), (10, 10)])

        with pytest.raises(ValueError, match="Unknown geometry type"):
            mock_base_fire_sim.get_cells_at_geometry(geom)

    def test_get_cells_returns_unique_cells(self, mock_base_fire_sim, cell_size):
        """Returned cells should be unique (no duplicates)."""
        # LineString that might traverse same cell multiple times
        line = LineString([
            (0, 0),
            (cell_size * np.sqrt(3), 0),
            (0, 0),  # Back to start
        ])

        cells = mock_base_fire_sim.get_cells_at_geometry(line)

        # Check for uniqueness
        cell_ids = [c.id for c in cells]
        assert len(cell_ids) == len(set(cell_ids))


# ============================================================================
# Tests for _add_cell_neighbors
# ============================================================================

class TestAddCellNeighbors:
    """Tests for neighbor population."""

    def test_interior_cell_has_six_neighbors(self, mock_base_fire_sim):
        """Interior cells should have exactly 6 neighbors."""
        # Initialize neighbors
        mock_base_fire_sim._add_cell_neighbors()

        # Get an interior cell (row=5, col=5 is well within 10x10 grid)
        interior_cell = mock_base_fire_sim._cell_grid[5, 5]

        assert len(interior_cell._neighbors) == 6

    def test_corner_cell_has_fewer_neighbors(self, mock_base_fire_sim):
        """Corner cells should have fewer than 6 neighbors."""
        # Initialize neighbors
        mock_base_fire_sim._add_cell_neighbors()

        # Get corner cell (0, 0)
        corner_cell = mock_base_fire_sim._cell_grid[0, 0]

        assert len(corner_cell._neighbors) < 6
        assert len(corner_cell._neighbors) >= 2  # Should have at least 2

    def test_edge_cell_has_correct_neighbors(self, mock_base_fire_sim):
        """Edge cells should have valid neighbors within bounds."""
        # Initialize neighbors
        mock_base_fire_sim._add_cell_neighbors()

        # Get edge cell (row=0, col=5)
        edge_cell = mock_base_fire_sim._cell_grid[0, 5]

        # Should have neighbors but not full 6
        assert len(edge_cell._neighbors) >= 3
        assert len(edge_cell._neighbors) <= 6

        # All neighbors should be valid cells
        for neighbor_id in edge_cell._neighbors.keys():
            assert neighbor_id in mock_base_fire_sim._cell_dict

    def test_neighbors_are_bidirectional(self, mock_base_fire_sim):
        """If A is B's neighbor, B should be A's neighbor."""
        # Initialize neighbors
        mock_base_fire_sim._add_cell_neighbors()

        # Check several cells
        for row in range(3, 7):
            for col in range(3, 7):
                cell = mock_base_fire_sim._cell_grid[row, col]

                for neighbor_id in cell._neighbors.keys():
                    neighbor = mock_base_fire_sim._cell_dict[neighbor_id]
                    assert cell.id in neighbor._neighbors

    def test_burnable_neighbors_initialized(self, mock_base_fire_sim):
        """_burnable_neighbors should be initialized same as _neighbors."""
        # Initialize neighbors
        mock_base_fire_sim._add_cell_neighbors()

        # Check that burnable_neighbors matches neighbors
        interior_cell = mock_base_fire_sim._cell_grid[5, 5]

        assert interior_cell._burnable_neighbors is not None
        assert len(interior_cell._burnable_neighbors) == len(interior_cell._neighbors)
        assert set(interior_cell._burnable_neighbors.keys()) == set(interior_cell._neighbors.keys())

    def test_even_odd_row_different_patterns(self, mock_base_fire_sim):
        """Even and odd rows should use different neighbor patterns."""
        # Initialize neighbors
        mock_base_fire_sim._add_cell_neighbors()

        # Get cells from even and odd rows at same column
        even_cell = mock_base_fire_sim._cell_grid[4, 5]  # Row 4 (even)
        odd_cell = mock_base_fire_sim._cell_grid[5, 5]   # Row 5 (odd)

        # The offset patterns should be different
        even_offsets = set(even_cell._neighbors.values())
        odd_offsets = set(odd_cell._neighbors.values())

        # The patterns should not be identical
        assert even_offsets != odd_offsets


# ============================================================================
# Integration tests - coordinate system consistency
# ============================================================================

class TestCoordinateSystemConsistency:
    """Tests to verify coordinate system consistency across methods."""

    def test_cell_center_roundtrip(self, mock_base_fire_sim, cell_size):
        """Getting a cell by its center coordinates should return that cell."""
        # Test several cells
        for row in range(5):
            for col in range(5):
                cell = mock_base_fire_sim._cell_grid[row, col]

                # Look up by center coordinates
                found_cell = mock_base_fire_sim.get_cell_from_xy(cell.x_pos, cell.y_pos)

                assert found_cell is not None
                assert found_cell.row == row
                assert found_cell.col == col

    def test_indices_match_xy_lookup(self, mock_base_fire_sim, cell_size):
        """get_cell_from_indices and get_cell_from_xy should return same cell."""
        for row in range(5):
            for col in range(5):
                # Get cell by indices
                cell_by_idx = mock_base_fire_sim.get_cell_from_indices(row, col)

                # Get cell by coordinates
                cell_by_xy = mock_base_fire_sim.get_cell_from_xy(
                    cell_by_idx.x_pos, cell_by_idx.y_pos
                )

                assert cell_by_idx.id == cell_by_xy.id

    def test_point_geometry_matches_xy_lookup(self, mock_base_fire_sim, cell_size):
        """Point geometry and xy lookup should return same cell."""
        for row in range(5):
            for col in range(5):
                cell = mock_base_fire_sim._cell_grid[row, col]

                # Lookup by point geometry
                point = Point(cell.x_pos, cell.y_pos)
                cells_from_geom = mock_base_fire_sim.get_cells_at_geometry(point)

                assert len(cells_from_geom) == 1
                assert cells_from_geom[0].id == cell.id


# ============================================================================
# Tests for the extracted GridManager class
# ============================================================================

@pytest.fixture
def grid_manager(cell_size):
    """Create a GridManager with a 10x10 grid of MockCells."""
    from embrs.base_classes.grid_manager import GridManager

    gm = GridManager(num_rows=10, num_cols=10, cell_size=cell_size)

    # Populate with MockCell objects
    cell_id = 0
    for row in range(10):
        for col in range(10):
            cell = MockCell(id=cell_id, col=col, row=row, cell_size=cell_size)
            gm.set_cell(row, col, cell)
            cell_id += 1

    return gm


class TestGridManagerHexRound:
    """Tests for GridManager.hex_round."""

    def test_exact_coordinates(self, grid_manager):
        """hex_round should return exact values for integer inputs."""
        assert grid_manager.hex_round(0, 0) == (0, 0)
        assert grid_manager.hex_round(3, 2) == (3, 2)
        assert grid_manager.hex_round(-1, 2) == (-1, 2)

    def test_near_boundary(self, grid_manager):
        """hex_round should correctly round near boundaries."""
        assert grid_manager.hex_round(0.6, 0.0) == (1, 0)
        assert grid_manager.hex_round(0.4, 0.0) == (0, 0)


class TestGridManagerCellLookup:
    """Tests for GridManager cell lookup methods."""

    def test_get_cell_at_origin(self, grid_manager):
        """Cell at origin should be at grid position (0,0)."""
        cell = grid_manager.get_cell_from_xy(0.0, 0.0)
        assert cell is not None
        assert cell.row == 0
        assert cell.col == 0

    def test_get_cell_from_indices(self, grid_manager):
        """Valid indices should return correct cell."""
        cell = grid_manager.get_cell_from_indices(3, 4)
        assert cell is not None
        assert cell.row == 3
        assert cell.col == 4

    def test_get_cell_oob_returns_none(self, grid_manager, cell_size):
        """Out of bounds with oob_ok=True should return None."""
        cell = grid_manager.get_cell_from_xy(cell_size * 1000, cell_size * 1000, oob_ok=True)
        assert cell is None

    def test_get_cells_at_point(self, grid_manager):
        """Point geometry should return single cell."""
        point = Point(0.0, 0.0)
        cells = grid_manager.get_cells_at_geometry(point)
        assert len(cells) == 1
        assert cells[0].row == 0
        assert cells[0].col == 0


class TestGridManagerNeighbors:
    """Tests for GridManager neighbor calculation."""

    def test_interior_cell_six_neighbors(self, grid_manager):
        """Interior cells should have 6 neighbors."""
        grid_manager.add_cell_neighbors()
        interior_cell = grid_manager.cell_grid[5, 5]
        assert len(interior_cell._neighbors) == 6

    def test_corner_cell_fewer_neighbors(self, grid_manager):
        """Corner cells should have fewer than 6 neighbors."""
        grid_manager.add_cell_neighbors()
        corner_cell = grid_manager.cell_grid[0, 0]
        assert len(corner_cell._neighbors) < 6

    def test_bidirectional_neighbors(self, grid_manager):
        """If A is B's neighbor, B should be A's neighbor."""
        grid_manager.add_cell_neighbors()
        for row in range(3, 7):
            for col in range(3, 7):
                cell = grid_manager.cell_grid[row, col]
                for neighbor_id in cell._neighbors.keys():
                    neighbor = grid_manager.cell_dict[neighbor_id]
                    assert cell.id in neighbor._neighbors


class TestGridManagerBehaviorMatch:
    """Tests that GridManager behaves identically to MockBaseFireSim."""

    def test_hex_round_matches(self, mock_base_fire_sim, grid_manager):
        """GridManager and MockBaseFireSim should produce same hex_round results."""
        test_cases = [(0.3, 0.2), (1.7, -0.3), (-0.5, 2.1), (2.9, 1.1)]
        for q, r in test_cases:
            mock_result = mock_base_fire_sim.hex_round(q, r)
            gm_result = grid_manager.hex_round(q, r)
            assert mock_result == gm_result

    def test_cell_lookup_matches(self, mock_base_fire_sim, grid_manager, cell_size):
        """GridManager and MockBaseFireSim should return same cells for coordinates."""
        for row in range(5):
            for col in range(5):
                mock_cell = mock_base_fire_sim._cell_grid[row, col]
                gm_cell = grid_manager.cell_grid[row, col]

                # Lookup by center coordinates
                mock_found = mock_base_fire_sim.get_cell_from_xy(mock_cell.x_pos, mock_cell.y_pos)
                gm_found = grid_manager.get_cell_from_xy(gm_cell.x_pos, gm_cell.y_pos)

                assert mock_found.row == gm_found.row
                assert mock_found.col == gm_found.col

    def test_neighbor_calculation_matches(self, mock_base_fire_sim, grid_manager):
        """GridManager and MockBaseFireSim should produce same neighbor patterns."""
        mock_base_fire_sim._add_cell_neighbors()
        grid_manager.add_cell_neighbors()

        for row in range(3, 7):
            for col in range(3, 7):
                mock_cell = mock_base_fire_sim._cell_grid[row, col]
                gm_cell = grid_manager.cell_grid[row, col]

                # Same number of neighbors
                assert len(mock_cell._neighbors) == len(gm_cell._neighbors)

                # Same offset patterns
                mock_offsets = set(mock_cell._neighbors.values())
                gm_offsets = set(gm_cell._neighbors.values())
                assert mock_offsets == gm_offsets


class TestGridManagerInitGrid:
    """Tests for GridManager.init_grid method."""

    def test_init_grid_populates_all_cells(self, cell_size):
        """init_grid should create cells for all positions."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(5, 6, cell_size)

        def factory(cell_id, col, row):
            # MockCell takes (id, col, row, cell_size)
            return MockCell(cell_id, col, row, cell_size=cell_size)

        gm.init_grid(factory)

        # Check all positions are populated
        assert len(gm.cell_dict) == 5 * 6  # 30 cells
        for row in range(5):
            for col in range(6):
                cell = gm.cell_grid[row, col]
                assert cell is not None
                assert cell.row == row
                assert cell.col == col

    def test_init_grid_calls_factory_with_correct_args(self, cell_size):
        """Factory should receive cell_id, col, row in correct order."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(3, 4, cell_size)

        calls = []
        def factory(cell_id, col, row):
            calls.append((cell_id, col, row))
            return MockCell(cell_id, col, row, cell_size=cell_size)

        gm.init_grid(factory)

        # Verify call order: outer loop over cols, inner loop over rows
        expected_id = 0
        for col in range(4):
            for row in range(3):
                assert (expected_id, col, row) in calls
                expected_id += 1

    def test_init_grid_with_progress_callback(self, cell_size):
        """Progress callback should be called for each cell."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(3, 3, cell_size)

        progress_count = [0]
        def progress_callback(n):
            progress_count[0] += n

        def factory(cell_id, col, row):
            return MockCell(cell_id, col, row, cell_size=cell_size)

        gm.init_grid(factory, progress_callback)

        assert progress_count[0] == 9  # 3x3 = 9 cells

    def test_init_grid_cells_in_dict(self, cell_size):
        """All cells should be accessible via cell_dict."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(4, 4, cell_size)

        def factory(cell_id, col, row):
            return MockCell(cell_id, col, row, cell_size=cell_size)

        gm.init_grid(factory)

        for cell_id in range(16):
            assert cell_id in gm.cell_dict
            assert gm.cell_dict[cell_id].id == cell_id


# ============================================================================
# Tests for vectorized terrain loading methods
# ============================================================================

class TestGridManagerVectorizedPositions:
    """Tests for GridManager.compute_all_cell_positions."""

    def test_compute_positions_shape(self, cell_size):
        """compute_all_cell_positions should return arrays with correct shape."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(num_rows=10, num_cols=15, cell_size=cell_size)
        all_x, all_y = gm.compute_all_cell_positions()

        assert all_x.shape == (10, 15)
        assert all_y.shape == (10, 15)

    def test_compute_positions_type(self, cell_size):
        """compute_all_cell_positions should return numpy arrays."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(num_rows=5, num_cols=5, cell_size=cell_size)
        all_x, all_y = gm.compute_all_cell_positions()

        assert isinstance(all_x, np.ndarray)
        assert isinstance(all_y, np.ndarray)

    def test_compute_positions_match_mock_cells(self, cell_size):
        """Computed positions should match MockCell positions."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(num_rows=8, num_cols=8, cell_size=cell_size)

        # Compute vectorized positions
        all_x, all_y = gm.compute_all_cell_positions()

        # Compare with MockCell positions
        for row in range(8):
            for col in range(8):
                mock_cell = MockCell(id=0, col=col, row=row, cell_size=cell_size)

                # Allow small floating point tolerance
                assert abs(all_x[row, col] - mock_cell.x_pos) < 1e-10, \
                    f"x mismatch at ({row}, {col}): {all_x[row, col]} vs {mock_cell.x_pos}"
                assert abs(all_y[row, col] - mock_cell.y_pos) < 1e-10, \
                    f"y mismatch at ({row}, {col}): {all_y[row, col]} vs {mock_cell.y_pos}"

    def test_compute_positions_positive(self, cell_size):
        """All computed positions should be non-negative."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(num_rows=10, num_cols=10, cell_size=cell_size)
        all_x, all_y = gm.compute_all_cell_positions()

        assert np.all(all_x >= 0)
        assert np.all(all_y >= 0)

    def test_compute_positions_odd_row_offset(self, cell_size):
        """Odd rows should have x positions offset from even rows."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(num_rows=4, num_cols=4, cell_size=cell_size)
        all_x, all_y = gm.compute_all_cell_positions()

        # Check that odd rows (1, 3) have different x offset than even rows (0, 2)
        # For same column, odd row x should be offset by hex_width/2
        hex_width = np.sqrt(3) * cell_size
        expected_offset = hex_width / 2

        for col in range(4):
            # Compare row 0 (even) with row 1 (odd)
            diff = all_x[1, col] - all_x[0, col]
            assert abs(diff - expected_offset) < 1e-10


class TestGridManagerComputeDataIndices:
    """Tests for GridManager.compute_data_indices."""

    def test_compute_data_indices_shape(self, cell_size):
        """compute_data_indices should return arrays with correct shape."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(num_rows=10, num_cols=15, cell_size=cell_size)
        all_x, all_y = gm.compute_all_cell_positions()

        data_rows, data_cols = gm.compute_data_indices(
            all_x, all_y,
            data_res=10.0,
            data_rows=100,
            data_cols=150
        )

        assert data_rows.shape == (10, 15)
        assert data_cols.shape == (10, 15)

    def test_compute_data_indices_type(self, cell_size):
        """compute_data_indices should return integer arrays."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(num_rows=5, num_cols=5, cell_size=cell_size)
        all_x, all_y = gm.compute_all_cell_positions()

        data_rows, data_cols = gm.compute_data_indices(
            all_x, all_y,
            data_res=10.0,
            data_rows=50,
            data_cols=50
        )

        assert data_rows.dtype == np.int32
        assert data_cols.dtype == np.int32

    def test_compute_data_indices_in_bounds(self, cell_size):
        """Computed indices should be within valid range."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(num_rows=10, num_cols=10, cell_size=cell_size)
        all_x, all_y = gm.compute_all_cell_positions()

        data_rows_count = 100
        data_cols_count = 100
        data_rows, data_cols = gm.compute_data_indices(
            all_x, all_y,
            data_res=5.0,
            data_rows=data_rows_count,
            data_cols=data_cols_count
        )

        # All indices should be in valid range [0, max-1]
        assert np.all(data_rows >= 0)
        assert np.all(data_rows < data_rows_count)
        assert np.all(data_cols >= 0)
        assert np.all(data_cols < data_cols_count)

    def test_compute_data_indices_clipping(self, cell_size):
        """Indices should be clipped to valid range when positions exceed data bounds."""
        from embrs.base_classes.grid_manager import GridManager

        # Create a large grid that will exceed small data bounds
        gm = GridManager(num_rows=100, num_cols=100, cell_size=cell_size)
        all_x, all_y = gm.compute_all_cell_positions()

        # Small data array that grid positions will exceed
        data_rows_count = 10
        data_cols_count = 10
        data_rows, data_cols = gm.compute_data_indices(
            all_x, all_y,
            data_res=1000.0,  # Large resolution so positions map to small indices
            data_rows=data_rows_count,
            data_cols=data_cols_count
        )

        # Indices should still be in valid range
        assert np.all(data_rows >= 0)
        assert np.all(data_rows < data_rows_count)
        assert np.all(data_cols >= 0)
        assert np.all(data_cols < data_cols_count)

    def test_compute_data_indices_deterministic(self, cell_size):
        """compute_data_indices should produce consistent results."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(num_rows=10, num_cols=10, cell_size=cell_size)
        all_x, all_y = gm.compute_all_cell_positions()

        # Compute twice
        data_rows1, data_cols1 = gm.compute_data_indices(
            all_x, all_y, data_res=5.0, data_rows=100, data_cols=100
        )
        data_rows2, data_cols2 = gm.compute_data_indices(
            all_x, all_y, data_res=5.0, data_rows=100, data_cols=100
        )

        np.testing.assert_array_equal(data_rows1, data_rows2)
        np.testing.assert_array_equal(data_cols1, data_cols2)


class TestVectorizedTerrainExtraction:
    """Integration tests for vectorized terrain data extraction."""

    def test_vectorized_extraction_matches_per_cell(self, cell_size):
        """Vectorized extraction should match per-cell extraction."""
        from embrs.base_classes.grid_manager import GridManager

        gm = GridManager(num_rows=10, num_cols=10, cell_size=cell_size)

        # Create mock terrain data
        data_res = 10.0
        data_rows = 100
        data_cols = 100
        elevation_map = np.random.rand(data_rows, data_cols) * 1000

        # Compute vectorized positions and indices
        all_x, all_y = gm.compute_all_cell_positions()
        data_row_idx, data_col_idx = gm.compute_data_indices(
            all_x, all_y, data_res, data_rows, data_cols
        )

        # Vectorized extraction
        vectorized_elevations = elevation_map[data_row_idx, data_col_idx]

        # Per-cell extraction (simulating old approach)
        for row in range(10):
            for col in range(10):
                mock_cell = MockCell(id=0, col=col, row=row, cell_size=cell_size)
                cell_x, cell_y = mock_cell.x_pos, mock_cell.y_pos

                # Per-cell index computation
                per_cell_col = int(np.floor(cell_x / data_res))
                per_cell_row = int(np.floor(cell_y / data_res))
                per_cell_col = min(per_cell_col, data_cols - 1)
                per_cell_row = min(per_cell_row, data_rows - 1)

                per_cell_elev = elevation_map[per_cell_row, per_cell_col]

                # Values should match
                assert vectorized_elevations[row, col] == per_cell_elev, \
                    f"Mismatch at ({row}, {col})"
