"""Tests for Cell class.

These tests validate the hexagonal cell implementation including
position calculations, state management, and neighbor relationships.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from embrs.fire_simulator.cell import Cell
from embrs.utilities.fire_util import CellStates, CrownStatus
from embrs.utilities.data_classes import CellData
from embrs.models.fuel_models import Anderson13


class TestCellInitialization:
    """Tests for Cell initialization and position calculations."""

    def test_basic_initialization(self):
        """Cell should initialize with correct basic properties."""
        cell = Cell(id=0, col=5, row=5, cell_size=30.0)

        assert cell.id == 0
        assert cell._col == 5
        assert cell._row == 5
        assert cell._cell_size == 30.0

    def test_position_calculation_even_row(self):
        """Even row cells should have standard x position."""
        cell = Cell(id=0, col=3, row=4, cell_size=30.0)  # Row 4 is even

        # For even rows: x = col * cell_size * sqrt(3)
        expected_x = 3 * 30.0 * np.sqrt(3)
        expected_y = 4 * 30.0 * 1.5

        assert cell._x_pos == pytest.approx(expected_x, abs=0.001)
        assert cell._y_pos == pytest.approx(expected_y, abs=0.001)

    def test_position_calculation_odd_row(self):
        """Odd row cells should have offset x position."""
        cell = Cell(id=0, col=3, row=5, cell_size=30.0)  # Row 5 is odd

        # For odd rows: x = (col + 0.5) * cell_size * sqrt(3)
        expected_x = (3 + 0.5) * 30.0 * np.sqrt(3)
        expected_y = 5 * 30.0 * 1.5

        assert cell._x_pos == pytest.approx(expected_x, abs=0.001)
        assert cell._y_pos == pytest.approx(expected_y, abs=0.001)

    def test_cell_area_calculation(self):
        """Cell area should be calculated correctly for hexagon."""
        cell = Cell(id=0, col=0, row=0, cell_size=30.0)

        # Hexagon area = (3 * sqrt(3) / 2) * edge_length^2
        expected_area = (3 * np.sqrt(3) / 2) * 30.0 ** 2

        assert cell._cell_area == pytest.approx(expected_area, abs=0.01)

    def test_initial_state_defaults(self):
        """Cell should initialize with correct default states."""
        cell = Cell(id=0, col=0, row=0, cell_size=30.0)

        # Default states
        assert cell._retardant == False
        assert cell._retardant_factor == 1.0
        assert cell.retardant_expiration_s == -1.0
        assert cell.local_rain == 0.0
        assert cell._break_width == 0
        assert cell.breached == True
        assert cell.lofted == False
        assert cell._parent is None
        assert cell._arrival_time == -999


class TestCellData:
    """Tests for cell data configuration."""

    @pytest.fixture
    def basic_cell(self):
        """Create a basic cell for testing."""
        return Cell(id=1, col=5, row=5, cell_size=30.0)

    @pytest.fixture
    def sample_cell_data(self):
        """Create sample cell data."""
        return CellData(
            fuel_type=Anderson13(1),
            elevation=500.0,
            aspect=180.0,
            slope_deg=15.0,
            canopy_cover=0.0,
            canopy_height=0.0,
            canopy_base_height=0.0,
            canopy_bulk_density=0.0,
            init_dead_mf=[0.08, 0.09, 0.10],
            live_h_mf=0.30,
            live_w_mf=0.30
        )

    def test_set_cell_data_fuel(self, basic_cell, sample_cell_data):
        """Cell data should set fuel model correctly."""
        basic_cell._set_cell_data(sample_cell_data)

        assert basic_cell.fuel is not None
        assert isinstance(basic_cell.fuel, Anderson13)

    def test_set_cell_data_terrain(self, basic_cell, sample_cell_data):
        """Cell data should set terrain properties correctly."""
        basic_cell._set_cell_data(sample_cell_data)

        assert basic_cell.elevation_m == 500.0
        assert basic_cell.aspect == 180.0
        assert basic_cell.slope_deg == 15.0


class TestCellProperties:
    """Tests for cell property accessors."""

    @pytest.fixture
    def configured_cell(self):
        """Create a fully configured cell."""
        cell = Cell(id=1, col=5, row=5, cell_size=30.0)
        cell_data = CellData(
            fuel_type=Anderson13(4),
            elevation=300.0,
            aspect=225.0,
            slope_deg=25.0,
            canopy_cover=0.6,
            canopy_height=15.0,
            canopy_base_height=3.0,
            canopy_bulk_density=0.1,
            init_dead_mf=[0.06, 0.07, 0.08],
            live_h_mf=0.50,
            live_w_mf=0.80
        )
        cell._set_cell_data(cell_data)
        return cell

    def test_col_property(self, configured_cell):
        """col property should return column index."""
        assert configured_cell.col == 5

    def test_row_property(self, configured_cell):
        """row property should return row index."""
        assert configured_cell.row == 5

    def test_x_property(self, configured_cell):
        """x_pos property should return x position."""
        assert configured_cell.x_pos == configured_cell._x_pos

    def test_y_property(self, configured_cell):
        """y_pos property should return y position."""
        assert configured_cell.y_pos == configured_cell._y_pos


class TestCellParentReference:
    """Tests for parent simulation reference."""

    def test_set_parent(self):
        """set_parent should create weak reference."""
        cell = Cell(id=0, col=0, row=0, cell_size=30.0)
        mock_parent = MagicMock()

        cell.set_parent(mock_parent)

        # Should be able to dereference
        assert cell._parent() is mock_parent

    def test_parent_weak_reference(self):
        """Parent reference should be weak (not prevent garbage collection)."""
        import weakref

        cell = Cell(id=0, col=0, row=0, cell_size=30.0)

        # Create parent and set
        mock_parent = MagicMock()
        cell.set_parent(mock_parent)

        # Verify weak ref
        assert isinstance(cell._parent, weakref.ref)


class TestCellStateTransitions:
    """Tests for cell state transitions (FUEL -> FIRE -> BURNT)."""

    @pytest.fixture
    def fuel_cell(self):
        """Create a cell in FUEL state."""
        cell = Cell(id=0, col=5, row=5, cell_size=30.0)
        cell_data = CellData(
            fuel_type=Anderson13(1),
            elevation=100.0,
            aspect=0.0,
            slope_deg=0.0,
            canopy_cover=0.0,
            canopy_height=0.0,
            canopy_base_height=0.0,
            canopy_bulk_density=0.0,
        )
        cell._set_cell_data(cell_data)
        return cell

    def test_initial_state_is_fuel(self, fuel_cell):
        """Newly configured cell should be in FUEL state."""
        assert fuel_cell.state == CellStates.FUEL

    def test_cell_states_values(self):
        """Cell states should have expected integer values."""
        assert CellStates.BURNT == 0
        assert CellStates.FUEL == 1
        assert CellStates.FIRE == 2


class TestCellGridPosition:
    """Tests for cell position in different grid configurations."""

    @pytest.mark.parametrize("col,row,cell_size", [
        (0, 0, 30.0),
        (10, 10, 30.0),
        (5, 3, 25.0),
        (0, 1, 30.0),  # Odd row at origin
    ])
    def test_y_position_formula(self, col, row, cell_size):
        """Y position should follow y = row * cell_size * 1.5."""
        cell = Cell(id=0, col=col, row=row, cell_size=cell_size)

        expected_y = row * cell_size * 1.5
        assert cell._y_pos == pytest.approx(expected_y, abs=0.001)

    def test_adjacent_row_y_offset(self):
        """Adjacent rows should be offset by 1.5 * cell_size in y."""
        cell_size = 30.0
        cell1 = Cell(id=0, col=5, row=4, cell_size=cell_size)
        cell2 = Cell(id=1, col=5, row=5, cell_size=cell_size)

        y_offset = cell2._y_pos - cell1._y_pos

        assert y_offset == pytest.approx(1.5 * cell_size, abs=0.001)


class TestCellRetardant:
    """Tests for fire retardant application on cells."""

    def test_initial_no_retardant(self):
        """Cell should have no retardant initially."""
        cell = Cell(id=0, col=0, row=0, cell_size=30.0)

        assert cell._retardant == False
        assert cell._retardant_factor == 1.0

    def test_retardant_factor_range(self):
        """Retardant factor should be between 0 and 1."""
        cell = Cell(id=0, col=0, row=0, cell_size=30.0)

        # Simulate retardant application
        cell._retardant = True
        cell._retardant_factor = 0.5

        assert 0.0 <= cell._retardant_factor <= 1.0


class TestCellBreakWidth:
    """Tests for fuel discontinuities (roads, firebreaks)."""

    def test_initial_no_break(self):
        """Cell should have no fuel break initially."""
        cell = Cell(id=0, col=0, row=0, cell_size=30.0)

        assert cell._break_width == 0
        assert cell.breached == True

    def test_break_width_affects_breached(self):
        """Cells with breaks should track breach status."""
        cell = Cell(id=0, col=0, row=0, cell_size=30.0)

        cell._break_width = 10.0
        cell.breached = False

        assert cell._break_width == 10.0
        assert cell.breached == False


class TestCellResetToFuel:
    """Tests for Cell.reset_to_fuel() method."""

    @pytest.fixture
    def configured_cell(self):
        """Create a fully configured cell with neighbors.

        Uses Anderson13(4) which has fuel classes at indices 0, 1, 2, 5
        so moisture reset can be tested across multiple fuel classes.
        """
        cell = Cell(id=42, col=5, row=5, cell_size=30.0)
        cell_data = CellData(
            fuel_type=Anderson13(4),
            elevation=500.0,
            aspect=180.0,
            slope_deg=15.0,
            canopy_cover=0.0,
            canopy_height=0.0,
            canopy_base_height=0.0,
            canopy_bulk_density=0.0,
            init_dead_mf=[0.08, 0.09, 0.10],
            live_h_mf=0.30,
            live_w_mf=0.30
        )
        cell._set_cell_data(cell_data)
        # Simulate neighbors being set
        cell._neighbors = {100: (1, 0), 101: (-1, 0), 102: (0, 1)}
        cell._burnable_neighbors = dict(cell._neighbors)
        return cell

    def test_reset_restores_fuel_state(self, configured_cell):
        """reset_to_fuel should restore CellStates.FUEL."""
        configured_cell._state = CellStates.FIRE
        configured_cell.fully_burning = True
        configured_cell._crown_status = CrownStatus.PASSIVE

        configured_cell.reset_to_fuel()

        assert configured_cell._state == CellStates.FUEL
        assert configured_cell.fully_burning == False
        assert configured_cell._crown_status == CrownStatus.NONE
        assert configured_cell.cfb == 0
        assert configured_cell.reaction_intensity == 0

    def test_reset_clears_spread_arrays(self, configured_cell):
        """reset_to_fuel should reset all spread tracking arrays."""
        # Simulate fire spread state
        configured_cell.directions = np.array([0, 60, 120, 180, 240, 300])
        configured_cell.distances = np.array([10, 10, 10, 10, 10, 10])
        configured_cell.fire_spread = np.array([1.0, 2.0, 3.0])
        configured_cell.r_t = np.array([0.5, 0.6, 0.7])
        configured_cell.has_steady_state = True
        configured_cell.e = 0.8
        configured_cell.intersections = {1, 2, 3}

        configured_cell.reset_to_fuel()

        assert configured_cell.directions is None
        assert configured_cell.distances is None
        assert configured_cell.end_pts is None
        assert configured_cell.r_h_ss is None
        assert configured_cell.I_h_ss is None
        assert len(configured_cell.fire_spread) == 0
        assert configured_cell.r_t[0] == 0
        assert configured_cell.has_steady_state == False
        assert configured_cell.e == 0
        assert configured_cell.alpha is None
        assert len(configured_cell.intersections) == 0

    def test_reset_clears_suppression_effects(self, configured_cell):
        """reset_to_fuel should reset retardant, rain, and firebreak state."""
        configured_cell._retardant = True
        configured_cell._retardant_factor = 0.3
        configured_cell.retardant_expiration_s = 5000.0
        configured_cell.local_rain = 2.5
        configured_cell._break_width = 10
        configured_cell.breached = False
        configured_cell.lofted = True
        configured_cell._arrival_time = 1234.5

        configured_cell.reset_to_fuel()

        assert configured_cell._retardant == False
        assert configured_cell._retardant_factor == 1.0
        assert configured_cell.retardant_expiration_s == -1.0
        assert configured_cell.local_rain == 0.0
        assert configured_cell._break_width == 0
        assert configured_cell.breached == True
        assert configured_cell.lofted == False
        assert configured_cell._arrival_time == -999

    def test_reset_clears_wind_forecast(self, configured_cell):
        """reset_to_fuel should clear wind forecast arrays."""
        configured_cell.forecast_wind_speeds = [5.0, 6.0, 7.0]
        configured_cell.forecast_wind_dirs = [90.0, 95.0, 100.0]

        configured_cell.reset_to_fuel()

        assert configured_cell.forecast_wind_speeds == []
        assert configured_cell.forecast_wind_dirs == []

    def test_reset_restores_moisture(self, configured_cell):
        """reset_to_fuel should reset moisture to initial values."""
        # Modify moisture values
        configured_cell.fmois = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        configured_cell.moist_update_time_s = 99999

        configured_cell.reset_to_fuel()

        assert configured_cell.moist_update_time_s == 0
        # Anderson13(4) has rel_indices [0, 1, 2, 5]
        assert configured_cell.fmois[0] == pytest.approx(0.08)  # 1-hr dead
        assert configured_cell.fmois[1] == pytest.approx(0.09)  # 10-hr dead
        assert configured_cell.fmois[2] == pytest.approx(0.10)  # 100-hr dead
        assert configured_cell.fmois[5] == pytest.approx(0.30)  # live woody

    def test_reset_restores_burnable_neighbors(self, configured_cell):
        """reset_to_fuel should restore full neighbor set to _burnable_neighbors."""
        # Simulate neighbors being removed during fire spread
        del configured_cell._burnable_neighbors[100]
        del configured_cell._burnable_neighbors[101]
        assert len(configured_cell._burnable_neighbors) == 1

        configured_cell.reset_to_fuel()

        assert len(configured_cell._burnable_neighbors) == 3
        assert set(configured_cell._burnable_neighbors.keys()) == {100, 101, 102}

    def test_reset_preserves_immutable_attributes(self, configured_cell):
        """reset_to_fuel should NOT change position, fuel, terrain, or polygon."""
        # Capture immutable state before reset
        orig_id = configured_cell.id
        orig_col = configured_cell.col
        orig_row = configured_cell.row
        orig_x = configured_cell.x_pos
        orig_y = configured_cell.y_pos
        orig_fuel = configured_cell.fuel
        orig_elevation = configured_cell.elevation_m
        orig_slope = configured_cell.slope_deg
        orig_aspect = configured_cell.aspect
        orig_polygon = configured_cell.polygon
        orig_cell_size = configured_cell.cell_size
        orig_cell_area = configured_cell.cell_area

        # Dirty up the cell
        configured_cell._state = CellStates.FIRE
        configured_cell.fully_burning = True

        configured_cell.reset_to_fuel()

        assert configured_cell.id == orig_id
        assert configured_cell.col == orig_col
        assert configured_cell.row == orig_row
        assert configured_cell.x_pos == orig_x
        assert configured_cell.y_pos == orig_y
        assert configured_cell.fuel is orig_fuel
        assert configured_cell.elevation_m == orig_elevation
        assert configured_cell.slope_deg == orig_slope
        assert configured_cell.aspect == orig_aspect
        assert configured_cell.polygon is orig_polygon
        assert configured_cell.cell_size == orig_cell_size
        assert configured_cell.cell_area == orig_cell_area

    def test_reset_resets_dfm_objects(self, configured_cell):
        """reset_to_fuel should reset DFM stick state (not recreate objects)."""
        # Get original DFM object references
        orig_dfms = configured_cell.dfms[:]

        # Dirty up moisture
        configured_cell.fmois = np.ones(6)

        configured_cell.reset_to_fuel()

        # Same DFM objects, not recreated
        for orig, current in zip(orig_dfms, configured_cell.dfms):
            assert orig is current
        # DFMs should be re-initialized (m_init should be False)
        for dfm in configured_cell.dfms:
            assert dfm.initialized() == False
