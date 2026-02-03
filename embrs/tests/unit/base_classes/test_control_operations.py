"""Unit tests for control action operations in BaseFireSim and ControlActionHandler.

These tests verify the behavior of control action methods (retardant,
water drops, firelines) and ensure ControlActionHandler matches the original.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, Mock, patch
from shapely.geometry import LineString, Point

from embrs.base_classes.control_handler import ControlActionHandler
from embrs.base_classes.grid_manager import GridManager


class MockFuel:
    """Mock fuel model for testing."""
    def __init__(self, burnable=True, model_num=1, dead_mx=0.25):
        self.burnable = burnable
        self.model_num = model_num
        self.dead_mx = dead_mx


class MockCell:
    """Mock cell for control action testing."""
    def __init__(self, id, row, col, x_pos=0.0, y_pos=0.0):
        self.id = id
        self._row = row
        self._col = col
        self.x_pos = x_pos
        self.y_pos = y_pos
        self._fuel = MockFuel()
        self._retardant = False
        self._retardant_factor = 1.0
        self.retardant_expiration_s = -1.0
        self._break_width = 0.0
        self.fmois = [0.08, 0.10, 0.12]

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

    @property
    def fuel(self):
        return self._fuel

    def add_retardant(self, duration_hr, effectiveness):
        """Mock retardant application."""
        self._retardant = True
        self._retardant_factor = 1.0 - effectiveness
        self.retardant_expiration_s = duration_hr * 3600

    def water_drop_as_rain(self, water_depth_cm):
        """Mock water drop as rain."""
        pass

    def water_drop_as_moisture_bump(self, moisture_inc):
        """Mock moisture bump."""
        pass

    def _set_fuel_type(self, fuel):
        """Mock fuel type setter."""
        self._fuel = fuel


class MockControlActionHandler:
    """Mock class containing control action methods from BaseFireSim.

    This replicates the exact control action logic for testing.
    """

    def __init__(self, cell_size=30.0):
        self._cell_size = cell_size
        self._time_step = 60
        self._curr_time_s = 0
        self._long_term_retardants = set()
        self._active_water_drops = []
        self._active_firelines = {}
        self._fire_break_cells = []
        self._fire_breaks = []
        self.fire_break_dict = {}
        self._updated_cells = {}
        self._new_fire_break_cache = []
        self._cell_dict = {}
        self._cell_grid = None
        self.logger = None
        self._visualizer = None
        self.FuelClass = MockFuel

        # Mock grid for testing
        self._setup_mock_grid()

    def _setup_mock_grid(self, rows=10, cols=10):
        """Set up a mock cell grid for testing."""
        self._cell_grid = np.empty((rows, cols), dtype=object)
        cell_id = 0
        for i in range(rows):
            for j in range(cols):
                cell = MockCell(cell_id, i, j, j * 30.0, i * 30.0)
                self._cell_grid[i, j] = cell
                self._cell_dict[cell_id] = cell
                cell_id += 1

    def get_cell_from_xy(self, x_m, y_m, oob_ok=False):
        """Mock cell lookup by coordinates."""
        row = int(y_m / 30.0)
        col = int(x_m / 30.0)
        if 0 <= row < self._cell_grid.shape[0] and 0 <= col < self._cell_grid.shape[1]:
            return self._cell_grid[row, col]
        if oob_ok:
            return None
        raise ValueError("Out of bounds")

    def get_cell_from_indices(self, row, col):
        """Mock cell lookup by indices."""
        return self._cell_grid[row, col]

    @property
    def curr_time_s(self):
        return self._curr_time_s

    def add_retardant_at_xy(self, x_m: float, y_m: float, duration_hr: float, effectiveness: float):
        """Apply long-term fire retardant at the specified coordinates."""
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            self.add_retardant_at_cell(cell, duration_hr, effectiveness)

    def add_retardant_at_indices(self, row: int, col: int, duration_hr: float, effectiveness: float):
        """Apply long-term fire retardant at the specified grid indices."""
        cell = self.get_cell_from_indices(row, col)
        self.add_retardant_at_cell(cell, duration_hr, effectiveness)

    def add_retardant_at_cell(self, cell: MockCell, duration_hr: float, effectiveness: float):
        """Apply long-term fire retardant to the specified cell."""
        # Ensure that effectiveness is between 0 and 1
        effectiveness = min(max(effectiveness, 0), 1)

        if cell.fuel.burnable:
            cell.add_retardant(duration_hr, effectiveness)
            self._long_term_retardants.add(cell)
            self._updated_cells[cell.id] = cell

    def update_long_term_retardants(self):
        """Update long-term retardant effects and remove expired retardants."""
        for cell in self._long_term_retardants.copy():
            if cell.retardant_expiration_s <= self._curr_time_s:
                cell._retardant = False
                cell._retardant_factor = 1.0
                cell.retardant_expiration_s = -1.0

                self._updated_cells[cell.id] = cell

                self._long_term_retardants.remove(cell)

    def water_drop_at_xy_as_rain(self, x_m: float, y_m: float, water_depth_cm: float):
        """Apply water drop as equivalent rainfall at the specified coordinates."""
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            self.water_drop_at_cell_as_rain(cell, water_depth_cm)

    def water_drop_at_cell_as_rain(self, cell: MockCell, water_depth_cm: float):
        """Apply water drop as equivalent rainfall to the specified cell."""
        if water_depth_cm < 0:
            raise ValueError(f"Water depth must be >=0, {water_depth_cm} passed in")

        if cell.fuel.burnable:
            cell.water_drop_as_rain(water_depth_cm)
            self._active_water_drops.append(cell)
            self._updated_cells[cell.id] = cell

    def water_drop_at_xy_as_moisture_bump(self, x_m: float, y_m: float, moisture_inc: float):
        """Apply water drop as direct moisture increase at the specified coordinates."""
        cell = self.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            self.water_drop_at_cell_as_moisture_bump(cell, moisture_inc)

    def water_drop_at_cell_as_moisture_bump(self, cell: MockCell, moisture_inc: float):
        """Apply water drop as direct moisture increase to the specified cell."""
        if moisture_inc < 0:
            raise ValueError(f"Moisture increase must be >0, {moisture_inc} passed in")

        if cell.fuel.burnable:
            cell.water_drop_as_moisture_bump(moisture_inc)
            self._active_water_drops.append(cell)
            self._updated_cells[cell.id] = cell

    def construct_fireline(self, line: LineString, width_m: float,
                           construction_rate: float = None, id: str = None) -> str:
        """Construct a fire break along a line geometry."""
        if construction_rate is None:
            # Add fire break instantly
            self._apply_firebreak(line, width_m)

            if id is None:
                id = str(len(self._fire_breaks) + 1)

            self._fire_breaks.append((line, width_m, id))
            self.fire_break_dict[id] = (line, width_m)

            # Add to cache
            cache_entry = {
                "id": id,
                "line": line,
                "width": width_m,
                "time": self.curr_time_s,
                "logged": False,
                "visualized": False
            }
            self._new_fire_break_cache.append(cache_entry)
        else:
            if id is None:
                id = str(len(self._fire_breaks) + len(self._active_firelines) + 1)

            # Create an active fireline to be updated over time
            self._active_firelines[id] = {
                "line": line,
                "width": width_m,
                "rate": construction_rate,
                "progress": 0.0,
                "partial_line": LineString([]),
                "cells": set()
            }

        return id

    def _apply_firebreak(self, line: LineString, break_width: float):
        """Apply a fire break along a line geometry."""
        cells = self._get_cells_at_line(line)
        for cell in cells:
            if cell not in self._fire_break_cells:
                self._fire_break_cells.append(cell)
            cell._break_width += break_width

    def _get_cells_at_line(self, line: LineString):
        """Get cells along a line (simplified for testing)."""
        cells = []
        step_size = self._cell_size / 4.0
        for i in range(int(line.length / step_size) + 1):
            point = line.interpolate(i * step_size)
            cell = self.get_cell_from_xy(point.x, point.y, oob_ok=True)
            if cell is not None and cell not in cells:
                cells.append(cell)
        return cells

    def stop_fireline_construction(self, fireline_id: str):
        """Stop construction of an active fireline."""
        if self._active_firelines.get(fireline_id) is not None:
            fireline = self._active_firelines[fireline_id]
            partial_line = fireline["partial_line"]
            self._fire_breaks.append((partial_line, fireline["width"], fireline_id))
            self.fire_break_dict[fireline_id] = (partial_line, fireline["width"])
            del self._active_firelines[fireline_id]


class TestAddRetardant:
    """Tests for retardant application methods."""

    def test_add_retardant_at_cell_applies_retardant(self):
        """Retardant should be applied to a burnable cell."""
        handler = MockControlActionHandler()
        cell = handler.get_cell_from_indices(5, 5)

        handler.add_retardant_at_cell(cell, 2.0, 0.5)

        assert cell._retardant is True
        assert cell._retardant_factor == pytest.approx(0.5)
        assert cell in handler._long_term_retardants
        assert cell.id in handler._updated_cells

    def test_add_retardant_clamps_effectiveness_to_one(self):
        """Effectiveness should be clamped to maximum 1.0."""
        handler = MockControlActionHandler()
        cell = handler.get_cell_from_indices(5, 5)

        handler.add_retardant_at_cell(cell, 2.0, 1.5)  # >1

        assert cell._retardant_factor == pytest.approx(0.0)  # 1 - 1.0 = 0

    def test_add_retardant_clamps_effectiveness_to_zero(self):
        """Effectiveness should be clamped to minimum 0.0."""
        handler = MockControlActionHandler()
        cell = handler.get_cell_from_indices(5, 5)

        handler.add_retardant_at_cell(cell, 2.0, -0.5)  # <0

        assert cell._retardant_factor == pytest.approx(1.0)  # 1 - 0 = 1

    def test_add_retardant_at_xy(self):
        """Retardant should be applied at xy coordinates."""
        handler = MockControlActionHandler()

        handler.add_retardant_at_xy(150.0, 150.0, 2.0, 0.5)

        # Cell at approximately (5, 5)
        cell = handler.get_cell_from_xy(150.0, 150.0)
        assert cell._retardant is True

    def test_add_retardant_at_indices(self):
        """Retardant should be applied at grid indices."""
        handler = MockControlActionHandler()

        handler.add_retardant_at_indices(3, 4, 2.0, 0.5)

        cell = handler.get_cell_from_indices(3, 4)
        assert cell._retardant is True

    def test_add_retardant_not_applied_to_unburnable(self):
        """Retardant should not be applied to non-burnable cells."""
        handler = MockControlActionHandler()
        cell = handler.get_cell_from_indices(5, 5)
        cell._fuel = MockFuel(burnable=False)

        handler.add_retardant_at_cell(cell, 2.0, 0.5)

        assert cell._retardant is False
        assert cell not in handler._long_term_retardants


class TestUpdateLongTermRetardants:
    """Tests for long-term retardant update method."""

    def test_retardant_removed_when_expired(self):
        """Expired retardant should be removed."""
        handler = MockControlActionHandler()
        cell = handler.get_cell_from_indices(5, 5)

        # Apply retardant with 1 hour duration
        handler.add_retardant_at_cell(cell, 1.0, 0.5)

        # Advance time past expiration
        handler._curr_time_s = 3601  # Just past 1 hour

        handler.update_long_term_retardants()

        assert cell._retardant is False
        assert cell._retardant_factor == pytest.approx(1.0)
        assert cell not in handler._long_term_retardants

    def test_retardant_not_removed_before_expiration(self):
        """Retardant should remain before expiration."""
        handler = MockControlActionHandler()
        cell = handler.get_cell_from_indices(5, 5)

        handler.add_retardant_at_cell(cell, 1.0, 0.5)

        # Advance time but not past expiration
        handler._curr_time_s = 3000  # Before 1 hour

        handler.update_long_term_retardants()

        assert cell._retardant is True
        assert cell in handler._long_term_retardants


class TestWaterDrop:
    """Tests for water drop methods."""

    def test_water_drop_as_rain_at_cell(self):
        """Water drop as rain should be applied to a burnable cell."""
        handler = MockControlActionHandler()
        cell = handler.get_cell_from_indices(5, 5)

        handler.water_drop_at_cell_as_rain(cell, 1.0)

        assert cell in handler._active_water_drops
        assert cell.id in handler._updated_cells

    def test_water_drop_negative_depth_raises(self):
        """Negative water depth should raise ValueError."""
        handler = MockControlActionHandler()
        cell = handler.get_cell_from_indices(5, 5)

        with pytest.raises(ValueError, match="Water depth must be >=0"):
            handler.water_drop_at_cell_as_rain(cell, -1.0)

    def test_water_drop_as_moisture_bump(self):
        """Water drop as moisture bump should be applied."""
        handler = MockControlActionHandler()
        cell = handler.get_cell_from_indices(5, 5)

        handler.water_drop_at_cell_as_moisture_bump(cell, 0.1)

        assert cell in handler._active_water_drops
        assert cell.id in handler._updated_cells

    def test_water_drop_negative_moisture_raises(self):
        """Negative moisture increment should raise ValueError."""
        handler = MockControlActionHandler()
        cell = handler.get_cell_from_indices(5, 5)

        with pytest.raises(ValueError, match="Moisture increase must be >0"):
            handler.water_drop_at_cell_as_moisture_bump(cell, -0.1)

    def test_water_drop_at_xy_as_rain(self):
        """Water drop at xy coordinates."""
        handler = MockControlActionHandler()

        handler.water_drop_at_xy_as_rain(150.0, 150.0, 1.0)

        cell = handler.get_cell_from_xy(150.0, 150.0)
        assert cell in handler._active_water_drops

    def test_water_drop_not_applied_to_unburnable(self):
        """Water drop should not be applied to non-burnable cells."""
        handler = MockControlActionHandler()
        cell = handler.get_cell_from_indices(5, 5)
        cell._fuel = MockFuel(burnable=False)

        handler.water_drop_at_cell_as_rain(cell, 1.0)

        assert cell not in handler._active_water_drops


class TestConstructFireline:
    """Tests for fireline construction methods."""

    def test_instant_fireline_construction(self):
        """Instant fireline should be added immediately."""
        handler = MockControlActionHandler()
        line = LineString([(0, 0), (90, 0)])  # 90m horizontal line

        fireline_id = handler.construct_fireline(line, 2.0)

        assert len(handler._fire_breaks) == 1
        assert fireline_id in handler.fire_break_dict
        assert len(handler._new_fire_break_cache) == 1

    def test_instant_fireline_with_custom_id(self):
        """Instant fireline should use custom ID if provided."""
        handler = MockControlActionHandler()
        line = LineString([(0, 0), (90, 0)])

        fireline_id = handler.construct_fireline(line, 2.0, id="custom_id")

        assert fireline_id == "custom_id"
        assert "custom_id" in handler.fire_break_dict

    def test_progressive_fireline_construction(self):
        """Progressive fireline should be added to active firelines."""
        handler = MockControlActionHandler()
        line = LineString([(0, 0), (90, 0)])

        fireline_id = handler.construct_fireline(line, 2.0, construction_rate=1.0)

        assert len(handler._fire_breaks) == 0  # Not yet complete
        assert fireline_id in handler._active_firelines
        assert handler._active_firelines[fireline_id]["rate"] == 1.0

    def test_fireline_cells_receive_break_width(self):
        """Cells along fireline should have increased break width."""
        handler = MockControlActionHandler()
        line = LineString([(0, 0), (90, 0)])

        handler.construct_fireline(line, 5.0)

        # Check that at least some cells have break width
        cells_with_break = [c for c in handler._fire_break_cells if c._break_width > 0]
        assert len(cells_with_break) > 0

    def test_stop_fireline_construction(self):
        """Stopping construction should finalize the fireline."""
        handler = MockControlActionHandler()
        line = LineString([(0, 0), (90, 0)])

        fireline_id = handler.construct_fireline(line, 2.0, construction_rate=1.0)

        assert fireline_id in handler._active_firelines

        handler.stop_fireline_construction(fireline_id)

        assert fireline_id not in handler._active_firelines
        assert fireline_id in handler.fire_break_dict


class TestFirelineProgression:
    """Tests for active fireline progression."""

    def test_fireline_auto_id_generation(self):
        """Firelines should auto-generate IDs correctly."""
        handler = MockControlActionHandler()
        line1 = LineString([(0, 0), (90, 0)])
        line2 = LineString([(0, 30), (90, 30)])

        id1 = handler.construct_fireline(line1, 2.0)
        id2 = handler.construct_fireline(line2, 2.0)

        assert id1 != id2


class TestControlActionIntegration:
    """Integration tests for control action methods."""

    def test_multiple_retardants(self):
        """Multiple retardant applications should be tracked."""
        handler = MockControlActionHandler()

        handler.add_retardant_at_indices(1, 1, 2.0, 0.5)
        handler.add_retardant_at_indices(2, 2, 2.0, 0.5)
        handler.add_retardant_at_indices(3, 3, 2.0, 0.5)

        assert len(handler._long_term_retardants) == 3

    def test_multiple_water_drops(self):
        """Multiple water drops should be tracked."""
        handler = MockControlActionHandler()

        handler.water_drop_at_xy_as_rain(30.0, 30.0, 1.0)
        handler.water_drop_at_xy_as_rain(60.0, 60.0, 1.0)
        handler.water_drop_at_xy_as_moisture_bump(90.0, 90.0, 0.1)

        assert len(handler._active_water_drops) == 3

    def test_updated_cells_tracking(self):
        """All control actions should track updated cells."""
        handler = MockControlActionHandler()

        handler.add_retardant_at_indices(1, 1, 2.0, 0.5)
        handler.water_drop_at_indices_as_rain(2, 2, 1.0)

        assert len(handler._updated_cells) == 2

    def water_drop_at_indices_as_rain(self, row, col, water_depth_cm):
        """Helper to add water drop at indices."""
        cell = self.get_cell_from_indices(row, col)
        self.water_drop_at_cell_as_rain(cell, water_depth_cm)


# Add the missing method to MockControlActionHandler
MockControlActionHandler.water_drop_at_indices_as_rain = lambda self, row, col, water_depth_cm: \
    self.water_drop_at_cell_as_rain(self.get_cell_from_indices(row, col), water_depth_cm)


# ==============================================================================
# Tests for the real ControlActionHandler class
# ==============================================================================

class MockGridManagerForControl:
    """Mock grid manager for ControlActionHandler testing."""

    def __init__(self, rows=10, cols=10, cell_size=30.0):
        self._rows = rows
        self._cols = cols
        self._cell_size = cell_size
        self._cell_grid = np.empty((rows, cols), dtype=object)
        self._cell_dict = {}
        self._setup_grid()

    def _setup_grid(self):
        """Set up a mock cell grid."""
        cell_id = 0
        for i in range(self._rows):
            for j in range(self._cols):
                cell = MockCell(cell_id, i, j, j * self._cell_size, i * self._cell_size)
                self._cell_grid[i, j] = cell
                self._cell_dict[cell_id] = cell
                cell_id += 1

    def get_cell_from_xy(self, x_m, y_m, oob_ok=False):
        """Mock cell lookup by coordinates."""
        row = int(y_m / self._cell_size)
        col = int(x_m / self._cell_size)
        if 0 <= row < self._rows and 0 <= col < self._cols:
            return self._cell_grid[row, col]
        if oob_ok:
            return None
        raise ValueError("Out of bounds")

    def get_cell_from_indices(self, row, col):
        """Mock cell lookup by indices."""
        return self._cell_grid[row, col]

    def get_cells_at_geometry(self, geom):
        """Get cells along a line geometry (simplified)."""
        cells = []
        step_size = self._cell_size / 4.0
        for i in range(int(geom.length / step_size) + 1):
            point = geom.interpolate(i * step_size)
            cell = self.get_cell_from_xy(point.x, point.y, oob_ok=True)
            if cell is not None and cell not in cells:
                cells.append(cell)
        return cells


@pytest.fixture
def control_handler():
    """Create a ControlActionHandler with mocked dependencies."""
    grid_manager = MockGridManagerForControl(rows=10, cols=10, cell_size=30.0)

    def fuel_factory(model_num):
        return MockFuel(burnable=(model_num != 91), model_num=model_num)

    handler = ControlActionHandler(
        grid_manager=grid_manager,
        cell_size=30.0,
        time_step=60,
        fuel_class_factory=fuel_factory
    )
    handler.set_updated_cells_ref({})
    handler.set_time_accessor(lambda: 0)

    return handler


class TestControlActionHandlerRetardant:
    """Tests for ControlActionHandler retardant methods."""

    def test_add_retardant_at_cell(self, control_handler):
        """Retardant should be applied to a burnable cell."""
        cell = control_handler._grid_manager.get_cell_from_indices(5, 5)

        control_handler.add_retardant_at_cell(cell, 2.0, 0.5)

        assert cell._retardant is True
        assert cell._retardant_factor == pytest.approx(0.5)
        assert cell in control_handler.long_term_retardants

    def test_add_retardant_clamps_effectiveness(self, control_handler):
        """Effectiveness should be clamped to [0, 1]."""
        cell1 = control_handler._grid_manager.get_cell_from_indices(5, 5)
        cell2 = control_handler._grid_manager.get_cell_from_indices(5, 6)

        control_handler.add_retardant_at_cell(cell1, 2.0, 1.5)  # >1
        control_handler.add_retardant_at_cell(cell2, 2.0, -0.5)  # <0

        assert cell1._retardant_factor == pytest.approx(0.0)  # 1 - 1.0 = 0
        assert cell2._retardant_factor == pytest.approx(1.0)  # 1 - 0 = 1

    def test_add_retardant_at_xy(self, control_handler):
        """Retardant should be applied at xy coordinates."""
        control_handler.add_retardant_at_xy(150.0, 150.0, 2.0, 0.5)

        cell = control_handler._grid_manager.get_cell_from_xy(150.0, 150.0)
        assert cell._retardant is True

    def test_add_retardant_at_indices(self, control_handler):
        """Retardant should be applied at grid indices."""
        control_handler.add_retardant_at_indices(3, 4, 2.0, 0.5)

        cell = control_handler._grid_manager.get_cell_from_indices(3, 4)
        assert cell._retardant is True

    def test_add_retardant_not_applied_to_unburnable(self, control_handler):
        """Retardant should not be applied to non-burnable cells."""
        cell = control_handler._grid_manager.get_cell_from_indices(5, 5)
        cell._fuel = MockFuel(burnable=False)

        control_handler.add_retardant_at_cell(cell, 2.0, 0.5)

        assert cell._retardant is False
        assert cell not in control_handler.long_term_retardants


class TestControlActionHandlerRetardantUpdate:
    """Tests for ControlActionHandler retardant update method."""

    def test_retardant_removed_when_expired(self, control_handler):
        """Expired retardant should be removed."""
        cell = control_handler._grid_manager.get_cell_from_indices(5, 5)
        control_handler.add_retardant_at_cell(cell, 1.0, 0.5)

        # Update with time past expiration
        control_handler.update_long_term_retardants(curr_time_s=3601)

        assert cell._retardant is False
        assert cell._retardant_factor == pytest.approx(1.0)
        assert cell not in control_handler.long_term_retardants

    def test_retardant_not_removed_before_expiration(self, control_handler):
        """Retardant should remain before expiration."""
        cell = control_handler._grid_manager.get_cell_from_indices(5, 5)
        control_handler.add_retardant_at_cell(cell, 1.0, 0.5)

        # Update with time before expiration
        control_handler.update_long_term_retardants(curr_time_s=3000)

        assert cell._retardant is True
        assert cell in control_handler.long_term_retardants


class TestControlActionHandlerWaterDrop:
    """Tests for ControlActionHandler water drop methods."""

    def test_water_drop_as_rain_at_cell(self, control_handler):
        """Water drop as rain should be applied to a burnable cell."""
        cell = control_handler._grid_manager.get_cell_from_indices(5, 5)

        control_handler.water_drop_at_cell_as_rain(cell, 1.0)

        assert cell in control_handler.active_water_drops

    def test_water_drop_negative_depth_raises(self, control_handler):
        """Negative water depth should raise ValueError."""
        cell = control_handler._grid_manager.get_cell_from_indices(5, 5)

        with pytest.raises(ValueError, match="Water depth must be >=0"):
            control_handler.water_drop_at_cell_as_rain(cell, -1.0)

    def test_water_drop_as_moisture_bump(self, control_handler):
        """Water drop as moisture bump should be applied."""
        cell = control_handler._grid_manager.get_cell_from_indices(5, 5)

        control_handler.water_drop_at_cell_as_moisture_bump(cell, 0.1)

        assert cell in control_handler.active_water_drops

    def test_water_drop_negative_moisture_raises(self, control_handler):
        """Negative moisture increment should raise ValueError."""
        cell = control_handler._grid_manager.get_cell_from_indices(5, 5)

        with pytest.raises(ValueError, match="Moisture increase must be >0"):
            control_handler.water_drop_at_cell_as_moisture_bump(cell, -0.1)

    def test_water_drop_not_applied_to_unburnable(self, control_handler):
        """Water drop should not be applied to non-burnable cells."""
        cell = control_handler._grid_manager.get_cell_from_indices(5, 5)
        cell._fuel = MockFuel(burnable=False)

        control_handler.water_drop_at_cell_as_rain(cell, 1.0)

        assert cell not in control_handler.active_water_drops


class TestControlActionHandlerFireline:
    """Tests for ControlActionHandler fireline construction."""

    def test_instant_fireline_construction(self, control_handler):
        """Instant fireline should be added immediately."""
        line = LineString([(0, 0), (90, 0)])

        fireline_id = control_handler.construct_fireline(line, 2.0, curr_time_s=0)

        assert len(control_handler.fire_breaks) == 1
        assert fireline_id in control_handler.fire_break_dict
        assert len(control_handler.new_fire_break_cache) == 1

    def test_instant_fireline_with_custom_id(self, control_handler):
        """Instant fireline should use custom ID if provided."""
        line = LineString([(0, 0), (90, 0)])

        fireline_id = control_handler.construct_fireline(
            line, 2.0, fireline_id="custom_id", curr_time_s=0
        )

        assert fireline_id == "custom_id"
        assert "custom_id" in control_handler.fire_break_dict

    def test_progressive_fireline_construction(self, control_handler):
        """Progressive fireline should be added to active firelines."""
        line = LineString([(0, 0), (90, 0)])

        fireline_id = control_handler.construct_fireline(line, 2.0, construction_rate=1.0)

        assert len(control_handler.fire_breaks) == 0  # Not yet complete
        assert fireline_id in control_handler.active_firelines
        assert control_handler.active_firelines[fireline_id]["rate"] == 1.0

    def test_stop_fireline_construction(self, control_handler):
        """Stopping construction should finalize the fireline."""
        line = LineString([(0, 0), (90, 0)])

        fireline_id = control_handler.construct_fireline(line, 2.0, construction_rate=1.0)
        assert fireline_id in control_handler.active_firelines

        control_handler.stop_fireline_construction(fireline_id)

        assert fireline_id not in control_handler.active_firelines
        assert fireline_id in control_handler.fire_break_dict


class TestControlActionHandlerBehaviorMatch:
    """Tests that verify ControlActionHandler matches MockControlActionHandler behavior."""

    def test_retardant_behavior_matches(self, control_handler):
        """ControlActionHandler retardant should match mock behavior."""
        mock = MockControlActionHandler()

        # Apply same retardant to both
        mock_cell = mock.get_cell_from_indices(5, 5)
        handler_cell = control_handler._grid_manager.get_cell_from_indices(5, 5)

        mock.add_retardant_at_cell(mock_cell, 2.0, 0.7)
        control_handler.add_retardant_at_cell(handler_cell, 2.0, 0.7)

        # Verify state matches
        assert mock_cell._retardant == handler_cell._retardant
        assert mock_cell._retardant_factor == pytest.approx(handler_cell._retardant_factor)

    def test_fireline_id_generation_matches(self, control_handler):
        """ControlActionHandler fireline ID generation should match mock."""
        mock = MockControlActionHandler()
        line = LineString([(0, 0), (90, 0)])

        mock_id = mock.construct_fireline(line, 2.0)
        handler_id = control_handler.construct_fireline(line, 2.0, curr_time_s=0)

        # Both should generate "1" as the first ID
        assert mock_id == handler_id

    def test_water_drop_error_behavior_matches(self, control_handler):
        """ControlActionHandler water drop errors should match mock."""
        mock = MockControlActionHandler()
        mock_cell = mock.get_cell_from_indices(5, 5)
        handler_cell = control_handler._grid_manager.get_cell_from_indices(5, 5)

        # Both should raise same error for negative values
        with pytest.raises(ValueError, match="Water depth must be >=0"):
            mock.water_drop_at_cell_as_rain(mock_cell, -1.0)

        with pytest.raises(ValueError, match="Water depth must be >=0"):
            control_handler.water_drop_at_cell_as_rain(handler_cell, -1.0)
