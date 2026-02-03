"""Control action handling for fire simulation.

This module provides the ControlActionHandler class which manages all fire
suppression control actions including retardant application, water drops,
and fireline construction.

Classes:
    - ControlActionHandler: Handles fire suppression control actions.
"""

from typing import Optional, Set, List, Dict, Callable, TYPE_CHECKING
from shapely.geometry import LineString

if TYPE_CHECKING:
    from embrs.fire_simulator.cell import Cell
    from embrs.base_classes.grid_manager import GridManager


class ControlActionHandler:
    """Handles fire suppression control actions for fire simulation.

    Manages retardant application, water drops, and fireline construction.
    Tracks active suppression effects and handles their updates over time.

    Attributes:
        long_term_retardants (Set[Cell]): Cells with active long-term retardant.
        active_water_drops (List[Cell]): Cells with active water drop effects.
        active_firelines (Dict): Firelines currently under construction.
        fire_break_cells (List[Cell]): Cells along fire breaks.
        fire_breaks (List): Completed fire breaks as (line, width, id) tuples.
    """

    def __init__(self,
                 grid_manager: 'GridManager',
                 cell_size: float,
                 time_step: float,
                 fuel_class_factory: Callable[[int], object]):
        """Initialize the control action handler.

        Args:
            grid_manager: GridManager instance for cell lookups.
            cell_size: Cell size in meters.
            time_step: Simulation time step in seconds.
            fuel_class_factory: Callable that creates a fuel class from a model number.
        """
        self._grid_manager = grid_manager
        self._cell_size = cell_size
        self._time_step = time_step
        self._fuel_class_factory = fuel_class_factory

        # Control action containers
        self._long_term_retardants: Set['Cell'] = set()
        self._active_water_drops: List['Cell'] = []
        self._active_firelines: Dict[str, dict] = {}
        self._fire_break_cells: List['Cell'] = []
        self._fire_breaks: List = []
        self._fire_break_dict: Dict[str, tuple] = {}
        self._new_fire_break_cache: List[dict] = []

        # Reference to track updated cells (set by parent)
        self._updated_cells: Dict[int, 'Cell'] = {}

        # Current time accessor (set by parent)
        self._get_curr_time: Callable[[], float] = lambda: 0

        # Reference to logger and visualizer for cache cleanup
        self.logger = None
        self._visualizer = None

    @property
    def long_term_retardants(self) -> Set['Cell']:
        """Cells with active long-term retardant."""
        return self._long_term_retardants

    @property
    def active_water_drops(self) -> List['Cell']:
        """Cells with active water drop effects."""
        return self._active_water_drops

    @property
    def active_firelines(self) -> Dict[str, dict]:
        """Firelines currently under construction."""
        return self._active_firelines

    @property
    def fire_break_cells(self) -> List['Cell']:
        """Cells along fire breaks."""
        return self._fire_break_cells

    @property
    def fire_breaks(self) -> List:
        """Completed fire breaks as (line, width, id) tuples."""
        return self._fire_breaks

    @property
    def fire_break_dict(self) -> Dict[str, tuple]:
        """Dictionary mapping fire break IDs to (line, width) tuples."""
        return self._fire_break_dict

    @property
    def new_fire_break_cache(self) -> List[dict]:
        """Cache of newly constructed fire breaks for logging/visualization."""
        return self._new_fire_break_cache

    def set_updated_cells_ref(self, updated_cells: Dict[int, 'Cell']) -> None:
        """Set reference to the simulation's updated cells dictionary.

        Args:
            updated_cells: Dictionary to track cells that have been modified.
        """
        self._updated_cells = updated_cells

    def set_time_accessor(self, time_func: Callable[[], float]) -> None:
        """Set the function to get current simulation time.

        Args:
            time_func: Callable that returns current time in seconds.
        """
        self._get_curr_time = time_func

    def add_retardant_at_xy(self, x_m: float, y_m: float,
                            duration_hr: float, effectiveness: float) -> None:
        """Apply long-term fire retardant at the specified coordinates.

        Args:
            x_m: X position in meters.
            y_m: Y position in meters.
            duration_hr: Duration of retardant effect in hours.
            effectiveness: Retardant effectiveness factor (0.0-1.0).
        """
        cell = self._grid_manager.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            self.add_retardant_at_cell(cell, duration_hr, effectiveness)

    def add_retardant_at_indices(self, row: int, col: int,
                                  duration_hr: float, effectiveness: float) -> None:
        """Apply long-term fire retardant at the specified grid indices.

        Args:
            row: Row index in the cell grid.
            col: Column index in the cell grid.
            duration_hr: Duration of retardant effect in hours.
            effectiveness: Retardant effectiveness factor (0.0-1.0).
        """
        cell = self._grid_manager.get_cell_from_indices(row, col)
        self.add_retardant_at_cell(cell, duration_hr, effectiveness)

    def add_retardant_at_cell(self, cell: 'Cell', duration_hr: float,
                               effectiveness: float) -> None:
        """Apply long-term fire retardant to the specified cell.

        Effectiveness is clamped to the range [0.0, 1.0]. Only applies
        to burnable cells.

        Args:
            cell: Cell to apply retardant to.
            duration_hr: Duration of retardant effect in hours.
            effectiveness: Retardant effectiveness factor (0.0-1.0).
        """
        # Ensure that effectiveness is between 0 and 1
        effectiveness = min(max(effectiveness, 0), 1)

        if cell.fuel.burnable:
            cell.add_retardant(duration_hr, effectiveness)
            self._long_term_retardants.add(cell)
            self._updated_cells[cell.id] = cell

    def update_long_term_retardants(self, curr_time_s: float) -> None:
        """Update long-term retardant effects and remove expired retardants.

        Args:
            curr_time_s: Current simulation time in seconds.
        """
        # First, clear retardant from expired cells and track them as updated
        for cell in self._long_term_retardants:
            if cell.retardant_expiration_s <= curr_time_s:
                cell._retardant = False
                cell._retardant_factor = 1.0
                cell.retardant_expiration_s = -1.0
                self._updated_cells[cell.id] = cell

        # Then filter to keep only non-expired cells
        self._long_term_retardants = {
            cell for cell in self._long_term_retardants
            if cell._retardant  # Still has retardant (wasn't cleared above)
        }

    def water_drop_at_xy_as_rain(self, x_m: float, y_m: float,
                                  water_depth_cm: float) -> None:
        """Apply water drop as equivalent rainfall at the specified coordinates.

        Args:
            x_m: X position in meters.
            y_m: Y position in meters.
            water_depth_cm: Equivalent rainfall depth in centimeters.
        """
        cell = self._grid_manager.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            self.water_drop_at_cell_as_rain(cell, water_depth_cm)

    def water_drop_at_indices_as_rain(self, row: int, col: int,
                                       water_depth_cm: float) -> None:
        """Apply water drop as equivalent rainfall at the specified grid indices.

        Args:
            row: Row index in the cell grid.
            col: Column index in the cell grid.
            water_depth_cm: Equivalent rainfall depth in centimeters.
        """
        cell = self._grid_manager.get_cell_from_indices(row, col)
        self.water_drop_at_cell_as_rain(cell, water_depth_cm)

    def water_drop_at_cell_as_rain(self, cell: 'Cell', water_depth_cm: float) -> None:
        """Apply water drop as equivalent rainfall to the specified cell.

        Only applies to burnable cells. Adds cell to active water drops
        for moisture tracking.

        Args:
            cell: Cell to apply water to.
            water_depth_cm: Equivalent rainfall depth in centimeters.

        Raises:
            ValueError: If water_depth_cm is negative.
        """
        if water_depth_cm < 0:
            raise ValueError(f"Water depth must be >=0, {water_depth_cm} passed in")

        if cell.fuel.burnable:
            cell.water_drop_as_rain(water_depth_cm)
            self._active_water_drops.append(cell)
            self._updated_cells[cell.id] = cell

    def water_drop_at_xy_as_moisture_bump(self, x_m: float, y_m: float,
                                           moisture_inc: float) -> None:
        """Apply water drop as direct moisture increase at the specified coordinates.

        Args:
            x_m: X position in meters.
            y_m: Y position in meters.
            moisture_inc: Moisture content increase as a fraction.
        """
        cell = self._grid_manager.get_cell_from_xy(x_m, y_m, oob_ok=True)
        if cell is not None:
            self.water_drop_at_cell_as_moisture_bump(cell, moisture_inc)

    def water_drop_at_indices_as_moisture_bump(self, row: int, col: int,
                                                moisture_inc: float) -> None:
        """Apply water drop as direct moisture increase at the specified grid indices.

        Args:
            row: Row index in the cell grid.
            col: Column index in the cell grid.
            moisture_inc: Moisture content increase as a fraction.
        """
        cell = self._grid_manager.get_cell_from_indices(row, col)
        self.water_drop_at_cell_as_moisture_bump(cell, moisture_inc)

    def water_drop_at_cell_as_moisture_bump(self, cell: 'Cell',
                                             moisture_inc: float) -> None:
        """Apply water drop as direct moisture increase to the specified cell.

        Only applies to burnable cells. Adds cell to active water drops
        for moisture tracking.

        Args:
            cell: Cell to apply water to.
            moisture_inc: Moisture content increase as a fraction.

        Raises:
            ValueError: If moisture_inc is negative.
        """
        if moisture_inc < 0:
            raise ValueError(f"Moisture increase must be >0, {moisture_inc} passed in")

        if cell.fuel.burnable:
            cell.water_drop_as_moisture_bump(moisture_inc)
            self._active_water_drops.append(cell)
            self._updated_cells[cell.id] = cell

    def construct_fireline(self, line: LineString, width_m: float,
                           construction_rate: Optional[float] = None,
                           fireline_id: Optional[str] = None,
                           curr_time_s: float = 0) -> str:
        """Construct a fire break along a line geometry.

        If construction_rate is None, the fire break is applied instantly.
        Otherwise, it is constructed progressively over time.

        Args:
            line: Shapely LineString defining the fire break path.
            width_m: Width of the fire break in meters.
            construction_rate: Construction rate in m/s. If None, instant.
            fireline_id: Unique identifier. Auto-generated if not provided.
            curr_time_s: Current simulation time in seconds.

        Returns:
            Identifier of the constructed fire break.
        """
        if construction_rate is None:
            # Add fire break instantly
            self._apply_firebreak(line, width_m)

            if fireline_id is None:
                fireline_id = str(len(self._fire_breaks) + 1)

            self._fire_breaks.append((line, width_m, fireline_id))
            self._fire_break_dict[fireline_id] = (line, width_m)

            # Add to cache for visualization and logging
            cache_entry = {
                "id": fireline_id,
                "line": line,
                "width": width_m,
                "time": curr_time_s,
                "logged": False,
                "visualized": False
            }
            self._new_fire_break_cache.append(cache_entry)
        else:
            if fireline_id is None:
                fireline_id = str(len(self._fire_breaks) + len(self._active_firelines) + 1)

            # Create an active fireline to be updated over time
            self._active_firelines[fireline_id] = {
                "line": line,
                "width": width_m,
                "rate": construction_rate,
                "progress": 0.0,
                "partial_line": LineString([]),
                "cells": set()
            }

        return fireline_id

    def stop_fireline_construction(self, fireline_id: str) -> None:
        """Stop construction of an active fireline.

        Finalizes the partially constructed fireline and adds it to the
        permanent fire breaks list.

        Args:
            fireline_id: Identifier of the fireline to stop constructing.
        """
        if self._active_firelines.get(fireline_id) is not None:
            fireline = self._active_firelines[fireline_id]
            partial_line = fireline["partial_line"]
            self._fire_breaks.append((partial_line, fireline["width"], fireline_id))
            self._fire_break_dict[fireline_id] = (partial_line, fireline["width"])
            del self._active_firelines[fireline_id]

    def update_active_firelines(self) -> None:
        """Update progress of active fireline construction.

        Extends partially constructed fire lines based on their
        construction rate. Completes fire lines that reach their
        full length.
        """
        step_size = self._cell_size / 4.0
        firelines_to_remove = []

        for fid in list(self._active_firelines.keys()):
            fireline = self._active_firelines[fid]

            full_line = fireline["line"]
            length = full_line.length

            # Get the progress from previous update
            prev_progress = fireline["progress"]

            # Update progress based on line construction rate
            fireline["progress"] += fireline["rate"] * self._time_step

            # Cap progress at full length
            fireline["progress"] = min(fireline["progress"], length)

            # Interpolate new points between prev_progress and current progress
            num_steps = int(fireline["progress"] / step_size)
            prev_steps = int(prev_progress / step_size)

            for i in range(prev_steps, num_steps):
                # Get the interpolated point
                point = fireline["line"].interpolate(i * step_size)

                # Find the cell containing the new point
                cell = self._grid_manager.get_cell_from_xy(point.x, point.y, oob_ok=True)

                if cell is not None:
                    # Add cell to fire break cell container
                    if cell not in self._fire_break_cells:
                        self._fire_break_cells.append(cell)

                    # Add cell to the line's container
                    if cell not in fireline["cells"]:
                        cell._break_width += fireline["width"]
                        fireline["cells"].add(cell)

                        # Set to urban fuel if break width exceeds cell size
                        if cell._break_width > self._cell_size:
                            cell._set_fuel_type(self._fuel_class_factory(91))

            # If line has met its full length, add to permanent and remove from active
            if fireline["progress"] == length:
                fireline["partial_line"] = full_line
                self._fire_breaks.append((full_line, fireline["width"], fid))
                self._fire_break_dict[fid] = (full_line, fireline["width"])
                firelines_to_remove.append(fid)
            else:
                # Store the truncated line based on progress
                fireline["partial_line"] = self._truncate_linestring(
                    fireline["line"], fireline["progress"]
                )

        # Remove completed firelines from active
        for fireline_id in firelines_to_remove:
            del self._active_firelines[fireline_id]

    def _apply_firebreak(self, line: LineString, break_width: float) -> None:
        """Apply a fire break along a line geometry.

        Args:
            line: Shapely LineString defining the fire break path.
            break_width: Width of the fire break in meters.
        """
        cells = self._grid_manager.get_cells_at_geometry(line)

        for cell in cells:
            if cell not in self._fire_break_cells:
                self._fire_break_cells.append(cell)

            cell._break_width += break_width
            if cell._break_width > self._cell_size:
                cell._set_fuel_type(self._fuel_class_factory(91))

    def _truncate_linestring(self, line: LineString, length: float) -> LineString:
        """Truncate a LineString to the specified length.

        Args:
            line: Original line to truncate.
            length: Desired length in meters.

        Returns:
            Truncated line, or original if length exceeds line length.
        """
        if length <= 0:
            return LineString([line.coords[0]])
        if length >= line.length:
            return line

        coords = list(line.coords)
        accumulated = 0.0
        new_coords = [coords[0]]

        for i in range(1, len(coords)):
            seg = LineString([coords[i - 1], coords[i]])
            seg_len = seg.length
            if accumulated + seg_len >= length:
                ratio = (length - accumulated) / seg_len
                x = coords[i - 1][0] + ratio * (coords[i][0] - coords[i - 1][0])
                y = coords[i - 1][1] + ratio * (coords[i][1] - coords[i - 1][1])
                new_coords.append((x, y))
                break
            else:
                new_coords.append(coords[i])
                accumulated += seg_len

        return LineString(new_coords)
