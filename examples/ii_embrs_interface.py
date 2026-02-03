"""Demo class demonstrating some useful functions of the fire's interface.

To run this example code, start a fire sim and select this file as the "User Module"
"""

from shapely.geometry import LineString, Point, Polygon

from embrs.base_classes.agent_base import AgentBase
from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim
from embrs.utilities.fire_util import CellStates


class InterfaceDemo(ControlClass):
    def __init__(self, fire: FireSim):
        # Map dimensions and grid details
        width_m, height_m = fire.size
        rows, cols = fire.shape
        self.cell_size = fire.cell_size
        self.time_step_s = fire.time_step
        self.sim_duration_s = fire.sim_duration

        print(f"Map is {width_m} m x {height_m} m")
        print(f"Backing array has {rows} rows and {cols} cols")
        print(f"Cell size: {self.cell_size} m | Time step: {self.time_step_s} s | Sim duration: {self.sim_duration_s} s")
        print(f"Is FireSim: {fire.is_firesim()} | Is Prediction: {fire.is_prediction()}")
        print(f"Existing fire breaks: {len(fire.fire_breaks)}, roads: {len(fire.roads) if fire.roads else 0}, initial ignitions: {len(fire.initial_ignition)}")
        print(f"Cell grid shape: {fire.cell_grid.shape} | Cell dict entries: {len(fire.cell_dict)}")

        # Coordinates and indices we will reuse across interface calls
        self.x = width_m / 4
        self.y = height_m / 4
        self.row = rows // 2
        self.col = cols // 2
        self.alt_row = min(rows - 1, self.row + 1)
        self.alt_col = min(cols - 1, self.col + 1)
        self.edge_row = 0
        self.edge_col = 0
        self.near_boundary_x = min(width_m - self.cell_size, width_m * 0.1)
        self.near_boundary_y = min(height_m - self.cell_size, height_m * 0.1)

        # Sample geometries for get_cells_at_geometry
        box_half = self.cell_size * 1.5
        base_x = min(width_m - box_half, width_m * 0.6)
        base_y = min(height_m - box_half, height_m * 0.3)
        self.point_geom = Point(self.x, self.y)
        self.line_geom = LineString([(width_m * 0.1, height_m * 0.1), (width_m * 0.4, height_m * 0.4)])
        self.poly_geom = Polygon(
            [
                (base_x - box_half, base_y - box_half),
                (base_x + box_half, base_y - box_half),
                (base_x + box_half, base_y + box_half),
                (base_x - box_half, base_y + box_half),
            ]
        )

        # Register an example agent
        demo_agent = AgentBase(id="demo_agent", x=self.near_boundary_x, y=self.near_boundary_y, label="Demo", marker="^", color="green")
        fire.add_agent(demo_agent)

        self.demo_ran = False
        self.active_fireline_id = None
        self.fireline_stop_after_iter = None

    def process_state(self, fire: FireSim):
        # Run the one-time interface showcase after the first iteration completes
        if not self.demo_ran:
            self._demo_interface_calls(fire)
            self.demo_ran = True

        # Stop an active, progressive fireline after a couple of iterations
        if self.active_fireline_id and self.fireline_stop_after_iter is not None and fire.iters >= self.fireline_stop_after_iter:
            fire.stop_fireline_construction(self.active_fireline_id)
            print(f"Stopped active fireline construction for id={self.active_fireline_id}")
            self.active_fireline_id = None

        # Periodically report frontier and burning cell information
        if fire.iters % 50 == 0:
            print(f"Iter {fire.iters} | Frontier size: {len(fire.get_frontier())} | Burning cells: {len(fire.burning_cells)} | Finished: {fire.finished}")
            if fire.burning_cells:
                x_avg, y_avg = fire.get_avg_fire_coord()
                print(f"Avg. fire coordinate: ({x_avg} m, {y_avg} m)")

            # Demonstrate average coordinate retrieval only when fire is active
            if fire.burning_cells:
                fire.set_surface_accel_constant(fire.burning_cells[0])

    def _demo_interface_calls(self, fire: FireSim):
        """Run through all public interface functions with their supported input styles."""
        print(f"Sim time (s/m/h): {fire.curr_time_s}, {fire.curr_time_m}, {fire.curr_time_h}")
        print(f"Bounds: x_lim={fire.x_lim}, y_lim={fire.y_lim}")

        # Locate cells from coordinates and indices (including out-of-bounds tolerant lookup)
        def clamp(val, min_val, max_val):
            return max(min_val, min(val, max_val))

        cell_from_xy = fire.get_cell_from_xy(self.x, self.y)
        cell_from_indices = fire.get_cell_from_indices(self.row, self.col)
        neighbor_cell = fire.get_cell_from_indices(self.alt_row, self.alt_col)
        edge_cell = fire.get_cell_from_indices(self.edge_row, self.edge_col)
        print(f"OOB lookup returns: {fire.get_cell_from_xy(-1, -1, oob_ok=True)}")
        print(f"Rounded hex coords for ({self.x}, {self.y}): {fire.hex_round(self.x / self.cell_size, self.y / self.cell_size)}")

        # Retrieve cells intersecting sample geometries (Point, LineString, Polygon)
        point_cells = fire.get_cells_at_geometry(self.point_geom)
        line_cells = fire.get_cells_at_geometry(self.line_geom)
        polygon_cells = fire.get_cells_at_geometry(self.poly_geom)
        print(f"Cells from point/line/polygon: {len(point_cells)}, {len(line_cells)}, {len(polygon_cells)}")

        # Set explicit states using each input style
        fire.set_state_at_xy(self.x, self.y, CellStates.FUEL)
        fire.set_state_at_indices(self.row, self.col, CellStates.FUEL)
        fire.set_state_at_cell(edge_cell, CellStates.BURNT)

        # Ignitions using xy, indices, and cell inputs
        ignite_x = clamp(self.x + self.cell_size, 0, fire.x_lim - self.cell_size)
        ignite_y = clamp(self.y + self.cell_size, 0, fire.y_lim - self.cell_size)
        fire.set_state_at_xy(ignite_x, ignite_y, CellStates.FUEL)
        fire.set_ignition_at_xy(ignite_x, ignite_y)
        fire.set_state_at_indices(self.alt_row, self.alt_col, CellStates.FUEL)
        fire.set_ignition_at_indices(self.alt_row, self.alt_col)
        fire.set_state_at_cell(cell_from_xy, CellStates.FUEL)
        fire.set_ignition_at_cell(cell_from_xy)

        # Long-term retardant applications via xy, indices, and cell
        fire.add_retardant_at_xy(self.near_boundary_x, self.near_boundary_y, duration_hr=1.0, effectiveness=0.75)
        fire.add_retardant_at_indices(self.row, self.col, duration_hr=0.5, effectiveness=1.0)  # effectiveness will be clamped to 1
        fire.add_retardant_at_cell(neighbor_cell, duration_hr=0.25, effectiveness=0.5)

        # Water drops as rain
        rain_x = clamp(self.x + (2 * self.cell_size), 0, fire.x_lim - self.cell_size)
        rain_y = clamp(self.y - self.cell_size, 0, fire.y_lim - self.cell_size)
        fire.water_drop_at_xy_as_rain(rain_x, rain_y, water_depth_cm=2.0)
        fire.water_drop_at_indices_as_rain(self.edge_row, self.edge_col, water_depth_cm=1.0)
        fire.water_drop_at_cell_as_rain(cell_from_indices, water_depth_cm=0.5)

        # Water drops as moisture bumps
        bump_y = clamp(self.y + (2 * self.cell_size), 0, fire.y_lim - self.cell_size)
        fire.water_drop_at_xy_as_moisture_bump(self.x, bump_y, moisture_inc=0.1)
        fire.water_drop_at_indices_as_moisture_bump(self.alt_row, self.alt_col, moisture_inc=0.05)
        fire.water_drop_at_cell_as_moisture_bump(neighbor_cell, moisture_inc=0.02)

        # Fireline construction: instantaneous and progressive
        instant_line = LineString([(fire.x_lim * 0.1, fire.y_lim * 0.2), (fire.x_lim * 0.3, fire.y_lim * 0.2)])
        fire.construct_fireline(instant_line, width_m=self.cell_size / 2.0, id="static_demo")
        active_line = LineString([(fire.x_lim * 0.2, fire.y_lim * 0.3), (fire.x_lim * 0.8, fire.y_lim * 0.3)])
        self.active_fireline_id = fire.construct_fireline(
            active_line,
            width_m=self.cell_size / 2.0,
            construction_rate=self.cell_size / max(self.time_step_s, 1),
            id="active_demo",
        )
        self.fireline_stop_after_iter = fire.iters + 2

        # Action logging for any suppression work just submitted
        actions = fire.get_action_entries()
        print(f"Action entries collected: {[entry.to_dict() for entry in actions]}")

        # Prediction logging (example prediction uses a single known cell)
        fire.curr_prediction = {cell_from_xy.id: (cell_from_xy.row, cell_from_xy.col)}
        prediction_entry = fire.get_prediction_entry()
        print(f"Prediction entry: {prediction_entry.to_dict()}")

        # Frontier, fire break, and road snapshots
        print(f"Frontier ids: {list(fire.get_frontier())}")
        print(f"Fire break cell count: {len(fire.fire_break_cells)}, fire break definitions: {len(fire.fire_breaks)}")
        print(f"Road list length: {len(fire.roads) if fire.roads else 0}")
