"""Control Class implementing the burnout example strategy described in the EMBRS paper."""

from embrs.base_classes.control_base import ControlClass
from embrs.base_classes.agent_base import AgentBase
from embrs.fire_simulator.fire import FireSim
from embrs.tools.fire_predictor import FirePredictor
from embrs.utilities.data_classes import PredictorParams
from embrs.utilities.fire_util import CellStates

from scipy.spatial import KDTree
from shapely.geometry import LineString, Point
from shapely.affinity import translate

import numpy as np
import matplotlib.pyplot as plt

TRAVEL, WAIT, BURN, PLAN, MONITOR, IDLE = 0, 1, 2, 3, 4, 5

class Burnout(ControlClass):

    def __init__(self, fire: FireSim):
        self.fire = fire

        self.burn_state = PLAN
        self.soak_state = WAIT

        # Define firefighting crews
        x, y = 3000, 540
        self.burn_crew = AgentBase(0, x, y, color='red')
        fire.add_agent(self.burn_crew)
        self.soak_crew = AgentBase(1, x, y, color='blue')
        fire.add_agent(self.soak_crew)

        # Set prediction length to whole sim for now
        self.t_horizon = self.fire._sim_duration / 3600

        # Parameters        
        self.hold_prob_thresh = 0.965
        self.seg_length = 500
        self.wind_speed_thresh = 2.5 # equivalent to ~5 mph winds
        self.travel_vel = 1.34 # m/s

        self.line_segments = self.create_line_segments()

        self.curr_burn_line = None
        self.burn_cell = None
        self.burn_queue = []

        self.monitor_queue = []

    def process_state(self, fire):

        if self.fire._curr_time_s < 30:
            return

        # Burn Crew State Machine
        if self.burn_state == PLAN:
            self.plan()

        elif self.burn_state == TRAVEL:
            self.travel()

        elif self.burn_state == WAIT:
            self.wait()

        elif self.burn_state == BURN:
            self.burn()

        elif self.burn_state == IDLE:
            pass
        
        # Soak Crew State Machine
        if self.soak_state == WAIT:
            pass

        if self.soak_state == TRAVEL:
            self.soak_travel()

        elif self.soak_state == MONITOR:
            self.monitor()

    def create_line_segments(self):
        segments = []

        for fireline, _, _ in self.fire.fire_breaks:
            segments.extend(split_line(fireline, self.seg_length))

        return segments
    
    def plan(self):
        pred_input = PredictorParams(
            time_horizon_hr=self.t_horizon,
            time_step_s=self.fire.time_step,
            cell_size_m=self.fire.cell_size,
            dead_mf=0.03,
            wind_speed_bias=1
        )

        self.pred_model = FirePredictor(pred_input, self.fire)
        self.pred_output = self.pred_model.run()
        
        self.burning_locs = [(cell.x_pos, cell.y_pos) for cell in self.fire._burning_cells]
        self.dist_tree = KDTree(self.burning_locs)

        self.burn_plan = self.create_burn_plan()

        self.burn_state = WAIT

    def create_burn_plan(self):
        # Gather relevant prediction output
        hold_probs = self.pred_output.hold_probs
        breaches = self.pred_output.breaches

        at_risk_segments = []
        for seg in self.line_segments:
            # Create buffer around segment (e.g., 1-2 cells wide)
            buffer = seg.buffer(self.fire.cell_size * 1.5)
            
            # Check hold probabilities within buffer
            risky = False
            for loc, prob in hold_probs.items():
                if buffer.contains(Point(loc)):
                    if prob < self.hold_prob_thresh or breaches.get(loc, False):
                        risky = True
                        break
            
            if risky:
                # Get the cells to ignite for this segment
                cells = self.get_cells_for_seg(seg)

                if cells:
                    at_risk_segments.append((seg, cells))

        print(f"{len(at_risk_segments)} burn lines generated")
        self.plot_burn_plan_with_baselines(
            at_risk_segments,
            base_linestrings=self.fire.fire_breaks)

        return at_risk_segments

    def get_cells_for_seg(self, seg: LineString):
        cells = set()

        break_cells = self.fire.get_cells_at_geometry(seg)

        for bc in break_cells:
            for n_id in bc.burnable_neighbors.keys():
                neighbor = self.fire.cell_dict[n_id]

                if neighbor._break_width > 0 or neighbor in cells:
                    # Don't add other break cells or already included cells
                    continue
                
                _, idx = self.dist_tree.query((neighbor.x_pos, neighbor.y_pos), k=1)
                fire_pt = self.burning_locs[idx]
                
                test_line = LineString([(neighbor.x_pos, neighbor.y_pos), fire_pt])

                keep = True
                for fire_break, _, _ in self.fire.fire_breaks:
                    if test_line.intersects(fire_break):
                        keep = False
                        break

                if keep:
                    cells.add(neighbor)

        return cells

    def travel(self):
        # Get target burn cell
        if not self.burn_queue:
            if not self.burn_plan:
                self.burn_state = IDLE
                return
            else:
                self.burn_state = WAIT
                return

        target_cell = self.burn_queue[self.burn_order[0]]
        cell_point = Point(target_cell.x_pos, target_cell.y_pos)

        # Find closest fireline and project agent onto it
        agent_point = Point(self.burn_crew.x, self.burn_crew.y)
        closest_line = None

        min_dist = float('inf')

        for fireline, _, _ in self.fire.fire_breaks:
            proj_dist = fireline.project(agent_point)
            proj_point = fireline.interpolate(proj_dist)
            dist = agent_point.distance(proj_point)

            if dist < min_dist:
                min_dist = dist
                closest_line = fireline

        # Determine direction along the fireline
        proj_dist = closest_line.project(agent_point)
        target_proj_dist = closest_line.project(cell_point)
        target_point = closest_line.interpolate(target_proj_dist)

        step = self.travel_vel * self.fire.time_step

        if target_proj_dist > proj_dist:
            new_proj_dist = proj_dist + step
        else:
            new_proj_dist = proj_dist - step

        # Clamp projection distance to fireline
        new_proj_dist = max(0, min(new_proj_dist, closest_line.length))

        # Move agent along fireline
        new_point = closest_line.interpolate(new_proj_dist)
        self.burn_crew.x, self.burn_crew.y = new_point.x, new_point.y

        # Check if agent reached target
        if agent_point.distance(target_point) < self.fire.cell_size:
            self.burn_state = BURN
            self.burn_cell = None

    def wait(self):
        qualified_lines = []
        dists = []
        for line, cells in list(self.burn_plan):

            speed_tot = 0
            dir_tot = 0

            for cell in cells:

                if (self.fire._curr_time_s - self.fire.weather_t_step * self.fire._curr_weather_idx) > (50 * 60):
                    wind_speed, wind_dir = (cell.forecast_wind_speeds[self.fire._curr_weather_idx + 1], cell.forecast_wind_dirs[self.fire._curr_weather_idx + 1 ])

                else:
                    wind_speed, wind_dir = cell.curr_wind()
                
                speed_tot += wind_speed
                dir_tot += wind_dir

            avg_speed = speed_tot / len(cells)
            avg_dir = dir_tot / len(cells)

            if avg_speed < self.wind_speed_thresh:
                # Check that dir is pointing from the fire to the line
                burn = True
                for cell in cells:
                    # Check if cell is already burning or burnt
                    if cell.state != CellStates.FUEL:
                        burn = False
                        self.burn_plan.remove((line, cells))
                        break

                    # Create a line from cell center extending in wind direction
                    cell_pt = Point((cell.x_pos, cell.y_pos))
                    dx = cell.cell_size * 2 * np.sin(np.deg2rad(avg_dir))
                    dy = cell.cell_size * 2 * np.cos(np.deg2rad(avg_dir))
                    test_pt = translate(cell_pt, xoff=dx, yoff=dy)
                    test_line = LineString([cell_pt, test_pt])

                    # Ensure that this line intersects the fireline
                    if not test_line.intersects(line.buffer(cell.cell_size * 1.5)):
                        burn = False
                        break

                if burn:
                    qualified_lines.append((line, cells))
                    avg_x = np.mean([cell.x_pos for cell in cells])
                    avg_y = np.mean([cell.y_pos for cell in cells])
                    dist = np.sqrt((avg_x - self.burn_crew.x) ** 2 + (avg_y - self.burn_crew.y) ** 2)

                    dists.append(dist)

        if qualified_lines:
            idx = np.argmin(dists)
            line, cells = qualified_lines[idx]
            self.curr_burn_line = line
            self.burn_queue = list(cells)
            self.b_tree = KDTree([(cell.x_pos, cell.y_pos) for cell in cells])
            _, burn_order = self.b_tree.query((self.burn_crew.x, self.burn_crew.y), k=len(cells))

            if len(cells) == 1:
                burn_order = [burn_order]

            self.burn_order = list(burn_order)
            self.burn_plan.remove((line, cells))
            self.burn_state = TRAVEL

    def burn(self):
        if self.burn_cell is None:
            self.burn_cell = self.burn_queue[self.burn_order[0]]
            self.burn_order.remove(self.burn_order[0])
        
        dist_to_cell = np.sqrt((self.burn_crew.x - self.burn_cell.x_pos) ** 2 + (self.burn_crew.y - self.burn_cell.y_pos) ** 2)

        if dist_to_cell > (self.fire.time_step * self.travel_vel):
            # Travel towards the cell
            vec_to_cell = np.array([self.burn_cell.x_pos - self.burn_crew.x, self.burn_cell.y_pos - self.burn_crew.y])
            angle = np.atan2(vec_to_cell[1], vec_to_cell[0])

            # Move agent towards the cell
            self.burn_crew.x += (self.fire.time_step * self.travel_vel) * np.cos(angle)
            self.burn_crew.y += (self.fire.time_step * self.travel_vel) * np.sin(angle)

        else:
            # Update agent position to the cell's position
            self.burn_crew.x = self.burn_cell.x_pos
            self.burn_crew.y = self.burn_cell.y_pos

            # Ignite the cell
            self.fire.set_ignition_at_cell(self.burn_cell)
            self.burn_cell = None

        if len(self.burn_order) == 0:
            self.burn_cell = None
            self.burn_queue = []

            self.monitor_queue.append({
                'line': self.curr_burn_line,
                'start_time': self.fire._curr_time_s
            })

            self.burn_state = TRAVEL
            self.soak_state = TRAVEL

    def soak_travel(self):
        # Travel to first line in monitor queue
        if not self.monitor_queue:
            self.soak_state = WAIT
            return
        
        # Get the current line to monitor
        target_line = self.monitor_queue[0]['line']

        # Target point is the midpoint of the line
        target_point = target_line.interpolate(target_line.length / 2)

        # Find closest fireline and project agent onto it
        agent_point = Point(self.soak_crew.x, self.soak_crew.y)
        closest_line = None

        min_dist = float('inf')

        for fireline, _, _ in self.fire.fire_breaks:
            proj_dist = fireline.project(agent_point)
            proj_point = fireline.interpolate(proj_dist)
            dist = agent_point.distance(proj_point)

            if dist < min_dist:
                min_dist = dist
                closest_line = fireline

        # Determine direction along the fireline
        proj_dist = closest_line.project(agent_point)
        target_proj_dist = closest_line.project(target_point)
        target_point = closest_line.interpolate(target_proj_dist)

        step = self.travel_vel * self.fire.time_step

        if target_proj_dist > proj_dist:
            new_proj_dist = proj_dist + step
        else:
            new_proj_dist = proj_dist - step

        # Clamp projection distance to fireline
        new_proj_dist = max(0, min(new_proj_dist, closest_line.length))

        # Move agent along fireline
        new_point = closest_line.interpolate(new_proj_dist)
        self.soak_crew.x, self.soak_crew.y = new_point.x, new_point.y

        # Check if agent reached target
        if agent_point.distance(target_point) < 2 * self.fire.cell_size:
            self.soak_state = MONITOR
            self.move_soak_crew_away_from_line(closest_line)

    def monitor(self):
        # Get line and cells to monitor
        line = self.monitor_queue[0]['line']
        cells = self.fire.get_cells_at_geometry(line.buffer(self.fire.cell_size * 1.5))

        # Get the 10 closest burning cells to the soak crew
        burning_locs = [(cell.x_pos, cell.y_pos) for cell in self.fire._burning_cells if cell._break_width == 0]
        tree = KDTree(burning_locs)

        k = min(10, len(burning_locs))
        fire_dist, idx = tree.query((self.soak_crew.x, self.soak_crew.y), k=k)

        breaches = []
        for i in idx:
            # Test if the cell has breached the firebreak
            breach = True
            fireloc = burning_locs[i]
            fire_pt = Point(fireloc)

            test_line = LineString([fire_pt, Point(self.soak_crew.x, self.soak_crew.y)])

            for fire_break, _, _ in self.fire.fire_breaks:
                if test_line.intersects(fire_break):
                    breach = False

            if breach:
                breaches.append(self.fire.get_cell_from_xy(fireloc[0], fireloc[1]))

        # If any breaches have occurred, soak neighbors of burning cells
        if breaches:
            for cell in breaches:
                # Soak neighbors of the burning cell
                for n_id in cell.burnable_neighbors.keys():
                    neighbor = self.fire.cell_dict[n_id]
                    self.fire.water_drop_at_cell_as_rain(neighbor, 0.057)
            
        else:
            elapsed_time = self.fire._curr_time_s - self.monitor_queue[0]['start_time']
            if elapsed_time > 1200 and not any(cell.state == CellStates.FIRE for cell in cells):
                # If no breaches and no burning cells, remove from queue
                self.monitor_queue.pop(0)
                self.soak_state = TRAVEL
                return
            
    def move_soak_crew_away_from_line(self, line):
        agent_point = Point(self.soak_crew.x, self.soak_crew.y)
        proj_dist = line.project(agent_point)
        proj_point = line.interpolate(proj_dist)

        delta = 0.01 * line.length
        pt_before = line.interpolate(max(0, proj_dist - delta))
        pt_after = line.interpolate(min(line.length, proj_dist + delta))

        tangent_vec = np.array([pt_after.x - pt_before.x, pt_after.y - pt_before.y])
        tangent_vec /= np.linalg.norm(tangent_vec)
        normal_vec = np.array([-tangent_vec[1], tangent_vec[0]])

        # Use nearest initial ignition point to determine fire side
        fire_vec = np.array([self.fire.initial_ignition[0].x - proj_point.x,
                            self.fire.initial_ignition[0].y - proj_point.y])

        side = np.sign(np.cross(tangent_vec, fire_vec))
        safe_vec = -side * normal_vec

        offset = 6 * self.fire.cell_size
        new_pos = np.array([proj_point.x, proj_point.y]) + offset * safe_vec

        self.soak_crew.x, self.soak_crew.y = new_pos



def plot_burn_plan_with_baselines(
        self,
        burn_plan_lines,             # List of (LineString, [Cell]) tuples
        base_linestrings=None,       # List of LineString objects
        ax=None,
        show=True,
        base_color='black',
        burn_color='tab:red',
        linewidth_base=1,
        linewidth_burn=2.5,
        point_style='o',
        point_size=10,
        title="Burn Plan",
        annotate=False,
        alpha=1.0
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal')

        # ── Base lines: anchor/control lines ─────────────────────────────
        if base_linestrings:
            for idx, (base_line, _, _) in enumerate(base_linestrings):
                x, y = base_line.xy
                ax.plot(
                    x, y,
                    color=base_color,
                    linewidth=linewidth_base,
                    linestyle='--',
                    alpha=1,
                    label="Control Lines" if idx == 0 else None
                )

        # ── Burn plan lines + cell markers ───────────────────────────────
        if isinstance(burn_color, str):
            burn_color = [burn_color] * len(burn_plan_lines)

        for i, (line, cell_list) in enumerate(burn_plan_lines):
            x, y = line.xy
            line_color = burn_color[i % len(burn_color)]
            ax.plot(
                x, y,
                color=line_color,
                linewidth=linewidth_burn,
                alpha=alpha,
                label="Burn Plan Lines" if i == 0 else None
            )

            if cell_list:
                xs = [cell.x_pos for cell in cell_list]
                ys = [cell.y_pos for cell in cell_list]
                ax.scatter(
                    xs, ys,
                    color=line_color,
                    s=point_size,
                    marker=point_style,
                    alpha=alpha,
                    label="Burn Plan Cells" if i == 0 else None
                )

            if annotate:
                mid_idx = len(x) // 2
                ax.text(x[mid_idx], y[mid_idx], f"{i}", fontsize=10, color='black', ha='center')

        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlim(0, self.fire.x_lim)
        ax.set_ylim(0, self.fire.y_lim)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        # ── Legend ───────────────────────────────────────────────────────
        ax.legend(loc='upper right', fontsize=10, frameon=True)

        if show:
            plt.tight_layout()
            plt.show()

        return ax



def split_line(line, segment_length):
    """
    Split a LineString into sub-segments of approximately segment_length.
    """
    coords = list(line.coords)
    result = []
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i+1]])
        if segment.length > segment_length:
            # Further subdivide this segment
            num_subsegments = int(np.ceil(segment.length / segment_length))
            for j in range(num_subsegments):
                start = segment.interpolate(j / num_subsegments, normalized=True)
                end = segment.interpolate((j + 1) / num_subsegments, normalized=True)
                result.append(LineString([start, end]))
        else:
            result.append(segment)

    return result
