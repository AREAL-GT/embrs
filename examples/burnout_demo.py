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

TRAVEL, WAIT, BURN, PLAN, IDLE = 0, 1, 2, 3, 4


class Burnout(ControlClass):

    def __init__(self, fire: FireSim):
        self.fire = fire

        self.state = PLAN

        # TODO: may be worth setting up more than one agent
        x, y = 800, 0
        self.agent = AgentBase(0, x, y)

        fire.add_agent(self.agent)

        # Set prediction length to whole sim for now
        self.t_horizon = self.fire._sim_duration / 3600

        # Parameters        
        self.hold_prob_thresh = 0.99
        self.seg_length = 500
        self.wind_speed_thresh = 2.5 # equivalent to ~5 mph winds

        self.travel_vel = 1.34 # m/s


        self.line_segments = self.create_line_segments()

        self.curr_burn_line = None
        self.burn_cell = None
        self.burn_queue = []


    def process_state(self, fire):
        
        if self.state == PLAN:
            self.plan()

        elif self.state == TRAVEL:
            self.travel()

        elif self.state == WAIT:
            self.wait()

        elif self.state == BURN:
            self.burn()

        elif self.state == IDLE:
            pass


    def create_line_segments(self):

        segments = []

        for fireline, _, _ in self.fire.fire_breaks:
            segments.extend(split_line(fireline, self.seg_length))

        return segments
    
    def plan(self):

        # TODO: play with parameters
        pred_input = PredictorParams(
            time_horizon_hr=self.t_horizon,
            time_step_s=self.fire.time_step,
            cell_size_m=self.fire.cell_size,
            dead_mf=0.03,
            wind_bias_factor=1
        )

        self.pred_model = FirePredictor(pred_input, self.fire)
        self.pred_output = self.pred_model.run()
        
        self.burning_locs = [(cell.x_pos, cell.y_pos) for cell in self.fire._burning_cells]
        self.dist_tree = KDTree(self.burning_locs)

        self.burn_plan = self.create_burn_plan()

        self.state = WAIT

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
        self.plot_linestrings(at_risk_segments)

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
                
                n_dist, idx = self.dist_tree.query((neighbor.x_pos, neighbor.y_pos), k=1)
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
        # this should manage the agents travel 
        fire_tree = KDTree([(cell.x_pos, cell.y_pos) for cell in self.fire._burning_cells])
        dist, idx = fire_tree.query((self.agent.x, self.agent.y), k=1)
        fire_cell = self.fire._burning_cells[idx]

        agent_to_fire_line = LineString([(self.agent.x, self.agent.y), (fire_cell.x_pos, fire_cell.y_pos)])

        safe = False
        for fire_break, _, _ in self.fire.fire_breaks:
            if agent_to_fire_line.intersects(fire_break):
                safe = True
                break
        
        if safe:
            if not self.burn_queue:
                if not self.burn_plan:
                    # all done
                    self.state = IDLE
                else:
                    # wait for next burn assignment
                    self.state = WAIT

            else:
                # travel towards the closest cell in burn_queue while staying safe
                target_cell = self.burn_queue[self.burn_order[0]]
                target_pt = (target_cell.x_pos, target_cell.y_pos)

                atr_vec = np.array([target_pt[0] - self.agent.x, target_pt[1] - self.agent.y])
                atr_vec /= np.linalg.norm(atr_vec)


                line_coords = []
                for ln, _, _ in self.fire.fire_breaks:
                    line_coords.extend(list(ln.coords))
                
                tree = KDTree(line_coords)
                dist, idx = tree.query((self.agent.x, self.agent.y), k=1)
                rep_pt = line_coords[idx]

                rep_vec = np.array([self.agent.x - rep_pt[0], self.agent.y - rep_pt[1]])
                rep_vec /= np.linalg.norm(rep_vec)

                d = self.fire.time_step * self.travel_vel

                r = np.exp(-1 * (dist - d)/(10*d - d))
                r = min(r, 1)

                vec = (1 - r) * atr_vec + r * rep_vec

                theta = np.arctan2(vec[1], vec[0])

                dx = self.fire.time_step * self.travel_vel * np.cos(theta)
                dy = self.fire.time_step * self.travel_vel * np.sin(theta)

                self.agent.x += dx
                self.agent.y += dy

                dist_to_goal = np.sqrt((target_cell.x_pos - self.agent.x) ** 2 + (target_cell.y_pos - self.agent.y) ** 2)
                if dist_to_goal <= self.fire.time_step * self.travel_vel:
                    self.state = BURN

        else:
            # focus on getting safe first
            closest_dist = np.inf
            closest_point = None
            for fire_break, _, _ in self.fire.fire_breaks:
                point = fire_break.interpolate(fire_break.project(Point((self.agent.x, self.agent.y))))

                dist = np.sqrt((point.x - self.agent.x) ** 2 + (point.y - self.agent.y) ** 2)

                if dist < closest_dist:
                    closest_dist = dist
                    closest_point = point

            
            atr_vec = np.array([closest_point.x- self.agent.x, closest_point.y - self.agent.y])
            atr_vec /= np.linalg.norm(atr_vec)

            theta = np.arctan2(atr_vec[1], atr_vec[0])

            dx = self.fire.time_step * self.travel_vel * np.cos(theta)
            dy = self.fire.time_step * self.travel_vel * np.sin(theta)

            self.agent.x += dx
            self.agent.y += dy



    def wait(self):
        # TODO: implement

        for line, cells in list(self.burn_plan):

            speed_tot = 0
            dir_tot = 0

            for cell in cells:
                cell._update_weather(self.fire._curr_weather_idx, self.fire._weather_stream, False)

                speed_tot += cell.curr_wind[0]
                dir_tot += cell.curr_wind[1]

            avg_speed = speed_tot / len(cells)
            avg_dir = dir_tot / len(cells)

            if avg_speed < self.wind_speed_thresh:
                # Check that dir is pointing from the fire to the line
                burn = True
                for cell in cells:
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
                    # self.curr_burn_line = (line, cells)
                    self.burn_queue.extend(cells)
                    self.b_tree = KDTree([(cell.x_pos, cell.y_pos) for cell in cells])
                    _, burn_order = self.b_tree.query((self.agent.x, self.agent.y), k=len(cells))
                    

                    if len(cells) == 1:
                        burn_order = [burn_order]

                    self.burn_order = list(burn_order)
                    self.burn_plan.remove((line, cells))
                    self.state = TRAVEL
                    return

    def burn(self):
        
        if self.burn_cell is None:
            self.burn_cell = self.burn_queue[self.burn_order[0]]
            self.burn_order.remove(self.burn_order[0])
        
        dist_to_cell = np.sqrt((self.agent.x - self.burn_cell.x_pos) ** 2 + (self.agent.y - self.burn_cell.y_pos) ** 2)

        if dist_to_cell > (self.fire.time_step * self.travel_vel):
            # Travel towards the cell
            vec_to_cell = np.array([self.burn_cell.x_pos - self.agent.x, self.burn_cell.y_pos - self.agent.y])
            
            angle = np.atan2(vec_to_cell[1], vec_to_cell[0])

            self.agent.x += (self.fire.time_step * self.travel_vel) * np.cos(angle) # Check cosine v sine
            self.agent.y += (self.fire.time_step * self.travel_vel) * np.sin(angle)


        else:
            self.agent.x = self.burn_cell.x_pos
            self.agent.y = self.burn_cell.y_pos

            # self.burn_queue.remove(self.burn_cell)
            self.fire.set_ignition_at_cell(self.burn_cell)
            self.burn_cell = None


        if len(self.burn_order) == 0:

            self.burn_queue = []
            
            self.state = TRAVEL
            # # Change state
            # if self.burn_plan:
            #     self.state = TRAVEL

            # else:
            #     self.state = TRAVEL


    def plot_linestrings(self, linestrings, ax=None, show=True, color=['red', 'green', 'blue'], linewidth=1):
        """
        Plot a list of Shapely LineString objects using matplotlib.

        Parameters:
        ----------
        linestrings : list
            A list of shapely.geometry.LineString objects.
        ax : matplotlib.axes.Axes, optional
            Existing matplotlib axes to plot on. If None, a new figure is created.
        show : bool, default=True
            Whether to call plt.show() after plotting.
        color : str or list, default='blue'
            Color of the LineStrings. Can be a single color or a list of colors.
        linewidth : float, default=1
            Width of the lines.

        Returns:
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axes containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')

        if isinstance(color, str):
            color = [color] * len(linestrings)

        for i, line in enumerate(linestrings):
            x, y = line[0].xy
            ax.plot(x, y, color=color[i % len(color)], linewidth=linewidth)

            xs = [cell.x_pos for cell in line[1]]
            ys = [cell.y_pos for cell in line[1]]

            ax.scatter(xs, ys, color=color[i%len(color)])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("LineString Plot")
        ax.set_xlim(0, self.fire.x_lim)
        ax.set_ylim(0, self.fire.y_lim)

        if show:
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
