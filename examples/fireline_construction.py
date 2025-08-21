from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim
from embrs.base_classes.agent_base import AgentBase

from embrs.utilities.data_classes import PredictorParams
from embrs.tools.fire_predictor import FirePredictor


from scipy.spatial import KDTree

from shapely.geometry import LineString, Point
from shapely.affinity import translate

import numpy as np

from embrs.utilities.unit_conversions import *

TRAVEL, CONSTRUCTION, PLANNING, ADAPT, IDLE = 0, 1, 2, 3, 4


class FirelineConstruction(ControlClass):

    def __init__(self, fire: FireSim):
        self.fire = fire

        self.state = PLANNING

        # Get the target and anchor fire breaks
        self.target_line = fire.fire_break_dict["target"][0]
        self.anchor_line = fire.fire_break_dict["anchor"][0]

        # Get the starting coordinate of the crew (set to midpoint of anchor line for now)
        xs, ys = self.anchor_line.xy
        min_x = min(xs)
        max_x = max(xs)
        x_pos = (min_x + max_x) / 4
        x, y = self.get_xy_from(50)
        self.pos = (x,y)

        self.agent = AgentBase(0, x, y)

        fire.add_agent(self.agent)

        self.transit_speed_ms = 1.34112 # Equivalent to 3 mph (walking speed)

        # Working with Hardwood Litter (Type I Crew Indirect: 455 ft/hr)
        self.ln_prod_rate_ms = ft_to_m(376 / 3600) # convert to m/s

        self.plan_buffer = 500 # TODO: find a citable value for this
        self.adapt_buffer = 1500
        self.abs_min = 500
        self.k = 4

        self.line_width = 10 # TODO: find a citable value for this
        
    def process_state(self, fire: FireSim):

        if self.state == PLANNING:
            self.plan()

        elif self.state == TRAVEL:
            self.travel()

        elif self.state == CONSTRUCTION:
            self.construct()

        elif self.state == ADAPT:
            self.adapt()

        elif self.state == IDLE:
            pass


    def adapt(self):
        d = self.fire.time_step * self.ln_prod_rate_ms

        prev_line = self.fire.fire_break_dict[self.line_id][0]

        self.pos = prev_line.coords[-1]
        self.agent.x = self.pos[0]
        self.agent.y = self.pos[1]
        
        if prev_line.intersects(self.target_line):
            self.state = IDLE

        # Apply the potential field approach in real-time
        closest_point = self.target_line.interpolate(self.target_line.project(Point(self.pos)))
        direction_vector = np.array([closest_point.x - self.pos[0], closest_point.y - self.pos[1]])
        norm = np.linalg.norm(direction_vector)

        if norm != 0:
            atr_vec = direction_vector / norm
        else:
            atr_vec = direction_vector  # Zero vector if point is on line

        if self.fire._burning_cells:
            burning_locs = [(cell.x_pos, cell.y_pos) for cell in self.fire._burning_cells]
            tree = KDTree(burning_locs)
            fire_dist, idx = tree.query(self.pos, k=1)
            closest_point = burning_locs[idx]

            vec_from_fire = np.array([self.pos[0] - closest_point[0], self.pos[1] - closest_point[1]])
            norm = np.linalg.norm(vec_from_fire)
            if norm != 0:
                rep_vec = vec_from_fire / norm
            else:
                rep_vec = vec_from_fire

            # Compute the mixing constant
            r = np.exp(-self.k * (fire_dist - self.abs_min)/(self.adapt_buffer - self.abs_min))
            r = min(r, 1)

            vec = (1 - r) * atr_vec + r * rep_vec
            
            # Direction that minimizes distance to target
            theta = np.arctan2(vec[1], vec[0])

            # D length vector pointing towards target
            dx = d * np.cos(theta)
            dy = d * np.sin(theta)

            next_pos = translate(Point(self.pos), xoff=dx, yoff=dy)
            
            line = LineString([Point(self.pos), Point(next_pos)])

            self.line_id = self.fire.construct_fireline(line, self.line_width)

        
    def plan(self):
        time_est = Point(self.pos).distance(self.target_line) / self.ln_prod_rate_ms # time to nearest point on target line in seconds

        time_horizon = (time_est / 3600) + 2 # add 2 hours buffer to the estimated time
        time_step = self.fire.time_step
        
        # TODO: play with the bias parameters and such
        pred_input = PredictorParams(
            time_horizon_hr=time_horizon,
            time_step_s=time_step*3,
            cell_size_m=self.fire.cell_size*2,
            dead_mf=0.10,
            wind_speed_bias=-0.5
        )

        self.pred_model = FirePredictor(pred_input, self.fire)
        pred_output = self.pred_model.run()

        self.spread_prediction = pred_output.spread

        # Construct nominal plan for control line
        # Note: this assumes fire is moving left to right
        x_arr = np.linspace(self.pos[0], self.fire.x_lim, 10)
        
        self.nominal_line, self.eta = self.choose_nominal_plan(x_arr)
        self.prev_progress = self.anchor_line.project(Point(self.pos))
        self.state = TRAVEL

    def travel(self):
        progress = self.prev_progress + self.transit_speed_ms * self.fire.time_step

        point = self.anchor_line.interpolate(progress)

        if point.x > self.nominal_line.xy[0][0]:
            
            self.pos = self.nominal_line.coords[0]
            self.state = CONSTRUCTION

            self.agent.x = self.pos[0]
            self.agent.y = self.pos[1]

            self.line_id = self.fire.construct_fireline(self.nominal_line, self.line_width, self.ln_prod_rate_ms)
            
            return

        self.pos = (point.x, point.y)
        self.agent.x = point.x
        self.agent.y = point.y

        self.prev_progress = progress

    def construct(self):
        # If no active firelines, mission is complete
        if self.fire._active_firelines.get(self.line_id) is None:
            self.state = IDLE
            return

        line = self.fire._active_firelines[self.line_id]

        line_so_far = line["partial_line"]
        if line_so_far.coords:
            self.pos = line_so_far.coords[-1]

            self.agent.x = self.pos[0]
            self.agent.y = self.pos[1]

        # Check if the new location is still outside safe buffer
        if self.fire._burning_cells:
            burning_locs = [(cell.x_pos, cell.y_pos) for cell in self.fire._burning_cells]
            tree = KDTree(burning_locs)
            fire_dist, _ = tree.query(self.pos, k=1)

            if fire_dist < self.adapt_buffer:
                # Need to stop and replan
                self.fire.stop_fireline_construction(self.line_id)
                self.state = ADAPT

    def choose_nominal_plan(self, x_vals):
        best_line = None
        min_time = np.inf

        for x_start in x_vals:
            x, y = self.get_xy_from(x_start)

            if x is None or y is None:
                time = np.inf
                line = None

            else:
                line, time = self.construct_valid_line_from((x, y))

            if time < min_time:
                min_time = time
                best_line = line

        return best_line, time

    def get_xy_from(self, x):
        # Note: this only works for 2 control lines roughly horizontal
        vert_line = LineString([(x, 0), (x, self.fire.y_lim)])

        intersection = self.anchor_line.intersection(vert_line)
        if intersection.is_empty:
            return None, None
        elif intersection.geom_type == 'Point':
            return intersection.x, intersection.y
        elif intersection.geom_type == 'MultiPoint':
            # If there are multiple intersection points, return the first one
            return intersection.geoms[0].x, intersection.geoms[0].y
        else:
            raise ValueError("Unexpected intersection type: {}".format(intersection.geom_type))

    def construct_valid_line_from(self, start_pos: tuple):
        # Get start point
        x0, y0 = start_pos

        # Left line end
        x1, y1 = self.target_line.coords[0]

        # Right line end
        x2, y2 = self.target_line.coords[-1]

        # Compute angles
        angle_left = np.atan2(y1 - y0, x1 - x0)
        angle_right = np.atan2(y2 - y0, x2 - x0)

        # Normalize to [0, 2*pi) if you want
        angle_left = angle_left % (2 * np.pi)
        angle_right = angle_right % (2 * np.pi)

        # Find min and max
        min_angle = min(angle_left, angle_right) + 0.01
        max_angle = max(angle_left, angle_right) - 0.01

        angles = np.linspace(min_angle, max_angle, 10)

        artificial_line_length = self.fire.y_lim * 10

        travel_dist = Point(start_pos).distance(Point(self.pos))
        travel_time = travel_dist / self.transit_speed_ms
        
        min_dist = np.inf
        min_time = np.inf
        best_line = None

        for theta in angles:
            dx = artificial_line_length * np.cos(theta)
            dy = artificial_line_length * np.sin(theta)

            art_line = LineString([start_pos, (start_pos[0] + dx, start_pos[1] + dy)])
            
            point = art_line.intersection(self.target_line)

            act_line = LineString([Point(start_pos), point])

            if act_line.length >= min_dist:
                # No need to compute if distance already worse than current best
                continue

            fires = []

            valid = True
            for t_step in list(self.spread_prediction.keys()):
                fires.extend(self.spread_prediction[t_step])
                
                travel_adj_time = min(t_step - travel_time, 0)

                progress = travel_adj_time * self.ln_prod_rate_ms

                point_at_t = act_line.interpolate(progress)

                tree = KDTree(fires)
                fire_dist, idx = tree.query(point_at_t.coords, k=1)

                if fire_dist < self.plan_buffer:
                    valid = False
                    break

            if valid:
                    min_dist = act_line.length
                    min_time = (act_line.length / self.ln_prod_rate_ms) + travel_time
                    best_line = act_line

        return best_line, min_time
