from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim

from embrs.utilities.data_classes import PredictorParams
from embrs.tools.fire_predictor import FirePredictor


from shapely.geometry import LineString, Point
from shapely.affinity import translate

import numpy as np

class FirelineConstruction(ControlClass):

    def __init__(self, fire: FireSim):
        self.fire = fire

        # TODO: change according to scenario
        self.pos = (500, 500) 

        # TODO: these can likely be grabbed directly from the map_params
        self.anchor_line = LineString() # TODO: grab the anchor line, this is where the firefighters are starting from
        self.target_line = LineString() # TODO: grab the target control line

        self.transit_speed_ms = 1.34 # Equivalent to 3 mph (walking speed)
        self.ln_prod_rate_ms = 0.5 # TODO: find a citable value for this (https://www.frames.gov/documents/behaveplus/publications/NWCG_2021_FireLineProductionRates.pdf)

        self.fire_buffer_dist = 100 # TODO: find a citable value for this

        # Incremental distance to build lines by
        self.d = 10 # meters
        
        self.dtheta = np.pi / 2 / 10
        
        time_est = Point(self.pos).distance(self.target_line) / self.ln_prod_rate_ms # time to nearest point on target line in seconds

        time_horizon = (time_est / 3600) + 1 # add an hour buffer to the estimated time
        time_step = self.d / self.ln_prod_rate_ms
        
        # TODO: play with the bias parameters and such
        pred_input = PredictorParams(
            time_horizon_hr=time_horizon,
            time_step_s=time_step,
            cell_size_m=fire.cell_size
        )

        self.pred_model = FirePredictor(pred_input, fire)
        self.curr_prediction = self.pred_model.run()

        # Construct nominal plan for control line
        # TODO: this assumes fire is moving left to right
        # TODO: will want a smarter approach to choosing start and stop here
        x_arr = np.linspace(self.pos[0], fire.x_lim, 10)
        
        nominal_line, eta = self.choose_nominal_plan(x_arr)

    def process_state(self, fire: FireSim):
        # TODO: this should essentially construct segments of the fireline step by step
        # TODO: this should force replanning if the fire is approaching the crew

        pass

    def choose_nominal_plan(self, x_vals):
        best_line = None
        min_time = np.inf

        for x in x_vals:
            x, y = self.get_xy_from(x)

            line, time = self.construct_valid_line_from((x, y))

            if time < min_time:
                min_time = time
                best_line = line

        return best_line, time

    def get_xy_from(self, x):
        # TODO: this only works for horizontal anchors lines, # need to generalize for all orientations

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
        dist = Point(start_pos).distance(Point(self.pos))
        time = dist / self.transit_speed_ms
        pos = Point(start_pos)

        # Distance remaining to target anchor line
        gap = pos.distance(self.target_line)

        path = [pos]

        fires = []

        while gap > 0:
            fires.extend(self.curr_prediction[time]) # TODO: need to round this time to the nearest time-step

            # Find the unit vector pointing to the closest point on the target line from current position
            closest_point = self.target_line.interpolate(self.target_line.project(pos))
            direction_vector = np.array([closest_point.x - pos.x, closest_point.y - pos.y])
            norm = np.linalg.norm(direction_vector)
            if norm != 0:
                unit_vector = direction_vector / norm
            else:
                unit_vector = direction_vector  # Zero vector if point is on line

            # Direction that minimizes distance to target
            theta = np.arctan2(unit_vector[1], unit_vector[0])

            # D length vector pointing towards target
            dx = self.d * np.cos(theta)
            dy = self.d * np.sin(theta)

            tmp_pos = translate(pos, xoff=dx, yoff=dy)

            # TODO: can probably use that optimized search for the nearest point can't remember the library rn
            fire_dist = np.inf
            closest_point = None
            for fire in fires:
                fire_pt = Point(fire)
                dist = fire_pt.distance(tmp_pos)

                if dist < fire_dist:
                    fire_dist = dist
                    closest_point = fire_pt

            if fire_dist < self.fire_buffer_dist:
                
                vecx = tmp_pos.x - closest_point.x
                vecy = tmp_pos.y - closest_point.y

                vec = np.array([vecx, vecy])

                # TODO: should watchout for cases where the unit vector is sending us toward the fire
                cross = np.cross(unit_vector, vec)

                if cross > 0:
                    inc = False

                else:
                    inc = True

            delta_theta = 0
            while fire_dist < self.fire_buffer_dist:
                if inc:
                    theta += self.dtheta
                else:
                    theta -= self.dtheta
                
                delta_theta += self.dtheta

                dx = self.d * np.cos(theta)
                dy = self.d * np.sin(theta)

                tmp_pos = translate(pos, xoff=dx, yoff=dy)

                fire_dist = tmp_pos.distance(closest_point)

                if delta_theta >= np.pi/2:
                    return None, np.inf
            
            path.append(tmp_pos)
            pos = tmp_pos
            time += self.d/self.ln_prod_rate_ms
            gap = pos.distance(self.target_line)

            if LineString(path).intersects(self.target_line):
                gap = 0

        return LineString(path), time