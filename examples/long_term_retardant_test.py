from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim

import numpy as np

class TestRetardant(ControlClass):

    def __init__(self, fire):
        self.interval = 3600 * 1 # interval in seconds
    
        self.last_intervention = -1

        self.r = 0

    def process_state(self, fire: FireSim):

        if (fire.curr_time_s - self.last_intervention) >= self.interval:
            # 
            # x, y = fire.get_avg_fire_coord()

            # frontier = fire.frontier

            # xs = [cell.x_pos for cell in frontier]
            # ys = [cell.y_pos for cell in frontier]

            # angles = np.linspace(0, np.pi * 2, 200)

            # self.r += 500

            # xs = x + self.r * np.cos(angles)
            # ys = y + self.r * np.sin(angles)

            # locs = zip(xs, ys)

            for cell_id in fire.frontier:
                cell = fire.cell_dict[cell_id]
                fire.add_retardant_at_cell(cell, 3, 1)
                # fire.add_retardant_at_xy(loc[0], loc[1], 2, 1)

            self.last_intervention = fire.curr_time_s

