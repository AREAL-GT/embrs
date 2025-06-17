from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim

import numpy as np

class TestRetardant(ControlClass):

    def __init__(self, fire):
        self.interval = 3600 * 1 # interval in seconds
    
        self.last_intervention = -1

        self.r = 0

    def process_state(self, fire: FireSim):

        if (fire.curr_time_s - self.last_intervention) >= self.interval or self.last_intervention == -1:
            
            x, y = fire.get_avg_fire_coord()

            angles = np.linspace(0, np.pi * 2, 200)

            self.r += 500

            xs = x + self.r * np.cos(angles)
            ys = y + self.r * np.sin(angles)

            locs = zip(xs, ys)

            for loc in locs:
                fire.add_retardant_at_xy(loc[0], loc[1], 2, self.r/(500*10))

            self.last_intervention = fire.curr_time_s

