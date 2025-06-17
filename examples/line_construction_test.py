from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim
from shapely.geometry import LineString

import numpy as np

class ConstructLine(ControlClass):

    def __init__(self, fire):
        self.interval = 3600 * 1 # interval in seconds
    
        self.last_intervention = -1

        self.r = 0

    def process_state(self, fire: FireSim):

        if self.last_intervention == -1:
            
            xs = np.linspace(1000, fire.size[0] - 1000, 100)
            ys = np.sin(xs / 1000) * 500 + fire.size[1] / 2

            coords = zip(xs, ys)

            fireline = LineString(coords)

            fire.construct_fireline(fireline, 10, 0.75)

            # y_2s = ys + 1000
            # fire.construct_fireline(LineString(zip(xs, y_2s)), 10)

            self.last_intervention = fire.curr_time_s

