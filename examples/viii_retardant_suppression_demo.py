"""Demonstration of how to implement a retardant suppression system using the embrs library."""

from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim

class TestRetardant(ControlClass):

    def __init__(self, fire):
        self.interval = 3600 * 1 # interval in seconds
    
        self.last_intervention = -1

        self.r = 0

    def process_state(self, fire: FireSim):

        if (fire.curr_time_s - self.last_intervention) >= self.interval:
            for cell_id in fire.frontier:
                cell = fire.cell_dict[cell_id]
                fire.add_retardant_at_cell(cell, 3, 1)

            self.last_intervention = fire.curr_time_s
