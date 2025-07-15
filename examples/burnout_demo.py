from embrs.base_classes.control_base import ControlClass

from embrs.fire_simulator.fire import FireSim


class Burnout(ControlClass):

    def __init__(self, fire: FireSim):
        self.fire = fire


    def process_state(self, fire):
        pass


