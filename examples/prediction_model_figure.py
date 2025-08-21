from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim

from embrs.utilities.data_classes import PredictorParams
from embrs.tools.fire_predictor import FirePredictor

class PredictionFigures(ControlClass):
    def __init__(self, fire: FireSim):
        self.fire = fire
        self.done = False
        
        params = PredictorParams(
                time_horizon_hr=12,
                time_step_s=5,
                cell_size_m=fire.cell_size,
                dead_mf=0.06,
                live_mf=0.3,
                wind_dir_bias=0.0,
                wind_speed_bias=0.0,
                ros_bias=-0.25,
                model_spotting=False
        )

        self.pred = FirePredictor(params, fire)
        self.pred.run(visualize=True)

    def process_state(self, fire):
        return