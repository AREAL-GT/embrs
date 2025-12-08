from embrs.base_classes.control_base import ControlClass
from embrs.tools.fire_predictor import FirePredictor
from embrs.utilities.data_classes import PredictorParams
from embrs.fire_simulator.fire import FireSim

class TestPrediction(ControlClass):

    def __init__(self, fire):
        self.interval = 3600 * 4 # interval in seconds
        self.predictor = None
        self.last_prediction = -1

    def process_state(self, fire: FireSim):

        if self.last_prediction == -1 or (fire.curr_time_s - self.last_prediction) >= self.interval:
            if self.predictor is None:
                predictor_params = PredictorParams(
                    time_horizon_hr=4,
                    time_step_s=fire._time_step*4,
                    cell_size_m=fire._cell_size,
                    dead_mf=0.12,
                    live_mf=0.3,
                    spot_delay_s=1200,
                    model_spotting=True,
                    wind_bias_factor=-0.5,
                    wind_uncertainty_factor=0
                )

                # Construct a predictor
                self.predictor = FirePredictor(predictor_params, fire)

            prediction = self.predictor.run(visualize=True)
            self.last_prediction = fire.curr_time_s

