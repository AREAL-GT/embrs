"""Demo class demonstrating how the use of the fire prediction tool during a fire simulation.
The prediction tool can be used to inform firefighting decision making.

To run this example code, start a fire sim and select this file as the "User Module"
Adjust the PredictorParams block below to see how changes to wind, moisture, resolution,
or prediction horizon influence the forecast.
"""

from typing import Optional

from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim
from embrs.tools.fire_predictor import FirePredictor
from embrs.utilities.data_classes import PredictorParams

class PredictorCode(ControlClass):
    def __init__(self, fire: FireSim):
        self.prediction_done = False
        self.predictor: Optional[FirePredictor] = None

    def process_state(self, fire: FireSim) -> None:
        if self.prediction_done or fire.curr_time_h <= 1:
            return

        # Tune these PredictorParams values to explore how different inputs shift the forecast.
        # Examples: increase wind_speed_bias/wind_dir_bias to skew the wind, or raise dead_mf/live_mf
        # to dampen spread from wetter fuels. time_horizon_hr controls how far into the future to predict.
        params = PredictorParams(
            time_horizon_hr=2,
            time_step_s=fire.time_step * 2,
            cell_size_m=fire.cell_size,
            dead_mf=0.10,
            live_mf=0.30,
            wind_speed_bias=0.0,
            wind_dir_bias=0.0,
            ros_bias=0.0,
            wind_uncertainty_factor=0.0,
            model_spotting=False,
        )

        self.predictor = FirePredictor(params, fire)

        # Run prediction and visualize it on top of the live simulation
        future_state = self.predictor.run(visualize=True)
        self.prediction_done = True

        # Print out the predicted fire locations for each time step
        for time_step, fires in future_state.spread.items():
            print(f"Time: {time_step}, fires: {fires}")
