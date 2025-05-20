



from embrs.base_classes.base_fire import BaseFireSim
from embrs.fire_simulator.fire import FireSim
from embrs.fire_simulator.cell import Cell
from embrs.utilities.data_classes import PredictorParams, CellData
from embrs.utilities.fire_util import UtilFuncs, CellStates
from embrs.utilities.fuel_models import Anderson13

import copy
import numpy as np

class FirePredictor(BaseFireSim):



    def __init__(self, params: PredictorParams, fire: FireSim):

        # Live reference to the fire sim
        self.fire = fire
        
        self.set_params(params)

    
    def set_params(self, params: PredictorParams):

        generate_cell_grid = False

        self.time_horizon_hr = params.time_horizon_hr
        self.time_step = params.time_step_s
        cell_size = params.cell_size_m

        if cell_size != self.cell_size:
            generate_cell_grid = True

        self.cell_size = cell_size

        self.dead_mf = params.dead_mf
        self.live_mf = params.live_mf

        sim_params = copy.deepcopy(self.fire._sim_params)
        sim_params.cell_size = self.cell_size
        sim_params.t_step_s = self.time_step
        sim_params.duration_s = self.time_horizon_hr * 3600
        sim_params.map_params.scenario_data.initial_ign = UtilFuncs.get_cell_polygons(self.fire._burning_cells)

        self.model_spotting = sim_params.model_spotting

        burnt_region = UtilFuncs.get_cell_polygons(self.fire._burnt_cells)

        sim_params.predction_model = True

        if generate_cell_grid:
            super().__init__(sim_params, burnt_region=burnt_region)
            

    def run(self):
        
        
        # Catch states of predcictor cells up with the fire sim
        self.catch_up_with_fire()


        



        # Perform the prediction



    def catch_up_with_fire(self):
        # Set current time to fire sim time
        self._curr_time_s = self.fire._curr_time_s

        burnt_cells = self.fire._burnt_cells
        self._set_state_from_geometries(burnt_cells, CellStates.BURNT)


        burning_cells = self.fire._burning_cells
        # This populates self.starting_ignitions
        self._set_state_from_geometries(burning_cells, CellStates.FIRE)

        








