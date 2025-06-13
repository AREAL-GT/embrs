from embrs.base_classes.base_visualizer import BaseVisualizer
from embrs.fire_simulator.fire import FireSim
from embrs.utilities.data_classes import VisualizerInputs

class RealTimeVisualizer(BaseVisualizer):

    def __init__(self, sim: FireSim):
        self.sim = sim

        input_params = self.get_input_params()

        super().__init__(input_params)

    def set_sim(self, sim: FireSim):
        self.sim = sim

    def get_init_entries(self):
        entries = [cell.to_log_entry(self.sim._curr_time_s) for cell in self.sim.cell_dict.values()]
        return entries

    def update(self):
        entries = [cell.to_log_entry(self.sim._curr_time_s) for cell in self.sim._updated_cells.values()]
        agents = [agent.to_log_entry(self.sim._curr_time_s) for agent in self.sim.agent_list]
        actions = self.sim.get_action_entries(logger=False)
        self.update_grid(self.sim._curr_time_s, entries, agents, actions)
        self.sim._updated_cells.clear()

    def get_input_params(self):
        params = VisualizerInputs(
            cell_size=self.sim.cell_size,
            sim_shape=self.sim.shape,
            sim_size=self.sim.size,
            start_datetime=self.sim._start_datetime,
            north_dir_deg=self.sim._north_dir_deg,
            wind_forecast=self.sim.wind_forecast,
            wind_resolution=self.sim._wind_res,
            wind_t_step=self.sim.weather_t_step,
            wind_xpad=self.sim.wind_xpad,
            wind_ypad=self.sim.wind_ypad,
            elevation=self.sim.coarse_elevation,
            roads=self.sim.roads,
            fire_breaks=self.sim.fire_breaks,
            init_entries=self.get_init_entries()
        )

        return params
