"""Real-time visualization for running fire simulations.

This module provides a visualizer that renders fire simulation state in real-time
during simulation execution. It extends the base visualizer with live update
capabilities and caching for efficient rendering.

Classes:
    - RealTimeVisualizer: Live visualization interface for FireSim instances.

.. autoclass:: RealTimeVisualizer
    :members:
"""
from embrs.base_classes.base_visualizer import BaseVisualizer
from embrs.fire_simulator.fire import FireSim
from embrs.utilities.data_classes import VisualizerInputs

import numpy as np


class RealTimeVisualizer(BaseVisualizer):
    """Real-time visualization interface for running fire simulations.

    Provides live rendering of fire spread, agent positions, and weather data
    during simulation execution. Cell updates are cached between visualization
    frames for efficient batch rendering.

    Attributes:
        sim (FireSim): Reference to the associated fire simulation.
        cell_cache (list): Buffer of CellLogEntry objects awaiting visualization.
    """

    def __init__(self, sim: FireSim):
        """Initialize the real-time visualizer for a simulation.

        Args:
            sim (FireSim): The fire simulation instance to visualize.

        Side Effects:
            - Initializes the base visualizer with simulation parameters.
            - Creates an empty cell cache for buffering updates.
        """
        self.sim = sim
        self.cell_cache = []

        input_params = self.get_input_params()

        super().__init__(input_params)

    def set_sim(self, sim: FireSim):
        """Update the simulation reference for this visualizer.

        Args:
            sim (FireSim): New simulation instance to visualize.
        """
        self.sim = sim

    def get_init_entries(self) -> list:
        """Generate initial log entries for all cells in the simulation.

        Returns:
            list: List of CellLogEntry objects representing the initial state
                of every cell in the simulation grid.
        """
        entries = [cell.to_log_entry(self.sim._curr_time_s) for cell in self.sim.cell_dict.values()]
        return entries

    def cache_changes(self, updated_cells: list):
        """Add cell updates to the visualization cache.

        Buffers cell state changes for batch rendering on the next update call.

        Args:
            updated_cells (list): List of CellLogEntry objects to cache.

        Side Effects:
            - Extends self.cell_cache with new entries.
        """
        self.cell_cache.extend(updated_cells)

    def update(self):
        """Render all cached updates to the visualization.

        Collects agent positions and action entries from the simulation,
        then renders all cached cell changes along with current agent and
        action state.

        Side Effects:
            - Calls update_grid with cached cell data.
            - Clears the cell cache after rendering.
        """
        agents = [agent.to_log_entry(self.sim._curr_time_s) for agent in self.sim.agent_list]
        actions = self.sim.get_action_entries(logger=False)
        self.update_grid(self.sim._curr_time_s, self.cell_cache, agents, actions)

        self.cell_cache = []

    def get_input_params(self) -> VisualizerInputs:
        """Build visualization parameters from the current simulation state.

        Extracts grid geometry, weather forecasts, terrain data, and initial
        cell states from the simulation to configure the base visualizer.

        Returns:
            VisualizerInputs: Configuration dataclass for BaseVisualizer.
        """
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
            temp_forecast=np.array([entry.temp for entry in self.sim._weather_stream.stream[self.sim.sim_start_w_idx:]]),
            rh_forecast=np.array([entry.rel_humidity for entry in self.sim._weather_stream.stream[self.sim.sim_start_w_idx:]]),
            forecast_t_step=self.sim.weather_t_step,
            elevation=self.sim.coarse_elevation,
            roads=self.sim.roads,
            fire_breaks=self.sim.fire_breaks,
            init_entries=self.get_init_entries(),
            show_weather_data=True
        )

        return params
