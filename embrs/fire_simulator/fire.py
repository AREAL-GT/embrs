"""Core fire simulation model.

This module defines the FireSim class, which implements wildfire simulation
based on fire spread dynamics, wind conditions, and terrain influences. It extends
BaseFireSim and incorporates hexagonal grid modeling to track fire behavior
at a cellular level.

Classes:
    - FireSim: The main wildfire simulation model.

.. autoclass:: FireSim
    :members:
"""

from datetime import timedelta
from tqdm import tqdm
import numpy as np

from embrs.base_classes.base_fire import BaseFireSim
from embrs.utilities.fire_util import CellStates, CrownStatus, UtilFuncs
from embrs.utilities.data_classes import SimParams
from embrs.fire_simulator.cell import Cell

from embrs.models.rothermel import *
from embrs.models.crown_model import crown_fire

class FireSim(BaseFireSim):
    """A hexagonal grid-based wildfire simulation model.

    Extends BaseFireSim and models wildfire spread using Rothermel's fire spread
    equations. Handles core fire behavior simulation while inheriting grid management,
    weather, and terrain functionality from the base class.

    Attributes:
        logger (Logger): Logging utility for storing simulation outputs.
        progress_bar (tqdm): Progress bar for tracking simulation steps.
        agent_list (list): Agents interacting with the fire.
        agents_added (bool): Whether agents have been added to the simulation.
    """
    def __init__(self, sim_params: SimParams):
        """Initialize the wildfire simulation.

        Args:
            sim_params (SimParams): Simulation parameters including terrain, fuel,
                wind conditions, and ignition points.
        """
        print("Fire Simulation Initializing...")
        
        # Variable to store tqdm progress bar
        self.progress_bar = None

        # Variables to keep track of agents in sim
        self._agent_list = []
        self._agents_added = False

        # Reference to visualizer
        self._visualizer = None

        # Reference to active prediction
        self.curr_prediction = None

        super().__init__(sim_params)

        # Log frequency (set to 1 hour by default)
        self._log_freq = int(np.floor(3600 / self._time_step))

        # Dynamic GSI tracking (None if use_gsi is False)
        self._gsi_tracker = self._weather_stream.gsi_tracker
        self._last_gsi_weather_idx = -1
        self._live_mf_changed = False

        self._init_iteration(True)

    def iterate(self):
        """Advance the fire simulation by one time step.

        Updates fire propagation, weather conditions, and burning cell states.
        Handles new ignitions, calculates rate of spread, and transitions fully
        burned cells to BURNT state.

        Side Effects:
            - Updates cell states and fire spread distances.
            - May ignite neighboring cells.
            - Logs changes if logger is configured.
            - Updates visualizer if configured.
        """
        # Set-up iteration
        if self._init_iteration():
            self._finished = True
            return

        # Loop over surface fires
        # Cache frequently-accessed attributes and methods for tight loop
        weather_changed = self.weather_changed
        live_mf_changed = self._live_mf_changed
        weather_idx = self._curr_weather_idx
        weather_stream = self._weather_stream
        ts = self.time_step
        updated_cells = self._updated_cells
        burnt_cells = self._burnt_cells
        update_steady = self.update_steady_state
        propagate = self.propagate_fire
        remove_nbrs = self.remove_neighbors

        # Track cells to remove (avoids copying entire list)
        cells_to_remove = []
        for cell in self._burning_cells:
            if cell.fully_burning:
                cell._set_state(CellStates.BURNT)
                burnt_cells.add(cell)
                cells_to_remove.append(cell)
                updated_cells[cell.id] = cell
                continue

            # Check if conditions have changed
            needs_update = weather_changed or not cell.has_steady_state or live_mf_changed

            if needs_update:
                cell._update_moisture(weather_idx, weather_stream)

            # Van Wagner binary-threshold suppression check
            if cell.water_applied_kJ > 0:
                cell.decay_water_energy(self._curr_time_s)
                if cell.water_applied_kJ > 0 and cell.check_vw_extinguishment():
                    cell.extinguish_vw()
                    needs_update = True
                    # ROS will be zero after surface_fire recompute;
                    # propagate_fire handles suppress_to_fuel

            if needs_update:
                update_steady(cell)

            # Set real time ROS and fireline intensity
            accelerate(cell, ts)

            # Update extent of fire along each direction and check for ignition
            propagate(cell)

            # Remove any neighbors that are no longer burnable
            remove_nbrs(cell)

            updated_cells[cell.id] = cell

        # Remove fully burned cells after iteration completes
        if cells_to_remove:
            remove_set = set(cells_to_remove)
            self._burning_cells = [c for c in self._burning_cells if c not in remove_set]

        # Remove suppressed cells (transitioned back to FUEL via partial suppression)
        if self._suppressed_cells:
            suppressed_set = set(self._suppressed_cells)
            self._burning_cells = [c for c in self._burning_cells if c not in suppressed_set]
            for cell in self._suppressed_cells:
                self._updated_cells[cell.id] = cell
            self._suppressed_cells.clear()

        # Get set of spot fires started in this time step
        if self.model_spotting and self._spot_ign_prob > 0:
            self.propagate_embers()

        self.update_control_interface_elements()

        if self.logger:
            self._log_changes()

            if self._iters % self._log_freq == 0:
                self.logger.flush()

        if self._visualizer:
            self._visualizer.cache_changes(self._get_cell_updates())

        self._updated_cells.clear()
        self._iters += 1

    def _log_changes(self):
        """Cache all simulation updates for logging.

        Collects cell state changes, agent positions, prediction data, and
        action entries, then sends them to the logger for buffered output.

        Side Effects:
            - Caches cell updates via logger.cache_cell_updates.
            - Caches agent updates if agents are registered.
            - Caches current prediction if one exists (then clears it).
            - Caches action entries from the control interface.
        """
        self.logger.cache_cell_updates(self._get_cell_updates())

        if self.agents_added:
            self.logger.cache_agent_updates(self._get_agent_updates())

        if self.curr_prediction is not None:
            self.logger.cache_prediction(self.get_prediction_entry())
            self.curr_prediction = None

        self.logger.cache_action_updates(self.get_action_entries(logger=True))

    def _init_iteration(self, in_constructor: bool = False) -> bool:
        """Initialize or update simulation state for the current iteration.

        Handles first-time initialization and subsequent iteration updates including
        progress bar, weather conditions, new ignitions, and fire spread parameters.

        Args:
            in_constructor (bool): True when called from __init__, False otherwise.

        Returns:
            bool: True if simulation should terminate, False otherwise.
        """
        if in_constructor:
            if self.progress_bar is None:
                self.progress_bar = tqdm(total=self._sim_duration/self.time_step,
                                     desc='Current sim ', position=0, leave=False)

            self.weather_changed = True
            self._new_ignitions = []
            for cell, loc in self.starting_ignitions:
                cell._arrival_time = self.curr_time_m
                cell.get_ign_params(loc)

                if not cell.fuel.burnable:
                    continue
                
                cell._update_moisture(self._curr_weather_idx, self._weather_stream)
                cell._set_state(CellStates.FIRE)
                surface_fire(cell)
                crown_fire(cell, self.fmc)

                cell.has_steady_state = True
                accelerate(cell, self.time_step)

                self._updated_cells[cell.id] = cell
                self._new_ignitions.append(cell)
        
        else:
            for cell in self._new_ignitions:
                cell._arrival_time = self.curr_time_m
                surface_fire(cell)
                crown_fire(cell, self.fmc)

                for neighbor in list(cell.burnable_neighbors.keys()):
                    self._frontier.add(neighbor)

                if cell._break_width > 0:
                    flame_len_ft = calc_flame_len(cell)
                    flame_len_m = ft_to_m(flame_len_ft)

                    hold_prob = cell.calc_hold_prob(flame_len_m)
                    
                    rand = np.random.random()

                    cell.breached = rand > hold_prob

                else:
                    cell.breached = True

                cell.has_steady_state = True

                if self.model_spotting:
                    if not cell.lofted and cell._crown_status != CrownStatus.NONE and self._spot_ign_prob > 0:
                        self.embers.loft(cell, self.curr_time_m)

                accelerate(cell, self.time_step)

        # Update current time
        self._curr_time_s = self.time_step * self._iters
        if self.progress_bar:
            self.progress_bar.update()

        # Add any new ignitions to the current set of burning cells
        self._burning_cells.extend(self._new_ignitions)
        # Reset new ignitions
        self._new_ignitions = []

        if self._curr_time_s >= self._sim_duration or (self._iters != 0 and len(self._burning_cells) == 0):
            if self.progress_bar:
                self.progress_bar.close()

            return True

        # Update wind if necessary
        self.weather_changed = self._update_weather()

        # Reset live moisture flag and check for GSI day boundary
        self._live_mf_changed = False
        if self._gsi_tracker is not None:
            self._update_gsi()

        return False

    def _update_gsi(self):
        """Feed hourly weather to the GSI tracker and trigger daily update.

        Called each iteration from :meth:`_init_iteration`. Only acts
        when the weather index has advanced (new hourly entry). If the
        calendar day changed, finalizes the previous day's summary and
        triggers :meth:`_apply_gsi_update` to recompute live moistures.
        """
        weather_idx = self._curr_weather_idx

        if weather_idx == self._last_gsi_weather_idx:
            return

        entry = self._weather_stream.stream[weather_idx]
        sim_dt = self._start_datetime + timedelta(seconds=self._curr_time_s)

        day_changed = self._gsi_tracker.ingest_hourly(entry, sim_dt)
        self._last_gsi_weather_idx = weather_idx

        if day_changed:
            self._apply_gsi_update()

    def _apply_gsi_update(self):
        """Recompute GSI and update fuel curing and sim-level live moistures.

        Does NOT iterate over cells. Sets :attr:`_live_mf_changed` so
        that burning cells sync their ``fmois`` lazily via
        :meth:`Cell._update_moisture`. Cold cells pick up new values
        when they ignite.
        """
        gsi = self._gsi_tracker.compute_gsi()
        if gsi < 0:
            return

        new_h, new_w = self._weather_stream.set_live_moistures(gsi)

        if abs(new_h - self._live_h_mf) < 1e-6 and abs(new_w - self._live_w_mf) < 1e-6:
            return

        self._live_h_mf = new_h
        self._live_w_mf = new_w

        # Update shared fuel objects (~40 cache entries)
        for fuel in self._fuel_cache.values():
            fuel.update_curing(new_h)

        self._live_mf_changed = True

    def _get_agent_updates(self) -> list:
        """Collect log entries for all registered agents.

        Returns:
            list: List of AgentLogEntry objects, one per agent, containing
                position and display properties at the current simulation time.
        """
        agent_data = [agent.to_log_entry(self.curr_time_s) for agent in self.agent_list]
        return agent_data
    
    def _get_cell_updates(self) -> list:
        """Collect log entries for all cells modified this iteration.

        Returns:
            list: List of CellLogEntry objects for cells in _updated_cells.
        """
        cell_data = [cell.to_log_entry(self.curr_time_s) for cell in list(self._updated_cells.values())]

        return cell_data

    def set_visualizer(self, visualizer):
        """Set the visualizer reference for this simulation.

        Args:
            visualizer (RealTimeVisualizer): Visualizer instance to use.
        """
        self._visualizer = visualizer

    def visualize_prediction(self, prediction_grid: dict):
        """Display a prediction grid on the visualization.

        Args:
            prediction_grid (dict): Prediction data mapping timestamps to
                coordinate lists.

        Side Effects:
            - Passes prediction to visualizer if available.
            - Stores prediction in self.curr_prediction for logging.
        """
        if self._visualizer is not None:
            self._visualizer.visualize_prediction(prediction_grid)

        self.curr_prediction = prediction_grid

    def visualize_ensemble_prediction(self, prediction_grid: dict):
        """Visualize ensemble prediction results on the current visualization.

        Args:
            prediction_grid (dict): Ensemble prediction data structure containing
                multiple forecast scenarios.

        Side Effects:
            - Passes prediction to visualizer if available.
            - Stores prediction in self.curr_prediction for logging.
        """
        if self._visualizer is not None:
            self._visualizer.visualize_ensemble_prediction(prediction_grid)

        self.curr_prediction = prediction_grid

    @property
    def agent_list(self) -> list:
        """List of agents registered with the simulation."""
        return self._agent_list

    @property
    def agents_added(self) -> bool:
        """True if agents have been registered, False otherwise."""
        return self._agents_added