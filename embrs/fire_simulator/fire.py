"""
Core fire simulation model.

This module defines the `FireSim` class, which implements a **wildfire simulation** 
based on fire spread dynamics, wind conditions, and terrain influences. It extends 
`BaseFireSim` and incorporates **hexagonal grid modeling** to track fire behavior 
at a cellular level.

Classes:
    - FireSim: The main wildfire simulation model.

.. autoclass:: FireSim
    :members:
"""

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

    `FireSim` extends `BaseFireSim` and models wildfire spread using **Rothermel's 
    fire spread equations**. It focuses on the core fire behavior simulation while
    inheriting grid management, weather, and terrain functionality from BaseFire.

    Attributes:
        burnout_thresh (float): The fraction of fuel consumption at which a cell is 
                                considered fully burned.
        logger (Optional[Logger]): A logging utility for storing simulation outputs.
        progress_bar (Optional[tqdm]): A progress bar for tracking simulation steps.
        _updated_cells (dict): Stores cells that have been modified during the current iteration.
        _soaked (list): Stores cells that have been suppressed or extinguished.
        _burning_cells (list): Contains all currently burning cells.
        _new_ignitions (list): Stores new ignitions to be processed in the next iteration.
        _agent_list (list): Tracks agents (e.g., firefighters, sensors) interacting with the fire.
        _agents_added (bool): Indicates whether agents have been added to the simulation.
        _visualizer (Optional[Visualizer]): Reference to the visualizer for visualization.

    Methods:
        iterate(): Advances the simulation by one time step.
        ignite_neighbors(): Attempts to ignite neighboring cells based on spread conditions.
        _init_iteration(): Resets and updates key variables at the start of each time step.
        log_changes(): Records simulation updates for logging and visualization.
        set_visualizer(): Sets the visualizer reference for this simulation.
        visualize_prediction(): Visualizes a prediction grid on top of the current simulation visualization.

    Notes:
        - The fire simulation operates on a **point-up hexagonal grid** managed by BaseFire.
        - Fire spread is calculated using **Rothermel's fire behavior model**.
        - The simulation tracks both fire progression and suppression efforts.
    """
    def __init__(self, sim_params: SimParams):
        """Initializes the wildfire simulation with input parameters and sets up core tracking structures.

        This constructor initializes key attributes related to fire progression, cell state tracking, 
        and agent interactions. It also sets up logging and a progress bar for monitoring 
        the simulation.

        Args:
            sim_params (SimParams): A structured input object containing all necessary simulation parameters, 
                                including terrain, fuel, wind conditions, and ignition points.

        Attributes Initialized:
            - **Logging & Monitoring:**
                - `logger` (Optional[Logger]): Handles simulation logging.
                - `progress_bar` (Optional[tqdm]): Tracks simulation progress visually.

            - **Agent Tracking:**
                - `_agent_list` (list): Holds agents (firefighters, sensors, etc.) interacting with the fire.
                - `_agents_added` (bool): Indicates whether agents have been added to the simulation.
        """
        print("Fire Simulation Initializing...")
        
        # Variable to store tqdm progress bar
        self.progress_bar = None

        # Variables to keep track of agents in sim
        self._agent_list = []
        self._agents_added = False

        # Boolean indicating if sim is finished
        self._finished = False

        # Reference to visualizer
        self._visualizer = None

        super().__init__(sim_params)
        
        # Log frequency
        self._log_freq = int(np.floor(3600 / self._time_step))

        self._init_iteration()

    def iterate(self):
        """Advances the fire simulation by one time step.

        This function updates fire propagation, wind conditions, and the state of burning cells.
        It handles new ignitions, spreads fire based on calculated rates of spread (ROS), 
        and removes cells that have fully burned.

        Behavior:
            - On the first iteration (`_iters == 0`):
                - Marks `self.weather_changed` as `True`.
                - Initializes `_new_ignitions` with `starting_ignitions`.
                - Sets the state of newly ignited cells to `CellStates.FIRE` and computes 
                their initial fire spread parameters.
            - Updates wind conditions if necessary (`_update_weather()`).
            - Adds new ignitions to `_burning_cells` and resets `_new_ignitions`.
            - Calls `_init_iteration()` to prepare for the time step.
            - Iterates through burning cells and:
                1. **Updates wind and fire spread parameters** if wind has changed or if 
                the cell hasn't reached steady-state ROS.
                2. **Calculates steady-state rate of spread (`r_ss`) and fireline intensity (`I_ss`)** 
                using `calc_propagation_in_cell()`.
                3. **Computes real-time ROS (`r_t`) and fireline intensity (`I_t`)**.
                4. **Advances fire spread** by updating `fire_spread` distances.
                5. **Determines if fire has reached the cell edge**, calling `ignite_neighbors()` 
                to attempt ignition of adjacent cells.
                6. **Removes fully burned cells** from `_burning_cells` and updates their state 
                to `CellStates.BURNT`.
                7. **Increments elapsed time for fire acceleration calculations**.
            - Calls `log_changes()` to record updates.

        Notes:
            - Fire spread is calculated using a **hexagonal grid model**.
            - ROS is updated dynamically based on wind changes and fire behavior.
            - Fire can only ignite neighboring cells that are still in a **burnable** state.
            - A mass-loss approach for fuel consumption is **not yet implemented**.
        """
        # Set-up iteration
        if self._init_iteration():
            self._finished = True
            return
        
        # Loop over surface fires
        for cell, loc in self._burning_cells:
            if cell.fully_burning:
                self.update_fuel_in_burning_cell(cell, loc)
                # No need to compute spread for these cells
                continue

            # Check if conditions have changed
            if self.weather_changed or not cell.has_steady_state: 
                # Reset the elapsed time counters
                cell.t_elapsed_min = 0

                # Update wind in cell
                cell._update_weather(self._curr_weather_idx, self._weather_stream, self._uniform_map)
                
                # Set previous rate of spreads to the most recent value
                if cell.r_t is not None:
                    cell.r_prev_list = cell.r_t
                else:
                    cell.r_prev_list = np.zeros(len(cell.directions))

                # Updates the cell's steady-state rate of spread and fireline intensity
                # Also checks for crown fire initiation
                self.update_steady_state(cell)

            # Set real time ROS and fireline intensity (vals stored in cell.r_t, cell.I_t)
            cell.set_real_time_vals()

            # Update extent of fire along each direction and check for ignition
            self.propagate_fire(cell)

            # Remove any neighbors that are no longer burnable
            self.remove_neighbors(cell)

            # Update time since conditions have changed for fire acceleration
            cell.t_elapsed_min += self.time_step / 60

            self.updated_cells[cell.id] = cell

        # Get set of spot fires started in this time step
        if self.model_spotting and self._spot_ign_prob > 0:
            self.propagate_embers()

        if self.logger:
            self._log_changes()

            if self._iters % self._log_freq == 0: # TODO: adjust this or make it settable
                self.logger.flush()

        self._iters += 1

    def _log_changes(self):
        # TODO: add a mechanism for treated cells to be added to updates

        self.logger.cache_cell_updates(self._get_cell_updates())

        if self.agents_added:
            self.logger.cache_agent_updates(self._get_agent_updates())

    def _init_iteration(self) -> bool:
        """Initialize or update the simulation state for the current iteration.

        This method handles both first-time initialization (when _iters == 0) and 
        subsequent iteration updates. It manages:
        - Progress bar initialization and updates
        - Weather condition updates
        - New ignition processing
        - Fire spread parameter calculations
        - Crown fire status updates
        - Fuel consumption history computation

        Returns:
            bool: True if the simulation should terminate (due to time limit or no active fires),
                 False otherwise.
        """
        if self._iters == 0:
            if self.progress_bar is None:
                self.progress_bar = tqdm(total=self._sim_duration/self.time_step,
                                     desc='Current sim ', position=0, leave=False)

            self.weather_changed = True
            self._new_ignitions = self.starting_ignitions
            for cell, loc in self._new_ignitions:
                cell.directions, cell.distances, cell.end_pts = UtilFuncs.get_ign_parameters(loc, self.cell_size)
                cell._set_state(CellStates.FIRE)

                r_list, I_list = calc_propagation_in_cell(cell) # r in m/s, I in BTU/ft/min
                cell.r_ss = r_list
                cell.I_ss = I_list
                cell.has_steady_state = True
                cell.set_real_time_vals()

                self._updated_cells[cell.id] = cell
        
        else:
            for cell, loc in self._new_ignitions:
                r_list, I_list = calc_propagation_in_cell(cell) # r in m/s, I in BTU/ft/min
                cell.r_ss = r_list
                cell.I_ss = I_list

                crown_fire(cell, self.fmc)

                if cell._crown_status != CrownStatus.ACTIVE:
                    cell.r_ss = r_list
                    cell.I_ss = I_list

                if cell._break_width > 0:
                    flame_len_ft = calc_flame_len(np.max(cell.I_ss))
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

                cell.set_real_time_vals()

        # Update current time
        self._curr_time_s = self.time_step * self._iters
        if self.progress_bar:
            self.progress_bar.update()
        
        # Compute the fuel consumption over time for each new ignition
        self.compute_burn_histories(self._new_ignitions)

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

        return False


    def update_fuel_in_burning_cell(self, cell: Cell, loc: int):
        # TODO: Need to figure out how we want to visualize this state
        cell.burn_idx += 1

        if cell.burn_idx == len(cell.burn_history):

            # Set static fuel load to new value
            cell.fuel.set_new_fuel_loading(cell.dynamic_fuel_load)

            # Check if there is enough fuel remaining to set back to fuel
            if cell.fuel.w_n_dead < self.burnout_thresh:
                # if not set to burnt
                self.set_state_at_cell(cell, CellStates.BURNT)
                self._burnt_cells.append(cell)
            else:
                self.set_state_at_cell(cell, CellStates.FUEL)
                                
            # remove from burning cells
            self._burning_cells.remove((cell, loc))
            
            cell.burn_idx = -1

        else:
            # TODO: If a fire intesects a cell in this state we need to set fuel load to this value for burn
            cell.dynamic_fuel_load = cell.burn_history[cell.burn_idx]
            
        # Add cell to update dictionary
        self._updated_cells[cell.id] = cell

    def _get_agent_updates(self):
        """Returns a list of dictionaries describing the location of each agent in __agent_list

        :return: List of dictionaries, describing the x,y position of each agent as well as their
                 display preferences
        :rtype: list
        """
        agent_data = [agent.to_log_entry(self.curr_time_s) for agent in self.agent_list]
        return agent_data
    
    def _get_cell_updates(self):
        cell_data = [cell.to_log_entry(self.curr_time_s) for cell in list(self._updated_cells.values())]
        return cell_data

    def set_visualizer(self, visualizer):
        """Sets the visualizer reference for this simulation.
        
        Args:
            visualizer: The Visualizer instance to use for visualization
        """
        self._visualizer = visualizer

    def visualize_prediction(self, prediction_grid):
        """Visualizes a prediction grid on top of the current simulation visualization.
        
        Args:
            prediction_grid (dict): Dictionary mapping timestamps to lists of (x,y) coordinates
                                  representing predicted fire spread
        """
        if self._visualizer is not None:
            self._visualizer.visualize_prediction(prediction_grid)

    @property
    def updated_cells(self) -> dict:
        """Dictionary containing cells updated since last time real-time visualization was updated.
        Dict keys are the ids of the :class:`~fire_simulator.cell.Cell` objects.
        """
        return self._updated_cells

    @property
    def agent_list(self) -> list:
        """List of :class:`~base_classes.agent_base.AgentBase` objects representing agents
        registered with the sim.
        """

        return self._agent_list

    @property
    def agents_added(self) -> bool:
        """`True` if agents have been registered in sim's agent list, returns `False` otherwise
        """
        return self._agents_added