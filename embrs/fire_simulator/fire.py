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
from embrs.utilities.fire_util import CellStates, FuelConstants, UtilFuncs, HexGridMath
from embrs.utilities.data_classes import SimParams
from embrs.fire_simulator.cell import Cell

from embrs.utilities.rothermel import *
from embrs.utilities.van_wagner_crown import calc_R10
from embrs.utilities.burnup import Burnup
from embrs.utilities.unit_conversions import *

from typing import Tuple

class FireSim(BaseFireSim):
    """A hexagonal grid-based wildfire simulation model.

    `FireSim` extends `BaseFireSim` and models wildfire spread using **Rothermel’s 
    fire spread equations**, wind influence, and terrain effects. It tracks individual 
    burning cells, manages ignition events, and simulates real-time fire growth over 
    a discrete simulation grid.

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
        _curr_weather_idx (int): The index of the current wind forecast entry.
        _last_wind_update (float): The last recorded time (in seconds) when wind was updated.

    Methods:
        iterate(): Advances the simulation by one time step.
        ignite_neighbors(): Attempts to ignite neighboring cells based on spread conditions.
        _update_weather(): Updates the current wind conditions if the forecast has changed.
        _init_iteration(): Resets and updates key variables at the start of each time step.
        log_changes(): Records simulation updates for logging and visualization.

    Notes:
        - The fire simulation operates on a **point-up hexagonal grid**.
        - Fire spread is calculated using **Rothermel’s fire behavior model**.
        - Wind conditions dynamically influence fire propagation.
        - The simulation tracks both fire progression and suppression efforts.

    """
    def __init__(self, sim_params: SimParams):
        """Initializes the wildfire simulation with input parameters and sets up core tracking structures.

        This constructor initializes key attributes related to fire progression, cell state tracking, 
        wind updates, and agent interactions. It also sets up logging and a progress bar for monitoring 
        the simulation.

        Args:
            sim_input (SimInput): A structured input object containing all necessary simulation parameters, 
                                including terrain, fuel, wind conditions, and ignition points.

        Attributes Initialized:
            - **Fire Behavior Tracking:**
                - `burnout_thresh` (float): Fuel fraction threshold at which a cell is considered fully burnt.
                - `_updated_cells` (dict): Stores cells modified during the current iteration.

            - **Cell State Management:**
                - `_soaked` (list): Stores cells affected by suppression efforts.
                - `_burning_cells` (list): Contains currently burning cells.
                - `_new_ignitions` (list): Stores new ignitions for the next iteration.

            - **Agent Tracking:**
                - `_agent_list` (list): Holds agents (firefighters, sensors, etc.) interacting with the fire.
                - `_agents_added` (bool): Indicates whether agents have been added to the simulation.

            - **Wind Conditions:**
                - `_curr_weather_idx` (int): Index tracking the current wind forecast entry.
                - `_last_wind_update` (float): Timestamp of the last wind update.

            - **Logging & Monitoring:**
                - `logger` (Optional[Logger]): Handles simulation logging.
                - `progress_bar` (Optional[tqdm]): Tracks simulation progress visually.

        Notes:
            - Calls `super().__init__(sim_input)` to inherit base functionality from `BaseFireSim`.
            - Calls `_init_iteration()` to set up initial conditions for the first simulation step.
            - The progress bar is initialized later when the simulation starts.

        """

        print("Simulation Initializing...")

        # Fuel fraction to be considered BURNT
        self.burnout_thresh = 0.01

        # Variable to store logger object
        self.logger = None

        # Variable to store tqdm progress bar
        self.progress_bar = None

        # Containers for keeping track of updates to cells 
        self._updated_cells = {}

        # Containers for cells
        self._soaked = []
        self._burning_cells = []
        self._new_ignitions = []

        # Crown fire containers
        self._active_crowns = []
        self._new_active_crown_ignitions = []
        self._passive_crowns = []

        # Foliar moisture as a percentage of dry weight
        self.foliar_moisture_content = 100 # TODO: Set this somewhere in base_fire and sim initilization process

        # Variables to keep track of agents in sim
        self._agent_list = []
        self._agents_added = False

        # Variables to keep track of current wind conditions
        self._curr_weather_idx = 0
        self._last_weather_update = 0

        super().__init__(sim_params)
        
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
                cell._update_weather(self._curr_weather_idx, self._weather_stream)
                
                # Set previous rate of spreads to the most recent value
                if cell.r_t is not None:
                    cell.r_prev_list = cell.r_t
                else:
                    cell.r_prev_list = np.zeros(len(cell.directions))

                self.update_surface_steady_state(cell)

            # Set real time ROS and fireline intensity (vals stored in cell.r_t, cell.I_t)
            cell.set_real_time_vals()

            # Update extent of surface fire along each direction and check for ignition
            self.propagate_surface_fire(cell)

            # Remove any neighbors that are no longer burnable
            self.remove_surface_neighbors(cell)

            # Check for crown fire ignition within cell
            self.check_for_crown_fire(cell)

            # Update time since conditions have changed for fire acceleration
            cell.t_elapsed_min += self.time_step / 60

            self.updated_cells[cell.id] = cell


        # TODO: should there be a separate state for crown fires?
        # TODO: Need to set up active crown spread directions (get ign_parameters etc.)
        self._active_crowns.extend(self._new_active_crown_ignitions)

        for active_crown in self._active_crowns:
            # TODO: implement spread of active crown fire
            pass


        self._iters += 1

    def generate_burn_history_entry(self, cell, fuel_loads):
        # TODO: this assumes that any live fuel will be totally consumed
        # TODO: verify this assumption

        entry = [0] * len(cell.fuel.w_0)
        j = 0
        for i in range(len(cell.fuel.w_0)):
            if i in cell.fuel.rel_indices:
                entry[i] = fuel_loads[j]
                j += 1

        return entry

    def compute_burn_histories(self, new_ignitions):
        # TODO: This assumes weather will be static across burn history
        curr_weather = self._weather_stream.stream[self._curr_weather_idx]

        wind_speed = curr_weather.wind_speed
    
        t_ambF = curr_weather.temp

        dt = self._time_step

        for cell, _ in new_ignitions:
            # Reset cell burn history
            cell.burn_history = []

            # Get cell duff loading (tons/acre)
            wdf = cell.wdf

            I_r = cell.reaction_intensity  # BTU/ft2/min

            # TODO: should we add wind speed to the burnup model?
            u = 0   #wind_speed * cell.wind_adj_factor
            
            # Get fuel bed depth
            depth = cell.fuel.fuel_depth_ft

            # Calculate duff moisture content
            if 2 in cell.fuel.rel_indices:
                mx = cell.fmois[2]
            elif 1 in cell.fuel.rel_indices:
                mx = cell.fmois[1]
            else:
                mx = cell.fmois[0]

            dfm = -0.347 + 6.42 * mx
            dfm = max(dfm, 0.10)

            # Calculate Residence time using FARSITE equation
            fli = np.max(cell.I_ss) # BTU/ft/min
            ros = m_s_to_ft_min(np.max(cell.r_ss)) #ft/min
            
            t_r = (fli*60) / (ros * I_r) # residence time in seconds
            
            # Clip to allowable values in FOFEM
            t_r = np.min([t_r, 120])
            t_r = np.max([t_r, 10])

            burn_mgr = Burnup(cell)
            burn_mgr.set_fire_data(3000, I_r, t_r, u, depth, t_ambF, dt, wdf, dfm)

            burn_mgr.arrays()
            now = 1
            d_time = burn_mgr.ti
            burn_mgr.duff_burn()

            if not (burn_mgr.start(d_time, now)):
                # Burnup does not predict ignition
                # Set to amount consumed in flaming front
                # This is how farsite does it
                fuel_loads = burn_mgr.get_flaming_front_consumption()
                entry = self.generate_burn_history_entry(cell, fuel_loads)
                cell.burn_history = [entry]

                continue

            burn_mgr.fi = burn_mgr.fire_intensity()

            if d_time > burn_mgr.tdf:
                burn_mgr.dfi = 0

            while now <= burn_mgr.ntimes:
                burn_mgr.step(burn_mgr.dt, d_time, burn_mgr.dfi)
                now += 1

                d_time += burn_mgr.dt
                if d_time > burn_mgr.tdf:
                    burn_mgr.dfi = 0

                burn_mgr.fi = burn_mgr.fire_intensity()
                
                if burn_mgr.fi <= burn_mgr.fi_min:
                    break

                fuel_loads = burn_mgr.get_updated_fuel_loading()
                entry = self.generate_burn_history_entry(cell, fuel_loads)
                cell.burn_history.append(entry)

            if len(cell.burn_history) == 0:
                # Intensity was not high enough to ignite
                # Set to amount consumed in flaming front
                # This is how farsite does it
                fuel_loads = burn_mgr.get_flaming_front_consumption()
                entry = self.generate_burn_history_entry(cell, fuel_loads)
                cell.burn_history = [entry]

    def ignite_neighbors(self, cell: Cell, r_gamma: float, end_point: list) -> list:
        """Attempts to ignite neighboring cells based on fire spread conditions.

        This method evaluates fire spread from a burning cell to its neighbors. 
        If a neighboring cell meets ignition criteria, it is transitioned to the `FIRE` state 
        and its fire spread parameters are updated.

        Args:
            cell (Cell): The currently burning cell attempting to ignite its neighbors.
            r_gamma (float): The rate of spread within the burning cell along the ignition direction.
            end_point (list): A list of tuples representing fire spread endpoints, where each tuple 
                            contains:
                            - An integer indicating the ignition location along the neighboring cell.
                            - A letter (A-F) indicating which neighbor the fire is spreading to.

        Returns:
            list: A list of successfully ignited neighboring `Cell` objects.

        Behavior:
            - Iterates through `end_point` to identify potential ignition locations.
            - Calls `get_neighbor_from_end_point()` to retrieve the corresponding neighboring cell.
            - Checks if the neighbor is in a burnable state (`CellStates.FUEL` and has a burnable fuel type).
            - Computes the **ignition rate of spread** (`r_ign`) using `calc_ignition_ros()`.
            - If `r_ign > 1e-3`, the neighbor is ignited:
                - Adds the cell to `_new_ignitions`.
                - Initializes fire spread parameters (`directions`, `distances`, `end_pts`).
                - Updates wind conditions using `_update_wind()`.
                - Computes in-cell fire propagation using `calc_propagation_in_cell()`.
                - Logs the update to `_updated_cells`

        Notes:
            - The ignition threshold (`1e-3`) is a placeholder; consider using mass-loss calculations 
            or setting `R_min` dynamically.
            - If a neighboring cell is **not** ignitable but exists in `cell.burnable_neighbors`, 
            it is removed from that list.
    """

        # Loop through end points
        for pt in end_point:

            # Get the location of the potential ignition on the neighbor
            n_loc = pt[0]

            # Get the Cell object of the neighbor
            neighbor = self.get_neighbor_from_end_point(cell, pt)

            if neighbor:
                # Check that neighbor state is burnable
                if neighbor.state == CellStates.FUEL and neighbor.fuel.burnable:
                    # Make ignition calculation
                    neighbor._update_weather(self._curr_weather_idx, self._weather_stream)
                    r_ign = self.calc_ignition_ros(cell, neighbor, r_gamma) # ft/min
                    r_0, _ = calc_r_0(neighbor.fuel, neighbor.fmois) # ft/min

                    # Check that ignition ros is greater than no wind no slope ros
                    if 0 < r_0 < r_ign:
                        self._new_ignitions.append((neighbor, n_loc))
                        neighbor.directions, neighbor.distances, neighbor.end_pts = UtilFuncs.get_ign_parameters(n_loc, self.cell_size)
                        neighbor._set_state(CellStates.FIRE)
                        neighbor.r_prev_list, _ = calc_propagation_in_cell(neighbor, r_ign) # r in m/s, I in BTU/ft/min
                        
                        self._updated_cells[neighbor.id] = neighbor

    def calc_ignition_ros(self, cell: Cell, neighbor: Cell, r_gamma: float) -> float:
        """Calculates the rate of spread (ROS) required for ignition between a burning cell 
        and an unburnt neighboring cell.

        This method determines the ignition ROS by comparing the heat source of the burning 
        cell to the heat sink of the unburned neighbor. The calculation follows:

            r_ign = heat_source_of_burning_cell / heat_sink_of_unburned_neighbor

        where:
            - `heat_source_of_burning_cell` is calculated as:
                
                heat_source_of_burning_cell = r_gamma * heat_sink_of_burning_cell

            This accounts for the energy available for fire spread along the ignition direction.
            - `r_gamma` represents the rate of spread within the burning cell in the ignition direction.
            - `heat_sink_of_burning_cell` and `heat_sink_of_unburned_neighbor` are computed 
            using the **Rothermel fire spread model**, which accounts for fuel properties and moisture content.

        Args:
            cell (Cell): The burning cell acting as the heat source.
            neighbor (Cell): The adjacent unburned cell receiving heat (potential ignition target).
            r_gamma (float): The rate of spread within the burning cell along the igniting direction.

        Returns:
            float: The calculated ignition rate of spread (ROS), representing the minimum 
                fire spread rate required for ignition of the neighboring cell.

        Notes:
            - The `calc_heat_sink` function is used to compute both heat source and sink values.
            - This method assumes that `r_gamma` is precomputed and valid.
            - The accuracy of this calculation depends on correct fuel moisture modeling.
            - Currently, fuel moisture content updates are not implemented.
        """

        m_f = get_working_m_f(cell.fuel, cell.fmois)
        neighbor_m_f = get_working_m_f(neighbor.fuel, neighbor.fmois)

        # Get the rate of spread in ft/s
        r_ft_s = m_s_to_ft_min(r_gamma)

        # Get the heat source in the direction of question by eliminating denominator
        heat_source = r_ft_s * calc_heat_sink(cell.fuel, m_f) # TODO: make sure this computation is valid (I think it is)

        # Get the heat sink using the neighbors fuel and moisture content
        heat_sink = calc_heat_sink(neighbor.fuel, neighbor_m_f)
        
        # Calculate a ignition rate of spread
        r_ign = heat_source / heat_sink

        return r_ign

    def get_neighbor_from_end_point(self, cell: Cell, end_point: Tuple[int, str]) -> Cell:
        """Retrieves the neighboring cell corresponding to a fire spread endpoint.

            This method identifies which neighboring cell is adjacent to a given fire spread 
            endpoint within the burning cell. The endpoint location is represented as a tuple:

                (position_index, neighbor_letter)

            where:
                - `position_index` (int) is a number from `1-12` indicating the fire spread endpoint 
                on the neighboring cell.
                - `neighbor_letter` (str) is a letter from `A-F` indicating which of the six 
                neighboring cells the endpoint borders.

            The mapping of these conventions is defined in **HexGridMath** (see `utilities.fire_util`).

            Args:
                cell (Cell): The burning cell from which the fire spreads.
                end_point (Tuple[int, str]): A tuple representing the endpoint of the fire spread direction.

            Returns:
                Optional[Cell]: The neighboring cell that the endpoint borders if it exists and is burnable, 
                                otherwise `None`.

            Notes:
                - Even-row and odd-row hexagonal grids use different neighbor mappings (handled via `HexGridMath`).
                - The method ensures that the retrieved neighbor exists within the simulation grid bounds.
                - Only neighbors listed in `cell.burnable_neighbors` are considered valid.
            """
        # Get the letter representing the neighbor location relative to cell
        neighbor_letter = end_point[1]

        # Get neighbor based on neighbor_letter
        if cell._row % 2 == 0:
            diff_to_letter_map = HexGridMath.even_neighbor_letters
            
        else:
            diff_to_letter_map = HexGridMath.odd_neighbor_letters

        # Get the row and col difference between cell and neighbor
        dx, dy = diff_to_letter_map[neighbor_letter]

        row_n = int(cell.row + dy)
        col_n = int(cell.col + dx)

        if self._grid_height >= row_n >=0 and self._grid_width >= col_n >= 0:
            # Retrieve neighbor from cell grid
            neighbor = self._cell_grid[row_n, col_n]

            # If neighbor in cell's neighbors return it
            if neighbor.id in cell.burnable_neighbors:
                return neighbor

        return None

    def _init_iteration(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        if self._iters == 0:
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
                cell.has_steady_state = True
                cell.set_real_time_vals()

        # Update current time
        self._curr_time_s = self.time_step * self._iters
        self.progress_bar.update()

        # Update wind if necessary
        self.weather_changed = self._update_weather()
        
        # Compute the fuel consumption over time for each new ignition
        self.compute_burn_histories(self._new_ignitions)

        # Add any new ignitions to the current set of burning cells
        self._burning_cells.extend(self._new_ignitions)
        # Reset new ignitions
        self._new_ignitions = []

        if self._curr_time_s >= self._sim_duration or (self._iters != 0 and len(self._burning_cells) == 0):
            self.progress_bar.close()

            return True

        return False
    

    def propagate_surface_fire(self, cell: Cell):
        
        if np.all(cell.r_t == 0) and np.all(cell.r_ss == 0) and self._iters != 0:
            cell.fully_burning = True

        # Update extent of fire spread along each direction
        cell.fire_spread = cell.fire_spread + (cell.r_t * self._time_step)

        # TODO: is there a way to prevent distances that are done from being computed?
        intersections = np.where(cell.fire_spread > cell.distances)[0]

        # TODO: Check if fireline intensity along any direction is high to initiate crown fire

        # Check where fire spread has reached edge of cell
        if len(intersections) >= int(len(cell.distances)):
            # Set cell to fully burning when all edges reached
            cell.fully_burning = True

        for idx in intersections:
            # Check if ignition signal should be sent to each intersecting neighbor
            self.ignite_neighbors(cell, cell.r_t[idx], cell.end_pts[idx])


    def remove_surface_neighbors(self, cell: Cell):
        # Remove any neighbors which are no longer burnable
            neighbors_to_rem = []
            for n_id in cell.burnable_neighbors:
                neighbor = self._cell_dict[n_id]

                if neighbor.state != CellStates.FUEL:
                    neighbors_to_rem.append(n_id)

            if neighbors_to_rem:
                for n_id in neighbors_to_rem:
                    del cell.burnable_neighbors[n_id]

            if len(cell.burnable_neighbors) == 0:
                # Set cell to fully burning when no burnable neighbors remain
                cell.fully_burning = True        

    def check_for_crown_fire(self, cell: Cell):
        # TODO: Check all calculations (units etc.)
        # TODO: Get it in a case where both passive and active crown fires initiated

        # Return if crown fire not possible 
        # TODO: checking for active crowns this way may not be correct
        if not cell.has_canopy or cell in self._active_crowns:
            return
        
        # Calculate crown fire intensity threshold
        I_o = (0.01 * cell.canopy_base_height * (460 + 25.9 * self.foliar_moisture_content))**(3/2) # kW/m

        # TODO: should rewrite the Rothermel functions so that we can get fireline intensity more directly
        # Calculate the maximum fireline intensity in the cell (along the heading direction)
        R, _, I_r, _ = calc_r_h(cell)
        t_r = 384 / cell.fuel.sav_ratio
        H_a = I_r * t_r
        I_t = H_a * R

        I_t = BTU_ft_min_to_kW_m(I_t) # kW/m

        # Check if fireline intensity is high enough to initiate crown fire
        if I_t >= I_o:
            # Surface fire will initiate a crown fire

            # Check if crown should be passive or active

            rac = 3.0 / cell.canopy_bulk_density

            R_0 = I_o - (R/I_t)

            a_c = -np.log(0.1) / (0.9 * (rac - R_0))

            cfb = 1 - np.exp(-a_c * (R - R_0))

            R_10 = calc_R10(cell)

            R_cmax = 3.34 * R_10

            r_actual = R + cfb * (R_cmax - R)

            # TODO: should use the same checks farsite has for values of R_cmax etc.
            if r_actual >= rac:
                # Active crown fire
                self._new_active_crown_ignitions.append(cell)
            else:
                # Passive crown fire
                self._passive_crowns.append(cell)

    def update_surface_steady_state(self, cell: Cell):
        """_summary_

        Args:
            cell (Cell): _description_
        """
        r_list, I_list = calc_propagation_in_cell(cell) # r in m/s, I in BTU/ft/min
        cell.r_ss = r_list
        cell.I_ss = I_list
        cell.has_steady_state = True

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
    
    def _update_weather(self) -> bool:
        """Updates the current wind conditions based on the forecast.


        This method checks whether the time elapsed since the last wind update exceeds 
        the wind forecast time step. If so, it updates the wind index and retrieves 
        the next forecasted wind condition. If the forecast has no remaining entries, 
        it raises a `ValueError`.

        Returns:
            bool: `True` if the wind conditions were updated, `False` otherwise.

        Raises:
            ValueError: If the wind forecast runs out of entries.

        Side Effects:
            - Updates `_last_wind_update` to the current simulation time.
            - Increments `_curr_weather_idx` to the next wind forecast entry.
            - Resets `_curr_weather_idx` to `0` if out of bounds and raises an error.
        """
        # Check if a wind forecast time step has elapsed since last update
        weather_changed = self.curr_time_s - self._last_weather_update >= self.weather_t_step

        if weather_changed:
            # Reset last wind update to current time
            self._last_weather_update = self.curr_time_s

            # Increment wind index
            self._curr_weather_idx += 1

            # Check for out of bounds index
            if self._curr_weather_idx >= len(self._weather_stream.stream):
                self._curr_weather_idx = 0
                raise ValueError("Weather forecast has no more entries!")
        
        return weather_changed

    def _get_agent_updates(self):
        """Returns a list of dictionaries describing the location of each agent in __agent_list

        :return: List of dictionaries, describing the x,y position of each agent as well as their
                 display preferences
        :rtype: list
        """
        agent_data = []

        for agent in self.agent_list:
            agent_data.append(agent.to_log_format())

        return agent_data

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
