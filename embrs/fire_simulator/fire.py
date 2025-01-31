"""Core fire simulation model.

.. autoclass:: FireSim
    :members:
"""

from tqdm import tqdm
import numpy as np

from embrs.base_classes.base_fire import BaseFireSim
from embrs.utilities.fire_util import CellStates, FireTypes, FuelConstants, UtilFuncs, HexGridMath
from embrs.utilities.sim_input import SimInput
from embrs.fire_simulator.cell import Cell

from embrs.utilities.rothermel import *

from typing import Tuple

class FireSim(BaseFireSim):
    def __init__(self, sim_input: SimInput):
        """_summary_

        Args:
            sim_input (SimInput): _description_
        """

        print("Simulation Initializing...")

        # Fuel fraction to be considered BURNT
        self.burnout_thresh = FuelConstants.burnout_thresh

        # Variable to store logger object
        self.logger = None

        # Variable to store tqdm progress bar
        self.progress_bar = None

        # Containers for keeping track of updates to cells 
        self._updated_cells = {}
        self._curr_updates = []

        # Containers for cells
        self._partially_burnt = []
        self._soaked = []
        self._burning_cells = []
        self._new_ignitions = []

        # Variables to keep track of agents in sim
        self._agent_list = []
        self._agents_added = False

        # Variables to keep track of current wind conditions
        self._curr_wind_idx = 0
        self._last_wind_update = 0

        super().__init__(sim_input)
        
        self._init_iteration()

    def iterate(self):
        """Step forward the fire simulation a single time-step

        """
        
        if self._iters == 0:
            self.wind_changed = True
            self._new_ignitions = self.starting_ignitions
            for cell, loc in self._new_ignitions:
                cell.directions, cell.distances, cell.end_pts = UtilFuncs.get_ign_parameters(loc, self.cell_size)
                cell._set_state(CellStates.FIRE)
                self._updated_cells[cell.id] = cell
        
        # Update wind if necessary
        self.wind_changed = self._update_wind()
        
        # Add any new ignitions to the current set of burning cells
        self._burning_cells.extend(self._new_ignitions)
        # Reset new ignitions
        self._new_ignitions = []

        # Set-up iteration
        if self._init_iteration():
            self._finished = True
            return

        for cell, loc in self._burning_cells:
            # Check if conditions have changed
            if self.wind_changed or not cell.has_steady_state: 
                # Reset the elapsed time counters
                cell.t_elapsed_min = 0

                # Update wind in cell
                cell._update_wind(self._curr_wind_idx)
                
                # Set previous rate of spreads to the most recent value
                if cell.r_t is not None:
                    cell.r_prev_list = cell.r_t
                else:
                    cell.r_prev_list = np.zeros(len(cell.directions))

                # Get steady state ROS (m/s) and I(kW/m), along each of cell's directions
                r_list, I_list = calc_propagation_in_cell(cell) 

                # Store steady-state values
                cell.r_ss = r_list
                cell.I_ss = I_list
                cell.has_steady_state = True

            # Set real time ROS and fireline intensity (vals stored in cell.r_t, cell.I_t)
            cell.set_real_time_vals()

            # Update extent of fire spread along each direction
            cell.fire_spread = cell.fire_spread + (cell.r_t * self._time_step)
            
            # TODO: Check if fireline intensity along any direction is high to initiate crown fire

            # Check where fire spread has reached edge of cell
            intersections = np.where(cell.fire_spread > cell.distances)[0]

            for idx in intersections:
                # TODO: This will be called unnecessarily a few times since end points and neighbors not matched well
                ignited_neighbors = self.ignite_neighbors(cell, cell.r_t[idx], cell.end_pts[idx])

                for neighbor in ignited_neighbors:
                    # Remove neighbor if its already ignited
                    if neighbor.id in cell.burnable_neighbors:
                        del cell.burnable_neighbors[neighbor.id]
                    
            # TODO: Likely need to check for if a cell has been burning for too long as well
            # TODO: Do we want to implement mass-loss approach from before?
            if not cell.burnable_neighbors:
                self._burning_cells.remove((cell, loc))
                cell._set_state(CellStates.BURNT)
                self.updated_cells[cell.id] = cell

            # Update time since conditions have changed for fire acceleration
            cell.t_elapsed_min += self.time_step / 60

        self.log_changes()

    def ignite_neighbors(self, cell: Cell, r_gamma: float, end_point: list) -> list:
        """_summary_

        Args:
            cell (Cell): _description_
            r_gamma (float): _description_
            end_point (list): _description_

        Returns:
            list: _description_
        """

        # Calculate how long fire will take to reach end point given local ROS 

        ignited_neighbors = []

        for pt in end_point:
            n_loc = pt[0]
            neighbor = self.get_neighbor_from_end_point(cell, pt)

            if neighbor:
                # Check that neighbor state is burnable
                if neighbor.state == CellStates.FUEL and neighbor.fuel_type.burnable:
                    # Make ignition calculation
                    r_ign = self.calc_ignition_ros(cell, neighbor, r_gamma)
                    
                    # Check that ignition ros meets threshold
                    if r_ign > 1e-3: # TODO: Think about using mass-loss calculation to do this or set R_min another way
                        self._new_ignitions.append((neighbor, n_loc))
                        neighbor.directions, neighbor.distances, neighbor.end_pts = UtilFuncs.get_ign_parameters(n_loc, self.cell_size)
                        neighbor._set_state(CellStates.FIRE)
                        neighbor._update_wind(self._curr_wind_idx)
                        neighbor.r_prev_list, _ = calc_propagation_in_cell(neighbor, r_ign) # TODO: does it make sense to use r_ign for r_h here

                        neighbor._set_fire_type(FireTypes.WILD) # TODO: Deprecate wild vs. prescribed fire 

                        self._updated_cells[neighbor.id] = neighbor

                        if neighbor.to_log_format() not in self._curr_updates:
                            self._curr_updates.append(neighbor.to_log_format())

                        ignited_neighbors.append(neighbor)
                else:
                    if neighbor.id in cell.burnable_neighbors:
                        del cell.burnable_neighbors[neighbor.id]

        return ignited_neighbors

    def calc_ignition_ros(self, cell: Cell, neighbor: Cell, r_gamma: float) -> float:
        """_summary_

        Args:
            cell (Cell): _description_
            neighbor (Cell): _description_
            r_gamma (float): _description_

        Returns:
            float: _description_
        """

        # Get the heat source in the direction of question by eliminating denominator
        heat_source = r_gamma * calc_heat_sink(cell.fuel_type, cell.dead_m) # TODO: make sure this computation is valid (I think it is)

        # Get the heat sink using the neighbors fuel and moisture content
        heat_sink = calc_heat_sink(neighbor.fuel_type, neighbor.dead_m) # TODO: need to implement updating fuel moisture in each cell
        
        # Calculate a ignition rate of spread
        r_ign = heat_source / heat_sink

        return r_ign

    def get_neighbor_from_end_point(self, cell: Cell, end_point: Tuple) -> Cell:
        """_summary_

        Args:
            cell (Cell): _description_
            end_point (Tuple): _description_

        Returns:
            Cell: _description_
        """

        neighbor_letter = end_point[1]
        # Get neighbor based on neighbor_letter
        if cell._row % 2 == 0:
            diff_to_letter_map = HexGridMath.even_neighbor_letters
            
        else:
            diff_to_letter_map = HexGridMath.odd_neighbor_letters

        dx, dy = diff_to_letter_map[neighbor_letter]

        row_n = int(cell.row + dy)
        col_n = int(cell.col + dx)

        if self._grid_height >= row_n >=0 and self._grid_width >= col_n >= 0:
            neighbor = self._cell_grid[row_n, col_n]

            if neighbor.id in cell.burnable_neighbors:
                return neighbor

        return None

    def log_changes(self):
        """Log the changes in state from the current iteration
        """
        self._curr_updates.extend(self._partially_burnt)
        self._curr_updates.extend(self._soaked)
        self._soaked = []
        
        if self.logger:
            self.logger.add_to_cache(self._curr_updates.copy(), self.curr_time_s)

            if self.agents_added:
                self.logger.add_to_agent_cache(self._get_agent_updates(), self.curr_time_s)

        self._iters += 1

    def _init_iteration(self) -> bool:
        """Set up the next iteration. Reset and update relevant data structures based on last 
        iteration. 

        :return: Boolean value representing if the simulation should be terminated.
        :rtype: bool
        """
        if self._iters == 0:
            self.progress_bar = tqdm(total=self._sim_duration/self.time_step,
                                     desc='Current sim ', position=0, leave=False)

        self._curr_updates.clear()

        # Update current time
        self._curr_time_s = self.time_step * self._iters
        self.progress_bar.update()

        self.w_vals = []
        self.fuels_at_ignition = []
        self.ignition_clocks = []
        
        if self._curr_time_s >= self._sim_duration or (self._iters != 0 and len(self._burning_cells) == 0):
            self.progress_bar.close()

            return True

        return False
    
    def _update_wind(self) -> bool:
        """_summary_

        Raises:
            ValueError: _description_

        Returns:
            bool: _description_
        """

        wind_changed = self.curr_time_s - self._last_wind_update >= self.wind_forecast_t_step

        if wind_changed:
            self._last_wind_update = self.curr_time_s
            self._curr_wind_idx += 1

            if self._curr_wind_idx >= len(self.wind_forecast):
                self._curr_wind_idx = 0
                raise ValueError("Wind forecast has no more entries!")
        
        return wind_changed

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
    def curr_updates(self) -> list:
        """List of cells updated during the most recent iteration.
        
        Cells are in their log format as generated by 
        :func:`~fire_simulator.cell.Cell.to_log_format()`
        """
        return self._curr_updates

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
