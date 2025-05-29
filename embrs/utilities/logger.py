"""Logger for saving sim data and creating log files

.. autoclass:: Logger
    :members:
"""

import datetime
import pickle
import json
import sys
import os
import msgpack

from embrs.fire_simulator.fire import FireSim
from embrs.utilities.data_classes import SimParams

class Logger:
    """Logger class used to generate log files from sim data

    :param log_folder: String representing the path to the folder where the logger instance will
                       write log files.
    :type log_folder: str
    """
    def __init__(self, log_folder: str):
        """Constructor method that saves input data and creates a logger instance.
        """
        self.time = 0
        self.log_ctr = 0
        self.cache = {}
        self.agent_cache = {}
        self.messages = []
        self.agent_data = False
        self._log_folder = log_folder
        self._session_folder = self.generate_session_folder()
        os.makedirs(self.session_folder, exist_ok=True)

        self.data = None
        self.metadata = None

    def generate_session_folder(self) -> str:
        """Generates the path for the current sim's log files based on current datetime

        :return: Session folder path string
        :rtype: str
        """
        date_time_str = datetime.datetime.now().strftime('%d-%b-%Y-%H-%M-%S')
        return os.path.join(self._log_folder, f"log_{date_time_str}")

    def store_init_fire_obj(self, fire_obj: FireSim):
        """Store the initial state of the fire in a .pkl within the log folder.
        
        This is done to speed up the post-processing of log data significantly.

        :param fire_obj: :class:`~fire_simulator.fire.FireSim` object just after being initialized.
        :type fire_obj: :class:`~fire_simulator.fire.FireSim`
        """
        fire = {
            'cell_size': fire_obj.cell_size,
            'grid_width': fire_obj.grid_width,
            'grid_height': fire_obj.grid_height,
            'time_step': fire_obj.time_step,
            'cell_dict': fire_obj.cell_dict,
            'cell_grid': fire_obj.cell_grid,
            'fire_breaks': fire_obj.fire_breaks,
            'elevation_map': fire_obj.elevation_map,
            'coarse_elevation': fire_obj.coarse_elevation,
            'fuel_map': fire_obj.fuel_map,
            'wind_forecast': fire_obj.wind_forecast,
            'wind_forecast_time_step': fire_obj.wind_forecast_t_step
        }

        if fire_obj.roads is not None:
            fire['roads'] = fire_obj.roads

        filename = f"{self.session_folder}/init_fire_state.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(fire, f)

    def add_to_cache(self, updated_cells: list, time: int):
        """Add the list of updated cells from the previous iteration of the sim to the logger's
        cache to be stored in a log file later on.

        :param updated_cells: list of :class:`~fire_simulator.cell.Cell` objects that were changed
                              in the last iteration in their log format. See
                              :func:`~fire_simulator.cell.Cell.to_log_format()` for more info.
        :type updated_cells: list
        :param time: sim time (in seconds) when these cells were updated and stored.
        :type time: int
        """
        self.cache[time] = updated_cells

    def add_to_agent_cache(self, agents: list, time: int):
        """Add list of agents' state from the previous iteration of the sim to the logger's agent
        cache to be stored in a log file later on.

        :param agents: list of :class:`~agent_base.AgentBase` instances in their log format see the
                       :class:`~agent_base.AgentBase`:func:`~agent_base.AgentBase.to_log_format()`
                       function for more info.
        :type agents: list
        :param time: sim time (in seconds) when agents' were stored.
        :type time: int
        """
        self.agent_data = True
        self.agent_cache[time] = agents

    def _dump_cache(self):
        """Dump the logger's cache in a .msgpack file in the specified file location to be used
        later in post-processing.
        """

        if not self.cache:
            return

        run_folder = f"{self.session_folder}/run_{self.log_ctr}"
        os.makedirs(run_folder, exist_ok=True)

        file_name = f"{run_folder}/log.msgpack"

        if self.agent_data:
            agent_filename = f"{run_folder}/agents.msgpack"

        try:
            with open(file_name, 'wb') as f:
                msgpack.pack(self.cache, f)

            if self.agent_data:
                with open(agent_filename, 'wb') as f:
                    msgpack.pack(self.agent_cache, f)

            self.log_message("Log data saved successfully!")

        except Exception as e:
            self.log_message(f"Exception {e} occurred while attempting to save data.")

    # ~~~ STATUS LOGGING ~~~ #
    def store_metadata(self, sim_params: SimParams, fire: FireSim):
        """Store metadata for the status log describing the parameters used to run a simulation.
        
        Resulting metadata will be used in all status logs generated with a given simulation
        setup.

        :param params: dictionary containing input parameters for the simulation.
        :type params: dict
        :param fire: :class:`~fire_simulator.fire.FireSim` instance for which metadata is being
                     stored.
        :type fire: :class:`~fire_simulator.fire.FireSim`
        """
        # inputs
        cell_size = params['cell_size']
        time_step = params['t_step'] # seconds
        duration = params['sim_time']
        wind_file = params['wind']
        import_roads = fire.roads is not None

        # sim size
        rows = fire.shape[0]
        cols = fire.shape[1]
        total_cells = rows*cols
        width_m = fire.size[0]
        height_m = fire.size[1]

        # Load map file
        map_folder = params['input']
        foldername =  os.path.basename(map_folder)
        map_file_path = os.path.join(map_folder, foldername + ".json")
        with open(map_file_path, 'r') as f:
            map_data = json.load(f)

        user_code_path = params['user_path']
        user_code_class = params['class_name']

        metadata = {
                "inputs": {
                    "cell size": cell_size,
                    "time step (sec)": time_step,
                    "duration (sec)": duration,
                    "roads imported": import_roads
                },

                "sim size": {
                    "rows": rows,
                    "cols": cols,
                    "total cells": total_cells,
                    "width (m)": width_m,
                    "height (m)": height_m  
                },

                "wind forecast": {
                    "file location": wind_file
                },

                "imported_code": {
                    "imported module location": user_code_path,
                    "imported class name": user_code_class
                },

                "map": {
                    "map file location": map_file_path,
                    "map contents": map_data
                }
        }

        self.metadata = metadata

    def start_status_log(self):
        """Start a status log by loading the relevant metadata and creating a dictionary to store
        all relevant status data.
        """
        # loads metadata and creates a dictionary to store status data
        self.data = {}
        self.data["sim start"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.data["metadata"] = self.metadata
        self.messages = []
        self.log_message("Initialization successful, simulation started.")

    def log_message(self, message: str):
        """Log a custom message in the current status log.

        :param message: Message to be logged
        :type message: str
        """
        # log a time-stamped message
        timestamp = self.create_timestamp()
        status = f"[{timestamp}]:{message} "

        self.messages.append(status)

    def _write_status_log(self, fire: FireSim, on_interrupt: bool = False):
        """Writes the status log after a simulation has been completed.
        
        :param fire: :class:`~fire_simulator.fire.FireSim` instance to write status log for.
        :type fire: :class:`~fire_simulator.fire.FireSim`
        """
        self.data["messages"] = self.messages

        if fire is not None:
            # log results of simulation run
            burnt_cells = len(fire.burnt_cells)
            fire_extinguished = len(fire._burning_cells) == 0

            self.data["results"] = {
                "user interrupted": on_interrupt,
                "cells burnt": burnt_cells,
                "burnt area (m^2)": burnt_cells * fire.cell_dict[0].cell_area,
                "fire extinguished": fire_extinguished,
            }

            if not fire_extinguished:
                self.data["results"]["burning cells remaining"] = len(fire._burning_cells)
                self.data["results"]["burning area remaining (m^2)"] = len(fire._burning_cells) * fire.cell_dict[0].cell_area

        else:
            self.log_message("Unable to publish results due to early termination")
            self.data["results"] = None

        filename = f"{self.session_folder}/run_{self.log_ctr}/status_log.json"

        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=4)

        self.log_ctr += 1

    def finish(self, fire: FireSim, on_interrupt=False):
        """Finish logging all data, write to a .msgpack log file and a .json status log.

        :param fire: :class:`~fire_simulator.fire.FireSim` instance to write logs for.
        :type fire: :class:`~fire_simulator.fire.FireSim`
        :param on_interrupt: set to `True` when writing final log file, if `True` program will
                             terminate when finished logging, defaults to `False`.
        :type on_interrupt: bool, optional
        """
        self._dump_cache()
        self._write_status_log(fire, on_interrupt)

        if on_interrupt:
            sys.exit(0)

    @property
    def session_folder(self) -> str:
        """Path to the folder the logger is writing to
        """
        return self._session_folder

    def create_timestamp(self) -> str:
        """Function that creates a timestamp string based on the current datetime and returns it.

        :return: Timestamp string representing the current date and time.
        :rtype: str
        """
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
