import os
from embrs.utilities.logger_schemas import CellLogEntry, AgentLogEntry, ActionsEntry, PredictionEntry
from embrs.utilities.parquet_writer import ParquetWriter
import pyarrow as pa
import pyarrow.parquet as pq
from embrs.fire_simulator.fire import FireSim
from embrs.utilities.data_classes import SimParams
import datetime
import numpy as np
import json
import pandas as pd
import glob
import shutil
import sys
import io
import zlib
import base64

class Logger:
    def __init__(self, log_folder: str):
        
        self.log_ctr = 0

        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)
        
        self._session_folder = self.generate_session_folder()

        self.cell_writer = ParquetWriter(
            os.path.join(self._session_folder, "cell_logs"), schema=CellLogEntry
        )

        self.agent_writer = ParquetWriter(
            os.path.join(self._session_folder, "agent_logs"), schema=AgentLogEntry
        )

        self.action_writer = ParquetWriter(
            os.path.join(self._session_folder, "action_logs"), schema=ActionsEntry
        )

        self.prediction_writer = ParquetWriter(
            os.path.join(self._session_folder, "prediction_logs"), schema=PredictionEntry
        )

        self._cell_cache = []
        self._agent_cache = []
        self._action_cache = []
        self._prediction_cache = []
        
        self._status_log = {
            "sim_start": datetime.datetime.now().isoformat(),
            "messages": [],
            "latest_flush": None,
            "results": None
        }

    def cache_cell_updates(self, entries):
        self._cell_cache.extend(entries)

    def cache_agent_updates(self, entries):
        self._agent_cache.extend(entries)

    def cache_action_updates(self, entries):
        self._action_cache.extend(entries)

    def cache_prediction(self, entry):
        self._prediction_cache.append(entry)

    def flush(self):
        self.cell_writer.write_batch(self._cell_cache)
        self._cell_cache.clear() 

        self.agent_writer.write_batch(self._agent_cache)
        self._agent_cache.clear()

        self.action_writer.write_batch(self._action_cache)
        self._action_cache.clear()

        self.prediction_writer.write_batch(self._prediction_cache)
        self._prediction_cache.clear()

        self._status_log["latest_flush"] = datetime.datetime.now().isoformat()
        self._write_status_log()

    def write_results(self, fire: FireSim, on_interrupt: bool = False):
        if fire is not None:
            # Log results
            burnt_cells = len(fire._burnt_cells)
            fire_extinguised = len(fire._burning_cells) == 0

            self._status_log["results"] = {
                "user interrupted": on_interrupt,
                "cells burnt": burnt_cells,
                "burnt area (m^2)": burnt_cells * fire.cell_dict[0]._cell_area,
                "fire extinguished": fire_extinguised
            }

            if not fire_extinguised:
                self._status_log["results"]["burning cells remaining"] = len(fire._burning_cells)
                self._status_log["results"]["burning area remaining (m^2)"] = len(fire._burning_cells) * fire.cell_dict[0]._cell_area


    def finish(self, fire: FireSim, on_interrupt: bool = False):  
        self.write_results(fire)
        self.flush()

        cell_log_path = os.path.join(self._session_folder, "cell_logs")
        agent_log_path = os.path.join(self._session_folder, "agent_logs")
        action_log_path = os.path.join(self._session_folder, "action_logs")
        prediction_log_path = os.path.join(self._session_folder, "prediction_logs")

        self._merge_parquet_files(
            cell_log_path,
            os.path.join(self._run_folder, "cell_logs.parquet")
        )

        self._merge_parquet_files(
            agent_log_path,
            os.path.join(self._run_folder, "agent_logs.parquet")
        )

        self._merge_parquet_files(
            action_log_path,
            os.path.join(self._run_folder, "action_logs.parquet")
        )

        self._merge_parquet_files(
            prediction_log_path,
            os.path.join(self._run_folder, "prediction_logs.parquet")
        )

        # Delete the temporary folders after merging
        if os.path.exists(cell_log_path):
            shutil.rmtree(cell_log_path)
        
        if os.path.exists(agent_log_path):
            shutil.rmtree(agent_log_path)

        if os.path.exists(action_log_path):
            shutil.rmtree(action_log_path)

        if os.path.exists(prediction_log_path):
            shutil.rmtree(prediction_log_path)

        if on_interrupt:
            sys.exit(0)

    def _merge_parquet_files(self, folder_path: str, output_file: str):

        parquet_files = sorted(glob.glob(os.path.join(folder_path, "part-*.parquet")))

        if not parquet_files:
            print(f"No parquet files found in {folder_path}")
            return
        
        dfs = [pd.read_parquet(f) for f in parquet_files]
        combined_df = pd.concat(dfs, ignore_index=True)

        table = pa.Table.from_pandas(combined_df)
        pq.write_table(table, output_file, compression='snappy')

    def generate_session_folder(self) -> str:
        """Generates the path for the current sim's log files based on current datetime

        :return: Session folder path string
        :rtype: str
        """
        date_time_str = datetime.datetime.now().strftime('%d-%b-%Y-%H-%M-%S')
        return os.path.join(self.log_folder, f"log_{date_time_str}")
    
    def start_new_run(self):
        self._run_folder = f"{self._session_folder}/run_{self.log_ctr}"
        os.makedirs(self._run_folder, exist_ok=True)

        self.log_ctr += 1

    def log_metadata(self, sim_params: SimParams, fire: FireSim):
        # inputs
        cell_size = sim_params.cell_size
        time_step = sim_params.t_step_s
        duration = sim_params.duration_s
        import_roads = sim_params.map_params.import_roads
        init_mf = sim_params.init_mf
        spotting = sim_params.model_spotting

        if spotting:
            canopy_species = sim_params.canopy_species
            dbh_cm = sim_params.dbh_cm
            min_spot_dist = sim_params.min_spot_dist
            spot_delay_s = sim_params.spot_delay_s

        # Weather inputs
        weather_input_type = sim_params.weather_input.input_type
        weather_file = sim_params.weather_input.file
        wind_mesh_resolution = sim_params.weather_input.mesh_resolution
        start_datetime = sim_params.weather_input.start_datetime
        end_datetime = sim_params.weather_input.end_datetime

        # sim size
        rows = fire.shape[0]
        cols = fire.shape[1]
        total_cells = rows*cols
        width_m = fire.size[0]
        height_m = fire.size[1]

        # load map file
        map_folder = sim_params.map_params.folder
        foldername = os.path.basename(map_folder)
        map_file_path = os.path.join(map_folder, foldername + ".json")
        with open(map_file_path, 'r') as f:
            map_data = json.load(f)

        # imported code
        user_path = sim_params.user_path
        user_class = sim_params.user_class

        metadata = {
            "inputs": {
                "cell size": cell_size,
                "time step (sec)": time_step,
                "duration (sec)": duration,
                "init dead moisture": init_mf,
                "roads imported": import_roads,
                "spotting modeled": spotting
            },

            "sim size": {
                "rows": rows,
                "cols": cols,
                "total cells": total_cells,
                "width (m)": width_m,
                "height (m)": height_m
            },

            "weather": {
                "input type": weather_input_type,
                "weather file": weather_file,
                "wind mesh resolution": wind_mesh_resolution,
                "start datetime": start_datetime,
                "end datetime": end_datetime
            },

            "imported code": {
                "imported module location": user_path,
                "imported class name": user_class
            },

            "map": {
                "map file": map_file_path,
                "map contents": map_data
            }
        }

        if spotting:

            metadata["spotting inputs"] = {
                "canopy species": canopy_species,
                "dbh (cm)": dbh_cm,
                "min. spot dist": min_spot_dist,
                "spot delay (s)": spot_delay_s
            }


        safe_dict = make_json_serializable(metadata)

        metadata_path = os.path.join(self._session_folder, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(safe_dict, f, indent=2)

    def save_initial_state(self, fire_obj: FireSim):

        init_cells = [cell.to_log_entry(0) for cell in fire_obj.cell_dict.values()]

        df = pd.DataFrame(init_cells)

        encoded_wind = serialize_array(fire_obj.wind_forecast)

        dict = {
            'cell_size': fire_obj.cell_size,
            'cols': fire_obj.shape[1],
            'rows': fire_obj.shape[0],
            'width_m': fire_obj.size[0],
            'height_m': fire_obj.size[1],
            'time_step': fire_obj.time_step,
            'fire_breaks': fire_obj.fire_breaks,
            'roads': fire_obj.roads,
            'wind_time_step': fire_obj.weather_t_step,
            'wind_xpad': fire_obj.wind_xpad,
            'wind_ypad': fire_obj.wind_ypad,
            'wind_forecast': encoded_wind,
            'wind_resolution': fire_obj._wind_res,
            'temp_forecast': [entry.temp for entry in fire_obj._weather_stream.stream],
            'rh_forecast': [entry.rel_humidity for entry in fire_obj._weather_stream.stream],
            'forecast_t_step': fire_obj.weather_t_step,
            'elevation': fire_obj.coarse_elevation,
            'start_datetime': fire_obj._start_datetime,
            'north_dir_deg': fire_obj._north_dir_deg
        }

        table = pa.Table.from_pandas(df)
        meta_json = json.dumps(make_json_serializable(dict))
        table = table.replace_schema_metadata({"init_metadata": meta_json})

        pq.write_table(table, os.path.join(self._session_folder, "init_state.parquet"), compression='snappy')


    def log_message(self, message: str):
        timestamp = datetime.datetime.now().isoformat()
        entry = f"[{timestamp}]: {message}"
        self._status_log["messages"].append(entry)

    def _write_status_log(self):
        status_path = os.path.join(self._run_folder, "status_log.json")
        with open(status_path, 'w') as f:
            json.dump(self._status_log, f, indent = 2)

def serialize_array(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    compressed = zlib.compress(buffer.getvalue())
    encoded = base64.b64encode(compressed).decode('utf-8')
    return encoded
        
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj
