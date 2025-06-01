
from embrs.base_classes.base_visualizer import BaseVisualizer
from embrs.utilities.data_classes import PlaybackVisualizerParams, VisualizerInputs
from embrs.utilities.logger_schemas import CellLogEntry
from embrs.utilities.file_io import VizFolderSelector

import pyarrow.parquet as pq
import pandas as pd
import json
import sys
import os
import io
import zlib
import base64
from datetime import datetime
import numpy as np

# TODO: implement looping like old tool
# TODO: This should stop itself on its own
# TODO: Agent visualization
# TODO: Make wind colorbar optional

class PlaybackVisualizer(BaseVisualizer):
    def __init__(self, params: PlaybackVisualizerParams):

        if params is None:
            sys.exit(0)
        
        self.config_params = params

        self.log_file = params.file
        self.update_freq_s = params.freq
        self.show_legend = params.legend
        init_location = params.init_location
        self.has_agents = params.has_agents

        if self.has_agents:
            agent_file = params.agent_file

        run_folder = os.path.dirname(self.log_file)
        log_folder = os.path.dirname(run_folder)

        if init_location:
            init_path = f"{log_folder}/init_state.parquet"
        else:
            init_path = f"{os.path.dirname(log_folder)}/init_state.parquet"

        input_params = self.get_input_params(init_path)

        super().__init__(input_params)

    def get_input_params(self, init_path: str):
        table = pq.read_table(init_path)
        metadata = table.schema.metadata
        
        if metadata and b"init_metadata" in metadata:
            meta_dict = json.loads(metadata[b"init_metadata"].decode("utf-8"))
        else:
            meta_dict = {}

        decoded_wind = self.deserialize_array(meta_dict['wind_forecast'])

        params = VisualizerInputs(
            cell_size=meta_dict['cell_size'],
            sim_shape=(meta_dict['rows'], meta_dict['cols']),
            sim_size=(meta_dict['width_m'], meta_dict['height_m']),
            start_datetime=datetime.fromisoformat(meta_dict['start_datetime']),
            north_dir_deg=meta_dict['north_dir_deg'],
            scale_bar_km=self.config_params.scale_km,
            show_legend=self.show_legend,
            wind_forecast=decoded_wind,
            wind_resolution=meta_dict['wind_resolution'],
            wind_t_step=meta_dict['wind_time_step'],
            wind_xpad=meta_dict['wind_xpad'],
            wind_ypad=meta_dict['wind_ypad'],
            elevation=meta_dict['elevation'],
            roads=meta_dict['roads'],
            fire_breaks=meta_dict['fire_breaks'],
            init_entries=self.get_init_entries(init_path)     
        )

        return params

    def deserialize_array(self, encoded_str: str):
        compressed = base64.b64decode(encoded_str)
        decompressed = zlib.decompress(compressed)
        buffer = io.BytesIO(decompressed)
        return np.load(buffer, allow_pickle=False)

    def get_init_entries(self, init_pq_file: str) -> list[CellLogEntry]:
        df = pd.read_parquet(init_pq_file)
        entries = [CellLogEntry(**row._asdict()) for row in df.itertuples(index=False)]
        return entries
    
    def run_animation(self):
        t = 0
        done = False

        while not done:
            entries = self.get_entries_between(t, t+self.update_freq_s)

            if len(entries) == 0:
                done = True
                break
            
            t+=self.update_freq_s

            self.update_grid(t, entries)        

    def get_entries_between(self, start_time: float, end_time: float):

        df = pd.read_parquet(self.log_file)

        # Filter entries where timestamp is in [start_time, end_time)
        filtered = df[(df["timestamp"] >= start_time) & (df["timestamp"] < end_time)]
        
        entries = [CellLogEntry(**row.to_dict()) for _, row in filtered.iterrows()]
        
        return entries

def run(params: PlaybackVisualizerParams):
    viz = PlaybackVisualizer(params)
    viz.run_animation()

def main():
    folder_selector = VizFolderSelector(run)
    folder_selector.run()

if __name__ == "__main__":
    main()