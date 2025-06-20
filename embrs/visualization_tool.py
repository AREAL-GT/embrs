
from embrs.base_classes.base_visualizer import BaseVisualizer
from embrs.utilities.data_classes import PlaybackVisualizerParams, VisualizerInputs
from embrs.utilities.logger_schemas import CellLogEntry, AgentLogEntry, ActionsEntry, PredictionEntry
from embrs.utilities.file_io import VizFolderSelector

import matplotlib.animation as animation
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
import json
import sys
import os
import io
import zlib
import base64
from datetime import datetime
import numpy as np

class PlaybackVisualizer(BaseVisualizer):
    def __init__(self, params: PlaybackVisualizerParams):

        if params is None:
            sys.exit(0)
        
        self.config_params = params

        self.log_file = params.cell_file
        self.update_freq_s = params.freq
        self.show_legend = params.show_legend
        init_location = params.init_location
        self.has_agents = params.has_agents
        self.has_actions = params.has_actions
        self.has_predictions = params.has_predictions

        if self.has_agents:
            self.agent_file = params.agent_file

        if self.has_actions:
            self.action_file = params.action_file

        if self.has_predictions:
            self.prediction_file = params.prediction_file

        self.save_video = params.save_video
        
        run_folder = os.path.dirname(self.log_file)
        log_folder = os.path.dirname(run_folder)

        if self.save_video:
            self.video_name = params.video_name
            self.video_fps = params.video_fps
            self.save_folder = params.video_folder
            self.save_path = os.path.join(self.save_folder, f"{self.video_name}")
        
        self.show_viz = params.show_visualization

        if init_location:
            init_path = f"{log_folder}/init_state.parquet"
        else:
            init_path = f"{os.path.dirname(log_folder)}/init_state.parquet"
        
        input_params = self.get_input_params(init_path)

        super().__init__(input_params, render=self.show_viz)

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
            wind_forecast=decoded_wind,
            wind_resolution=meta_dict['wind_resolution'],
            wind_t_step=meta_dict['wind_time_step'],
            wind_xpad=meta_dict['wind_xpad'],
            wind_ypad=meta_dict['wind_ypad'],
            temp_forecast=meta_dict['temp_forecast'],
            rh_forecast=meta_dict['rh_forecast'],
            forecast_t_step=meta_dict['forecast_t_step'],
            elevation=meta_dict['elevation'],
            roads=meta_dict['roads'],
            fire_breaks=meta_dict['fire_breaks'],
            init_entries=self.get_init_entries(init_path),     
            scale_bar_km=self.config_params.scale_km,
            show_legend=self.show_legend,
            show_wind_cbar=self.config_params.show_wind_cbar,
            show_wind_field=self.config_params.show_wind_field,
            show_weather_data=self.config_params.show_weather_data,
            show_compass=self.config_params.show_compass,
            show_temp_in_F=self.config_params.show_temp_in_F
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

        writer = None
        if self.save_video:
            FFMpegWriter = animation.writers['ffmpeg']
            writer = FFMpegWriter(fps=self.video_fps, metadata=dict(artist='EMBRS'), bitrate=1800)
            writer.setup(self.fig, self.save_path, dpi=100)
            writer.grab_frame()

        df = pd.read_parquet(self.log_file)
        max_time = df["timestamp"].max()
        total_steps = int(np.ceil(max_time / self.update_freq_s))
        
        pbar = tqdm(total=total_steps, desc="Playback Progress:", unit="step")
        while not done:
            entries, agents, actions = self.get_entries_between(t, t+self.update_freq_s)

            if len(entries) == 0:
                done = True
                break
            
            t+=self.update_freq_s

            self.update_grid(t, entries, agents, actions)     

            if writer:
                writer.grab_frame()
            
            pbar.update(1)
        
        if writer:
            writer.finish()

        self.close()

    def get_entries_between(self, start_time: float, end_time: float):
        agents = []
        actions = []

        df = pd.read_parquet(self.log_file)

        # Filter entries where timestamp is in [start_time, end_time)
        filtered = df[(df["timestamp"] >= start_time) & (df["timestamp"] < end_time)]
        
        # Get only the most recent entry for each cell
        filtered = filtered.sort_values("timestamp").groupby("id", as_index=False).tail(1)

        entries = [CellLogEntry(**row.to_dict()) for _, row in filtered.iterrows()]

        if self.has_agents:
            df = pd.read_parquet(self.agent_file)
            filtered = df[(df["timestamp"] >= start_time) & (df["timestamp"] < end_time)]
            agents = [AgentLogEntry(**row.to_dict()) for _, row in filtered.iterrows()]
        
        if self.has_actions:
            df = pd.read_parquet(self.action_file)
            filtered = df[(df["timestamp"] >= start_time) & (df["timestamp"] < end_time)]
            actions = [ActionsEntry(**row.to_dict()) for _, row in filtered.iterrows()]

        if self.has_predictions:
            df = pd.read_parquet(self.prediction_file)

            # Convert back from string to original dict[int, Tuple[int, int]]
            df["prediction"] = df["prediction"].apply(
                lambda s: {int(k): tuple(v) for k, v in json.loads(s).items()})

            filtered = df[(df["timestamp"] >= start_time) & (df["timestamp"] < end_time)]
            predictions = [PredictionEntry(**row.to_dict()) for _, row in filtered.iterrows()]

            if predictions:
                output = predictions[-1].prediction
                self.visualize_prediction(output)

        return entries, agents, actions

def run(params: PlaybackVisualizerParams):
    viz = PlaybackVisualizer(params)
    viz.run_animation()

def main():
    folder_selector = VizFolderSelector(run)
    folder_selector.run()

if __name__ == "__main__":
    main()