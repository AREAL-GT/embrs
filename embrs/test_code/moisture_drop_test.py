from embrs.base_classes.base_fire import BaseFireSim
from embrs.utilities.data_classes import SimParams, WeatherParams
from embrs.models.rothermel import get_characteristic_moistures

import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import argparse
import configparser
import os
import pickle
from datetime import datetime, timedelta
import json
import numpy as np


class MoistureDropTest(BaseFireSim):

    def __init__(self, sim_params: SimParams, burnt_region=None):
        sim_params.map_params.scenario_data.initial_ign = []
        super().__init__(sim_params)
        self.fig, self.ax = plt.subplots()
        self.cbar = None  # Colorbar for moisture map

    def run_moisture_test(self):
        while self.curr_time_s < self._sim_duration:

            if self.curr_time_s == 3600 * 3: 
                self.perform_test_drop()

            weather_changed = self._update_weather()
            
            if weather_changed:
                self._update_moisture()
                self._display_fuel_moisture_map()

            self._curr_time_s += self.weather_t_step

    def _update_moisture(self):
        for i in range(self._shape[1]):
            for j in range(self._shape[0]):
                cell = self._cell_grid[j, i]
                if cell.fuel.burnable:
                    cell._update_moisture(self._curr_weather_idx, self._weather_stream)

        print(f"Done updating at: {self.curr_time_s:.0f}s")

    def _display_fuel_moisture_map(self):
        x_vals, y_vals, moisture_vals = [], [], []

        for i in range(self._shape[1]):
            for j in range(self._shape[0]):
                cell = self._cell_grid[j, i]
                if cell.fuel.burnable:
                    dead_m, _ = get_characteristic_moistures(cell.fuel, cell.fmois)
                    x_vals.append(cell.x_pos)
                    y_vals.append(cell.y_pos)
                    moisture_vals.append(dead_m)

        self.ax.clear()
        sc = self.ax.scatter(x_vals, y_vals, c=moisture_vals, cmap='jet', s=15, vmin=0, vmax=0.35)
        self.ax.set_title(f"Fuel Moisture at t = {self.curr_time_s:.0f}s")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        
        if self.cbar is None:
            self.cbar = self.fig.colorbar(sc, ax=self.ax, label='Dead Fuel Moisture')


        self.fig.canvas.draw()
        plt.pause(0.1)  # Pause to allow the plot to update

    def perform_test_drop(self):
        center_x = self._size[0] // 2
        center_y = self._size[1] // 2
        radius = 2000

        angles = np.linspace(0, 2 * np.pi, 100)
        xs = center_x + radius * np.cos(angles)
        ys = center_y + radius * np.sin(angles)

        for x, y in zip(xs, ys):
            self.water_drop_at_xy_as_rain(x, y, 1)

        self._display_fuel_moisture_map()

def load_sim_params(cfg_path: str) -> SimParams:
    config = configparser.ConfigParser()
    config.read(cfg_path)

    map_params = None
    if "Map" in config:
        folder = config["Map"]["folder"]
        pkl_path = os.path.join(folder, "map_params.pkl")

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                map_params = pickle.load(f)
        else:
            raise FileNotFoundError(f"MapParams file not found in '{folder}'.")

    weather_input_type = config["Weather"].get("input_type", None)
    mesh_resolution = config["Weather"].getint("mesh_resolution", 250)
    start_iso_datetime = config["Weather"].get("start_datetime", None)

    if start_iso_datetime is None:
        raise ValueError("Missing start_datetime in config.")

    start_datetime = datetime.fromisoformat(start_iso_datetime)
    weather_file = config["Weather"].get("file", None)

    if weather_input_type == "File":
        if weather_file is None:
            raise ValueError("Weather file required for 'File' input_type.")
        
        with open(weather_file, "rb") as f:
            weather = json.load(f)

        weather_time_step_hr = weather['time_step_min'] / 60
        weather_len = len(weather['weather_entries']["wind_speed"])
        for key in weather['weather_entries']:
            if len(weather['weather_entries'][key]) != weather_len:
                raise ValueError("Mismatched weather entry lengths.")

        duration_s = weather_time_step_hr * weather_len * 3600
        end_datetime = start_datetime + timedelta(seconds=duration_s)

    elif weather_input_type == "OpenMeteo":
        end_iso_datetime = config["Weather"].get("end_datetime", None)
        if end_iso_datetime:
            end_datetime = datetime.fromisoformat(end_iso_datetime)
            duration_s = (end_datetime - start_datetime).total_seconds()
        else:
            duration_s = config["Simulation"].getint("duration_s", None)
            end_datetime = start_datetime + timedelta(seconds=duration_s)

    weather_params = WeatherParams(
        input_type=weather_input_type,
        file=weather_file,
        mesh_resolution=mesh_resolution,
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )

    init_mf = config["Weather"].getfloat("init_mf", 0.08)
    write_logs = config["Simulation"].getboolean("write_logs", False)
    log_folder = config["Simulation"].get("log_folder", None)

    t_step_s = config["Simulation"].getint("t_step_s", None)
    cell_size = config["Simulation"].getint("cell_size_m", None)

    model_spotting = config["Simulation"].getboolean("model_spotting", False)
    canopy_species = config["Simulation"].getint("canopy_species", 5)
    dbh_cm = config["Simulation"].getfloat("dbh_cm", 20)
    spot_ign_prob = config["Simulation"].getfloat("spot_ign_prob", 0.05)
    min_spot_dist_m = config["Simulation"].getfloat("min_spot_dist_m", 50)
    spot_delay_s = config["Simulation"].getint("spot_delay_s", 30)

    user_class = config["Simulation"].get("user_class", "")
    user_path = config["Simulation"].get("user_path", "")

    sim_params = SimParams(
        map_params=map_params,
        log_folder=log_folder,
        weather_input=weather_params,
        t_step_s=t_step_s,
        cell_size=cell_size,
        init_mf=init_mf,
        model_spotting=model_spotting,
        canopy_species=canopy_species,
        dbh_cm=dbh_cm,
        spot_ign_prob=spot_ign_prob,
        min_spot_dist=min_spot_dist_m,
        spot_delay_s=spot_delay_s,
        duration_s=duration_s,
        visualize=config["Simulation"].getboolean("visualize", False),
        num_runs=config["Simulation"].getint("num_runs", 1),
        user_path=user_path,
        user_class=user_class,
        write_logs=write_logs
    )

    return sim_params


def main():
    parser = argparse.ArgumentParser(description="Run moisture drop fire simulation")
    parser.add_argument("--config", type=str, help="Path to .cfg file")

    args = parser.parse_args()

    if not args.config:
        raise ValueError("No configuration file provided. Use --config to specify a .cfg file.")

    print(f"Loading simulation params from {args.config}...")
    sim_params = load_sim_params(args.config)

    drop_test = MoistureDropTest(sim_params)
    drop_test.run_moisture_test()


if __name__ == "__main__":
    main()
