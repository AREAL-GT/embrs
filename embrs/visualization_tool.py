from embrs.base_classes.base_visualizer import BaseVisualizer
from embrs.utilities.data_classes import PlaybackVisualizerParams, VisualizerInputs
from embrs.utilities.logger_schemas import CellLogEntry, AgentLogEntry, ActionsEntry, PredictionEntry
from embrs.utilities.file_io import VizFolderSelector
from shapely.geometry import Polygon, LineString
import matplotlib as mpl
import matplotlib.pyplot as plt
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
from dataclasses import fields as dc_fields, MISSING


class PlaybackVisualizer(BaseVisualizer):
    def __init__(self, params: PlaybackVisualizerParams):

        if params is None:
            sys.exit(0)

        self.config_params = params

        self.log_file = params.cell_file
        self.update_freq_s = float(params.freq)
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

        # Build all frame caches BEFORE calling BaseVisualizer (so we know max_frame etc.)
        self._build_frame_caches()

        super().__init__(input_params, render=self.show_viz)

    # ---------- Fast pre-load & conversion ----------

    def _dc_field_names(self, cls):
        return [f.name for f in dc_fields(cls)]

    def _dc_default_map(self, cls):
        d = {}
        for f in dc_fields(cls):
            if f.default is not MISSING:
                d[f.name] = f.default
            elif getattr(f, "default_factory", MISSING) is not MISSING:  # type: ignore[attr-defined]
                d[f.name] = f.default_factory()  # type: ignore[attr-defined]
            else:
                d[f.name] = None
        return d

    def _read_parquet(self, path, columns=None, sort_by=None):
        """Read parquet with optional column projection & stable sort."""
        if columns is None:
            df = pd.read_parquet(path)
        else:
            # Intersect requested columns with available ones
            schema = pq.read_schema(path)
            names = set(schema.names)
            keep = [c for c in columns if c in names]
            df = pd.read_parquet(path, columns=(keep if keep else None))
        if sort_by:
            keep = [c for c in sort_by if c in df.columns]
            if keep:
                df = df.sort_values(keep, kind="mergesort")
        return df

    def _to_entries(self, df: pd.DataFrame, cls):
        """Convert DataFrame rows to dataclass instances, selecting exactly the fields and filling defaults."""
        if df is None or len(df) == 0:
            return []
        names = self._dc_field_names(cls)
        defaults = self._dc_default_map(cls)
        # Make sure all fields exist
        for c in names:
            if c not in df.columns:
                df[c] = defaults.get(c, None)
        # Select exactly the dataclass fields (drop extras like 'frame')
        view = df[names]
        return [cls(**row._asdict()) for row in view.itertuples(index=False)]

    def _bin_by_frame(self, df: pd.DataFrame, id_col: str | None = "id"):
        """Compute frame index and keep last row per (frame,id)."""
        if "timestamp" not in df.columns:
            return pd.DataFrame(columns=df.columns.tolist() + ["frame"])
        # Clean timestamps
        df = df[pd.notnull(df["timestamp"])]
        df = df[df["timestamp"] >= 0]
        # Compute frame indices
        df = df.copy()
        df["frame"] = np.floor_divide(df["timestamp"].to_numpy(float), self.update_freq_s).astype(np.int64)
        # Keep last per (frame, id) if id exists, else last per frame
        if id_col and (id_col in df.columns):
            df = df.sort_values([id_col, "timestamp"], kind="mergesort")
            df = df.groupby(["frame", id_col], sort=False, as_index=False).tail(1)
        else:
            df = df.sort_values(["frame", "timestamp"], kind="mergesort")
            df = df.groupby(["frame"], sort=False, as_index=False).tail(1)
        return df

    def _build_frame_caches(self):
        """Load once, bin into frames once, convert to dataclasses once."""
        # ---- Cells (driver) ----
        # Read ALL columns to satisfy CellLogEntry (unknown fields) + id/timestamp
        cell_df = self._read_parquet(self.log_file)
        cell_df = self._bin_by_frame(cell_df, id_col="id")
        self.max_frame = int(cell_df["frame"].max()) if not cell_df.empty else -1

        self.cells_by_frame = {}
        for f, g in cell_df.groupby("frame", sort=True):
            self.cells_by_frame[int(f)] = self._to_entries(g, CellLogEntry)

        # ---- Agents (optional) ----
        self.agents_by_frame = {}
        if self.has_agents:
            ag_df = self._read_parquet(self.agent_file)
            ag_df = self._bin_by_frame(ag_df, id_col="id")
            for f, g in ag_df.groupby("frame", sort=True):
                self.agents_by_frame[int(f)] = self._to_entries(g, AgentLogEntry)
            if self.max_frame < 0 and not ag_df.empty:
                self.max_frame = int(ag_df["frame"].max())

        # ---- Actions (optional) ----
        self.actions_by_frame = {}
        if self.has_actions:
            act_df = self._read_parquet(self.action_file)
            act_df = self._bin_by_frame(act_df, id_col=None)  # actions may not have 'id'
            for f, g in act_df.groupby("frame", sort=True):
                self.actions_by_frame[int(f)] = self._to_entries(g, ActionsEntry)
            if self.max_frame < 0 and not act_df.empty:
                self.max_frame = int(act_df["frame"].max())

        # ---- Predictions (optional) ----
        self.prediction_by_frame = {}
        if self.has_predictions:
            pr_df = self._read_parquet(self.prediction_file, columns=["timestamp", "prediction"], sort_by=["timestamp"])
            if "timestamp" in pr_df.columns:
                pr_df = pr_df.copy()
                pr_df["frame"] = np.floor_divide(pr_df["timestamp"].to_numpy(float), self.update_freq_s).astype(np.int64)
                # Keep last per frame
                pr_df = pr_df.groupby("frame", sort=False, as_index=False).tail(1)
                # Decode JSON once
                if "prediction" in pr_df.columns:
                    pr_df["prediction"] = pr_df["prediction"].map(
                        lambda s: {int(k): tuple(v) for k, v in json.loads(s).items()} if isinstance(s, str) else s
                    )
                for f, row in pr_df.set_index("frame").iterrows():
                    self.prediction_by_frame[int(f)] = row.get("prediction", None)
                if self.max_frame < 0 and not pr_df.empty:
                    self.max_frame = int(pr_df["frame"].max())

        # Final guard
        if self.max_frame < 0:
            self.max_frame = 0

    # ---------- BaseVisualizer inputs / init_state ----------

    def get_input_params(self, init_path: str):
        table = pq.read_table(init_path)
        metadata = table.schema.metadata

        if metadata and b"init_metadata" in metadata:
            meta_dict = json.loads(metadata[b"init_metadata"].decode("utf-8"))
        else:
            meta_dict = {}

        decoded_wind = self.deserialize_array(meta_dict["wind_forecast"])

        fire_breaks = []
        for fb in meta_dict["fire_breaks"]:
            ls_dict = fb[0]
            fire_breaks.append((LineString(ls_dict["coordinates"]), fb[1], fb[2]))

        params = VisualizerInputs(
            cell_size=meta_dict["cell_size"],
            sim_shape=(meta_dict["rows"], meta_dict["cols"]),
            sim_size=(meta_dict["width_m"], meta_dict["height_m"]),
            start_datetime=datetime.fromisoformat(meta_dict["start_datetime"]),
            north_dir_deg=meta_dict["north_dir_deg"],
            wind_forecast=decoded_wind,
            wind_resolution=meta_dict["wind_resolution"],
            wind_t_step=meta_dict["wind_time_step"],
            wind_xpad=meta_dict["wind_xpad"],
            wind_ypad=meta_dict["wind_ypad"],
            temp_forecast=meta_dict["temp_forecast"],
            rh_forecast=meta_dict["rh_forecast"],
            forecast_t_step=meta_dict["forecast_t_step"],
            elevation=meta_dict["elevation"],
            roads=meta_dict["roads"],
            fire_breaks=fire_breaks,
            init_entries=self.get_init_entries(init_path),
            scale_bar_km=self.config_params.scale_km,
            show_legend=self.show_legend,
            show_wind_cbar=self.config_params.show_wind_cbar,
            show_wind_field=self.config_params.show_wind_field,
            show_weather_data=self.config_params.show_weather_data,
            show_compass=self.config_params.show_compass,
            show_temp_in_F=self.config_params.show_temp_in_F,
        )

        return params

    def deserialize_array(self, encoded_str: str):
        compressed = base64.b64decode(encoded_str)
        decompressed = zlib.decompress(compressed)
        buffer = io.BytesIO(decompressed)
        return np.load(buffer, allow_pickle=False)

    def get_init_entries(self, init_pq_file: str) -> list[CellLogEntry]:
        df = pd.read_parquet(init_pq_file)
        # Convert to dataclasses (in case schema contains extras)
        names = self._dc_field_names(CellLogEntry)
        for c in names:
            if c not in df.columns:
                df[c] = None
        df = df[names]
        entries = [CellLogEntry(**row._asdict()) for row in df.itertuples(index=False)]
        return entries

    # ---------- Fast animation loop (no parquet reads, no heavy work) ----------

    def run_animation(self):
        if self.save_video:
            os.makedirs(self.save_folder, exist_ok=True)

        writer = None
        if self.save_video:
            FFMpegWriter = animation.writers["ffmpeg"]
            writer = FFMpegWriter(
                fps=self.video_fps, metadata=dict(artist="EMBRS"), bitrate=1800
            )
            # Slightly lower DPI can speed up encoding notably
            writer.setup(self.fig, self.save_path, dpi=80)

        total_frames = self.max_frame + 1
        pbar = tqdm(total=total_frames, desc="Playback Progress:", unit="frame")

        for f in range(total_frames):
            t = f * self.update_freq_s

            cells = self.cells_by_frame.get(f, [])
            agents = self.agents_by_frame.get(f, []) if self.has_agents else []
            actions = self.actions_by_frame.get(f, []) if self.has_actions else []
            pred = self.prediction_by_frame.get(f, None) if self.has_predictions else None

            if pred is not None:
                self.visualize_prediction(pred)

            # IMPORTANT: BaseVisualizer.update_grid expects dataclass lists.
            self.update_grid(t, cells, agents, actions)

            if writer:
                writer.grab_frame()

            pbar.update(1)

        if writer:
            writer.finish()

        self.close()

    # ---------- One-off arrival plot (kept out of the loop) ----------

    def extract_sim_hexagons_and_times(self):
        """
        Extract hexagon geometries and arrival times from sim logs.
        Returns:
            hexagons (list of Polygon)
            arrival_times (list of float)
        """
        df = pd.read_parquet(self.log_file, columns=["x", "y", "arrival_time"])
        df = df[(df["arrival_time"].notnull()) & (df["arrival_time"] >= 0)]
        df = df.sort_values("arrival_time").drop_duplicates(subset=["x", "y"], keep="last")

        df["x_global"] = df["x"]
        df["y_global"] = df["y"]

        hexagons = [
            create_hexagon(row["x_global"], row["y_global"], self.cell_size)
            for _, row in df.iterrows()
        ]

        arrival_times = df["arrival_time"].tolist()
        return hexagons, arrival_times

    def display_arrival(self):
        hexes, times = self.extract_sim_hexagons_and_times()

        pts = [poly.centroid for poly in hexes]
        # Convert to minutes if your arrival_time is in seconds
        times = np.asarray(times, dtype=float) / 60.0

        cmap = mpl.cm.get_cmap("viridis")
        norm = plt.Normalize(np.nanmin(times), np.nanmax(times))

        sc = self.h_ax.scatter([pt.x for pt in pts], [pt.y for pt in pts], c=times, cmap=cmap, norm=norm, s=5)
        self.fig.colorbar(sc, ax=self.h_ax, label="Arrival Time (mins)", shrink=0.75)
        self.fig.canvas.draw()


# ---------- helpers outside the class ----------

def create_hexagon(center_x, center_y, radius):
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    x_vertices = center_x + radius * np.cos(angles)
    y_vertices = center_y + radius * np.sin(angles)
    return Polygon(zip(x_vertices, y_vertices))


def run(params: PlaybackVisualizerParams):
    viz = PlaybackVisualizer(params)
    viz.run_animation()


def display(params: PlaybackVisualizerParams):
    viz = PlaybackVisualizer(params)
    viz.display_arrival()


def main():
    folder_selector = VizFolderSelector(run, display)
    folder_selector.run()


if __name__ == "__main__":
    main()
