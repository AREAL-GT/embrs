"""Example: load a single simulation log and plot a few quick-look metrics.

What you get:
1) Burning and burnt area vs time (km^2)
2) Mean wind speed vs time (m/s)
3) Cumulative actions by type vs time (if action_logs.parquet is present)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from embrs.utilities.fire_util import CellStates


# -----------------------------------------------------------------------------
# Paths to your simulation output (edit these)
# -----------------------------------------------------------------------------

log_folder = "path/to/your/simulation/logs"
run_number = 0

init_path = f"{log_folder}/init_state.parquet"
cell_file = f"{log_folder}/run_{run_number}/cell_logs.parquet"
action_file = Path(log_folder) / f"run_{run_number}" / "action_logs.parquet"


# -----------------------------------------------------------------------------
# Helpers to compute burning/burnt counts from the change log
# -----------------------------------------------------------------------------

FIRE = CellStates.FIRE
BURNT = CellStates.BURNT


def burning_counts_from_change_log(df: pd.DataFrame) -> pd.Series:
    """
    Build a time series (indexed by timestamp) of concurrent burning cells
    using an event-based (start +1, stop -1) cumulative sum.
    """
    df_ev = (
        df[["id", "timestamp", "state"]]
        .sort_values(["id", "timestamp"])
        .drop_duplicates(subset=["id", "timestamp"], keep="last")
    )

    g = df_ev.groupby("id", sort=False)
    prev_state = g["state"].shift()
    first_row = g.cumcount() == 0

    start_fire = (df_ev["state"] == FIRE) & ((prev_state != FIRE) | first_row)
    end_fire = (df_ev["state"] == BURNT) & (prev_state == FIRE)

    starts = df_ev.loc[start_fire, ["timestamp"]].assign(delta=1)
    stops = df_ev.loc[end_fire, ["timestamp"]].assign(delta=-1)

    events = (
        pd.concat([starts, stops], ignore_index=True)
        .groupby("timestamp", sort=True)["delta"]
        .sum()
        .sort_index()
    )

    return events.cumsum()


def burnt_counts_from_change_log(df: pd.DataFrame) -> pd.Series:
    """
    Series indexed by timestamp: cumulative number of cells that have transitioned to BURNT.
    """
    ev = (
        df[["id", "timestamp", "state"]]
        .sort_values(["id", "timestamp"])
        .drop_duplicates(subset=["id", "timestamp"], keep="last")
    )

    g = ev.groupby("id", sort=False)
    prev_state = g["state"].shift()
    first_row = g.cumcount() == 0

    became_burnt = (ev["state"] == BURNT) & ((prev_state != BURNT) | first_row)

    events = (
        ev.loc[became_burnt, ["timestamp"]]
        .assign(delta=1)
        .groupby("timestamp", sort=True)["delta"]
        .sum()
        .sort_index()
    )

    cum_burnt = events.cumsum()
    if 0.0 not in cum_burnt.index:
        cum_burnt = (
            pd.concat([pd.Series([0], index=[0.0]), cum_burnt])
            .sort_index()
        )

    return cum_burnt


def reindex_to_grid(series: pd.Series, duration_s: float, dt_s: float) -> pd.DataFrame:
    """
    Reindex a stepwise series (timestamp index) to a regular grid [0, duration_s] with step dt_s.
    Forward-fills between events.
    """
    series = series.copy()
    series.index = series.index.astype(float)

    t_grid = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s, dtype=float)
    stepped = (
        series.reindex(np.union1d(series.index.values, t_grid))
        .sort_index()
        .ffill()
    )
    out = stepped.reindex(t_grid).fillna(method="ffill").fillna(0)
    return pd.DataFrame({"timestamp": t_grid, "value": out.to_numpy()})


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

table = pq.read_table(init_path)  # init metadata contains cell size
metadata = table.schema.metadata or {}
init_meta = json.loads(metadata.get(b"init_metadata", b"{}").decode("utf-8"))

cell_size_val = init_meta.get("cell_size", 1.0)
if isinstance(cell_size_val, (list, tuple)):
    cell_size = float(cell_size_val[0])
else:
    cell_size = float(cell_size_val)

cell_area_m2 = (3 * np.sqrt(3) * cell_size**2) / 2

df = pd.read_parquet(cell_file)

df_actions: pd.DataFrame | None = None
if action_file.exists():
    df_actions = pd.read_parquet(action_file)


# -----------------------------------------------------------------------------
# Build series on a regular grid
# -----------------------------------------------------------------------------

burning_step = burning_counts_from_change_log(df)
burnt_step = burnt_counts_from_change_log(df)

duration_s = float(df["timestamp"].max())
if df_actions is not None:
    duration_s = max(duration_s, float(df_actions["timestamp"].max()))

dt_s = 60.0  # 1 minute grid

burning_grid = reindex_to_grid(burning_step, duration_s, dt_s).rename(
    columns={"value": "num_burning"}
)
burnt_grid = reindex_to_grid(burnt_step, duration_s, dt_s).rename(
    columns={"value": "num_burnt"}
)

series_grid = (
    burning_grid.merge(burnt_grid, on="timestamp", how="outer")
    .sort_values("timestamp")
)
series_grid[["num_burning", "num_burnt"]] = (
    series_grid[["num_burning", "num_burnt"]].ffill().fillna(0).astype(int)
)

series_grid["burning_area_km2"] = (series_grid["num_burning"] * cell_area_m2) / 1e6
series_grid["burnt_area_km2"] = (series_grid["num_burnt"] * cell_area_m2) / 1e6


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

# -- Burning and burnt area --
def plot_burning_burnt_areas(
    t_hours,
    burning_area_km2,
    burnt_area_km2,
    title="Burning and Burnt Areas Over Time",
    ylims_burning=None,
    ylims_burnt=None,
    color_burning="red",
    color_burnt="black",
    label_burning="Burning",
    label_burnt="Burnt",
    show=True,
):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    ax1.plot(t_hours, burning_area_km2, color=color_burning, linewidth=2, label=label_burning)
    ax1.set_ylabel(r"Burning Area (km$^2$)", color=color_burning)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    if ylims_burning is not None:
        ax1.set_ylim(*ylims_burning)

    ax2.plot(t_hours, burnt_area_km2, color=color_burnt, linewidth=2, label=label_burnt)
    ax2.set_ylabel(r"Burnt Area (km$^2$)", color=color_burnt)
    ax2.set_xlabel("Time (hours)")
    ax2.grid(True, alpha=0.3)
    if ylims_burnt is not None:
        ax2.set_ylim(*ylims_burnt)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    if show:
        plt.show()

    return fig, (ax1, ax2)


t_hours = series_grid["timestamp"] / 3600.0

ylims_burning = (0, 1.05 * series_grid["burning_area_km2"].max())
ylims_burnt = (0, 1.05 * series_grid["burnt_area_km2"].max())

plot_burning_burnt_areas(
    t_hours,
    series_grid["burning_area_km2"],
    series_grid["burnt_area_km2"],
    title=f"Burning and Burnt Areas vs. Time (run {run_number})",
    ylims_burning=ylims_burning,
    ylims_burnt=ylims_burnt,
)


# -- Mean wind speed over time (from cell logs) --
wind_speed_step = (
    df.groupby("timestamp")["wind_speed"]
    .mean()
    .sort_index()
)
wind_speed_grid = reindex_to_grid(wind_speed_step, duration_s, dt_s)
wind_speed_grid = wind_speed_grid.rename(columns={"value": "mean_wind_speed_mps"})

plt.figure(figsize=(9, 3))
plt.plot(
    wind_speed_grid["timestamp"] / 3600.0,
    wind_speed_grid["mean_wind_speed_mps"],
    color="steelblue",
    linewidth=2,
)
plt.title(f"Mean Wind Speed vs. Time (run {run_number})")
plt.xlabel("Time (hours)")
plt.ylabel("Wind speed (m/s)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# -- Cumulative actions by type (if action logs are present) --
def plot_actions_by_type(actions_df: pd.DataFrame, duration_s: float, dt_s: float) -> None:
    """
    Plot cumulative counts of logged actions over time, split by action_type.
    """
    counts = (
        actions_df.assign(delta=1)
        .groupby(["timestamp", "action_type"])["delta"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
        .cumsum()
    )

    # Build a grid for each action type so we can step-plot smoothly
    grid = pd.DataFrame({"timestamp": np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)})
    for col in counts.columns:
        filled = reindex_to_grid(counts[col], duration_s, dt_s).rename(
            columns={"value": col}
        )
        grid[col] = filled[col]

    plt.figure(figsize=(9, 4))
    for col in counts.columns:
        plt.step(
            grid["timestamp"] / 3600.0,
            grid[col],
            where="post",
            linewidth=2,
            label=col.replace("_", " ").title(),
        )

    plt.title(f"Cumulative Actions vs. Time (run {run_number})")
    plt.xlabel("Time (hours)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if df_actions is not None and not df_actions.empty:
    plot_actions_by_type(df_actions, duration_s=duration_s, dt_s=dt_s)
else:
    print("No action logs found; skipping action plot.")
