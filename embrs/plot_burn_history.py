
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import json
from datetime import datetime

from embrs.utilities.logger_schemas import CellLogEntry, AgentLogEntry, ActionsEntry, PredictionEntry
from embrs.utilities.fire_util import CellStates


log_folder_with = "/Users/rjdp3/Library/Mobile Documents/com~apple~CloudDocs/Documents/Research/Thesis/Proposal/Figures/Firefighting/fire_logs/ex_1/ex_1_with_actions"
log_folder_without = "/Users/rjdp3/Library/Mobile Documents/com~apple~CloudDocs/Documents/Research/Thesis/Proposal/Figures/Firefighting/fire_logs/ex_1/ex_1_no_actions"


# Get data frame for "with actions" case
init_path_with = f"{log_folder_with}/init_state.parquet"
cell_file_with = f"{log_folder_with}/run_0/cell_logs.parquet"

table = pq.read_table(init_path_with)
metadata = table.schema.metadata

if metadata and b"init_metadata" in metadata:
    meta_dict = json.loads(metadata[b"init_metadata"].decode("utf-8"))

cell_size=meta_dict["cell_size"],
area_m2_with = (3 * np.sqrt(3) * cell_size[0] ** 2) / 2

df_with = pd.read_parquet(cell_file_with)


# Get data frame for "without actions" case
init_path_without = f"{log_folder_without}/init_state.parquet"
cell_file_without = f"{log_folder_without}/run_0/cell_logs.parquet"

table = pq.read_table(init_path_without)
metadata = table.schema.metadata

if metadata and b"init_metadata" in metadata:
    meta_dict = json.loads(metadata[b"init_metadata"].decode("utf-8"))

cell_size=meta_dict["cell_size"],
area_m2_without = (3 * np.sqrt(3) * cell_size[0] ** 2) / 2

df_without = pd.read_parquet(cell_file_without)


FIRE = CellStates.FIRE
BURNT = CellStates.BURNT

def burning_counts_from_change_log(df: pd.DataFrame) -> pd.Series:
    """
    Build a time series (indexed by timestamp) of concurrent burning cells
    using an event-based (start +1, stop -1) cumulative sum.

    Expects columns: ['id', 'timestamp', 'state'].
    """
    # Keep one record per (id, timestamp) – last one if duplicates
    df_ev = (df[['id', 'timestamp', 'state']]
             .sort_values(['id', 'timestamp'])  # stable by time
             .drop_duplicates(subset=['id', 'timestamp'], keep='last'))

    # Transitions by cell
    g = df_ev.groupby('id', sort=False)

    prev_state = g['state'].shift()
    first_row  = g.cumcount() == 0

    # Start burning: entered FIRE, or first row already FIRE
    start_fire = (df_ev['state'] == FIRE) & ((prev_state != FIRE) | first_row)

    # Stop burning: transitioned from FIRE to BURNT
    end_fire = (df_ev['state'] == BURNT) & (prev_state == FIRE)

    # Event table: +1 on starts, -1 on stops
    starts = df_ev.loc[start_fire, ['timestamp']].assign(delta=1)
    stops  = df_ev.loc[end_fire,  ['timestamp']].assign(delta=-1)

    events = (pd.concat([starts, stops], ignore_index=True)
                .groupby('timestamp', sort=True)['delta']
                .sum()
                .sort_index())

    # Cumulative sum over timestamps => concurrent burning count
    burning_counts = events.cumsum()
    return burning_counts

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIRE  = CellStates.FIRE
BURNT = CellStates.BURNT

def burnt_counts_from_change_log(df: pd.DataFrame) -> pd.Series:
    """
    Series indexed by timestamp: cumulative number of cells that have transitioned to BURNT.
    Expects columns: ['id', 'timestamp', 'state'].
    """
    ev = (df[['id', 'timestamp', 'state']]
          .sort_values(['id', 'timestamp'])
          .drop_duplicates(subset=['id', 'timestamp'], keep='last'))

    g = ev.groupby('id', sort=False)
    prev_state = g['state'].shift()
    first_row  = g.cumcount() == 0

    # A BURNT "event" occurs the first time a cell becomes BURNT
    became_burnt = (ev['state'] == BURNT) & ((prev_state != BURNT) | first_row)

    events = (ev.loc[became_burnt, ['timestamp']]
                .assign(delta=1)
                .groupby('timestamp', sort=True)['delta']
                .sum()
                .sort_index())

    # Cumulative total burnt
    cum_burnt = events.cumsum()

    # Ensure baseline at t=0
    if 0.0 not in cum_burnt.index:
        cum_burnt = pd.concat([pd.Series([0], index=[0.0]), cum_burnt]).sort_index()

    return cum_burnt


def reindex_to_grid(series: pd.Series, duration_s: float, dt_s: float) -> pd.DataFrame:
    """
    Reindex a stepwise series (indexed by timestamps) to a regular grid [0, duration_s] with step dt_s.
    Forward-fills between events.
    """
    series = series.copy()
    series.index = series.index.astype(float)

    t_grid = np.arange(0.0, duration_s + 0.5*dt_s, dt_s, dtype=float)
    # Include all event times + grid, then ffill and pick grid points
    stepped = (series.reindex(np.union1d(series.index.values, t_grid))
                     .sort_index()
                     .ffill())
    out = stepped.reindex(t_grid).fillna(method='ffill').fillna(0)
    return pd.DataFrame({'timestamp': t_grid, 'value': out.to_numpy()})


# ----- Build both series on a common grid -----

# For "with actions" case:

# 1) Burning (concurrent) from your earlier function
burning_step_with = burning_counts_from_change_log(df_with)  # from previous message

# 2) Burnt (cumulative total) from first BURNT transitions
burnt_step_with = burnt_counts_from_change_log(df_with)

# Choose duration & step
duration_s = 128 * 3600
dt_s = 60.0  # 1 minute grid

burning_grid_with = reindex_to_grid(burning_step_with, duration_s, dt_s).rename(columns={'value': 'num_burning'})
burnt_grid_with   = reindex_to_grid(burnt_step_with,   duration_s, dt_s).rename(columns={'value': 'num_burnt'})

# Merge to one frame (aligned timestamps)
series_grid_with = burning_grid_with.merge(burnt_grid_with, on='timestamp', how='outer').sort_values('timestamp')
series_grid_with[['num_burning','num_burnt']] = series_grid_with[['num_burning','num_burnt']].ffill().fillna(0)
series_grid_with[['num_burning','num_burnt']] = series_grid_with[['num_burning','num_burnt']].astype(int)

series_grid_with['burning_area_km2'] = (series_grid_with['num_burning'] * area_m2_with) / 1e6
series_grid_with['burnt_area_km2']   = (series_grid_with['num_burnt']   * area_m2_with) / 1e6


# For "without actions" case:

# 1) Burning (concurrent) from your earlier function
burning_step_without = burning_counts_from_change_log(df_without)  # from previous message

# 2) Burnt (cumulative total) from first BURNT transitions
burnt_step_without = burnt_counts_from_change_log(df_without)

# Choose duration & step
duration_s = 128 * 3600
dt_s = 60.0  # 1 minute grid

burning_grid_without = reindex_to_grid(burning_step_without, duration_s, dt_s).rename(columns={'value': 'num_burning'})
burnt_grid_without   = reindex_to_grid(burnt_step_without,   duration_s, dt_s).rename(columns={'value': 'num_burnt'})

# Merge to one frame (aligned timestamps)
series_grid_without = burning_grid_without.merge(burnt_grid_without, on='timestamp', how='outer').sort_values('timestamp')
series_grid_without[['num_burning','num_burnt']] = series_grid_without[['num_burning','num_burnt']].ffill().fillna(0)
series_grid_without[['num_burning','num_burnt']] = series_grid_without[['num_burning','num_burnt']].astype(int)

series_grid_without['burning_area_km2'] = (series_grid_without['num_burning'] * area_m2_without) / 1e6
series_grid_without['burnt_area_km2']   = (series_grid_without['num_burnt']   * area_m2_without) / 1e6


import matplotlib.pyplot as plt

def plot_burning_burnt_areas(
    t_hours,
    burning_area_km2,
    burnt_area_km2,
    title="Burning and Burnt Areas Over Time",
    ylims_burning=None,   # (ymin, ymax)
    ylims_burnt=None,     # (ymin, ymax)
    color_burning='red',
    color_burnt='black',
    label_burning='Burning',
    label_burnt='Burnt',
    show=True
):
    """
    Plot burning and burnt area in stacked subplots with optional fixed y-axis limits.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1]}
    )

    # --- Burning (top) ---
    ax1.plot(t_hours, burning_area_km2, color=color_burning, linewidth=2, label=label_burning)
    ax1.set_ylabel(r'Burning Area (km$^2$)', color=color_burning)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    if ylims_burning is not None:
        ax1.set_ylim(*ylims_burning)

    # --- Burnt (bottom) ---
    ax2.plot(t_hours, burnt_area_km2, color=color_burnt, linewidth=2, label=label_burnt)
    ax2.set_ylabel(r'Burnt Area (km$^2$)', color=color_burnt)
    ax2.set_xlabel('Time (hours)')
    ax2.grid(True, alpha=0.3)
    if ylims_burnt is not None:
        ax2.set_ylim(*ylims_burnt)

    # --- Legends ---
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    plt.tight_layout()
    if show:
        plt.show()

    return fig, (ax1, ax2)

# Time (hours)
t_hours = series_grid_with['timestamp'] / 3600.0

# Get overall limits from both datasets
max_burning = max(series_grid_with['burning_area_km2'].max(),
                  series_grid_without['burning_area_km2'].max())

max_burnt = max(series_grid_with['burnt_area_km2'].max(),
                series_grid_without['burnt_area_km2'].max())

ylims_burning = (0, 1.05 * max_burning)
ylims_burnt   = (0, 1.05 * max_burnt)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def _align_to_union_time(t_h_1: np.ndarray, y1: pd.Series,
                         t_h_2: np.ndarray, y2: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align two time series to the union of their time axes with forward-fill.
    Returns t_union_hours, y1_aligned, y2_aligned (as numpy arrays).
    """
    # Build DataFrame indexed by the union of times
    t_union = np.union1d(t_h_1, t_h_2)
    df = pd.DataFrame(index=t_union)
    s1 = pd.Series(y1.to_numpy(), index=t_h_1)
    s2 = pd.Series(y2.to_numpy(), index=t_h_2)

    # Reindex and forward-fill
    df['y1'] = s1.reindex(df.index).ffill().fillna(0.0)
    df['y2'] = s2.reindex(df.index).ffill().fillna(0.0)

    return df.index.values, df['y1'].to_numpy(), df['y2'].to_numpy()

def plot_unified_stacked_with_vs_without(
    series_grid_with: pd.DataFrame,
    series_grid_without: pd.DataFrame,
    ylims_burning=None,  # (ymin, ymax)
    ylims_burnt=None,    # (ymin, ymax)
    title="Burning and Burnt Areas — With (solid) vs Without (dashed) Backburn"
):
    # Time axes (hours)
    t_with = (series_grid_with['timestamp'] / 3600.0).to_numpy()
    t_without = (series_grid_without['timestamp'] / 3600.0).to_numpy()

    # Align both datasets to a common unioned time axis (forward-filled)
    t_union_h, burn_with, burn_without = _align_to_union_time(
        t_with,   series_grid_with['burning_area_km2'],
        t_without, series_grid_without['burning_area_km2']
    )
    _, burnt_with, burnt_without = _align_to_union_time(
        t_with,   series_grid_with['burnt_area_km2'],
        t_without, series_grid_without['burnt_area_km2']
    )

    # Figure & axes
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1]}
    )
    fig.suptitle(title, fontsize=13)

    # ---- Burning (top) ----
    ax1.plot(t_union_h, burn_with,   color='red',   linewidth=2, linestyle='-',  label='With actions')
    ax1.plot(t_union_h, burn_without, color='blue',   linewidth=2, linestyle='-', label='Without actions')
    ax1.set_ylabel(r'Burning Area (km$^2$)', color='black')
    ax1.grid(True, alpha=0.3)
    if ylims_burning is not None:
        ax1.set_ylim(*ylims_burning)
    ax1.legend(loc='upper right', frameon=True)

    # ---- Burnt (bottom) ----
    ax2.plot(t_union_h, burnt_with,    color='red', linewidth=2, linestyle='-',  label='With actions')
    ax2.plot(t_union_h, burnt_without, color='blue', linewidth=2, linestyle='-', label='Without actions')
    ax2.set_ylabel(r'Burnt Area (km$^2$)', color='black')
    ax2.set_xlabel('Time (hours)')
    ax2.grid(True, alpha=0.3)
    if ylims_burnt is not None:
        ax2.set_ylim(*ylims_burnt)
    ax2.legend(loc='upper right', frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for suptitle
    plt.show()

# ---- Call it ----
plot_unified_stacked_with_vs_without(
    series_grid_with,
    series_grid_without,
    ylims_burning=ylims_burning,
    ylims_burnt=ylims_burnt,
    title="Burning and Burnt Areas vs. Time"
)
