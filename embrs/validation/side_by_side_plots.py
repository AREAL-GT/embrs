import pickle
import rasterio
from rasterio.enums import Resampling
from embrs.map_generator import resample_raster
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


# Paths
map_params_path = f"/Users/rui/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Research/embrs_inputs/embrs_maps/cougar_creek/map_params.pkl"
flammap_path = "/Users/rui/Documents/Research/Code/embrs/embrs/validation/cougar_1/fmp_cougar.tif"
cell_logs_path = "/Users/rui/Documents/Research/Code/embrs/embrs/validation/cougar_1/cougar_1/run_0/cell_logs.parquet"


sim_time_hr = 80

# ── Load Map Transform ────────────────────────────────────────────────
with open(map_params_path, "rb") as f:
    map_params = pickle.load(f)

map_transform = map_params.lcp_data.transform  # Affine transform from sim space to UTM (EPSG:26910)
sim_height = map_params.lcp_data.rows

# ── Load FlamMap Raster ───────────────────────────────────────────────
with rasterio.open(flammap_path) as src:
    flammap_arrival_unclipped = src.read(1)
    flammap_transform = src.transform
    flammap_crs = src.crs
    nodata = src.nodata

# Mask out invalid values BEFORE resampling
flammap_arrival = np.where(
    (flammap_arrival_unclipped == nodata) | (flammap_arrival_unclipped > sim_time_hr * 60) | (flammap_arrival_unclipped == 0), 
    np.nan, 
    flammap_arrival_unclipped
)

# ── Resample FlamMap Raster to Match Sim Resolution ───────────────────
resampled_flammap, new_transform = resample_raster(
    flammap_arrival,
    crs=flammap_crs,
    transform=flammap_transform,
    target_resolution=10,  # Match sim's DATA_RES
    method=Resampling.bilinear
)

# ── Compute Extent from Resampled Transform ───────────────────────────
extent = [
    new_transform.c,
    new_transform.c + resampled_flammap.shape[1] * new_transform.a,
    new_transform.f + resampled_flammap.shape[0] * new_transform.e,
    new_transform.f
]

# LOAD just necessary columns of logs
df_unclipped = pd.read_parquet(
    cell_logs_path,
    columns=["x", "y", "arrival_time"]
)

df = df_unclipped[(df_unclipped["arrival_time"].notnull()) & (df_unclipped["arrival_time"] <= (sim_time_hr * 60)) & (df_unclipped["arrival_time"] > 0)]

df["x_rounded"] = df["x"].round(3)
df["y_rounded"] = df["y"].round(3)

df_unique = df.sort_values("arrival_time").drop_duplicates(subset=["x_rounded", "y_rounded"], keep="last")
df_unique = df_unique[(df_unique["arrival_time"].notnull()) & (df_unique["arrival_time"] != -999)]

scaling_factor = 10  # Usually 3.0
df_unique["x_unscaled"] = df_unique["x"] / scaling_factor
df_unique["y_unscaled"] = df_unique["y"] / scaling_factor

df_unique["y_flipped"] = sim_height - df_unique["y_unscaled"]


# ── Project Simulator Coordinates to UTM Using map_transform ───────────
xy_proj = [map_transform * (x, y) for x, y in zip(df_unique["x_unscaled"], df_unique["y_flipped"])]
df_unique["x_proj"] = [pt[0] for pt in xy_proj]
df_unique["y_proj"] = [pt[1] for pt in xy_proj]

# Shared color scale
vmin = min(np.nanmin(resampled_flammap), df_unique["arrival_time"].min())
vmax = max(np.nanmax(resampled_flammap), df_unique["arrival_time"].max())
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "inferno"  # Better for papers than rainbow

fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

# ── 1. FlamMap ────────────────────────────────
im0 = axs[0].imshow(
    resampled_flammap,
    cmap=cmap,
    norm=norm,
    origin="upper",
    extent=extent
)
axs[0].set_title("FlamMap Arrival Time")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")

# ── 2. Simulator ──────────────────────────────
sc = axs[1].scatter(
    df_unique["x_proj"],
    df_unique["y_proj"],
    c=df_unique["arrival_time"],
    cmap=cmap,
    norm=norm,
    s=10,
    edgecolor="none"
)
axs[1].set_title("Simulator Arrival Time")
axs[1].set_xlabel("x (m)")
axs[1].set_ylabel("y (m)")

# ── Shared Colorbar ───────────────────────────
cbar = fig.colorbar(sc, ax=axs[:], location='right', shrink=0.8, pad=0.02)
cbar.set_label("Arrival Time (s)")

# Optional: match axes
axs[0].set_aspect("equal")
axs[1].set_aspect("equal")

# Compute simulator projected coordinate bounds
x_min, x_max = df_unique["x_proj"].min(), df_unique["x_proj"].max()
y_min, y_max = df_unique["y_proj"].min(), df_unique["y_proj"].max()

# Get center and largest span to make square axis limits
x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2
half_range = max((x_max - x_min + 1000), (y_max - y_min + 1000)) / 2

xlim = (x_center - half_range, x_center + half_range)
ylim = (y_center - half_range, y_center + half_range)

# Apply to both subplots
for ax in axs:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")


# Save as high-res PNG or vector
plt.savefig("arrival_comparison.png", dpi=300)
# plt.savefig("arrival_comparison.pdf")  # for LaTeX embedding

plt.show()


# Compute MAE in both directions
# i.e. if performing MAE for sim data use the unclipped (by time) flammap data to compare the predicted arrival times
# for all locations after in this case 48 hours for the sim to the arrival time that flammap predicts for the same location
# regardless if that arrival time is within the 48 hour clipping

















