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


sim_time_hr = 36

# ── Load Map Transform ────────────────────────────────────────────────
with open(map_params_path, "rb") as f:
    map_params = pickle.load(f)

map_transform = map_params.lcp_data.transform  # Affine transform from sim space to UTM (EPSG:26910)
sim_height = map_params.lcp_data.rows

# ── Load FlamMap Raster ───────────────────────────────────────────────
with rasterio.open(flammap_path) as src:
    flammap_raw = src.read(1)
    flammap_transform = src.transform
    flammap_crs = src.crs
    nodata = src.nodata

# Mask out invalid values BEFORE resampling
flammap_unclipped = np.where(
    (flammap_raw == nodata) | (flammap_raw == 0), 
    np.nan, 
    flammap_raw
)

# ── Resample FlamMap Raster to Match Sim Resolution ───────────────────
flammap_unclipped_resampled, new_transform = resample_raster(
    flammap_unclipped,
    crs=flammap_crs,
    transform=flammap_transform,
    target_resolution=10,  # Match sim's DATA_RES
    method=Resampling.bilinear
)

# Mask out invalid values BEFORE resampling
flammap_clipped = np.where(
    (flammap_raw == nodata) | (flammap_raw > sim_time_hr * 60) | (flammap_raw == 0), 
    np.nan, 
    flammap_raw
)

# ── Resample FlamMap Raster to Match Sim Resolution ───────────────────
flammap_clipped_resampled, new_transform = resample_raster(
    flammap_clipped,
    crs=flammap_crs,
    transform=flammap_transform,
    target_resolution=10,  # Match sim's DATA_RES
    method=Resampling.bilinear
)


# LOAD just necessary columns of logs
df_raw = pd.read_parquet(
    cell_logs_path,
    columns=["x", "y", "arrival_time"]
)

# df_raw["x_rounded"] = df_raw["x"].round(3)
# df_raw["y_rounded"] = df_raw["y"].round(3)



df_unclipped = df_raw[(df_raw["arrival_time"].notnull()) & (df_raw["arrival_time"] > 0)]
df_clipped = df_raw[(df_raw["arrival_time"].notnull()) & (df_raw["arrival_time"] <= (sim_time_hr * 60)) & (df_raw["arrival_time"] > 0)]


df_unclipped = df_unclipped.sort_values("arrival_time").drop_duplicates(subset=["x", "y"], keep="last")
df_unclipped = df_unclipped[(df_unclipped["arrival_time"].notnull()) & (df_unclipped["arrival_time"] != -999)]

df_clipped = df_clipped.sort_values("arrival_time").drop_duplicates(subset=["x", "y"], keep="last")
df_clipped = df_clipped[(df_clipped["arrival_time"].notnull()) & (df_clipped["arrival_time"] != -999)]

scaling_factor = 10  # Usually 3.0
df_unclipped["x_unscaled"] = df_unclipped["x"] / scaling_factor
df_unclipped["y_unscaled"] = df_unclipped["y"] / scaling_factor

df_clipped["x_unscaled"] = df_clipped["x"] / scaling_factor
df_clipped["y_unscaled"] = df_clipped["y"] / scaling_factor

df_clipped["y_flipped"] = sim_height - df_clipped["y_unscaled"]
df_unclipped["y_flipped"] = sim_height - df_unclipped["y_unscaled"]

# ── Project Simulator Coordinates to UTM Using map_transform ───────────
xy_proj = [map_transform * (x, y) for x, y in zip(df_unclipped["x_unscaled"], df_unclipped["y_flipped"])]
df_unclipped["x_proj"] = [pt[0] for pt in xy_proj]
df_unclipped["y_proj"] = [pt[1] for pt in xy_proj]

xy_proj = [map_transform * (x, y) for x, y in zip(df_clipped["x_unscaled"], df_clipped["y_flipped"])]
df_clipped["x_proj"] = [pt[0] for pt in xy_proj]
df_clipped["y_proj"] = [pt[1] for pt in xy_proj]

print("FlamMap arrival time stats:", np.nanmin(flammap_unclipped), np.nanmax(flammap_unclipped))
print("Simulator arrival time stats:", df_unclipped["arrival_time"].min(), df_unclipped["arrival_time"].max())

# Compute MAE in both directions
# i.e. if performing MAE for sim data use the unclipped (by time) flammap data to compare the predicted arrival times
# for all locations after in this case 48 hours for the sim to the arrival time that flammap predicts for the same location
# regardless if that arrival time is within the 48 hour clipping


from scipy.spatial import KDTree

# ── MAE: Sim → FlamMap ────────────────────────────────
# Build a tree for FlamMap resampled grid points
height, width = flammap_unclipped_resampled.shape
rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
flammap_x = new_transform.c + cols * new_transform.a
flammap_y = new_transform.f + rows * new_transform.e
flammap_coords = np.column_stack((flammap_x.ravel(), flammap_y.ravel()))
flammap_times = flammap_unclipped_resampled.ravel()

# Build KDTree for FlamMap coordinates
flammap_tree = KDTree(flammap_coords)

# Query FlamMap times at simulator points
sim_coords = np.column_stack((df_clipped["x_proj"], df_clipped["y_proj"]))
distances, indices = flammap_tree.query(sim_coords)

# Arrival times from FlamMap at sim points
flammap_at_sim = flammap_times[indices]

# Filter out NaNs in FlamMap data
valid_mask = ~np.isnan(flammap_at_sim)
sim_times = df_clipped["arrival_time"].values[valid_mask]
flammap_times_valid = flammap_at_sim[valid_mask]

# Compute MAE
mae_sim_to_flammap = np.mean(np.abs(sim_times - flammap_times_valid))
print(f"MAE (Simulator → FlamMap): {mae_sim_to_flammap:.2f} minutes")

median_error = np.median(np.abs(sim_times - flammap_times_valid))
print(f"Median Absolute Error: {median_error:.2f} minutes")

trimmed_mask = np.abs(sim_times - flammap_times_valid) < 300  # exclude big outliers
mae_trimmed = np.mean(np.abs(sim_times[trimmed_mask] - flammap_times_valid[trimmed_mask]))
print(f"Trimmed MAE (exclude >300 min diffs): {mae_trimmed:.2f} minutes")

within_60 = np.mean(np.abs(sim_times - flammap_times_valid) <= 60) * 100
print(f"{within_60:.2f}% of points within 60 minutes")



# ── MAE: FlamMap → Sim ────────────────────────────────
# Build KDTree for simulator points
sim_coords = np.column_stack((df_unclipped["x_proj"], df_unclipped["y_proj"]))
sim_tree = KDTree(sim_coords)

# Query simulator times at FlamMap points
height, width = flammap_clipped_resampled.shape
rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
flammap_x = new_transform.c + cols * new_transform.a
flammap_y = new_transform.f + rows * new_transform.e
flammap_coords = np.column_stack((flammap_x.ravel(), flammap_y.ravel()))

distances_sim, indices_sim = sim_tree.query(flammap_coords)

# Arrival times from simulator at FlamMap points
sim_at_flammap = df_unclipped["arrival_time"].values[indices_sim]

# Filter out NaNs in simulator data
valid_mask_sim = ~np.isnan(sim_at_flammap) & ~np.isnan(flammap_times)
flammap_times_valid_2 = flammap_times[valid_mask_sim]
sim_times_valid_2 = sim_at_flammap[valid_mask_sim]

# Compute MAE
mae_flammap_to_sim = np.mean(np.abs(flammap_times_valid_2 - sim_times_valid_2))
print(f"MAE (FlamMap → Simulator): {mae_flammap_to_sim:.2f} minutes")


plt.scatter(sim_times, flammap_times_valid, alpha=0.3, s=1)
plt.xlabel("Simulator Arrival Time (min)")
plt.ylabel("FlamMap Arrival Time (min)")
plt.plot([0, max(sim_times)], [0, max(sim_times)], 'r--')
plt.show()


difference = flammap_clipped_resampled - sim_at_flammap.reshape(flammap_clipped_resampled.shape)

abs_diff = np.abs(difference)
print(f'min difference: {np.nanmin(abs_diff)}, max difference: {np.nanmax(abs_diff)}')


plt.imshow(difference, cmap="RdBu", vmin=-500, vmax=500)
plt.colorbar(label="Arrival Time Difference (min)")
plt.title("FlamMap - Simulator Arrival Time Difference")
plt.show()

plt.imshow(flammap_clipped_resampled, extent=[new_transform.c, new_transform.c + new_transform.a * flammap_clipped_resampled.shape[1],
                                              new_transform.f + new_transform.e * flammap_clipped_resampled.shape[0], new_transform.f],
           origin="upper", cmap="inferno")
plt.scatter(df_clipped["x_proj"], df_clipped["y_proj"], s=1, color="cyan")
plt.title("Simulator Points over FlamMap Raster")
plt.show()











