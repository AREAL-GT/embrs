import rasterio
import rasterio.plot
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from rasterio.enums import Resampling
from embrs.map_generator import resample_raster  # Your existing resampling function
import pickle

# ── Load Map Transform ────────────────────────────────────────────────
map_params_path = "/Users/rui/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Research/embrs_inputs/embrs_maps/happy_validation_new/map_params.pkl"

with open(map_params_path, "rb") as f:
    map_params = pickle.load(f)

map_transform = map_params.lcp_data.transform  # Affine transform from sim space to UTM (EPSG:26910)
sim_height = map_params.lcp_data.rows

# ── Load FlamMap Raster ───────────────────────────────────────────────
flammap_path = "/Users/rui/Documents/Research/Code/embrs/embrs/validation/happy_1/fmp_happy1.tif"

with rasterio.open(flammap_path) as src:
    flammap_arrival = src.read(1)
    flammap_transform = src.transform
    flammap_crs = src.crs
    nodata = src.nodata

# ── Resample FlamMap Raster to Match Sim Resolution ───────────────────
resampled_flammap, new_transform = resample_raster(
    flammap_arrival,
    crs=flammap_crs,
    transform=flammap_transform,
    target_resolution=10,  # Match sim's DATA_RES
    method=Resampling.bilinear
)
resampled_flammap = np.where((resampled_flammap == nodata) | (resampled_flammap < 0), np.nan, resampled_flammap)

# import pandas as pd

# LOAD just necessary columns of logs
df = pd.read_parquet(
    "/Users/rui/Documents/Research/Code/embrs/embrs/validation/happy_1/happy_1/run_0/cell_logs.parquet",
    columns=["x", "y", "arrival_time"]
)

df = df[(df["arrival_time"].notnull()) & (df["arrival_time"] > 0)]

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


# ── Logging Info ───────────────────────────────────────────────────────
print("FlamMap CRS:", flammap_crs)
print("Sim x range (projected):", df_unique["x_proj"].min(), "-", df_unique["x_proj"].max())
print("Sim y range (projected):", df_unique["y_proj"].min(), "-", df_unique["y_proj"].max())

# ── Compute Extent from Resampled Transform ───────────────────────────
extent = [
    new_transform.c,
    new_transform.c + resampled_flammap.shape[1] * new_transform.a,
    new_transform.f + resampled_flammap.shape[0] * new_transform.e,
    new_transform.f
]
print("Extent:", extent)

# ── Plot Both Layers ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))

# FlamMap raster background
fmap = ax.imshow(
    resampled_flammap,
    cmap="Blues",
    origin="upper",
    extent=extent,
    alpha=0.7
)

# Normalize sim arrival times
sim_norm = mcolors.Normalize(
    vmin=df_unique["arrival_time"].min(),
    vmax=df_unique["arrival_time"].max()
)

# Overlay simulator points (in projected UTM coordinates)
sc = ax.scatter(
    df_unique["x_proj"], df_unique["y_proj"],
    c=df_unique["arrival_time"],
    cmap="autumn_r",
    norm=sim_norm,
    s=30,
    edgecolor="k",
    linewidth=0.2,
    label="Simulator Arrival"
)

# Colorbars
plt.colorbar(fmap, ax=ax, fraction=0.046, pad=0.04, label="FlamMap Arrival Time")
plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.08, label="Simulator Arrival Time")

# Final formatting
ax.set_title("Overlay: Simulator vs. FlamMap Arrival Time")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_aspect("equal")
plt.tight_layout()

plt.show()



import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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



import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from rasterio.enums import Resampling
from scipy.stats import pearsonr
import pickle

# # ── Load Map Transform ────────────────────────────────────────────────
# map_params_path = "/Users/rui/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Research/embrs_inputs/embrs_maps/flammap_validation_2/map_params.pkl"
# with open(map_params_path, "rb") as f:
#     map_params = pickle.load(f)

# map_transform = map_params.lcp_data.transform
# sim_height = map_params.lcp_data.rows

# # ── Load FlamMap Raster ───────────────────────────────────────────────
# flammap_path =  "/Users/rui/Documents/Research/Code/embrs/embrs/test_code/48hr_validation_test.tif"
# with rasterio.open(flammap_path) as src:
#     flammap_arrival = src.read(1)
#     flammap_transform = src.transform
#     flammap_crs = src.crs
#     nodata = src.nodata

# # ── Resample FlamMap Raster ───────────────────────────────────────────
# from embrs.map_generator import resample_raster
# resampled_flammap, new_transform = resample_raster(
#     flammap_arrival,
#     crs=flammap_crs,
#     transform=flammap_transform,
#     target_resolution=10,
#     method=Resampling.bilinear
# )
# resampled_flammap = np.where((resampled_flammap == nodata) | (resampled_flammap < 0), np.nan, resampled_flammap)

# # ── Load Simulator Output ─────────────────────────────────────────────
# df = pd.read_parquet("/Users/rui/Documents/Research/Code/embrs_logs/48hr_arrival_time_test/run_0/cell_logs.parquet")
# df["x_rounded"] = df["x"].round(3)
# df["y_rounded"] = df["y"].round(3)
# df_unique = df.sort_values("arrival_time").drop_duplicates(subset=["x_rounded", "y_rounded"], keep="last")
# df_unique = df_unique[(df_unique["arrival_time"].notnull()) & (df_unique["arrival_time"] != -999)]

# scaling_factor = 10
# df_unique["x_unscaled"] = df_unique["x"] / scaling_factor
# df_unique["y_unscaled"] = df_unique["y"] / scaling_factor
# df_unique["y_flipped"] = sim_height - df_unique["y_unscaled"]

# xy_proj = [map_transform * (x, y) for x, y in zip(df_unique["x_unscaled"], df_unique["y_flipped"])]
# df_unique["x_proj"] = [pt[0] for pt in xy_proj]
# df_unique["y_proj"] = [pt[1] for pt in xy_proj]

# ── Rasterize Simulator Data to Match Grid ────────────────────────────
sim_raster = np.full_like(resampled_flammap, np.nan)
for x, y, val in zip(df_unique["x_proj"], df_unique["y_proj"], df_unique["arrival_time"]):
    col, row = ~new_transform * (x, y)
    row, col = int(round(row)), int(round(col))
    if 0 <= row < sim_raster.shape[0] and 0 <= col < sim_raster.shape[1]:
        sim_raster[row, col] = val

# ── Compute Similarity Metrics ─────────────────────────────────────────
valid_mask = ~np.isnan(sim_raster) & ~np.isnan(resampled_flammap)
sim_valid = sim_raster[valid_mask]
flammap_valid = resampled_flammap[valid_mask]

mae = np.mean(np.abs(sim_valid - flammap_valid))
mse = np.mean((sim_valid - flammap_valid) ** 2)
corr, _ = pearsonr(sim_valid, flammap_valid)

# IoU on binary burn regions (>1s)
sim_burn = (sim_raster > 1)
flammap_burn = (resampled_flammap > 1)
# intersection = np.logical_and(sim_burn, flammap_burn).sum()
# union = np.logical_or(sim_burn, flammap_burn).sum()

# Define valid data mask
valid_union_mask = ~np.isnan(sim_raster) | ~np.isnan(resampled_flammap)

# Compute burn masks
sim_burn = (sim_raster > 1)
flammap_burn = (resampled_flammap > 1)

# Apply valid mask to both
intersection = np.logical_and(sim_burn, flammap_burn) & valid_union_mask
union = np.logical_or(sim_burn, flammap_burn) & valid_union_mask

# Compute IoU
iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else np.nan

# iou = intersection / union if union > 0 else np.nan


# ── Print Results ─────────────────────────────────────────────────────
print(f"Mean Absolute Error (MAE): {mae:.2f} s")
print(f"Mean Squared Error (MSE): {mse:.2f} s^2")
print(f"Pearson Correlation: {corr:.4f}")
print(f"IoU (arrival_time > 1s): {iou:.4f}")

# ── Visualization of Masks ────────────────────────────────────────────
fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

axs[0].imshow(sim_burn, cmap="Reds", origin="upper")
axs[0].set_title("Simulator Burn Mask (>1s)")
axs[0].axis("off")

axs[1].imshow(flammap_burn, cmap="Blues", origin="upper")
axs[1].set_title("FlamMap Burn Mask (>1s)")
axs[1].axis("off")

axs[2].imshow(np.logical_or(sim_burn, flammap_burn), cmap="Greys", origin="upper")
axs[2].imshow(np.logical_and(sim_burn, flammap_burn), cmap="Greens", alpha=0.6, origin="upper")
axs[2].set_title("Intersection (green) on Union (gray)")
axs[2].axis("off")

plt.suptitle("IoU Burn Region Visualization (Threshold: 1s)", fontsize=14)
plt.savefig("iou_burn_region_visualization_rasterized.png", dpi=300)
plt.show()








# import rasterio
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np
# from rasterio.enums import Resampling
# from scipy.interpolate import griddata
# from scipy.stats import pearsonr
# from embrs.map_generator import resample_raster
# import pickle

# # ── Load Map Transform ────────────────────────────────────────────────
# map_params_path = "/Users/rui/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Research/embrs_inputs/embrs_maps/flammap_validation_2/map_params.pkl"
# with open(map_params_path, "rb") as f:
#     map_params = pickle.load(f)

# map_transform = map_params.lcp_data.transform
# sim_height = map_params.lcp_data.rows

# # ── Load FlamMap Raster ───────────────────────────────────────────────
# flammap_path = "/Users/rui/Documents/Research/Code/embrs/embrs/test_code/validation_1.tif"
# with rasterio.open(flammap_path) as src:
#     flammap_arrival = src.read(1)
#     flammap_transform = src.transform
#     flammap_crs = src.crs
#     nodata = src.nodata

# # ── Resample FlamMap Raster ───────────────────────────────────────────
# resampled_flammap, new_transform = resample_raster(
#     flammap_arrival,
#     crs=flammap_crs,
#     transform=flammap_transform,
#     target_resolution=10,
#     method=Resampling.bilinear
# )
# resampled_flammap = np.where((resampled_flammap == nodata) | (resampled_flammap < 0), np.nan, resampled_flammap)

# # ── Load Simulator Output ─────────────────────────────────────────────
# df = pd.read_parquet("/Users/rui/Documents/Research/Code/embrs_logs/48hr_arrival_time_test/run_0/cell_logs.parquet")
# df["x_rounded"] = df["x"].round(3)
# df["y_rounded"] = df["y"].round(3)
# df_unique = df.sort_values("arrival_time").drop_duplicates(subset=["x_rounded", "y_rounded"], keep="last")
# df_unique = df_unique[(df_unique["arrival_time"].notnull()) & (df_unique["arrival_time"] != -999)]

# # ── Flip and Unscale ──────────────────────────────────────────────────
# scaling_factor = 10
# df_unique["x_unscaled"] = df_unique["x"] / scaling_factor
# df_unique["y_unscaled"] = df_unique["y"] / scaling_factor
# df_unique["y_flipped"] = sim_height - df_unique["y_unscaled"]

# # ── Project to UTM ────────────────────────────────────────────────────
# xy_proj = [map_transform * (x, y) for x, y in zip(df_unique["x_unscaled"], df_unique["y_flipped"])]
# df_unique["x_proj"] = [pt[0] for pt in xy_proj]
# df_unique["y_proj"] = [pt[1] for pt in xy_proj]

# # ── Interpolate Simulator Data to FlamMap Grid ────────────────────────
# sim_points = np.column_stack((df_unique["x_proj"], df_unique["y_proj"]))
# sim_values = df_unique["arrival_time"].values

# # Ensure full simulator domain is included
# x_min, x_max = df_unique["x_proj"].min(), df_unique["x_proj"].max()
# y_min, y_max = df_unique["y_proj"].min(), df_unique["y_proj"].max()

# # Make sure grid covers both FlamMap and sim domain
# extent = [
#     min(new_transform.c, x_min),
#     max(new_transform.c + resampled_flammap.shape[1] * new_transform.a, x_max),
#     min(new_transform.f + resampled_flammap.shape[0] * new_transform.e, y_min),
#     max(new_transform.f, y_max)
# ]

# height, width = resampled_flammap.shape
# xx = np.linspace(extent[0], extent[1], width)
# yy = np.linspace(extent[3], extent[2], height)  # flip y
# grid_x, grid_y = np.meshgrid(xx, yy)

# sim_interp = griddata(sim_points, sim_values, (grid_x, grid_y), method="linear")

# # ── Masked Difference Grid ─────────────────────────────────────────────
# flammap_masked = np.where(np.isnan(sim_interp), np.nan, resampled_flammap)
# difference = np.abs(sim_interp - flammap_masked)

# # ── Compute Metrics ────────────────────────────────────────────────────
# valid_mask = ~np.isnan(sim_interp) & ~np.isnan(flammap_masked)

# sim_valid = sim_interp[valid_mask]
# flammap_valid = flammap_masked[valid_mask]

# mae = np.mean(np.abs(sim_valid - flammap_valid))
# mse = np.mean((sim_valid - flammap_valid) ** 2)
# corr, _ = pearsonr(sim_valid, flammap_valid)

# # IoU — burn regions defined as arrival_time < 1800 s
# sim_burn = sim_valid > 0
# flammap_burn = flammap_valid > 0
# intersection = np.logical_and(sim_burn, flammap_burn).sum()
# union = np.logical_or(sim_burn, flammap_burn).sum()
# iou = intersection / union if union > 0 else np.nan

# # ── Print Results ──────────────────────────────────────────────────────
# print(f"Mean Absolute Error (MAE): {mae:.2f} s")
# print(f"Mean Squared Error (MSE): {mse:.2f} s^2")
# print(f"Pearson Correlation: {corr:.4f}")
# print(f"IoU (arrival_time < 1800s): {iou:.4f}")


# sim_burn_mask = (sim_interp > 0)
# flammap_burn_mask = (resampled_flammap > 0)

# # ── Visualization of Masks ────────────────────────────────────────────
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# axs[0].imshow(sim_burn_mask, cmap="Reds", origin="upper")
# axs[0].set_title("Simulator Burn Mask (>1s)")
# axs[0].axis("off")

# axs[1].imshow(flammap_burn_mask, cmap="Blues", origin="upper")
# axs[1].set_title("FlamMap Burn Mask (>1s)")
# axs[1].axis("off")

# axs[2].imshow(np.logical_or(sim_burn_mask, flammap_burn_mask), cmap="Greys", origin="upper")
# axs[2].imshow(np.logical_and(sim_burn_mask, flammap_burn_mask), cmap="Greens", alpha=0.6, origin="upper")
# axs[2].set_title("Intersection (green) on Union (gray)")
# axs[2].axis("off")

# plt.suptitle("IoU Burn Region Visualization (Threshold: 1s)", fontsize=14)
# plt.savefig("iou_burn_region_visualization_rasterized.png", dpi=300)
# plt.show()