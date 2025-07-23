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
map_params_path = "/Users/rui/Documents/Research/Code/embrs_maps/happy_validation/map_params.pkl"

with open(map_params_path, "rb") as f:
    map_params = pickle.load(f)

map_transform = map_params.lcp_data.transform  # Affine transform from sim space to UTM (EPSG:26910)
sim_height = map_params.lcp_data.rows

# ── Load FlamMap Raster ───────────────────────────────────────────────
flammap_path = "/Users/rui/Documents/Research/Code/embrs/embrs/validation/happy/fmp_happy1.tif"

with rasterio.open(flammap_path) as src:
    flammap_ros = src.read(3)
    flammap_transform = src.transform
    flammap_crs = src.crs
    nodata = src.nodata

# ── Resample FlamMap Raster to Match Sim Resolution ───────────────────
resampled_flammap, new_transform = resample_raster(
    flammap_ros,
    crs=flammap_crs,
    transform=flammap_transform,
    target_resolution=10,  # Match sim's DATA_RES
    method=Resampling.bilinear
)
resampled_flammap = np.where((resampled_flammap == nodata) | (resampled_flammap < 0), np.nan, resampled_flammap)

# ── Load and Filter Simulator Data ─────────────────────────────────────
df = pd.read_parquet("/Users/rui/Documents/Research/Code/embrs/embrs/validation/happy/happy_3/run_0/cell_logs.parquet")
df["x_rounded"] = df["x"].round(3)
df["y_rounded"] = df["y"].round(3)

df_unique = df.sort_values("arrival_time").drop_duplicates(subset=["x_rounded", "y_rounded"], keep="last")
df_unique = df_unique[(df_unique["ros"].notnull()) & (df_unique["arrival_time"] != -999)]

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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Shared color scale
vmin = min(np.nanmin(resampled_flammap), df_unique["ros"].min()*60)
vmax = max(np.nanmax(resampled_flammap), df_unique["ros"].max()*60)
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
axs[0].set_title("FlamMap Rate of Spread")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")

# ── 2. Simulator ──────────────────────────────
sc = axs[1].scatter(
    df_unique["x_proj"],
    df_unique["y_proj"],
    c=df_unique["ros"]*60,
    cmap=cmap,
    norm=norm,
    s=10,
    edgecolor="none"
)
axs[1].set_title("Simulator Rate of Spread")
axs[1].set_xlabel("x (m)")
axs[1].set_ylabel("y (m)")

# ── Shared Colorbar ───────────────────────────
cbar = fig.colorbar(sc, ax=axs[:], location='right', shrink=0.8, pad=0.02)
cbar.set_label("Rate of Spread (m/min)")

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
plt.savefig("ros_comparison.png", dpi=300)
# plt.savefig("arrival_comparison.pdf")  # for LaTeX embedding

plt.show()


