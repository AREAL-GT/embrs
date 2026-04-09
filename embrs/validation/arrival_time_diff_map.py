"""
Generate an arrival time difference map between EMBRS and FARSITE for a
validation case. Produces a single-panel figure with:

- Signed difference (t_sim - t_ref) where both models burned the cell, drawn
  with a diverging colormap centered on zero so that white indicates perfect
  agreement, red indicates EMBRS arrived late, and blue indicates EMBRS
  arrived early.
- Categorical overlay for cells burned by only one model: "EMBRS only" and
  "FARSITE only" regions, which represent disagreement in fire extent rather
  than timing.
- ATA and final Jaccard reported in the figure title for context.

This script mirrors the data loading pattern of ata_validation.py and
iou_v_time_validation.py and is intended to replace the side-by-side
arrival time scatter as the primary validation figure in the manuscript.
"""

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from shapely.ops import unary_union
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.transform import Affine, rowcol
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
import rasterio
import pickle
import json

# Change case number to run different validation case:
case_num = 2

# Spacing of the evaluation grid in metres. 10 m matches ata_validation.py.
eval_spacing = 10

# Maximum simulated duration in hours (matches ata_validation.py default).
max_time_hr = 72


# ---------------------------------------------------------------------------
# Helpers (mirrored from ata_validation.py / iou_v_time_validation.py for a
# self-contained script).
# ---------------------------------------------------------------------------

def create_hexagon(center_x, center_y, radius):
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    x_vertices = center_x + radius * np.cos(angles)
    y_vertices = center_y + radius * np.sin(angles)
    return Polygon(zip(x_vertices, y_vertices))


def extract_sim_hexagons_and_times(log_path, metadata_path, max_time_hr, sim_bounds):
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    cell_size = metadata['inputs']['cell size']

    df = pd.read_parquet(log_path, columns=["x", "y", "arrival_time"])
    df = df[(df["arrival_time"].notnull()) &
            (df["arrival_time"] >= 0) &
            (df["arrival_time"] <= max_time_hr * 60)]
    df = df.sort_values("arrival_time").drop_duplicates(subset=["x", "y"], keep="last")

    df["x_global"] = df["x"] + sim_bounds[0]
    df["y_global"] = df["y"] + sim_bounds[1]

    hexagons = [
        create_hexagon(row["x_global"], row["y_global"], cell_size)
        for _, row in df.iterrows()
    ]
    arrival_times = df["arrival_time"].tolist()
    final_time = float(np.nanmax(arrival_times)) if arrival_times else 0.0
    return hexagons, arrival_times, final_time


def crop_raster_to_sim(full_raster_path, sim_bounds):
    with rasterio.open(full_raster_path) as src:
        crs = src.crs
        nodata = src.nodata
        window = from_bounds(*sim_bounds, transform=src.transform)
        cropped_data = src.read(1, window=window)
        cropped_transform = src.window_transform(window)
    return cropped_data, cropped_transform, crs, nodata


def resample_raster(array, crs, transform, target_resolution, method, nodata):
    scale_factor = transform.a / target_resolution
    new_height = int(array.shape[0] * scale_factor)
    new_width = int(array.shape[1] * scale_factor)

    resampled_array = np.empty((new_height, new_width), dtype=np.float32)
    new_transform = transform * transform.scale(1 / scale_factor, 1 / scale_factor)

    reproject(
        source=array.astype(np.float32),
        destination=resampled_array,
        src_transform=transform,
        dst_transform=new_transform,
        src_crs=crs,
        dst_crs=crs,
        resampling=method,
        src_nodata=nodata,
        dst_nodata=nodata,
    )

    resampled_array[resampled_array == -9999] = np.nan
    return resampled_array, new_transform


def load_raster_arrival(raster_path, sim_bounds, max_time_hr):
    cropped_data, cropped_transform, crs, nodata = crop_raster_to_sim(
        raster_path, sim_bounds
    )
    new_transform = cropped_transform * Affine.translation(0, 0)
    resampled_data, transform = resample_raster(
        cropped_data, crs, new_transform, 10, Resampling.bilinear, nodata
    )
    max_time_sec = max_time_hr * 60
    resampled_data[resampled_data > max_time_sec] = np.nan
    return resampled_data, transform, float(np.nanmax(resampled_data))


def build_sim_tree(hexagons, arrival_times):
    tree = STRtree(hexagons)
    return tree, hexagons, arrival_times


def arrival_from_hexgrid(tree, hexagons, arrival_times, pt):
    matches = tree.query(pt)
    for idx in matches:
        geom = hexagons[idx]
        if geom.covers(pt):
            return arrival_times[idx]
    return np.inf


def arrival_from_raster(raster_arrival_array, transform, pt):
    try:
        row, col = rowcol(transform, pt.x, pt.y)
        t = raster_arrival_array[row, col]
        if np.isnan(t):
            return np.inf
        return float(t)
    except Exception:
        return np.inf


def compute_ATA(eval_pts_flat, sim_arr_flat, ref_arr_flat,
                final_time_sim, final_time_ref, init_time_ref=0):
    """ATA computed from already-evaluated arrival arrays (np.inf for unburned)."""
    max_time = max(final_time_sim, final_time_ref)
    discrepancy = 0.0
    count = 0

    for t_sim, t_ref in zip(sim_arr_flat, ref_arr_flat):
        sim_unburned = np.isinf(t_sim)
        ref_unburned = np.isinf(t_ref)

        if sim_unburned and ref_unburned:
            continue
        count += 1

        if not sim_unburned and not ref_unburned:
            discrepancy += max(t_sim - t_ref, 0)
        elif not sim_unburned:
            discrepancy += max_time - t_sim
        else:
            discrepancy += max_time - t_ref

    normalizer = count * (max_time - init_time_ref)
    return 1 - (discrepancy / normalizer) if normalizer > 0 else 0


def compute_jaccard(sim_arr_flat, ref_arr_flat):
    """Final Jaccard from burned/unburned masks."""
    sim_burned = ~np.isinf(sim_arr_flat)
    ref_burned = ~np.isinf(ref_arr_flat)
    intersection = np.logical_and(sim_burned, ref_burned).sum()
    union = np.logical_or(sim_burned, ref_burned).sum()
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# File paths
tif_file = f"embrs/validation/case_{case_num}/farsite.tif"
log_file = f"embrs/validation/case_{case_num}/embrs_data/run_0/cell_logs.parquet"
metadata_path = f"embrs/validation/case_{case_num}/embrs_data/metadata.json"
map_params_path = f"embrs/validation/case_{case_num}/embrs_data/map_params.pkl"

# Load simulation bounds
with open(map_params_path, "rb") as f:
    map_params = pickle.load(f)
bounding_box = map_params.geo_info.bounds
sim_bounds = (bounding_box.left, bounding_box.bottom, bounding_box.right, bounding_box.top)

# Load EMBRS hexagons + arrival times
hexes, times, final_time_sim = extract_sim_hexagons_and_times(
    log_file, metadata_path, max_time_hr=max_time_hr, sim_bounds=sim_bounds,
)
tree, hexes, _ = build_sim_tree(hexes, times)

# Load FARSITE raster arrival field
raster_arr, raster_transform, final_time_raster = load_raster_arrival(
    tif_file, sim_bounds, max_time_hr=max_time_hr,
)

# Build a regular evaluation grid that can be reshaped to a 2D image. Order
# the points such that flat[iy * nx + ix] corresponds to (xs[ix], ys[iy]).
x_min, y_min, x_max, y_max = sim_bounds
xs = np.arange(x_min, x_max, eval_spacing)
ys = np.arange(y_min, y_max, eval_spacing)
nx, ny = len(xs), len(ys)

eval_points = [Point(x, y) for y in ys for x in xs]

# Sample arrival times for both models
sim_arrivals = np.array([
    arrival_from_hexgrid(tree, hexes, times, pt) for pt in eval_points
])
ref_arrivals = np.array([
    arrival_from_raster(raster_arr, raster_transform, pt) for pt in eval_points
])

# Validation metrics
ata_score = compute_ATA(
    eval_points, sim_arrivals, ref_arrivals,
    final_time_sim, final_time_raster, init_time_ref=0,
)
jaccard_score = compute_jaccard(sim_arrivals, ref_arrivals)

print(f"ATA: {ata_score:.3f}")
print(f"Final Jaccard: {jaccard_score:.3f}")

# Reshape into 2D images (rows = y, cols = x)
sim_grid = sim_arrivals.reshape(ny, nx)
ref_grid = ref_arrivals.reshape(ny, nx)

sim_burned = ~np.isinf(sim_grid)
ref_burned = ~np.isinf(ref_grid)
both_burned = sim_burned & ref_burned
only_sim = sim_burned & ~ref_burned
only_ref = ref_burned & ~sim_burned

# Signed difference (minutes). Positive = EMBRS arrived late vs FARSITE.
diff = np.full_like(sim_grid, np.nan, dtype=np.float64)
diff[both_burned] = sim_grid[both_burned] - ref_grid[both_burned]

# Symmetric colormap range, robust to outliers via 98th percentile
finite_diff = diff[np.isfinite(diff)]
if finite_diff.size > 0:
    abs_limit = float(np.percentile(np.abs(finite_diff), 98))
    abs_limit = max(abs_limit, 1.0)  # avoid degenerate zero range
else:
    abs_limit = 1.0

mean_abs_diff = float(np.nanmean(np.abs(finite_diff))) if finite_diff.size else 0.0
median_abs_diff = float(np.nanmedian(np.abs(finite_diff))) if finite_diff.size else 0.0

extent = (x_min, x_min + nx * eval_spacing,
          y_min, y_min + ny * eval_spacing)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 7))

# Categorical overlay for disagreement regions: 0 = EMBRS only, 1 = FARSITE only.
overlay = np.full(diff.shape, np.nan)
overlay[only_sim] = 0
overlay[only_ref] = 1

overlay_cmap = ListedColormap(["#d7191c", "#2c7bb6"])  # red = EMBRS-only, blue = FARSITE-only
overlay_norm = BoundaryNorm([-0.5, 0.5, 1.5], overlay_cmap.N)

ax.imshow(
    overlay,
    extent=extent,
    origin="lower",
    cmap=overlay_cmap,
    norm=overlay_norm,
    interpolation="nearest",
    alpha=0.55,
)

# Main signed difference layer
diff_im = ax.imshow(
    diff,
    extent=extent,
    origin="lower",
    cmap="RdBu_r",
    vmin=-abs_limit,
    vmax=abs_limit,
    interpolation="nearest",
)

ax.set_aspect("equal")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_title(
    f"Arrival Time Difference (EMBRS − FARSITE)  |  Case {case_num}\n"
    f"ATA: {ata_score:.3f}    Final Jaccard: {jaccard_score:.3f}    "
    f"Mean |Δt|: {mean_abs_diff:.0f} min    Median |Δt|: {median_abs_diff:.0f} min"
)

cbar = fig.colorbar(diff_im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("EMBRS − FARSITE arrival time (minutes)")

# Legend for the categorical disagreement regions
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#d7191c", alpha=0.55, label="EMBRS only (over-extent)"),
    Patch(facecolor="#2c7bb6", alpha=0.55, label="FARSITE only (under-extent)"),
]
ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.savefig(f"arrival_time_diff_case_{case_num}.png", dpi=200, bbox_inches="tight")
plt.show()
