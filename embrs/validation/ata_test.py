from shapely.geometry import Point, Polygon
import numpy as np
import pickle
from shapely.ops import unary_union
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.transform import Affine
import matplotlib.pyplot as plt

def generate_eval_points(bounds, spacing=10):
    x_min, y_min, x_max, y_max = bounds
    xs = np.arange(x_min, x_max, spacing)
    ys = np.arange(y_min, y_max, spacing)
    points = [Point(x, y) for x in xs for y in ys]
    return points

from shapely.strtree import STRtree

def build_sim_tree(hexagons, arrival_times):
    tree = STRtree(hexagons)
    # geom_to_idx = {geom: i for i, geom in enumerate(hexagons)}
    return tree, hexagons, arrival_times


def arrival_from_hexgrid(tree, hexagons, arrival_times, pt):
    matches = tree.query(pt)
    for idx in matches:
        geom = hexagons[idx]
        if geom.covers(pt):
            return arrival_times[idx]
    return np.inf


from rasterio.transform import rowcol

def arrival_from_raster(raster_arrival_array, transform, pt):
    try:
        row, col = rowcol(transform, pt.x, pt.y)
        t = raster_arrival_array[row, col]

        if np.isnan(t):
            return np.inf
        else:
            return t
        
    except:
        return np.inf

def compute_ATA_from_points(eval_pts, sim_tree, hexes, sim_times,
                            raster_arrival_array, raster_transform,
                            final_time_sim, final_time_ref, init_time_ref=0):
    max_time = max(final_time_sim, final_time_ref)
    discrepancy = 0
    count = 0

    for pt in eval_pts:
        t_sim = arrival_from_hexgrid(sim_tree, hexes, sim_times, pt)
        t_ref = arrival_from_raster(raster_arrival_array, raster_transform, pt)

        if np.isinf(t_sim) and np.isinf(t_ref):
            continue  # Unburned in both

        count += 1  # part of union

        if not np.isinf(t_sim) and not np.isinf(t_ref):
            discrepancy += max(t_sim - t_ref, 0)
        elif not np.isinf(t_sim):  # overprediction
            discrepancy += max_time - t_sim
        elif not np.isinf(t_ref):  # underprediction
            discrepancy += max_time - t_ref

    normalizer = count * (max_time - init_time_ref)
    return 1 - (discrepancy / normalizer) if normalizer > 0 else 0

from shapely.geometry import Polygon
import pandas as pd
import numpy as np
import json

def create_hexagon(center_x, center_y, radius):
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    x_vertices = center_x + radius * np.cos(angles)
    y_vertices = center_y + radius * np.sin(angles)
    return Polygon(zip(x_vertices, y_vertices))

def extract_sim_hexagons_and_times(log_path, metadata_path, max_time_hr, sim_bounds):
    """
    Extract hexagon geometries and arrival times from sim logs.

    Returns:
        hexagons (list of Polygon)
        arrival_times (list of float)
        final_time (float): max arrival time in seconds
    """
    # Load metadata to get cell size
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    cell_size = metadata['inputs']['cell size']

    # Load and filter logs
    df = pd.read_parquet(log_path, columns=["x", "y", "arrival_time"])
    df = df[(df["arrival_time"].notnull()) &
            (df["arrival_time"] >= 0) &
            (df["arrival_time"] <= max_time_hr * 60)]
    df = df.sort_values("arrival_time").drop_duplicates(subset=["x", "y"], keep="last")

    # Translate local (0,0)-based sim coordinates to UTM
    df["x_global"] = df["x"] + sim_bounds[0]
    df["y_global"] = df["y"] + sim_bounds[1]

    hexagons = [
        create_hexagon(row["x_global"], row["y_global"], cell_size)
        for _, row in df.iterrows()
    ]

    arrival_times = df["arrival_time"].tolist()
    final_time = np.nanmax(arrival_times)
    return hexagons, arrival_times, final_time

import rasterio
import numpy as np

def load_raster_arrival(raster_path, sim_bounds, max_time_hr):
    """
    Crop the raster to sim bounds and extract arrival time data.

    Returns:
        arrival_array (2D np.ndarray)
        transform (Affine)
        final_time (float): max arrival time in seconds
    """
    cropped_data, cropped_transform, crs, nodata = crop_raster_to_sim(
        raster_path, sim_bounds
    )

    new_transform = cropped_transform * Affine.translation(0, 0)

    resampled_data, transform = resample_raster(
        cropped_data, crs, new_transform, 10, Resampling.bilinear, nodata
    )

    # Threshold data by max_time_hr
    max_time_sec = max_time_hr * 60
    resampled_data[resampled_data > max_time_sec] = np.nan  # Mask out late arrivals

    return resampled_data, transform, np.nanmax(resampled_data)

    
def resample_raster(array, crs, transform, target_resolution, method, nodata):
    scale_factor = transform.a / target_resolution

    new_height = int(array.shape[0] * scale_factor)
    new_width = int(array.shape[1] * scale_factor)
    
    # Create the resampled array
    resampled_array = np.empty((new_height, new_width), dtype=np.float32)

    # Compute the new transform (apply scale inversely)
    new_transform = transform * transform.scale(1 / scale_factor, 1 / scale_factor)
    # Perform resampling
    reproject(
        source=array.astype(np.float32),
        destination=resampled_array,
        src_transform=transform,
        dst_transform=new_transform,
        src_crs=crs,
        dst_crs=crs,
        resampling=method,
        src_nodata=nodata,
        dst_nodata=nodata
    )

    # Restore NoData values
    resampled_array[resampled_array == -9999] = np.nan 

    return resampled_array, new_transform


def crop_raster_to_sim(full_raster_path, sim_bounds, debug=False):
    """
    Crop the full raster to match the sim grid bounds and keep it in UTM coordinates.
    No rebasing or flipping; stays in global coordinate space.

    Args:
        full_raster_path (str): Path to the full raster (.tif).
        sim_bounds (tuple): (left, bottom, right, top) in UTM coordinates.
        debug (bool): If True, plots intermediate steps.

    Returns:
        (np.ndarray, Affine): Cropped raster array and transform (UTM).
    """
    with rasterio.open(full_raster_path) as src:
        # Step 1: Read full raster
        raw_data = src.read(1)
        raw_bounds = src.bounds
        crs = src.crs
    
    
        nodata = src.nodata

    
        # Step 2: Crop raster to sim_bounds
        window = from_bounds(*sim_bounds, transform=src.transform)
        cropped_data = src.read(1, window=window)
        cropped_transform = src.window_transform(window)


        return cropped_data, cropped_transform, crs, nodata


# Set file paths
tif_file = "/Users/rui/Documents/Research/Code/embrs/embrs/validation/happy/fmp_happy2.tif"
log_file = '/Users/rui/Documents/Research/Code/embrs/embrs/validation/happy/happy_7/run_0/cell_logs.parquet'
metadata_path = '/Users/rui/Documents/Research/Code/embrs/embrs/validation/happy/happy_7/metadata.json'
map_params_path = "/Users/rui/Documents/Research/Code/embrs_maps/happy_validation/map_params.pkl"

time = 60

# Load simulation bounds
with open(map_params_path, "rb") as f:
    map_params = pickle.load(f)
bounding_box = map_params.geo_info.bounds
sim_bounds = (bounding_box.left, bounding_box.bottom, bounding_box.right, bounding_box.top)

# Load sim hexagons and arrival times
hexes, times, final_time_sim = extract_sim_hexagons_and_times(
    log_file, metadata_path, max_time_hr=time, sim_bounds=sim_bounds 
)

# Build spatial index for hexes
tree, hexes, _ = build_sim_tree(hexes, times)


# Load raster arrival field
raster_arr, raster_transform, final_time_raster = load_raster_arrival(tif_file, sim_bounds, max_time_hr=time)

# Generate evaluation points (e.g., 10m spacing)
eval_points = generate_eval_points(sim_bounds, spacing=10)

sim_arrivals = np.array([arrival_from_hexgrid(tree, hexes, times, pt) for pt in eval_points])
ref_arrivals = np.array([arrival_from_raster(raster_arr, raster_transform, pt) for pt in eval_points])


fig, axs = plt.subplots(1, 2, figsize=(12, 5))

fig.suptitle("Arrival Time Comparison", fontsize=16)
sc1 = axs[0].scatter([pt.x for pt in eval_points], [pt.y for pt in eval_points], c=sim_arrivals, cmap='viridis', s=2)
axs[0].set_title('EMBRS')
axs[0].set_aspect('equal')

cbar1 = fig.colorbar(sc1, ax=axs[0])
cbar1.set_label("Arrival Time (minutes)")

sc2 = axs[1].scatter([pt.x for pt in eval_points], [pt.y for pt in eval_points], c=ref_arrivals, cmap='viridis', s=2)
axs[1].set_title('FARSITE')
axs[1].set_aspect('equal')

cbar2 = fig.colorbar(sc2, ax=axs[1])
cbar2.set_label("Arrival Time (minutes)")
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()


# Compute ATA
ata_score = compute_ATA_from_points(
    eval_points,
    tree, hexes, times,
    raster_arr, raster_transform,
    final_time_sim, final_time_raster,
    init_time_ref=0
)

print(f"Arrival Time Agreement (ATA): {ata_score:.3f}")
