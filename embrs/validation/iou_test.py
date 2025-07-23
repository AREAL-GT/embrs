from shapely.geometry import shape, MultiPolygon, Polygon
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from rasterio.features import shapes
import rasterio
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import pickle

max_time_hr = np.linspace(0, 4, 25)

def plot_raster_and_sim_bounds(raster_bounds, sim_bounds):
    fig, ax = plt.subplots()

    # Full raster bounds (blue)
    x_min, y_min, x_max, y_max = raster_bounds
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            color="blue", label="Full Raster Bounds")

    # Sim bounds (red)
    sx_min, sy_min, sx_max, sy_max = sim_bounds
    ax.plot([sx_min, sx_max, sx_max, sx_min, sx_min],
            [sy_min, sy_min, sy_max, sy_max, sy_min],
            color="red", label="Sim Cropped Bounds")

    ax.set_aspect('equal')
    ax.legend()
    plt.title("Full Raster vs. Sim Cropped Bounds")
    plt.show()

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

def raster_to_polygon_in_sim_coords(full_raster_path, sim_bounds, max_time, debug=False):
    """
    Generate a polygon from the full raster, cropped into UTM coordinates.
    """
    cropped_data, cropped_transform, crs, nodata = crop_raster_to_sim(
        full_raster_path, sim_bounds
    )

    rows, cols = cropped_data.shape
    

    print(f"Test transform: {cropped_transform}")


    new_transform = cropped_transform * Affine.translation(
        0, 0
    )

    resampled_data, transform = resample_raster(cropped_data, crs, new_transform, 10, Resampling.bilinear, nodata)


    data = np.flipud(resampled_data)

    if debug:
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap="viridis", origin='lower')
        plt.title("Step 6: flipup")
        plt.colorbar(label="Value")
        plt.show()


    # Mask valid data
    mask = (data <= max_time * 60) & (data >= 0) & (data != nodata)

    # Extract polygons
    results = shapes(data, mask=mask)

    polygons = [shape(geom) for geom, value in results] 

    transformed_geometries = []

    for poly in polygons:
        scaled_coords = [(x * 10, y * 10) for x, y in poly.exterior.coords]
        transformed_polygon = Polygon(scaled_coords)
        transformed_geometries.append(transformed_polygon)

    # Merge to MultiPolygon
    merged_polygon = unary_union(transformed_geometries)
    return merged_polygon


def create_hexagon(center_x, center_y, radius):
    """
    Create a Shapely hexagon centered at (center_x, center_y)
    with pointy-top orientation.
    """
    # Rotate angles by 30 degrees (π/6) for pointy-top
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    x_vertices = center_x + radius * np.cos(angles)
    y_vertices = center_y + radius * np.sin(angles)

    return Polygon(zip(x_vertices, y_vertices))

def cell_log_to_hex_polygons(log_path, metadata_path, map_params_path, max_time):
    """
    Create a polygon from sim hexagons in the cropped raster's coordinate space.
    """
    # Load map params
    with open(map_params_path, "rb") as f:
        map_params = pickle.load(f)

    df = pd.read_parquet(
        log_path,
        columns=["x", "y", "arrival_time"]
    )

    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    cell_size = metadata['inputs']['cell size']

    # Filter valid cells
    df = df[(df["arrival_time"].notnull()) & 
            (df["arrival_time"] <= (max_time * 60)) & 
            (df["arrival_time"] >= 0)]
    df_unique = df.sort_values("arrival_time").drop_duplicates(subset=["x", "y"], keep="last")
    df_unique = df_unique[df_unique["arrival_time"] != -999]

    hexagons = [
        create_hexagon(row['x'], row['y'], cell_size)
        for _, row in df_unique.iterrows()
    ]

    merged_polygon = unary_union(hexagons)
    return merged_polygon


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
        xres, yres = src.res
        print(f"Pixel size: {xres} × {yres}")
        nodata = src.nodata

        # If you need the number of rows/cols:
        rows, cols = src.read(1).shape
        print(f"Raster dimensions: {rows} rows × {cols} cols")


        if debug:
            plt.figure(figsize=(8, 6))
            plt.imshow(raw_data, cmap="viridis")
            plt.title("Step 1: Full Raw Raster (UTM)")
            plt.colorbar(label="Value")
            plt.show()
            print(f"Full raster bounds (UTM): {raw_bounds}")

        # Step 2: Crop raster to sim_bounds
        window = from_bounds(*sim_bounds, transform=src.transform)
        cropped_data = src.read(1, window=window)
        cropped_transform = src.window_transform(window)
        cropped_bounds = rasterio.windows.bounds(window, src.transform)

        if debug:
            plt.figure(figsize=(8, 6))
            plt.imshow(cropped_data, cmap="viridis")
            plt.title("Step 2: Cropped Raster (Still in UTM)")
            plt.colorbar(label="Value")
            plt.show()
            print(f"Cropped raster bounds (UTM): {cropped_bounds}")
            print(f"Cropped raster transform:\n{cropped_transform}")

        return cropped_data, cropped_transform, crs, nodata


def plot_polygons(polys):
    """
    Plot a shapely polygon using matplotlib.
    """
    fig, ax = plt.subplots()
    colors = ['blue', 'red']

    for i, polygon in enumerate(polys):
        if polygon.geom_type == 'Polygon':
            x, y = polygon.exterior.xy
            ax.fill(x, y, alpha=0.5, fc=colors[i], ec='black')
        elif polygon.geom_type == 'MultiPolygon':
            for poly in polygon.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc=colors[i], ec='black')

    ax.set_aspect('equal')
    plt.show()


# # Example usage
tif_file = "/Users/rui/Documents/Research/Code/embrs/embrs/validation/flatland/fmp_nowind.tif"


log_file = '/Users/rui/Documents/Research/Code/embrs/embrs/validation/flatland/flat_nowind_new_accel/run_0/cell_logs.parquet'
metadata_path = '/Users/rui/Documents/Research/Code/embrs/embrs/validation/flatland/flat_nowind_new_accel/metadata.json'
map_params_path = "/Users/rui/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Research/embrs_inputs/embrs_maps/flatland_test/map_params.pkl"


with open(map_params_path, "rb") as f:
    map_params = pickle.load(f)

with open(metadata_path, 'r') as f:
    metadata = json.load(f)


bounding_box = map_params.geo_info.bounds

sim_bounds = (bounding_box.left, bounding_box.bottom, bounding_box.right, bounding_box.top)

ious = []

for tm in max_time_hr:
    raster_polygon = raster_to_polygon_in_sim_coords(
        full_raster_path=tif_file,
        sim_bounds = sim_bounds,
        max_time = tm
    )

    log_polygon = cell_log_to_hex_polygons(log_file, metadata_path, map_params_path, tm)

    plot_polygons([raster_polygon, log_polygon])

    intersection_area = log_polygon.intersection(raster_polygon).area
    union_area = log_polygon.union(raster_polygon).area
    iou = intersection_area/union_area

    print(f"IOU: {iou}")

    ious.append(iou)



plt.plot(max_time_hr, ious)

plt.show()