"""Module used to run the application that allows users to generate a new map file.
"""

from typing import Tuple, List
import json
import pickle
import sys
import os
import rasterio
from rasterio.warp import reproject, transform_bounds
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import from_bounds
import requests
import pyproj
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import Polygon, LineString
from shapely.geometry import mapping
import numpy as np

from embrs.utilities.file_io import MapGenFileSelector
from embrs.utilities.map_drawer import PolygonDrawer, CropTiffTool
from embrs.utilities.fire_util import RoadConstants as rc
from embrs.utilities.fire_util import FuelConstants as fc
from embrs.utilities.data_classes import MapParams, MapDrawerData, GeoInfo, LandscapeData

def generate_map_from_file(map_params: MapParams):
    """_summary_

    Args:
        map_params (MapParams): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    # Get user choice for importing roads
    import_roads = map_params.import_roads

    bounds = geotiff_to_numpy(map_params)

    map_params.geo_info = GeoInfo()
    map_params.geo_info.bounds = bounds
    map_params.geo_info.calc_center_coords()
    map_params.geo_info.calc_time_zone()
    
    if import_roads:
        # Transform bounds to OSM projection
        osm_bounds = transform_bounds(map_params.lcp_data.crs, "EPSG:4326", *bounds)

        # Fetch OSM data from API
        roads = fetch_osm_roads(osm_bounds) 

        # Project OSM data to our coordinate system
        projected_roads = project_osm_roads(roads, "EPSG:4326", map_params.lcp_data.crs)

        # Convert world coordinates to pixel coordinates
        raster_roads = []
        for road, road_type in projected_roads:
            raster_coords = [world_to_pixel(x, y, map_params.lcp_data.transform, map_params.lcp_data.rows) for x, y in road]
            raster_roads.append((raster_coords, road_type))

        # Only keep parts of roads within sim boundaries
        map_params.roads = trim_roads(map_params, raster_roads)
        
    else:
        map_params.roads = None

    # Create a color list in the right order
    colors = [fc.fuel_color_mapping[key] for key in sorted(fc.fuel_color_mapping.keys())]

    # Create a colormap from the list
    cmap = ListedColormap(colors)

    # Create a norm object to map your data points to the colormap
    norm = BoundaryNorm(list(sorted(fc.fuel_color_mapping.keys())) + [100], cmap.N)

    plt.imshow(np.flipud(map_params.lcp_data.fuel_map), cmap=cmap, norm=norm)

def trim_roads(map: MapParams, raster_roads: List) -> List:
    """_summary_

    Args:
        map (MapParams): _description_
        raster_roads (List): _description_

    Returns:
        List: _description_
    """
    trimmed_roads = []
    for road, road_type in raster_roads:
        x_trimmed, y_trimmed = [], []
        x, y = zip(*road)

        for i in range(len(x)):
            if 0 < x[i] < map.lcp_data.width_m and 0 < y[i] < map.lcp_data.height_m:
                x_trimmed.append(x[i])
                y_trimmed.append(y[i])

        if x_trimmed and y_trimmed:
            trimmed_roads.append(((x_trimmed, y_trimmed), road_type))

        plt.plot(x_trimmed, y_trimmed, color=rc.road_color_mapping[road_type])

    return trimmed_roads

def project_osm_roads(roads: List[Tuple[List[Tuple[float, float]], str]], src_crs: str, dst_crs: str) -> List[Tuple[List[Tuple[float, float]], str]]:
    """
    Projects OSM road coordinates from WGS84 (EPSG:4326) to the LCP raster CRS.

    Args:
        roads (List): List of roads with coordinates in (lon, lat) (EPSG:4326).
        src_crs (str): Source CRS (typically "EPSG:4326").
        dst_crs (str): Destination CRS (LCP file CRS).

    Returns:
        List[Tuple[List[Tuple[float, float]], str]]: Roads with reprojected coordinates.
    """
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    projected_roads = []
    for road, road_type in roads:
        projected_coords = [transformer.transform(lon, lat) for lon, lat in road]
        projected_roads.append((projected_coords, road_type))

    return projected_roads


def world_to_pixel(x: float, y: float, transform: rasterio.Affine, raster_height) -> Tuple[int, int]:
    """
    Converts world (real-world meters) coordinates to raster (pixel) coordinates.

    Args:
        x (float): World X coordinate.
        y (float): World Y coordinate.
        transform (Affine): Rasterio affine transformation matrix.

    Returns:
        (int, int): (col, row) in raster coordinates.
    """
    col, row = ~transform * (x, y)
    row = raster_height - int(round(row)) - 1  
    return int(round(col)), int(round(row))

def find_warping_angle(array: np.ndarray) -> float:
    """_summary_

    Args:
        array (np.ndarray): _description_

    Returns:
        float: _description_
    """
    # find top left corner
    corner_1 = None
    for i in range(array.shape[1]):
        for x in range(array.shape[0]):
            if array[i][x] != -100:
                corner_1 = (x, i)
                break

        if corner_1 is not None:
            break

    # find bottom left corner
    corner_2 = None
    for i in range(array.shape[0]):
        for y in range(array.shape[1]):
            if array[y][i] != -100:
                corner_2 = (i, y)
                break
        
        if corner_2 is not None:
            break

    if corner_2[1] == 0:
        angle = 0
    else:
        angle = np.arctan(corner_1[0]/corner_2[1])

    return angle

def crop_map_data(map_params: MapParams) -> float:
    """_summary_

    Args:
        file_params (str): _description_

    Returns:
        float: _description_
    """
    # Get bounding box from user input

    crop_done = False

    lcp_path = map_params.lcp_filepath
    output_path = os.path.join(map_params.folder, "cropped_lcp.tif")

    while not crop_done:

        fig = plt.figure(figsize=(15, 10))

        crop_tool = CropTiffTool(fig, lcp_path)
        plt.show()

        bounds = crop_tool.get_coords()
        angle = find_warping_angle(crop_tool.fuel_data)
        crop_done = crop_and_save_tiff(lcp_path, output_path, bounds)

    map_params.cropped_lcp_path = output_path
    map_params.north_angle_deg = np.rad2deg(angle)
    map_params.geo_info = GeoInfo()

def crop_and_save_tiff(input_path: str, output_path: str, bounds: list) -> int:
    """_summary_

    Args:
        input_path (str): _description_
        output_path (str): _description_
        bounds (list): _description_

    Returns:
        int: _description_
    """

    left, bottom = bounds[0]
    right, top = bounds[1]

    with rasterio.open(input_path) as src:
        nodata_value = -9999

        window = from_bounds(left, bottom, right, top, src.transform)

        cropped_data = src.read(window=window)

        if np.any(cropped_data == nodata_value):
            return False

        new_transform = src.window_transform(window)

        out_meta = src.meta.copy()
        out_meta.update({
            "height": cropped_data.shape[1],
            "width": cropped_data.shape[2],
            "transform": new_transform,
            "crs": src.crs
        })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(cropped_data)

    return True
    
def save_to_file(map_params: MapParams, user_data: MapDrawerData):
    """_summary_

    Args:
        params (dict): _description_
        user_data (dict): _description_
        north_dir (float): _description_
    """

    # Extract relevant data from params
    save_path = map_params.folder
    lcp_data = map_params.lcp_data
    roads = map_params.roads

    if map_params.geo_info is not None:
        bounds = map_params.geo_info.bounds
    else:
        bounds = None

    data = {}

    data['north_angle_deg'] = map_params.north_angle_deg

    if bounds is None:
        data['geo_info'] = None
        map_params.geo_info = None

    else:
        data['geo_info'] = {
            'south_lim': bounds[0],
            'north_lim': bounds[1],
            'west_lim': bounds[2],
            'east_lim': bounds[3],
            'center_lat': map_params.geo_info.center_lat,
            'center_lon': map_params.geo_info.center_lon,
            'timezone': map_params.geo_info.timezone
        }

    # Save numpy arrays to files for debugging
    elev_path = save_path + '/elev.npy'
    np.save(elev_path, lcp_data.elevation_map)

    aspect_path = save_path + '/aspect.npy'
    np.save(aspect_path, lcp_data.aspect_map)

    slope_path = save_path + '/slope.npy'
    np.save(slope_path, lcp_data.slope_map)

    fuel_path = save_path + '/fuel.npy'
    np.save(fuel_path, lcp_data.fuel_map)
    
    canopy_cover_path = save_path + '/canopy_cover.npy'
    np.save(canopy_cover_path, lcp_data.canopy_cover_map)
    
    canopy_height_path = save_path + '/canopy_height.npy'
    np.save(canopy_height_path, lcp_data.canopy_height_map)

    canopy_base_height_path = save_path + '/canopy_base_height.npy'
    np.save(canopy_base_height_path, lcp_data.canopy_base_height_map)

    canopy_bulk_density_path = save_path + '/canopy_bulk_density.npy'
    np.save(canopy_bulk_density_path, lcp_data.canopy_bulk_density_map)

    data['landscape_info'] = {
        'elev_file': elev_path,
        'aspect_file': aspect_path,
        'slope_file': slope_path,
        'fuel_file': fuel_path,
        'canopy_cover_file': canopy_cover_path,
        'canopy_height_file': canopy_base_height_path,
        'canopy_base_height_file': canopy_base_height_path,
        'canopy_bulk_density_file': canopy_bulk_density_path,
        'rows': lcp_data.rows,
        'cols': lcp_data.cols,
        'resolution': lcp_data.resolution,
        'width_m': lcp_data.width_m,
        'height_m': lcp_data.height_m
    }

    # Save the roads data
    road_path = save_path + '/roads.pkl'
    with open(road_path, 'wb') as f:
        pickle.dump(roads, f)

    data['roads'] = {'file': road_path}

    data['initial_igntion'] = [mapping(polygon) for polygon in user_data.initial_ign]
    data['fire_breaks'] = [{"geometry": mapping(line), "fuel_value": fuel_value} for line, fuel_value in zip(user_data.fire_breaks, user_data.fuel_vals)]

    map_params.scenario_data = user_data

    with open(save_path + "/map_params.pkl", 'wb') as f:
        pickle.dump(map_params, f)

    # Save data to JSON
    folder_name = os.path.basename(save_path)
    with open(save_path + "/" + folder_name + ".json", 'w') as f:
        json.dump(data, f, indent=4)

def fetch_osm_roads(bounds: Tuple[float, float, float, float]) -> List[Tuple[List[Tuple[float, float]], str]]:
    """
    Fetches road data from OpenStreetMap (OSM) Overpass API within the specified bounding box.
    
    Args:
        bounds (Tuple): (left, bottom, right, top) in WGS84 coordinates (EPSG:4326).

    Returns:
        List[Tuple[List[Tuple[float, float]], str]]: List of roads with coordinate tuples and road types.
    """
    left, bottom, right, top = bounds
    overpass_url = "http://overpass-api.de/api/interpreter"

    overpass_query = f"""
    [out:json];
    (way["highway"]
    ({bottom}, {left}, {top}, {right});
    );
    out body;
    >;
    out skel qt;
    """

    print(f"Querying OSM for road data in bounds: {bounds}")
    response = requests.get(overpass_url, params={'data': overpass_query})

    if response.status_code != 200:
        print(f"WARNING: Request failed with status {response.status_code}")
        return []

    osm_data = response.json()

    # Extract node coordinates
    nodes = {elem['id']: (elem['lon'], elem['lat']) for elem in osm_data['elements'] if elem['type'] == 'node'}

    # Extract roads
    roads = []
    for elem in osm_data['elements']:
        if elem['type'] == 'way' and 'highway' in elem['tags']:
            road_type = elem['tags']['highway']
            road_coords = [nodes[node_id] for node_id in elem['nodes'] if node_id in nodes]
            if road_coords and road_type in rc.major_road_types:
                roads.append((road_coords, road_type))

    print(f"Fetched {len(roads)} roads from OSM.")
    return roads
    

def resample_raster(array, crs, transform, target_resolution, method):
    """TODO: insert docstring

    Args:
        array (_type_): _description_
        crs (_type_): _description_
        transform (_type_): _description_
        target_resolution (_type_): _description_

    Returns:
        _type_: _description_
    """
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
        resampling=method
    )

    # Restore NoData values
    resampled_array[resampled_array == -9999] = np.nan 

    return resampled_array, new_transform

def geotiff_to_numpy(map_params: MapParams, fill_value: int =-9999):
    """_summary_

    Args:
        filepath (str): _description_
        fill_value (int, optional): _description_. Defaults to -9999.

    Returns:
        Tuple[np.ndarray, list]: _description_
    """
    with rasterio.open(map_params.cropped_lcp_path) as src:
        # Read the lcp file
        array = src.read() 
        transform = src.transform

        # Remove NoData rows/cols for all bands
        non_empty_rows = np.any(array[0] != fill_value, axis=1)
        non_empty_cols = np.any(array[0] != fill_value, axis=0)
        array = array[:, non_empty_rows][:, :, non_empty_cols]

        # Adjust transform for cropped data
        new_transform = transform * Affine.translation(
            non_empty_cols.argmax(), non_empty_rows.argmax()
        )

        # Separate processing for categorical and continuous data 
        resampled_bands = []

        for i in range(array.shape[0]):  # Iterate through all bands
            if i + 1 == 4: # Check if processing fuel
                # Use nearest-neighbor resampling for categorical data
                resampling_method = Resampling.nearest
            else:
                # Use bilinear resampling for continuous data
                resampling_method = Resampling.bilinear
            
            resampled_band, transform = resample_raster(array[i], src.crs, new_transform, 1, resampling_method)
            
            resampled_bands.append(resampled_band)

        resampled_array = np.stack(resampled_bands, axis=0)

    rows, cols = resampled_array.shape[1:]


    map_params.lcp_data = LandscapeData(
        elevation_map=resampled_array[0],
        slope_map=resampled_array[1],
        aspect_map=resampled_array[2],
        fuel_map=resampled_array[3],
        canopy_cover_map=resampled_array[4],
        canopy_height_map=resampled_array[5]/10, # Adjust for how LANDFIRE handles canopy height
        canopy_base_height_map=resampled_array[6]/10, # Adjust for how LANDFIRE handles canopy base height
        canopy_bulk_density_map=resampled_array[7]/100, # Adjust for how LANDFIRE handles canopy bulk density
        rows=rows,
        cols=cols,
        resolution=1,
        width_m=cols,
        height_m=rows,
        transform=transform,
        crs = src.crs

    )

    return src.bounds

def get_user_data(fig: matplotlib.figure.Figure) -> dict:
    """Function that generates GUI for user to specify initial ignitions and fire-breaks. Returns
    dictionary containing user inputs.

    :param fig: figure object to deploy GUI in
    :type fig: matplotlib.figure.Figure
    :return: dictionary with all user input from GUI
    :rtype: dict
    """
    user_data = {}

    # display map
    drawer = PolygonDrawer(fig)
    plt.show()

    if not drawer.valid:
        print("Incomplete data provided. Not writing data to file, terminating...")
        sys.exit(0)

    polygons = drawer.get_ignitions()
    shapely_polygons = get_shapely_polys(polygons)

    lines, fuel_vals = drawer.get_fire_breaks()
    line_strings = [LineString(line) for line in lines]

    user_data = MapDrawerData(
        fire_breaks = line_strings,
        fuel_vals = fuel_vals,
        initial_ign = shapely_polygons
    )

    return user_data

def get_shapely_polys(polygons: list) -> list:
    """Convert list of polygons as generated by map drawer to a list of Shapely polygons

    :param polygons: list of polygons each defined as [(x1,y1),(x2,y2)...]
    :type polygons: list
    :return: list of equivalent polygons as shapely.Polygon objects
    :rtype: list
    """
    shapely_polygons = [Polygon(coords) for coords in polygons]
    return shapely_polygons

def main():
    file_selector = MapGenFileSelector()
    map_params = file_selector.run()

    if map_params is None:
        print("User exited before submitting necessary files.")
        sys.exit(0)

    # Prompt user to crop their ROI
    crop_map_data(map_params)

    # Initialize figure for user data GUI
    fig = plt.figure(figsize=(15, 10))
    plt.tick_params(left = False, right = False, bottom = False, labelleft = False,
                    labelbottom = False)

    generate_map_from_file(map_params)
    user_data = get_user_data(fig)
    save_to_file(map_params, user_data)

if __name__ == "__main__":
    main()
