"""Module used to run the application that allows users to generate a new map file.
"""

import subprocess
from typing import Tuple
import xml.etree.ElementTree as ET
import json
import pickle
import sys
import os
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling as ResamplingMethod
from rasterio.transform import Affine
from rasterio.windows import from_bounds
import requests
import pyproj
import utm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import Polygon, LineString
from shapely.geometry import mapping
from scipy import ndimage, stats
import numpy as np
from pyproj import Transformer
from timezonefinder import TimezoneFinder

from embrs.utilities.file_io import MapGenFileSelector
from embrs.utilities.map_drawer import PolygonDrawer, CropTiffTool
from embrs.utilities.fire_util import RoadConstants as rc
from embrs.utilities.fire_util import FuelConstants as fc
from embrs.utilities.data_classes import MapParams, DataProductParams, MapDrawerData
from typing import cast

DATA_RESOLUTION = 30 # meters

def generate_map_from_file(map_params: MapParams, data_res: float, min_cell_size: float):
    """Generate a simulation map. Take in user's selections of fuel and elevation files
    along with the drawings they overlay for initial ignitions and fire-breaks and generate a
    usable map.

    :param file_params: Dictionary containing the input parameters/files to generate map from
    :type file_parmas: dict
    :param data_res: Resolution of the fuel and elevation data
    :type data_res: float
    :param min_cell_size: minimum cell size for this map, used for interpolation of elevation
                          and fuel data
    :type min_cell_size: float
    :raises ValueError: if elevation and fuel data selected are not from the same region
    """

    # Get file paths for all data files
    uniform_fuel = map_params.uniform_fuel
    uniform_elev = map_params.uniform_elev

    # Get user choice for importing roads
    import_roads = map_params.import_roads

    if not uniform_elev and not uniform_fuel:
        fuel_bounds = parse_fuel_data(map_params.fuel_data, import_roads)
        elev_bounds = geotiff_to_numpy(map_params.elev_data)
        geotiff_to_numpy(map_params.asp_data)
        geotiff_to_numpy(map_params.slp_data)

        # Ensure that the elevation and fuel data are from same region
        for e_bound, f_bound in zip(elev_bounds, fuel_bounds):
            if np.abs(e_bound - f_bound) > 1:
                raise ValueError('The elevation and fuel data are not from the same region')

        bounds = elev_bounds

        widths = [map_params.fuel_data.width_m, map_params.elev_data.width_m, map_params.asp_data.width_m, map_params.slp_data.width_m]
        heights = [map_params.fuel_data.height_m, map_params.elev_data.height_m, map_params.asp_data.height_m, map_params.slp_data.height_m]

        widths_equal = all(x == widths[0] for x in widths)
        heights_equal = all(x == heights[0] for x in heights)

        if not widths_equal or not heights_equal:
            # TODO: See if this ever comes up, if it does figure out how to handle it
            raise ValueError('Widths or heights of cropped data do not match')

    elif uniform_fuel and not uniform_elev: # TODO: Handle uniform cases
        pass

    elif not uniform_fuel and uniform_elev: # TODO: Handle uniform cases
        pass

    else: # Both fuel and elevation are uniform # TODO: Handle uniform cases
        pass

    if import_roads:
        print("Starting road data retrieval")
        # get road data
        metadata_path = map_params.metadata_path
        road_data, bounds = get_road_data(metadata_path)
        if road_data is not None:
            roads = parse_road_data(road_data, bounds, map_params.fuel_data)

        print("Finished road retrieval")
    else:
        roads = None

    map_params.roads = roads
    map_params.bounds = bounds # TODO: this may not be the best way to get bounds

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

    fuel_path = map_params.fuel_data.tiff_filepath

    while not crop_done:

        fig = plt.figure(figsize=(15, 10))

        crop_tool = CropTiffTool(fig, fuel_path)
        plt.show()

        bounds = crop_tool.get_coords()

        crop_done = crop_data_products(map_params, bounds)

    angle = find_warping_angle(crop_tool.data)

    return angle


def crop_data_products(map_params: MapParams, bounds: list) -> bool:
    """_summary_

    Args:
        file_params (dict): _description_
        bounds (list): _description_

    Returns:
        bool: _description_
    """

    # TODO: How should we handle uniform fuel and elevation case?
    output_dir = map_params.output_folder

    data_attrs = ['fuel_data', 'elev_data', 'asp_data', 'slp_data']
    cropped_filenames = ['fuel.tif', 'elevation.tif', 'aspect.tif', 'slope.tif']

    for i, attr in enumerate(data_attrs):
        data_product = getattr(map_params, attr)  # Get the DataProductParams instance

        output_path = os.path.join(output_dir, cropped_filenames[i])
        
        flag = crop_and_save_tiff(data_product.tiff_filepath, output_path, bounds)
        if flag == -1:
            return False  # Return early if cropping fails

        # Update the cropped filepath inside the DataProductParams instance
        data_product.cropped_filepath = cast(str, output_path)

    return True

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
        nodata_value = 32767

        window = from_bounds(left, bottom, right, top, src.transform)

        cropped_data = src.read(1, window=window)

        if np.any(cropped_data == nodata_value):
            return -1

        new_transform = src.window_transform(window)

        out_meta = src.meta.copy()

        out_meta.update({
            "height": cropped_data.shape[0],
            "width": cropped_data.shape[1],
            "transform": new_transform,
            "crs": src.crs
        })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(cropped_data, 1)

    try:
        subprocess.run(["gdal_edit.py", "-a_srs", "EPSG:5070", output_path], check=True)
        print(f"Successfully set CRS to EPSG:5070 for {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running gdal_edit.py: {e}")

    return 0
    
def save_to_file(map_params: MapParams, user_data: MapDrawerData, north_dir: float):
    """_summary_

    Args:
        params (dict): _description_
        user_data (dict): _description_
        north_dir (float): _description_
    """

    # Extract relevant data from params
    save_path = map_params.output_folder
    bounds = map_params.bounds
    fuel_data = map_params.fuel_data
    elev_data = map_params.elev_data    
    slope_data = map_params.slp_data
    aspect_data = map_params.asp_data
    roads = map_params.roads

    data = {}

    # Save fuel data
    fuel_path = save_path + '/fuel.npy'
    np.save(fuel_path, fuel_data.map)
    fuel_data.np_filepath = fuel_path

    data['north_angle_deg'] = np.rad2deg(north_dir)

    if bounds is None:
        data['geo_info'] = None

    else:
        # Manually set the correct EPSG code for NAD83 / Conus Albers
        epsg_code = "EPSG:5070"

        # Compute midpoint in projected coordinates
        mid_x = (bounds.left + bounds.right) / 2
        mid_y = (bounds.bottom + bounds.top) / 2

        # Define the transformation from raster CRS (NAD83 / Conus Albers) to WGS84 (EPSG:4326)
        transformer = Transformer.from_crs(epsg_code, "EPSG:4326", always_xy=True)

        # Transform the midpoint from projected coordinates to lat/lon
        lon, lat = transformer.transform(mid_x, mid_y)

        # Get the time zone at the location to sample
        tf = TimezoneFinder()
        timezone = tf.timezone_at(lng=lon, lat=lat)

        data['geo_info'] = {
            'south_lim': bounds[0],
            'north_lim': bounds[1],
            'west_lim': bounds[2],
            'east_lim': bounds[3],
            'center_lat': lat,
            'center_lon': lon,
            'timezone': timezone
        }

    data['fuel'] = {'file': fuel_path,
                        'width_m': fuel_data.width_m,
                        'height_m': fuel_data.height_m,
                        'rows': fuel_data.rows,
                        'cols': fuel_data.cols,
                        'resolution': fuel_data.resolution,
                        'uniform': fuel_data.uniform,
                        'tif_file_path': fuel_data.tiff_filepath
                    }
    
    if fuel_data.uniform:
        data['fuel']['fuel type'] = map_params.fuel_type

    # Save elevation data
    elev_path = save_path + '/elev.npy'
    np.save(elev_path, elev_data.map)
    elev_data.np_filepath = elev_path


    data['elevation'] = {'file': elev_path,
                        'width_m': elev_data.width_m,
                        'height_m': elev_data.height_m,
                        'rows': elev_data.rows,
                        'cols': elev_data.cols,
                        'resolution': elev_data.resolution,
                        'uniform': elev_data.uniform,
                        'tif_file_path': elev_data.tiff_filepath
                        }

    aspect_path = save_path + '/aspect.npy'
    np.save(aspect_path, aspect_data.map)
    aspect_data.np_filepath = aspect_data

    data['aspect'] = {'file': aspect_path,
                      'width_m': aspect_data.width_m,
                      'height_m': aspect_data.height_m,
                      'rows': aspect_data.rows,
                      'cols': aspect_data.cols,
                      'resolution': aspect_data.resolution,
                      'uniform': aspect_data.uniform,
                      'tif_file_path': aspect_data.tiff_filepath
                      }

    slope_path = save_path + '/slope.npy'
    np.save(slope_path, slope_data.map)
    slope_data.np_filepath = slope_path

    data['slope'] = {'file': slope_path,
                      'width_m': slope_data.width_m,
                      'height_m': slope_data.height_m,
                      'rows': slope_data.rows,
                      'cols': slope_data.cols,
                      'resolution': slope_data.resolution,
                      'uniform': slope_data.uniform,
                      'tif_file_path': slope_data.tiff_filepath
                      }

    # Save the roads data
    road_path = save_path + '/roads.pkl'
    with open(road_path, 'wb') as f:
        pickle.dump(roads, f)

    data['roads'] = {'file': road_path}

    data['initial_igntion'] = user_data.initial_ign
    data['fire_breaks'] = user_data.fire_breaks

    map_params.scenario_data = user_data

    with open(save_path + "/map_params.pkl", 'wb') as f:
        pickle.dump(map_params, f)

    # Save data to JSON
    folder_name = os.path.basename(save_path)
    with open(save_path + "/" + folder_name + ".json", 'w') as f:
        json.dump(data, f, indent=4)

def get_road_data(path:str) -> Tuple[dict, list]:
    """Function that queries the openStreetMap API for the road data at the same region as the
    elevation and fuel maps
    
    :param path: path to the metadata file used to find the geographic region to pull from OSM API
    :type path: str
    :return: tuple with dictionary of road data and a list of the geobounds
    :rtype: Tuple[dict, list]
    """
    # TODO: Do this without metadata file
    
    # Load the xml file
    tree = ET.parse(path)
    root = tree.getroot()
    
    west_bounding = float(root.find(".//westbc").text)
    east_bounding = float(root.find(".//eastbc").text)
    north_bounding = float(root.find(".//northbc").text)
    south_bounding = float(root.find(".//southbc").text)

    overpass_url = "http://overpass-api.de/api/interpreter"

    overpass_query = f"""
    [out:json];
    (way["highway"]
    ({south_bounding}, {west_bounding}, {north_bounding}, {east_bounding});
    );
    out body;
    >;
    out skel qt;
    """

    print(f"Querying OSM for road data at: [west: {west_bounding}, north: {north_bounding}," +
          f"east: {east_bounding}, south: {south_bounding}]")

    print("Awaiting response...")

    response = requests.get(overpass_url, params={'data': overpass_query})

    if response.status_code != 200:
        print(f"WARNING: Request failed with status {response.status_code}," +
              "proceeding without road data")

        print(response.text)
        return None

    road_data = response.json()

    print("Road data retrieved successfully!")

    return road_data, [south_bounding, north_bounding, west_bounding, east_bounding]

def parse_road_data(road_data: dict, bounds: list, data: dict) -> list:
    """Function that takes the raw road data, trims it to fit with the map and interpolates it to
    decrease the spacing between points on roads.

    :param road_data: dictionary containing raw road data from openStreetMap
    :type road_data: dict
    :param bounds: geographic bounds of the region the data is from
    :type bounds: list
    :param data: dictionary containing data about the fuel or elevation map
    :type data: dict
    :return: list of roads in the form of [((x,y), road type)]
    :rtype: list
    """
    # Extract the bounding coordinates from the data dictionary
    bbox = {
        'south': bounds[0],
        'north': bounds[1],
        'west': bounds[2],
        'east': bounds[3] 
    }

    # Calculate the central point of the bounding box
    central_lat = (bbox['south'] + bbox['north']) / 2
    central_lon = (bbox['west'] + bbox['east']) / 2

    # Get the UTM zone of the central point
    _, _, utm_zone, _ = utm.from_latlon(central_lat, central_lon)

    # Get projection based on the utm_zone
    proj = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')

    # A dict to hold node coordinates
    nodes = {}

    # Get the origin in x,y coordinates
    origin_x, origin_y = proj(bbox['west'], bbox['south'])

    # First pass: get all node elements with their coordinates
    for element in road_data['elements']:
        if element['type'] == 'node':
            node_id = element['id']
            lat = element['lat']
            lon = element['lon']
            x, y = proj(lon, lat)

            # account for buffer on other data
            x -= 120
            y -= 120

            nodes[node_id] = (x - origin_x, y - origin_y)

    # Second pass: get all way elements that represent major roads
    roads = []
    for element in road_data['elements']:
        if element['type'] == 'way' and 'highway' in element['tags']:
            road_type = element['tags']['highway']
            node_ids = element['nodes']

            if road_type in rc.major_road_types:
                road = [nodes[node_id] for node_id in node_ids if node_id in nodes]
                if road:
                    roads.append((road, road_type))

    # Interpolate points so that each point in roads is at most 0.5m apart
    roads = interpolate_points(roads, 0.5)

    for road in roads:
        x_trimmed = []
        y_trimmed = []

        x, y = zip(*road[0])
        for i in range(len(x)):
            if 0 < x[i]/30 < data['cols']-1 and 0 < y[i]/30 < data['rows']-1:
                x_trimmed.append(x[i]/30)
                y_trimmed.append(y[i]/30)

        x = tuple(xi for xi in x_trimmed)
        y = tuple(yi for yi in y_trimmed)

        plt.plot(x, y, color=rc.road_color_mapping[road[1]])

    return roads

def interpolate_points(roads: list, max_spacing_m: float) -> list:
    """Interpolate points along a road so that every consecutive pair of points is less than a
    certain distance apart

    :param roads: list containing road data in the form [((x,y), road type)]
    :type roads: list
    :param max_spacing_m: maximum distance in meters two consecutive points can be apart
    :type max_spacing_m: float
    :return: list in the same form as 'roads' but with new interpolated points added in
    :rtype: list
    """
    interpolated_roads = []
    for road in roads:
        interpolated_road = []
        for i in range(len(road[0]) - 1):
            start = road[0][i]
            end = road[0][i + 1]

            # Calculate the distance between the two points
            dist_m = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)

            # If the distance is greater than max_spacing_m, interpolate points
            if dist_m > max_spacing_m:
                # Calculate the number of points to interpolate
                num_points = int(np.ceil(dist_m / max_spacing_m))

                # Interpolate the x and y coordinates
                x = np.linspace(start[0], end[0], num_points)
                y = np.linspace(start[1], end[1], num_points)

                # Add the interpolated points to the new road
                interpolated_road.extend(list(zip(x, y)))
            else:
                # If no interpolation is needed, just add the start point
                interpolated_road.append(start)

        # Add the last point of the road
        interpolated_road.append(road[0][-1])

        # Add the new road to the list of roads
        interpolated_roads.append((interpolated_road, road[1]))

    return interpolated_roads

def resample_raster(array, crs, transform, target_resolution):
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

    print(f"Resampling: Original Shape = {array.shape}, New Shape = ({new_height}, {new_width})")
    
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
        resampling=ResamplingMethod.bilinear
    )

    # Restore NoData values
    resampled_array[resampled_array == -9999] = np.nan 

    return resampled_array, new_transform

def geotiff_to_numpy(data_product: DataProductParams, fill_value: int =-9999) -> Tuple[DataProductParams, list]:
    """_summary_

    Args:
        filepath (str): _description_
        fill_value (int, optional): _description_. Defaults to -9999.

    Returns:
        Tuple[np.ndarray, list]: _description_
    """
    with rasterio.open(data_product.cropped_filepath) as src:
        # Read the data and metadata
        array = src.read(1)  # Read the first band
        transform = src.transform
        
        # Align to the axes (optional: crop to remove all NoData rows/columns)
        non_empty_rows = np.any(array != fill_value, axis=1)
        non_empty_cols = np.any(array != fill_value, axis=0)
        array = array[non_empty_rows][:, non_empty_cols]
        
        # Adjust the transform for the cropped array
        new_transform = transform * Affine.translation(
            non_empty_cols.argmax(), non_empty_rows.argmax()
        )

    resampled_array, transform = resample_raster(array, src.crs, new_transform, 1)

    rows, cols = resampled_array.shape

    width_m = cols
    height_m = rows

    resolution = 1

    data_product.width_m = width_m
    data_product.height_m = height_m
    data_product.rows = rows
    data_product.cols = cols
    data_product.resolution = resolution
    data_product.map = resampled_array
    data_product.uniform = False

    return src.bounds

def create_uniform_elev_map(rows: int, cols: int) -> DataProductParams:
    """Create an elevation map for the uniform case (all elevation set to 0)

    :param rows: number of rows to populate data for
    :type rows: int
    :param cols: number of columns to populate data for
    :type cols: int
    :return: dictionary containing relevant elevation data for uniform case
    :rtype: dict
    """
    elevation_map = np.full((rows, cols), 0)

    output = DataProductParams(
        width_m = cols,
        height_m = rows,
        rows = rows,
        cols = cols,
        resolution = 1,
        map = elevation_map,
        uniform = True,
        tiff_filepath = ""
    )

    return output

def parse_fuel_data(data_product: DataProductParams, import_roads: bool) -> Tuple[DataProductParams, list]:
    """Read fuel data file, rotate and buffer, outputs dictionary with relevant data

    :param fuel_path: path to the raw fuel data file
    :type fuel_path: str
    :param import_roads: boolean to indicate whether roads will be imported, if True function will
                         replace 'Urban' fuel type in raw data with nearby fuels
    :type import_roads: bool
    :raises ValueError: if fuel data contains values that are not one of the 13 Anderson FBFMs
    :return: dictionary with all relevant fuel data
    :rtype: dict
    """

    with rasterio.open(data_product.cropped_filepath) as src:
        # Read the data and metadata
        fuel_map = src.read(1)  # Read the first band

    if import_roads:
        fuel_map = replace_clusters(fuel_map)

    for i in range(fuel_map.shape[0]):
        for j in range(fuel_map.shape[1]):
            if fuel_map[i, j] not in fc.fbfm_13_keys:
                raise ValueError("One or more of the fuel values not valid for FBFM13")

    width_m = fuel_map.shape[1] * DATA_RESOLUTION
    height_m = fuel_map.shape[0] * DATA_RESOLUTION

    rows = fuel_map.shape[0]
    cols = fuel_map.shape[1]

    # Create a color list in the right order
    colors = [fc.fuel_color_mapping[key] for key in sorted(fc.fuel_color_mapping.keys())]

    # Create a colormap from the list
    cmap = ListedColormap(colors)

    # Create a norm object to map your data points to the colormap
    norm = BoundaryNorm(list(sorted(fc.fuel_color_mapping.keys())) + [100], cmap.N)

    plt.imshow(np.flipud(fuel_map), cmap=cmap, norm=norm)

    resolution = DATA_RESOLUTION
    
    data_product.width_m = width_m
    data_product.height_m = height_m
    data_product.rows = rows
    data_product.cols = cols
    data_product.resolution = resolution
    data_product.map = fuel_map
    data_product.uniform = False

    return src.bounds

def create_uniform_fuel_map(height_m: float, width_m: float, fuel_type: fc.fbfm_13_keys) -> DataProductParams:
    """Create a fuel map for the uniform case (all fuel set to same value)

    :param height_m: height in meters of the map that should be generated
    :type height_m: float
    :param width_m: width in meters of the map that should be generated
    :type width_m: float
    :param fuel_type: fuel type that should be applied across the entire map
    :type fuel_type: FuelConstants.fbfm_13_keys
    :return: dictionary containing all the necessary data for the uniform fuel case
    :rtype: dict
    """
    fuel_map = np.full((int(np.floor(height_m/DATA_RESOLUTION)),
                        int(np.floor(width_m/DATA_RESOLUTION))),
                        int(fuel_type))

    # Create a color list in the right order
    colors = [fc.fuel_color_mapping[key] for key in sorted(fc.fuel_color_mapping.keys())]

    # Create a colormap from the list
    cmap = ListedColormap(colors)

    # Create a norm object to map your data points to the colormap
    norm = BoundaryNorm(list(sorted(fc.fuel_color_mapping.keys())) + [100], cmap.N)

    plt.imshow(fuel_map, cmap=cmap, norm=norm)
    
    output = DataProductParams(
        width_m = width_m,
        height_m = height_m,
        rows = int(np.floor(height_m/DATA_RESOLUTION)),
        cols = int(np.floor(width_m/DATA_RESOLUTION)),
        resolution = DATA_RESOLUTION,
        map = fuel_map,
        uniform = False,
        tiff_filepath = ""
    )

    return output

def replace_clusters(fuel_map: np.ndarray, invalid_value=91) -> np.ndarray:
    """Function to replace clusters of invalid values with their neighboring values

    :param fuel_map: 2d array containing fuel data
    :type fuel_map: np.ndarray
    :param invalid_value: value to be replaced, defaults to 91 ('Urban')
    :type invalid_value: int, optional
    :return: 2d array containing fuel data with 'invalid_value' replaced
    :rtype: np.ndarray
    """
    # Check if fuel_map contains any invalid_value
    if invalid_value not in fuel_map:
        return fuel_map

    # Generate a mask for the invalid cells
    invalid_mask = fuel_map == invalid_value

    # Label each cluster of invalid cells
    labels, num_labels = ndimage.label(invalid_mask)

    # Create a dilation structuring element (SE)
    selem = ndimage.generate_binary_structure(2,2)  # 2x2 square SE

    # Initialize an output array to store the final fuel_map
    output_map = fuel_map.copy()

    # Process each label (cluster of invalid cells)
    for label in range(1, num_labels + 1):
        # Create a mask for this label only
        label_mask = labels == label

        # Dilation operation: for this label, expand it to its neighbors
        dilated_mask = ndimage.binary_dilation(label_mask, structure=selem)

        # Exclude the original label cells from the dilated mask
        outer_ring_mask = np.logical_and(dilated_mask, np.logical_not(label_mask))

        # Extract the values of the cells in the outer ring
        outer_ring_values = fuel_map[outer_ring_mask]

        # Exclude any invalid values in the outer ring
        outer_ring_values = outer_ring_values[outer_ring_values != invalid_value]

        # Find the most common value
        if len(outer_ring_values) > 0:
            most_common = stats.mode(outer_ring_values, keepdims=True)[0][0]
        else:
            most_common = invalid_value

        # Replace the label cells in the output_map with the most common value
        output_map[label_mask] = most_common

    return output_map

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
    transformed_polygons = transform_polygons(polygons)
    shapely_polygons = get_shapely_polys(transformed_polygons)

    polygons = [mapping(polygon) for polygon in shapely_polygons]

    lines, fuel_vals = drawer.get_fire_breaks()

    transformed_lines = transform_lines(lines, DATA_RESOLUTION)

    line_strings = [LineString(line) for line in transformed_lines]

    fire_breaks = [{"geometry": mapping(line), "fuel_value": fuel_value} for line, fuel_value in zip(line_strings, fuel_vals)]

    user_data = MapDrawerData(
        fire_breaks = fire_breaks,
        initial_ign = polygons
    )

    return user_data

def transform_polygons(polygons: list) -> list:
    """Function to transform polygons to the proper scale for the sim map

    :param polygons: list of polygons drawn by user
    :type polygons: list

    :return: list of polygons scaled up appropriately
    :rtype: list
    """
    transformed_polygons = []
    for polygon in polygons:

        transformed_polygon = [(x*DATA_RESOLUTION, y*DATA_RESOLUTION) for x,y in polygon]
        transformed_polygons.append(transformed_polygon)

    return transformed_polygons

def transform_lines(line_segments: list, scale_factor: float) -> list:
    """Function to transform lines to the proper scale for the sim map

    :param line_segments: list of lines drawn by user
    :type line_segments: list
    :param scale_factor: data resolution shown in drawing GUI, used to scale the line up
                         so they reflect the actually distances they were representing
    :type scale_factor: float
    :return: list of lines scaled up appropriately
    :rtype: list
    """
    transformed_lines = []
    for line in line_segments:
        transformed_line = []
        for pt in line:
            transformed_pt = [pt[0] * scale_factor, pt[1] * scale_factor]
            transformed_line.append(transformed_pt)

        transformed_line = remove_consec_duplicates(transformed_line)

        transformed_lines.append(transformed_line)

    return transformed_lines

def remove_consec_duplicates(line: list) -> list:
    """Function to remove consecutive duplicate points in a line

    :param line: list of (x,y) points that make up a line
    :type line: list
    :return: list of (x,y) points that make up a line with any duplicates removed
    :rtype: list
    """
    cleaned_line = [line[0]]  # start with the first point
    for point in line[1:]:  # for each subsequent point
        if point != cleaned_line[-1]:  # if it's not the same as the last point we added
            cleaned_line.append(point)  # add it to the list
    return cleaned_line

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


    # file_params = {'Output Map Folder': '/Users/rui/Documents/Research/Code/embrs_maps/denver_demo',
    #                'Metadata Path': '', 'Import Roads': False, 'Uniform Fuel': False,
    #                'Fuel Map Path': '/Users/rui/Documents/Research/Code/embrs_raw_data/west_of_denver/LF2023_FBFM13_240_CONUS/LC23_F13_240.tif',
    #                'Uniform Elev': False, 'elevation Map Path': '/Users/rui/Documents/Research/Code/embrs_raw_data/west_of_denver/LF2020_Elev_220_CONUS/LC20_Elev_220.tif',
    #                'Aspect Map Path': '/Users/rui/Documents/Research/Code/embrs_raw_data/west_of_denver/LF2020_Asp_220_CONUS/LC20_Asp_220.tif',
    #                'Slope Map Path': '/Users/rui/Documents/Research/Code/embrs_raw_data/west_of_denver/LF2020_SlpD_220_CONUS/LC20_SlpD_220.tif'}
    

    north_dir = crop_map_data(map_params)

    # Initialize figure for user data GUI
    fig = plt.figure(figsize=(15, 10))
    plt.tick_params(left = False, right = False, bottom = False, labelleft = False,
                    labelbottom = False)

    generate_map_from_file(map_params, DATA_RESOLUTION, 1)
    user_data = get_user_data(fig)
    save_to_file(map_params, user_data, north_dir)

if __name__ == "__main__":
    main()
