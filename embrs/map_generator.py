"""Module used to run the application that allows users to generate a new map file.
"""

from typing import Tuple
import xml.etree.ElementTree as ET
import json
import pickle
import sys
import os
import rasterio
import requests
import pyproj
import utm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import Polygon, LineString
from shapely.geometry import mapping
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage, stats
import numpy as np

from embrs.utilities.file_io import MapGenFileSelector
from embrs.utilities.map_drawer import PolygonDrawer
from embrs.utilities.fire_util import RoadConstants as rc
from embrs.utilities.fire_util import FuelConstants as fc

DATA_RESOLUTION = 30 # meters

def generate_map_from_file(file_params: dict, data_res: float, min_cell_size: float):
    """Generate a simulation map. Take in user's selections of fuel and topography files
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

    print(file_params)

    # Get file paths for all data files
    save_path = file_params['Output Map Folder']
    top_path  = file_params['Topography Map Path']
    fuel_path = file_params['Fuel Map Path']
    asp_path = file_params['Aspect Map Path']
    slope_path = file_params['Slope Map Path']

    uniform_fuel = file_params["Uniform Fuel"]
    uniform_elev = file_params["Uniform Elev"]

    # Get user choice for importing roads
    import_roads = file_params['Import Roads']

    if not uniform_elev and not uniform_fuel:
        print("Starting fuel parsing")
        fuel_data, fuel_bounds = parse_fuel_data(fuel_path, import_roads)
        fuel_data['tif_file_path'] = fuel_path
        print("Finished fuel parsing")
        topography_data, elev_bounds = parse_elevation_data(top_path, data_res, min_cell_size)
        topography_data['tif_file_path'] = top_path
        print("Finished topography parsing")
        aspect_data, _ = parse_elevation_data(asp_path, data_res, min_cell_size)
        aspect_data['tif_file_path'] = asp_path
        print("Finished aspect parsing")
        slope_data, _ = parse_elevation_data(slope_path, data_res, min_cell_size)
        slope_data['tif_file_path'] = slope_path

        print(f"Fuel data shape: {fuel_data['map'].shape}")
        print(f"Topography data shape: {topography_data['map'].shape}")
        print(f"Aspect data shape: {aspect_data['map'].shape}")
        print(f"Slope data shape: {slope_data['map'].shape}")

        # Ensure that the elevation and fuel data are from same region
        for e_bound, f_bound in zip(elev_bounds, fuel_bounds):
            if np.abs(e_bound - f_bound) > 1:
                raise ValueError('The elevation and fuel data are not from the same region')

        bounds = elev_bounds


        widths = [fuel_data['width_m'], topography_data['width_m'], aspect_data['width_m'], slope_data['width_m']]
        heights = [fuel_data['height_m'], topography_data['height_m'], aspect_data['height_m'], slope_data['height_m']]

        widths_equal = all(x == widths[0] for x in widths)
        heights_equal = all(x == heights[0] for x in heights)

        if not  widths_equal or not heights_equal:
            # set bounds to the smallest of each
            width_m = np.min(widths)
            height_m = np.min(heights)

            fuel_data['width_m'] = int(width_m)
            topography_data['width_m'] = int(width_m)
            aspect_data['width_m'] = int(width_m)
            slope_data['width_m'] = int(width_m)
            fuel_data['height_m'] = int(height_m)
            topography_data['height_m'] = int(height_m)
            aspect_data['height_m'] = int(height_m)
            slope_data['height_m'] = int(height_m)

    elif uniform_fuel and not uniform_elev:
        # parse elevation map first
        topography_data, elev_bounds = parse_elevation_data(top_path, data_res, min_cell_size)
        aspect_data, _ = parse_elevation_data(asp_path, data_res, min_cell_size)
        slope_data, _ = parse_elevation_data(slope_path, data_res, min_cell_size)
        bounds = elev_bounds

        # create uniform fuel map based on dimensions
        rows, cols = topography_data["rows"], topography_data["cols"]
        fuel_type = file_params["Fuel Type"]
        fuel_data = create_uniform_fuel_map(rows, cols, fuel_type)

    elif not uniform_fuel and uniform_elev:
        # parse fuel map first
        fuel_data, fuel_bounds = parse_fuel_data(fuel_path, import_roads)
        bounds = fuel_bounds

        # create uniform elevation map based on dimensions
        rows, cols = fuel_data["height_m"], fuel_data["width_m"]
        topography_data = create_uniform_elev_map(rows, cols)
        aspect_data = create_uniform_elev_map(rows, cols)
        slope_data = create_uniform_elev_map(rows, cols)
    else:
        # get sim size from parameters
        rows, cols = file_params["height m"], file_params["width m"]
        fuel_type = file_params["Fuel Type"]

        # create uniform elevation and fuel maps based on sim size
        topography_data = create_uniform_elev_map(rows, cols)
        aspect_data = create_uniform_elev_map(rows, cols)
        slope_data = create_uniform_elev_map(rows, cols)
        fuel_data = create_uniform_fuel_map(rows, cols, fuel_type)
        bounds = None

    if import_roads:
        print("Starting road data retrieval")
        # get road data
        metadata_path = file_params['Metadata Path']
        road_data, bounds = get_road_data(metadata_path)
        if road_data is not None:
            roads = parse_road_data(road_data, bounds, fuel_data)

        print("Finished road retrieval")
    else:
        roads = None

    return {
        'save_path': save_path,
        'fuel_data': fuel_data,
        'topography_data': topography_data,
        'aspect_data': aspect_data,
        'slope_data': slope_data,
        'roads': roads,
        'bounds': bounds
    }

    # Get user input from GUI
    user_data = get_user_data(fig)

    save_to_file(save_path, fuel_data, topography_data, roads, user_data, bounds)

def save_to_file(save_path: str, fuel_data: dict, topography_data: dict, aspect_data: dict, slope_data: dict,
                roads: list, user_data: dict, bounds: list):
    """Save all generated map data to a file specified in 'save_path'

    :param save_path: Path to save the map files in
    :type save_path: str
    :param fuel_data: Dictionary containing all data relevant to the fuel map
    :type fuel_data: dict
    :param topography_data: Dictionary containing all data relevant to the topography map
    :type topography_data: dict
    :param roads: list of points considered roads each element formatted as ((x, y), road_type)
    :type roads: list
    :param user_data: Dictionary containing the data generated by user in the drawing process
    :type user_data: dict
    :param bounds: list containing the geographic bounds of the map, corner coordinates listed in
    following order:[south, north, west, east]
    :type bounds: list
    """
    data = {}

    # Save fuel data
    fuel_path = save_path + '/fuel.npy'

    np.save(fuel_path, fuel_data['map'])

    if bounds is None:
        data['geo_bounds'] = None

    else:
        data['geo_bounds'] = {
            'south': bounds[0],
            'north': bounds[1],
            'west': bounds[2],
            'east': bounds[3] 
        }

    data['fuel'] = {'file': fuel_path,
                        'width_m': fuel_data['width_m'],
                        'height_m': fuel_data['height_m'],
                        'rows': fuel_data['rows'],
                        'cols': fuel_data['cols'],
                        'resolution': fuel_data['resolution'],
                        'uniform': fuel_data['uniform'],
                        'tif_file_path': fuel_data['tif_file_path']
                    }
    
    if fuel_data['uniform']:
        data['fuel']['fuel type'] = fuel_data['fuel type']

    # Save topography data
    topography_path = save_path + '/topography.npy'
    np.save(topography_path, topography_data['map'])

    data['topography'] = {'file': topography_path,
                        'width_m': topography_data['width_m'],
                        'height_m': topography_data['height_m'],
                        'rows': topography_data['rows'],
                        'cols': topography_data['cols'],
                        'resolution': topography_data['resolution'],
                        'uniform': topography_data['uniform'],
                        'tif_file_path': topography_data['tif_file_path']
                        }

    aspect_path = save_path + '/aspect.npy'
    np.save(aspect_path, aspect_data['map'])

    data['aspect'] = {'file': aspect_path,
                      'width_m': aspect_data['width_m'],
                      'height_m': aspect_data['height_m'],
                      'rows': aspect_data['rows'],
                      'cols': aspect_data['cols'],
                      'resolution': aspect_data['resolution'],
                      'uniform': aspect_data['uniform'],
                      'tif_file_path': aspect_data['tif_file_path']
                      }

    slope_path = save_path + '/aspect.npy'
    np.save(slope_path, slope_data['map'])

    data['slope'] = {'file': slope_path,
                      'width_m': slope_data['width_m'],
                      'height_m': slope_data['height_m'],
                      'rows': slope_data['rows'],
                      'cols': slope_data['cols'],
                      'resolution': slope_data['resolution'],
                      'uniform': slope_data['uniform'],
                      'tif_file_path': slope_data['tif_file_path']}

    # Save the roads data
    road_path = save_path + '/roads.pkl'
    with open(road_path, 'wb') as f:
        pickle.dump(roads, f)

    data['roads'] = {'file': road_path}

    data = data | user_data

    print(data)

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


def parse_elevation_data(top_path: str, data_res: float, cell_size: float) -> dict:
    """Read elevation data file, rotate and buffer, and interpolate so it can be used for smaller
    cell sizes.

    :param top_path: path to the elevation data file
    :type top_path: str
    :param data_res: resolution of the raw elevation data
    :type data_res: float
    :param cell_size: size the elevation data will interpolate to (new resolution)
    :type cell_size: float
    :return: dictionary containing all relevant topography data
    :rtype: dict
    """
    with rasterio.open(top_path) as elev_data:
        # read data
        array = elev_data.read(1)

    # Take out no data values
    no_data_value = 32767 #elev_data.nodatavals[0]

    print(f"shape before rotate and buffer: {array.shape}")

    topography_map = rotate_and_buffer_data(array, no_data_value)
    
    print(f"shape after rotate and buffer: {topography_map.shape}")


    width_m = topography_map.shape[1] * data_res
    height_m = topography_map.shape[0] * data_res

    x = np.arange(0, topography_map.shape[1])
    y = np.arange(0, topography_map.shape[0])
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, np.flipud(topography_map), colors='k')

    # upscale grid
    scale_factor = int(np.floor(data_res/cell_size))

    x = np.arange(topography_map.shape[1])
    y = np.arange(topography_map.shape[0])

    # construct the interpolator object
    interpolator = RectBivariateSpline(y, x, topography_map)

    xnew = np.linspace(0, topography_map.shape[1]-1, topography_map.shape[1]*scale_factor)
    ynew = np.linspace(0, topography_map.shape[0]-1, topography_map.shape[0]*scale_factor)

    # apply the interpolator
    topography_map_new = interpolator(ynew, xnew)
    topography_map_new = topography_map_new.reshape((topography_map.shape[0]*scale_factor,
                                                     topography_map.shape[1]*scale_factor))

    rows = topography_map_new.shape[0]
    cols = topography_map_new.shape[1]
    resolution = width_m / cols

    topography_output = {'width_m': width_m,
                        'height_m': height_m,
                        'rows': rows,
                        'cols': cols,
                        'resolution': resolution,
                        'map': topography_map_new,
                        'uniform': False
                        }

    return topography_output, elev_data.bounds

def create_uniform_elev_map(rows: int, cols: int) -> dict:
    """Create an elevation map for the uniform case (all elevation set to 0)

    :param rows: number of rows to populate data for
    :type rows: int
    :param cols: number of columns to populate data for
    :type cols: int
    :return: dictionary containing relevant topography data for uniform case
    :rtype: dict
    """
    topography_map = np.full((rows, cols), 0)

    topography_output = {'width_m': cols,
                    'height_m': rows,
                    'rows': rows,
                    'cols': cols,
                    'resolution': 1,
                    'map': topography_map,
                    'uniform': True
                    }

    return topography_output

def parse_fuel_data(fuel_path: str, import_roads: bool) -> dict:
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

    with rasterio.open(fuel_path) as fuel_data:
        array = fuel_data.read(1)
    no_data_value = fuel_data.nodatavals[0]

    fuel_map = rotate_and_buffer_data(array, no_data_value, order=0)

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

    fuel_output = {
                    'width_m': width_m,
                    'height_m': height_m,
                    'rows': rows,
                    'cols': cols,
                    'resolution': resolution,
                    'map': fuel_map,
                    'uniform': False
                }

    return fuel_output, fuel_data.bounds

def create_uniform_fuel_map(height_m: float, width_m: float, fuel_type: fc.fbfm_13_keys) -> dict:
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

    fuel_output = {
                    'width_m': width_m,
                    'height_m': height_m,
                    'rows': int(np.floor(height_m/DATA_RESOLUTION)),
                    'cols': int(np.floor(width_m/DATA_RESOLUTION)),
                    'resolution': DATA_RESOLUTION,
                    'map': fuel_map,
                    'uniform': True,
                    'fuel type': fc.fuel_names[fuel_type]
                }

    return fuel_output

def rotate_and_buffer_data(array: np.ndarray, no_data_value: int, order = 3) -> np.ndarray:
    """Rotates and buffers raw .tif fuel and elevation files so they are squares and contain no
    garbage data

    :param array: array of data to be cleaned
    :type array: np.ndarray
    :param no_data_value: value that represents no data within the array
    :type no_data_value: int
    :param order: order of interpolation for ndimage.rotate, order=0 does not interpolate, defaults
                  to 3
    :type order: int, optional
    :return: cleaned array
    :rtype: np.ndarray
    """
    array = np.where(array == no_data_value, -100, array)

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
        angle = np.arctan(corner_1[0]/corner_2[1]) * (180/np.pi)

    # rotate data to align it with x, y axes
    rotated_data = ndimage.rotate(array, angle, order=order)

    # Get the indices of valid elevation values
    valid_indices = np.where((rotated_data > 0))

    # Get rough min and max valid indices
    min_row = np.min(valid_indices[0])
    max_row = np.max(valid_indices[0])
    min_col = np.min(valid_indices[1])
    max_col = np.max(valid_indices[1])

    # Trim data
    new_array = rotated_data[min_row:max_row, min_col:max_col]

    # buffer 10 values on each side to remove any garbage data
    new_array = new_array[10:-10, 10:-10]

    return new_array

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

    user_data["initial_ignition"] = polygons

    lines, fuel_vals = drawer.get_fire_breaks()

    transformed_lines = transform_lines(lines, DATA_RESOLUTION)

    line_strings = [LineString(line) for line in transformed_lines]

    user_data["fire_breaks"] = [{"geometry": mapping(line), "fuel_value": fuel_value} for line, fuel_value in zip(line_strings, fuel_vals)]

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
    file_params = file_selector.run()

    if file_params is None:
        print("User exited before submitting necessary files.")
        sys.exit(0)

    # Initialize figure for GUI
    fig = plt.figure(figsize=(15, 10))
    plt.tick_params(left = False, right = False, bottom = False, labelleft = False,
                    labelbottom = False)

    # file_params = {'Output Map Folder': '/Users/rjdp3/Documents/Research/Code/full-stack-burnout/real_fire_eval_scenarios/soberanes_fire/map', 'Metadata Path': '', 'Import Roads': False, 'Uniform Fuel': False, 'Fuel Map Path': '/Users/rjdp3/Documents/Research/Code/full-stack-burnout/real_fire_eval_scenarios/soberanes_fire/raw_data_new/LF2019_FBFM13_200_CONUS/LC19_F13_200.tif', 'Uniform Elev': False, 'Topography Map Path': '/Users/rjdp3/Documents/Research/Code/full-stack-burnout/real_fire_eval_scenarios/soberanes_fire/raw_data_new/LF2020_Elev_220_CONUS/LC20_Elev_220.tif', 'Aspect Map Path': '/Users/rjdp3/Documents/Research/Code/full-stack-burnout/real_fire_eval_scenarios/soberanes_fire/raw_data_new/LF2020_Asp_220_CONUS/LC20_Asp_220.tif', 'Slope Map Path': '/Users/rjdp3/Documents/Research/Code/full-stack-burnout/real_fire_eval_scenarios/soberanes_fire/raw_data_new/LF2020_SlpD_220_CONUS/LC20_SlpD_220.tif'}

    params = generate_map_from_file(file_params, DATA_RESOLUTION, 1)
    user_data = get_user_data(fig)
    save_to_file(params['save_path'], params['fuel_data'], params['topography_data'], params['aspect_data'], params['slope_data'], params['roads'], user_data, params['bounds'])


if __name__ == "__main__":
    main()
