import sys
import matplotlib.pyplot as plt
import numpy as np

from embrs.utilities.file_io import UniformMapCreator
from embrs.utilities.data_classes import MapParams, GeoInfo, LandscapeData

from matplotlib.colors import ListedColormap, BoundaryNorm

from embrs.utilities.fire_util import FuelConstants as fc

from embrs.map_generator import get_user_data, save_to_file

def generate_uniform_map(map_params: MapParams):


    elevation_map = np.array([[map_params.uniform_data.elevation]])
    slope_map = np.array([[map_params.uniform_data.slope]])
    aspect_map = np.array([[map_params.uniform_data.aspect]])
    fuel_map = np.array([[map_params.uniform_data.fuel]])
    canopy_cover_map = np.array([[map_params.uniform_data.canopy_cover]])
    canopy_height_map = np.array([[map_params.uniform_data.canopy_height]])
    canopy_base_height_map = np.array([[map_params.uniform_data.canopy_base_height]])
    canopy_bulk_density_map = np.array([[map_params.uniform_data.canopy_bulk_density]])
    fccs_map = np.array([[map_params.uniform_data.fccs_id]])

    map_params.lcp_data = LandscapeData(
        elevation_map=elevation_map,
        slope_map=slope_map,
        aspect_map=aspect_map,
        fuel_map=fuel_map,
        canopy_cover_map=canopy_cover_map,
        canopy_height_map=canopy_height_map,
        canopy_base_height_map=canopy_base_height_map,
        canopy_bulk_density_map=canopy_bulk_density_map,
        fccs_map=fccs_map,
        rows=1,
        cols=1,
        resolution=10e10,
        width_m=map_params.uniform_data.width,
        height_m=map_params.uniform_data.height,
        transform=None,
        crs=None
    )

    # Create a color list in the right order
    colors = [fc.fuel_color_mapping[key] for key in sorted(fc.fuel_color_mapping.keys())]

    # Create a colormap from the list
    cmap = ListedColormap(colors)

    # Create a norm object to map your data points to the colormap
    norm = BoundaryNorm(list(sorted(fc.fuel_color_mapping.keys())) + [100], cmap.N)

    display_fuel_map = np.zeros((1, 1), dtype=np.uint8)
    display_fuel_map[0, 0] = map_params.uniform_data.fuel

    # Adjust the display_fuel_map to match the specified width and height
    display_fuel_map = np.full((int(map_params.uniform_data.height), int(map_params.uniform_data.width)), 
                               map_params.uniform_data.fuel, dtype=np.uint8)


    plt.imshow(np.flipud(display_fuel_map), cmap=cmap, norm=norm)


def main():
    file_selector = UniformMapCreator()
    map_params = file_selector.run()

    map_params.geo_info = GeoInfo()
    map_params.geo_info.north_angle_deg = 0
    map_params.geo_info.center_lat = map_params.uniform_data.latitude
    map_params.geo_info.center_lon = map_params.uniform_data.longitude
    map_params.geo_info.bounds = [map_params.uniform_data.longitude - (map_params.uniform_data.width / 2),
                                  map_params.uniform_data.latitude - (map_params.uniform_data.height / 2),
                                  map_params.uniform_data.longitude + (map_params.uniform_data.width / 2),
                                  map_params.uniform_data.latitude + (map_params.uniform_data.height / 2)]
    
    map_params.geo_info.calc_time_zone()

    if map_params is None:
        print("User exited before submitting necessary files.")
        sys.exit(0)

    # Initialize figure for user data GUI
    fig = plt.figure(figsize=(15, 10))
    plt.tick_params(left = False, right = False, bottom = False, labelleft = False,
                    labelbottom = False)

    generate_uniform_map(map_params)
    user_data = get_user_data(fig)
    save_to_file(map_params, user_data)

if __name__ == "__main__":
    main()