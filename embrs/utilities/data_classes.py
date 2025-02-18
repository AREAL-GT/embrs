from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np
from rasterio.coords import BoundingBox
from datetime import datetime, timedelta
from pyproj import Transformer
from timezonefinder import TimezoneFinder

# TODO: Add comments/docstrings for dataclasses

@dataclass
class DataProductParams:
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    resolution: Optional[int] = None
    map: Optional[np.ndarray] = None
    uniform: Optional[bool] = None
    tiff_filepath: Optional[str] = None
    cropped_filepath: Optional[str] = None
    np_filepath: Optional[str] = None

@dataclass
class MapDrawerData:
    fire_breaks: Optional[Dict] = field(default_factory=dict)
    fuel_vals: Optional[List] = field(default_factory=list)
    initial_ign: Optional[List] = field(default_factory=list)

@dataclass
class GeoInfo:
    bounds: Optional[BoundingBox] = None
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    timezone: Optional[str] = None

    def save_bounds(self, bounds: BoundingBox):
        epsg_code = "EPSG:5070"

        transformer = Transformer.from_crs(epsg_code, "EPSG:4326", always_xy=True)

        left, bottom = transformer.transform(bounds.left, bounds.bottom)
        right, top = transformer.transform(bounds.right, bounds.top)

        self.bounds = BoundingBox(left, bottom, right, top)

    def calc_center_coords(self):
        if self.bounds is None:
            raise ValueError("Can't perform this function without bounds")

        # Manually set the correct EPSG code for NAD83 / Conus Albers
        epsg_code = "EPSG:5070"

        # Compute midpoint in projected coordinates
        mid_x = (self.bounds.left + self.bounds.right) / 2
        mid_y = (self.bounds.bottom + self.bounds.top) / 2

        # Define the transformation from raster CRS (NAD83 / Conus Albers) to WGS84 (EPSG:4326)
        transformer = Transformer.from_crs(epsg_code, "EPSG:4326", always_xy=True)

        # Transform the midpoint from projected coordinates to lat/lon
        self.center_lon, self.center_lat = transformer.transform(mid_x, mid_y)

    def calc_time_zone(self):
        if self.center_lat is None or self.center_lon is None:
            raise ValueError("Center coordinates must be set before computing the time zone")

        # Get the time zone at the location to sample
        tf = TimezoneFinder()
        self.timezone = tf.timezone_at(lng=self.center_lon, lat=self.center_lat)
        
@dataclass
class MapParams:
    folder: Optional[str] = None
    metadata_path: Optional[str] = None
    import_roads: Optional[bool] = None
    uniform_fuel: Optional[bool] = None
    uniform_elev: Optional[bool] = None
    fuel_type: Optional[int] = None
    fuel_data: Optional[DataProductParams] = DataProductParams()
    elev_data: Optional[DataProductParams] = DataProductParams()
    asp_data: Optional[DataProductParams] = DataProductParams()
    slp_data: Optional[DataProductParams] = DataProductParams()
    cc_data: Optional[DataProductParams] = DataProductParams()
    ch_data: Optional[DataProductParams] = DataProductParams()
    roads: Optional[List] = field(default_factory=list)
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    geo_info: Optional[GeoInfo] = None
    north_angle_deg: Optional[float] = None
    scenario_data: Optional[MapDrawerData] = None

    def size(self) -> Tuple[float, float]:
        return (self.width_m, self.height_m)
    
    def shape(self, cell_size: int) -> Tuple[int, int]:
        rows = int(np.floor(self.elev_data.height_m/(1.5*cell_size))) 
        cols = int(np.floor(self.elev_data.width_m/(np.sqrt(3)*cell_size)))

        return (rows, cols)

@dataclass
class WeatherEntry:
    wind_speed: float
    wind_dir_deg: float
    temp: float
    rel_humidity: float
    cloud_cover: float

@dataclass
class WeatherParams:
    input_type: Optional[str] = None
    file: Optional[str] = "" # Populated only if user is passing a weather file
    mesh_resolution: Optional[int] = None
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None

@dataclass
class WeatherSeed:
    params: Optional[WeatherParams] = None
    time_step: Optional[float] = None
    input_wind_ht: Optional[float] = None
    input_wind_ht_units: Optional[str] = None
    input_wind_vel_units: Optional[str] = None
    input_temp_units: Optional[str] = None
    weather_entries: Optional[List[WeatherEntry]] = None

@dataclass
class SimParams:
    map_params: Optional[MapParams] = None
    log_folder: Optional[str] = None
    weather_input: Optional[WeatherParams] = None
    t_step_s: Optional[int] = None
    cell_size: Optional[int] = None
    duration_s: Optional[float] = None
    visualize: Optional[bool] = None
    num_runs: Optional[int] = None
    user_path: Optional[str] = None
    user_class: Optional[str] = None
    write_logs: Optional[bool] = None

@dataclass
class WindNinjaTask:
    index: int
    time_step: float
    entry: WeatherEntry
    elevation_path: str
    vegetation_path: str
    timezone: str
    north_angle: float
    mesh_resolution: float
    temp_file_path: str
    cli_path: str
    start_datetime: timedelta
    wind_height: float
    wind_height_units: str
    input_speed_units: str
    temperature_units: str