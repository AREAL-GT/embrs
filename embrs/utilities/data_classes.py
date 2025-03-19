from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np
from rasterio.coords import BoundingBox
from datetime import datetime, timedelta
from pyproj import Transformer
from timezonefinder import TimezoneFinder

# TODO: Add comments/docstrings for dataclasses

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
    north_angle_deg: Optional[float] = None

    def calc_center_coords(self):
        if self.bounds is None:
            raise ValueError("Can't perform this function without bounds")

        # Manually set the correct EPSG code for NAD83 / Conus Albers
        epsg_code = "EPSG:5070" # TODO: this should probably not be hard-coded

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
class LandscapeData:
    elevation_map: np.ndarray
    slope_map: np.ndarray
    aspect_map: np.ndarray
    fuel_map: np.ndarray
    canopy_cover_map: np.ndarray
    canopy_height_map: np.ndarray
    canopy_base_height_map: np.ndarray
    canopy_bulk_density_map: np.ndarray
    rows: int
    cols: int
    resolution: int
    width_m: float
    height_m: float
    transform: any
    crs: any

@dataclass
class MapParams:
    folder: Optional[str] = None
    lcp_filepath: Optional[str] = None
    cropped_lcp_path: Optional[str] = None
    import_roads: Optional[bool] = None
    lcp_data: Optional[LandscapeData] = None
    roads: Optional[List] = field(default_factory=list)
    geo_info: Optional[GeoInfo] = None
    scenario_data: Optional[MapDrawerData] = None

    def size(self) -> Tuple[float, float]:
        return (self.lcp_data.width_m, self.lcp_data.height_m)
    
    def shape(self, cell_size: int) -> Tuple[int, int]:
        rows = int(np.floor(self.lcp_data.height_m/(1.5*cell_size))) 
        cols = int(np.floor(self.lcp_data.width_m/(np.sqrt(3)*cell_size)))

        return (rows, cols)

@dataclass
class WeatherEntry:
    wind_speed: float
    wind_dir_deg: float
    temp: float
    rel_humidity: float
    cloud_cover: float
    rain: float
    dni: float
    dhi: float
    ghi: float
    solar_zenith: float
    solar_azimuth: float

@dataclass
class WeatherParams:
    input_type: Optional[str] = None
    file: Optional[str] = "" # Populated only if user is passing a weather file
    mesh_resolution: Optional[int] = None
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None

@dataclass
class WeatherSeed: # TODO: this is going to be replaced by WeatherStream
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
    init_mf: Optional[float] = 0.08
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

@dataclass
class BurnStruct:
    time: Optional[float] = None
    wdf: Optional[float] = None
    ff: Optional[float] = None