from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np
from rasterio.coords import BoundingBox
from datetime import datetime, timedelta
from pyproj import Transformer
from timezonefinder import TimezoneFinder
from shapely.geometry import Polygon


from embrs.models.fuel_models import Fuel

# TODO: Add comments/docstrings for dataclasses

@dataclass
class MapDrawerData:
    fire_breaks: Optional[Dict] = field(default_factory=dict)
    break_widths: Optional[List] = field(default_factory=list)
    break_ids: Optional[List] = field(default_factory=list)
    initial_ign: Optional[List] = field(default_factory=list)

@dataclass
class GeoInfo:
    bounds: Optional[BoundingBox] = None
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    timezone: Optional[str] = None
    north_angle_deg: Optional[float] = None

    def calc_center_coords(self, source_crs):
        if self.bounds is None:
            raise ValueError("Can't perform this function without bounds")

        # Compute midpoint in projected coordinates
        mid_x = (self.bounds.left + self.bounds.right) / 2
        mid_y = (self.bounds.bottom + self.bounds.top) / 2

        # Define the transformation from raster CRS to WGS84 (EPSG:4326)
        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)

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
    fbfm_type: Optional[str] = "Anderson"

    def size(self) -> Tuple[float, float]:
        return (self.lcp_data.width_m, self.lcp_data.height_m)
    
    def shape(self, cell_size: int) -> Tuple[int, int]:
        rows = int(np.floor(self.lcp_data.height_m/(1.5*cell_size))) + 1
        cols = int(np.floor(self.lcp_data.width_m/(np.sqrt(3)*cell_size))) + 1

        return (rows, cols)

@dataclass
class PlaybackVisualizerParams:
    cell_file: str
    init_location: bool
    save_video: bool
    video_folder: str
    video_name: str
    has_agents: bool
    has_actions: bool
    has_predictions: bool
    video_fps: Optional[int] = 10 
    agent_file: Optional[str] = None
    action_file: Optional[str] = None
    prediction_file: Optional[str] = None
    
    # Visualization Preferences
    freq: Optional[float] = 300
    scale_km: Optional[float] = 1.0
    show_legend: Optional[bool] = True
    show_wind_cbar: Optional[bool] = True
    show_wind_field: Optional[bool] = True
    show_weather_data: Optional[bool] = True
    show_compass: Optional[bool] = True
    show_visualization: Optional[bool] = True
    show_temp_in_F: Optional[bool] = True

@dataclass
class VisualizerInputs:
    cell_size: float
    sim_shape: Tuple[int, int]
    sim_size: Tuple[float, float]
    start_datetime: datetime
    north_dir_deg: float
    wind_forecast: np.ndarray
    wind_resolution: float
    wind_t_step: float
    wind_xpad: float
    wind_ypad: float
    temp_forecast: np.ndarray
    rh_forecast: np.ndarray
    forecast_t_step: float
    elevation: np.ndarray
    roads: list
    fire_breaks: list
    init_entries: list

    # Visualization Preferences
    scale_bar_km: Optional[float] = 1.0
    show_legend: Optional[bool] = True
    show_wind_cbar: Optional[bool] = True
    show_wind_field: Optional[bool] = True
    show_weather_data: Optional[bool] = True
    show_temp_in_F: Optional[bool] = True
    show_compass: Optional[bool] = True

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
    conditioning_start: Optional[datetime] = None
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None

@dataclass
class SimParams:
    map_params: Optional[MapParams] = None
    log_folder: Optional[str] = None
    weather_input: Optional[WeatherParams] = None
    t_step_s: Optional[int] = None
    cell_size: Optional[int] = None
    init_mf: Optional[List[float]] = field(default_factory=lambda: [0.06, 0.07, 0.08])
    fuel_moisture_map: Dict[int, List[float]] = field(default_factory=dict)
    fms_has_live: bool = False
    live_h_mf: Optional[float] = None
    live_w_mf: Optional[float] = None
    model_spotting: Optional[bool] = False
    canopy_species: Optional[int] = 5
    dbh_cm: Optional[float] = 20.0
    spot_ign_prob: Optional[float] = 0.05
    min_spot_dist: Optional[float] = 50
    spot_delay_s: Optional[float] = 1200
    duration_s: Optional[float] = None
    visualize: Optional[bool] = None
    num_runs: Optional[int] = None
    user_path: Optional[str] = None
    user_class: Optional[str] = None
    write_logs: Optional[bool] = None

@dataclass
class PredictorParams:
    time_horizon_hr: float = 2.0
    time_step_s: int = 30
    cell_size_m: float = 30
    dead_mf: float = 0.08
    live_mf: float = 0.30
    model_spotting: bool = False
    spot_delay_s: float = 1200
    wind_speed_bias: float = 0
    wind_dir_bias: float = 0
    wind_uncertainty_factor: float = 0
    ros_bias: float = 0

    # Advanced uncertainty settings
    max_wind_speed_bias: float = 2.5 # m/s
    max_wind_dir_bias: float = 20.0 # deg
    base_wind_spd_std: float = 1.0 # m/s
    base_wind_dir_std: float = 5.0 # deg
    max_beta: float = 0.95

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
class CellData:
    fuel_type: Optional[Fuel] = None
    elevation: Optional[float] = None
    aspect: Optional[float] = None
    slope_deg: Optional[float] = None
    canopy_cover:Optional[float] = None
    canopy_height: Optional[float] = None
    canopy_base_height: Optional[float] = None
    canopy_bulk_density: Optional[float] = None
    init_dead_mf: Optional[float] = 0.08
    live_h_mf: Optional[float] = 0.3
    live_w_mf: Optional[float] = 0.3

@dataclass
class PredictionOutput:
    spread: dict # keys: times, values: (x,y)
    flame_len_m: dict # keys: (x,y), values: float
    fli_kw_m: dict # keys: (x,y), values: float
    ros_ms: dict # keys: (x,y), values: float
    spread_dir: dict # keys: (x,y), values: float
    crown_fire: dict # keys: (x,y), values: active or passive
    hold_probs: dict # keys: (x,y), values: float
    breaches: dict # keys: (x,y), values: bool

@dataclass
class StateEstimate:
    burnt_polys: List[Polygon] = None
    burning_polys: List[Polygon] = None