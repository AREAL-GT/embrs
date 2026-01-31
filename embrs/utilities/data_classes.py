"""Dataclasses for simulation parameters and state.

This module defines dataclasses used throughout EMBRS for storing
configuration parameters, simulation state, and intermediate results.

Classes:
    - MapDrawerData: Data from the map drawing interface.
    - GeoInfo: Geographic information for a simulation area.
    - LandscapeData: Landscape raster data from LCP files.
    - MapParams: Parameters for map configuration.
    - PlaybackVisualizerParams: Parameters for playback visualization.
    - VisualizerInputs: Input data for the visualizer.
    - WeatherEntry: Single weather observation.
    - WeatherParams: Weather input configuration.
    - SimParams: Full simulation parameters.
    - PredictorParams: Fire predictor configuration.
    - WindNinjaTask: WindNinja processing task parameters.
    - CellData: Cell-level data for fire calculations.
    - PredictionOutput: Output from a single fire prediction.
    - StateEstimate: Estimated fire state from observations.
    - CellStatistics: Statistics for ensemble cell metrics.
    - EnsemblePredictionOutput: Output from ensemble predictions.
    - ForecastData: Container for a single wind forecast.
    - ForecastPool: Collection of pre-computed wind forecasts for ensemble use.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from embrs.models.weather import WeatherStream
from rasterio.coords import BoundingBox
from datetime import datetime, timedelta
from pyproj import Transformer
from timezonefinder import TimezoneFinder
from shapely.geometry import Polygon


from embrs.models.fuel_models import Fuel


@dataclass
class MapDrawerData:
    """Data collected from the interactive map drawing interface.

    Attributes:
        fire_breaks (Dict): Fire break geometries keyed by ID.
        break_widths (List): Width in meters for each fire break.
        break_ids (List): Identifiers for fire breaks.
        initial_ign (List): Initial ignition point coordinates.
    """

    fire_breaks: Optional[Dict] = field(default_factory=dict)
    break_widths: Optional[List] = field(default_factory=list)
    break_ids: Optional[List] = field(default_factory=list)
    initial_ign: Optional[List] = field(default_factory=list)


@dataclass
class GeoInfo:
    """Geographic information for a simulation area.

    Attributes:
        bounds (BoundingBox): Spatial bounds from rasterio.
        center_lat (float): Center latitude in degrees (WGS84).
        center_lon (float): Center longitude in degrees (WGS84).
        timezone (str): IANA timezone string (e.g., 'America/Denver').
        north_angle_deg (float): Rotation from grid north to true north in degrees.
    """

    bounds: Optional[BoundingBox] = None
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    timezone: Optional[str] = None
    north_angle_deg: Optional[float] = None

    def calc_center_coords(self, source_crs) -> None:
        """Calculate center lat/lon from bounds and source CRS.

        Args:
            source_crs: Coordinate reference system of the bounds.

        Raises:
            ValueError: If bounds is None.
        """
        if self.bounds is None:
            raise ValueError("Can't perform this function without bounds")

        mid_x = (self.bounds.left + self.bounds.right) / 2
        mid_y = (self.bounds.bottom + self.bounds.top) / 2

        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)

        self.center_lon, self.center_lat = transformer.transform(mid_x, mid_y)

    def calc_time_zone(self) -> None:
        """Calculate timezone from center coordinates.

        Raises:
            ValueError: If center coordinates are not set.
        """
        if self.center_lat is None or self.center_lon is None:
            raise ValueError("Center coordinates must be set before computing the time zone")

        tf = TimezoneFinder()
        self.timezone = tf.timezone_at(lng=self.center_lon, lat=self.center_lat)

@dataclass
class LandscapeData:
    """Landscape raster data extracted from LCP files.

    Attributes:
        elevation_map (np.ndarray): Elevation in meters.
        slope_map (np.ndarray): Slope in degrees.
        aspect_map (np.ndarray): Aspect in degrees (0=N, 90=E).
        fuel_map (np.ndarray): Fuel model IDs.
        canopy_cover_map (np.ndarray): Canopy cover as fraction.
        canopy_height_map (np.ndarray): Canopy height in meters.
        canopy_base_height_map (np.ndarray): Canopy base height in meters.
        canopy_bulk_density_map (np.ndarray): Canopy bulk density in kg/m^3.
        rows (int): Number of raster rows.
        cols (int): Number of raster columns.
        resolution (int): Raster resolution in meters.
        width_m (float): Total width in meters.
        height_m (float): Total height in meters.
        transform (any): Rasterio affine transform.
        crs (any): Coordinate reference system.
    """

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
    """Parameters for map configuration.

    Attributes:
        folder (str): Path to the map data folder.
        lcp_filepath (str): Path to the source LCP file.
        cropped_lcp_path (str): Path to cropped LCP file if applicable.
        import_roads (bool): Whether to import roads from OSM.
        lcp_data (LandscapeData): Extracted landscape data.
        roads (List): List of road geometries.
        geo_info (GeoInfo): Geographic information.
        scenario_data (MapDrawerData): User-drawn scenario elements.
        fbfm_type (str): Fuel model type ('Anderson' or 'ScottBurgan').
    """

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
        """Get the map size in meters.

        Returns:
            Tuple[float, float]: (width_m, height_m).
        """
        return (self.lcp_data.width_m, self.lcp_data.height_m)

    def shape(self, cell_size: int) -> Tuple[int, int]:
        """Calculate grid shape for a given cell size.

        Args:
            cell_size (int): Hexagon side length in meters.

        Returns:
            Tuple[int, int]: (rows, cols) for the hexagonal grid.
        """
        rows = int(np.floor(self.lcp_data.height_m/(1.5*cell_size))) + 1
        cols = int(np.floor(self.lcp_data.width_m/(np.sqrt(3)*cell_size))) + 1

        return (rows, cols)

@dataclass
class PlaybackVisualizerParams:
    """Parameters for playback visualization from log files.

    Attributes:
        cell_file (str): Path to cell_logs.parquet file.
        init_location (bool): Whether init_state.parquet was found.
        save_video (bool): Whether to save visualization as video.
        video_folder (str): Folder to save video output.
        video_name (str): Video filename.
        has_agents (bool): Whether agent logs are available.
        has_actions (bool): Whether action logs are available.
        has_predictions (bool): Whether prediction logs are available.
        video_fps (int): Video frames per second.
        agent_file (str): Path to agent_logs.parquet.
        action_file (str): Path to action_logs.parquet.
        prediction_file (str): Path to prediction_logs.parquet.
        freq (float): Update frequency in seconds.
        scale_km (float): Scale bar size in kilometers.
        show_legend (bool): Whether to display fuel legend.
        show_wind_cbar (bool): Whether to show wind colorbar.
        show_wind_field (bool): Whether to show wind field.
        show_weather_data (bool): Whether to show weather info.
        show_compass (bool): Whether to show compass.
        show_visualization (bool): Whether to render visualization.
        show_temp_in_F (bool): Whether to display temperature in Fahrenheit.
    """

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
    """Input data for the real-time visualizer.

    Attributes:
        cell_size (float): Hexagon side length in meters.
        sim_shape (Tuple[int, int]): Grid shape (rows, cols).
        sim_size (Tuple[float, float]): Map size (width_m, height_m).
        start_datetime (datetime): Simulation start time.
        north_dir_deg (float): Rotation to true north in degrees.
        wind_forecast (np.ndarray): Wind field data array.
        wind_resolution (float): Wind mesh resolution in meters.
        wind_t_step (float): Wind time step in seconds.
        wind_xpad (float): Wind field x padding in meters.
        wind_ypad (float): Wind field y padding in meters.
        temp_forecast (np.ndarray): Temperature forecast values.
        rh_forecast (np.ndarray): Relative humidity forecast values.
        forecast_t_step (float): Forecast time step in seconds.
        elevation (np.ndarray): Coarse elevation data for display.
        roads (list): Road geometries for display.
        fire_breaks (list): Fire break geometries for display.
        init_entries (list): Initial cell state entries.
        scale_bar_km (float): Scale bar size in kilometers.
        show_legend (bool): Whether to show fuel legend.
        show_wind_cbar (bool): Whether to show wind colorbar.
        show_wind_field (bool): Whether to show wind field.
        show_weather_data (bool): Whether to show weather info.
        show_temp_in_F (bool): Whether to show temperature in Fahrenheit.
        show_compass (bool): Whether to show compass.
    """

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

    scale_bar_km: Optional[float] = 1.0
    show_legend: Optional[bool] = True
    show_wind_cbar: Optional[bool] = True
    show_wind_field: Optional[bool] = True
    show_weather_data: Optional[bool] = True
    show_temp_in_F: Optional[bool] = True
    show_compass: Optional[bool] = True


@dataclass
class WeatherEntry:
    """Single weather observation at a point in time.

    Attributes:
        wind_speed (float): Wind speed in m/s.
        wind_dir_deg (float): Wind direction in degrees (meteorological).
        temp (float): Temperature in Celsius.
        rel_humidity (float): Relative humidity as fraction (0-1).
        cloud_cover (float): Cloud cover as fraction (0-1).
        rain (float): Rainfall in mm.
        dni (float): Direct normal irradiance in W/m^2.
        dhi (float): Diffuse horizontal irradiance in W/m^2.
        ghi (float): Global horizontal irradiance in W/m^2.
        solar_zenith (float): Solar zenith angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.
    """

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
    """Weather input configuration.

    Attributes:
        input_type (str): Weather source type ('OpenMeteo' or 'File').
        file (str): Path to weather file if using file input.
        mesh_resolution (int): WindNinja mesh resolution in meters.
        conditioning_start (datetime): Start time for fuel moisture conditioning.
        start_datetime (datetime): Simulation start time.
        end_datetime (datetime): Simulation end time.
    """

    input_type: Optional[str] = None
    file: Optional[str] = ""
    mesh_resolution: Optional[int] = None
    conditioning_start: Optional[datetime] = None
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None

@dataclass
class SimParams:
    """Full simulation parameters.

    Attributes:
        map_params (MapParams): Map configuration.
        log_folder (str): Path to log output folder.
        weather_input (WeatherParams): Weather configuration.
        t_step_s (int): Simulation time step in seconds.
        cell_size (int): Hexagon side length in meters.
        init_mf (List[float]): Initial dead fuel moisture [1hr, 10hr, 100hr].
        fuel_moisture_map (Dict): Per-fuel-model moisture values.
        fms_has_live (bool): Whether FMS file includes live moisture.
        live_h_mf (float): Live herbaceous moisture fraction.
        live_w_mf (float): Live woody moisture fraction.
        model_spotting (bool): Whether to model spotting.
        canopy_species (int): Canopy species ID for spotting.
        dbh_cm (float): Diameter at breast height in cm.
        spot_ign_prob (float): Spot ignition probability (0-1).
        min_spot_dist (float): Minimum spotting distance in meters.
        spot_delay_s (float): Spot ignition delay in seconds.
        duration_s (float): Simulation duration in seconds.
        visualize (bool): Whether to show real-time visualization.
        num_runs (int): Number of simulation iterations.
        user_path (str): Path to user control module.
        user_class (str): Name of user control class.
        write_logs (bool): Whether to write log files.
    """

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
    """Fire predictor configuration.

    Attributes:
        time_horizon_hr (float): Prediction time horizon in hours.
        time_step_s (int): Prediction time step in seconds.
        cell_size_m (float): Cell size in meters.
        dead_mf (float): Dead fuel moisture fraction.
        live_mf (float): Live fuel moisture fraction.
        model_spotting (bool): Whether to model spotting.
        spot_delay_s (float): Spot ignition delay in seconds.
        wind_speed_bias (float): Wind speed bias in m/s.
        wind_dir_bias (float): Wind direction bias in degrees.
        wind_uncertainty_factor (float): Wind uncertainty scaling factor.
        ros_bias (float): Rate of spread bias factor.
        max_wind_speed_bias (float): Maximum wind speed bias in m/s.
        max_wind_dir_bias (float): Maximum wind direction bias in degrees.
        base_wind_spd_std (float): Base wind speed std dev in m/s.
        base_wind_dir_std (float): Base wind direction std dev in degrees.
        max_beta (float): Maximum uncertainty beta value.
    """

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

    max_wind_speed_bias: float = 2.5
    max_wind_dir_bias: float = 20.0
    base_wind_spd_std: float = 1.0
    base_wind_dir_std: float = 5.0
    max_beta: float = 0.95


@dataclass
class WindNinjaTask:
    """Parameters for a WindNinja processing task.

    Attributes:
        index (int): Task index in the processing queue.
        time_step (float): Time step for this wind field.
        entry (WeatherEntry): Weather data for this time step.
        elevation_path (str): Path to elevation raster file.
        timezone (str): IANA timezone string.
        north_angle (float): Rotation to true north in degrees.
        mesh_resolution (float): WindNinja mesh resolution in meters.
        temp_file_path (str): Path for temporary output files.
        cli_path (str): Path to WindNinja CLI executable.
        start_datetime (timedelta): Time offset from simulation start.
        wind_height (float): Wind measurement height.
        wind_height_units (str): Units for wind height.
        input_speed_units (str): Units for input wind speed.
        temperature_units (str): Units for temperature.
    """

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
    """Cell-level data for fire behavior calculations.

    Attributes:
        fuel_type (Fuel): Fuel model for this cell.
        elevation (float): Elevation in meters.
        aspect (float): Aspect in degrees (0=N, 90=E).
        slope_deg (float): Slope in degrees.
        canopy_cover (float): Canopy cover as fraction.
        canopy_height (float): Canopy height in meters.
        canopy_base_height (float): Canopy base height in meters.
        canopy_bulk_density (float): Canopy bulk density in kg/m^3.
        init_dead_mf (float): Initial dead fuel moisture fraction.
        live_h_mf (float): Live herbaceous moisture fraction.
        live_w_mf (float): Live woody moisture fraction.
    """

    fuel_type: Optional[Fuel] = None
    elevation: Optional[float] = None
    aspect: Optional[float] = None
    slope_deg: Optional[float] = None
    canopy_cover: Optional[float] = None
    canopy_height: Optional[float] = None
    canopy_base_height: Optional[float] = None
    canopy_bulk_density: Optional[float] = None
    init_dead_mf: Optional[float] = 0.08
    live_h_mf: Optional[float] = 0.3
    live_w_mf: Optional[float] = 0.3

@dataclass
class PredictionOutput:
    """Output from a single fire spread prediction.

    Attributes:
        spread (dict): Maps time in seconds to list of (x, y) positions
            where fire is predicted to arrive at that time.
        flame_len_m (dict): Maps (x, y) to flame length in meters.
        fli_kw_m (dict): Maps (x, y) to fireline intensity in kW/m.
        ros_ms (dict): Maps (x, y) to rate of spread in m/s.
        spread_dir (dict): Maps (x, y) to spread direction in degrees.
        crown_fire (dict): Maps (x, y) to crown fire status ('active' or 'passive').
        hold_probs (dict): Maps (x, y) to hold probability (0-1).
        breaches (dict): Maps (x, y) to breach status (bool).
    """

    spread: dict
    flame_len_m: dict
    fli_kw_m: dict
    ros_ms: dict
    spread_dir: dict
    crown_fire: dict
    hold_probs: dict
    breaches: dict


@dataclass
class StateEstimate:
    """Estimated fire state from observations.

    Attributes:
        burnt_polys (List[Polygon]): Polygons of burnt area.
        burning_polys (List[Polygon]): Polygons of actively burning area.
        start_time_s (Optional[float]): Start time in seconds from simulation start.
            If None, uses current fire simulation time.
    """

    burnt_polys: List[Polygon] = None
    burning_polys: List[Polygon] = None
    start_time_s: Optional[float] = None

@dataclass
class CellStatistics:
    """Statistics for a single metric across ensemble members.

    Attributes:
        mean (float): Mean value across ensemble members.
        std (float): Standard deviation across ensemble members.
        min (float): Minimum value across ensemble members.
        max (float): Maximum value across ensemble members.
        count (int): Number of ensemble members with data for this cell.
    """

    mean: float
    std: float
    min: float
    max: float
    count: int


@dataclass
class EnsemblePredictionOutput:
    """Output from ensemble fire prediction runs.

    Aggregates statistics across multiple prediction runs with varying
    parameters to quantify prediction uncertainty.

    Attributes:
        n_ensemble (int): Number of ensemble members.
        burn_probability (dict): Maps time (s) to dict of (x, y) to burn
            probability (0-1).
        flame_len_m_stats (dict): Maps (x, y) to CellStatistics for flame length.
        fli_kw_m_stats (dict): Maps (x, y) to CellStatistics for fireline intensity.
        ros_ms_stats (dict): Maps (x, y) to CellStatistics for rate of spread.
        spread_dir_stats (dict): Maps (x, y) to dict with 'mean_x' and 'mean_y'
            for circular mean spread direction.
        crown_fire_frequency (dict): Maps (x, y) to crown fire probability (0-1).
        hold_prob_stats (dict): Maps (x, y) to CellStatistics for hold probability.
        breach_frequency (dict): Maps (x, y) to breach probability (0-1).
        individual_predictions (List[PredictionOutput]): Individual prediction
            outputs for inspection. Optional.
    """

    n_ensemble: int
    burn_probability: dict
    flame_len_m_stats: dict
    fli_kw_m_stats: dict
    ros_ms_stats: dict
    spread_dir_stats: dict
    crown_fire_frequency: dict
    hold_prob_stats: dict
    breach_frequency: dict
    individual_predictions: Optional[List[PredictionOutput]] = None


@dataclass
class ForecastData:
    """Container for a single wind forecast and its generating parameters.

    Stores a WindNinja output array along with the perturbation parameters
    used to generate it, enabling reproducibility and reuse of forecasts
    across ensemble predictions.

    Attributes:
        wind_forecast: WindNinja output array.
            Shape: (n_timesteps, height, width, 2) where [..., 0] = speed (m/s),
            [..., 1] = direction (degrees).
        weather_stream: The perturbed weather stream used to generate this forecast.
        wind_speed_bias: Constant wind speed bias applied (m/s).
        wind_dir_bias: Constant wind direction bias applied (degrees).
        speed_error_seed: Random seed used for AR(1) speed noise.
        dir_error_seed: Random seed used for AR(1) direction noise.
        forecast_id: Unique identifier for this forecast within the pool.
        generation_time: Unix timestamp when forecast was generated.
    """

    wind_forecast: np.ndarray
    weather_stream: 'WeatherStream'
    wind_speed_bias: float
    wind_dir_bias: float
    speed_error_seed: int
    dir_error_seed: int
    forecast_id: int
    generation_time: float


@dataclass
class ForecastPool:
    """A collection of pre-computed wind forecasts for ensemble use.

    Provides storage and sampling methods for a pool of perturbed wind
    forecasts that can be reused across global predictions and rollouts.

    Attributes:
        forecasts: List of ForecastData objects.
        base_weather_stream: Original unperturbed weather stream.
        map_params: Map parameters used for WindNinja.
        predictor_params: Predictor parameters at time of pool creation.
        created_at_time_s: Simulation time (seconds) when pool was created.
        forecast_start_datetime: Local datetime that index 0 of forecasts corresponds to.
    """

    forecasts: List[ForecastData]
    base_weather_stream: 'WeatherStream'
    map_params: 'MapParams'
    predictor_params: 'PredictorParams'
    created_at_time_s: float
    forecast_start_datetime: 'datetime'

    def __len__(self) -> int:
        """Return the number of forecasts in the pool."""
        return len(self.forecasts)

    def __getitem__(self, idx: int) -> ForecastData:
        """Get a forecast by index."""
        return self.forecasts[idx]

    def sample(self, n: int, replace: bool = True, seed: int = None) -> List[int]:
        """Sample n indices from the pool.

        Args:
            n: Number of indices to sample.
            replace: If True, sample with replacement (default). If False,
                n must not exceed pool size.
            seed: Random seed for reproducibility.

        Returns:
            List of forecast indices.

        Raises:
            ValueError: If replace=False and n > pool size.
        """
        rng = np.random.default_rng(seed)
        return rng.choice(len(self.forecasts), size=n, replace=replace).tolist()

    def get_forecast(self, idx: int) -> ForecastData:
        """Get a specific forecast by index.

        Args:
            idx: Index of the forecast to retrieve.

        Returns:
            ForecastData at the specified index.
        """
        return self.forecasts[idx]

    def get_weather_scenarios(self) -> List['WeatherStream']:
        """Return all perturbed weather streams for time window calculation.

        Returns:
            List of WeatherStream objects, one per forecast in the pool.
        """
        return [f.weather_stream for f in self.forecasts]