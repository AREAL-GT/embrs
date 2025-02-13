from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
from rasterio.coords import BoundingBox

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
    initial_ign: Optional[List] = field(default_factory=list)

@dataclass
class MapParams:
    output_folder: Optional[str] = None
    metadata_path: Optional[str] = None
    import_roads: Optional[bool] = None
    uniform_fuel: Optional[bool] = None
    uniform_elev: Optional[bool] = None
    fuel_type: Optional[int] = None
    fuel_data: Optional[DataProductParams] = DataProductParams()
    elev_data: Optional[DataProductParams] = DataProductParams()
    asp_data: Optional[DataProductParams] = DataProductParams()
    slp_data: Optional[DataProductParams] = DataProductParams()
    roads: Optional[List] = field(default_factory=list)
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    bounds: Optional[BoundingBox] = None
    north_angle_deg: Optional[float] = None
    scenario_data: Optional[MapDrawerData] = None
