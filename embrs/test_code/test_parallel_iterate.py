import numpy as np
from datetime import datetime
import sys
import types

# Provide a very small stub for tqdm to avoid an external dependency during tests.
tqdm_stub = types.ModuleType("tqdm")

def _tqdm(*args, **kwargs):  # pragma: no cover - trivial stub
    class _Dummy:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def update(self, *args, **kwargs):
            pass

        def close(self):
            pass

    return _Dummy()

tqdm_stub.tqdm = _tqdm
sys.modules["tqdm"] = tqdm_stub

# Minimal shapely stubs
shapely_stub = types.ModuleType("shapely")
geometry_stub = types.ModuleType("geometry")

class _Geom:
    pass

class Point(_Geom):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Polygon(_Geom):
    def __init__(self, coords=None):
        self.coords = coords or []

class LineString(_Geom):
    pass

class MultiPolygon(_Geom):
    pass

geometry_stub.Point = Point
geometry_stub.Polygon = Polygon
geometry_stub.LineString = LineString
geometry_stub.MultiPolygon = MultiPolygon
shapely_stub.geometry = geometry_stub
ops_stub = types.ModuleType("ops")

def unary_union(*args, **kwargs):
    return None

ops_stub.unary_union = unary_union
shapely_stub.ops = ops_stub
sys.modules["shapely"] = shapely_stub
sys.modules["shapely.geometry"] = geometry_stub
sys.modules["shapely.ops"] = ops_stub

# Minimal rasterio stubs
rasterio_stub = types.ModuleType("rasterio")
coords_stub = types.ModuleType("coords")

class BoundingBox:
    pass

coords_stub.BoundingBox = BoundingBox
rasterio_stub.coords = coords_stub
sys.modules["rasterio"] = rasterio_stub
sys.modules["rasterio.coords"] = coords_stub

# Minimal pyproj stub
pyproj_stub = types.ModuleType("pyproj")

class _Transformer:
    @staticmethod
    def from_crs(src, dst, always_xy=False):
        class _T:
            def transform(self, x, y):
                return x, y

        return _T()

pyproj_stub.Transformer = _Transformer
sys.modules["pyproj"] = pyproj_stub

# Minimal timezonefinder stub
tzf_stub = types.ModuleType("timezonefinder")

class TimezoneFinder:
    def timezone_at(self, lng, lat):
        return "UTC"

tzf_stub.TimezoneFinder = TimezoneFinder
sys.modules["timezonefinder"] = tzf_stub

# Stub weather and wind forecast modules used during initialization
from embrs.utilities.data_classes import WeatherEntry

weather_stub = types.ModuleType("embrs.models.weather")


class WeatherStream:
    def __init__(self, params, geo, use_gsi=True):
        self.time_step = 60
        self.live_h_mf = 1.4
        self.live_w_mf = 1.25
        self.fmc = 100
        self.stream = [
            WeatherEntry(
                wind_speed=0,
                wind_dir_deg=0,
                temp=None,
                rel_humidity=None,
                cloud_cover=None,
                rain=0,
                dni=None,
                dhi=None,
                ghi=None,
                solar_zenith=None,
                solar_azimuth=None,
            )
        ]


weather_stub.WeatherStream = WeatherStream
def apply_site_specific_correction(*args, **kwargs):
    return None


def calc_local_solar_radiation(*args, **kwargs):
    return 0


weather_stub.apply_site_specific_correction = apply_site_specific_correction
weather_stub.calc_local_solar_radiation = calc_local_solar_radiation
sys.modules["embrs.models.weather"] = weather_stub

wind_stub = types.ModuleType("embrs.models.wind_forecast")


def create_uniform_wind(weather):
    return np.zeros((1, 1, 1, 2))


def run_windninja(weather, map):
    return np.zeros((1, 1, 1, 2)), 0


wind_stub.create_uniform_wind = create_uniform_wind
wind_stub.run_windninja = run_windninja
sys.modules["embrs.models.wind_forecast"] = wind_stub

from embrs.fire_simulator.fire import FireSim
from embrs.utilities.data_classes import (
    SimParams,
    MapParams,
    UniformMapParams,
    MapDrawerData,
    LandscapeData,
    WeatherParams,
)


def build_sim(parallel: bool = False) -> FireSim:
    """Construct a minimal simulation for testing."""
    uniform = UniformMapParams(fuel=2, width=300, height=300)
    lcp = LandscapeData(
        elevation_map=np.array([[0]]),
        slope_map=np.array([[0]]),
        aspect_map=np.array([[0]]),
        fuel_map=np.array([[2]]),
        canopy_cover_map=np.array([[0]]),
        canopy_height_map=np.array([[0]]),
        canopy_base_height_map=np.array([[0]]),
        canopy_bulk_density_map=np.array([[0]]),
        fccs_map=np.array([[0]]),
        rows=1,
        cols=1,
        resolution=int(1e9),
        width_m=300,
        height_m=300,
        transform=None,
        crs=None,
    )
    scenario = MapDrawerData(
        fire_breaks={}, break_widths=[], break_ids=[], initial_ign=[]
    )
    map_params = MapParams(
        uniform_map=True,
        uniform_data=uniform,
        lcp_data=lcp,
        scenario_data=scenario,
        import_roads=False,
        geo_info=None,
    )
    weather_params = WeatherParams(
        input_type="File",
        file="examples/burnout_test_weather.wxs",
        start_datetime=datetime(2020, 1, 1),
        end_datetime=datetime(2020, 1, 2),
    )
    sim_params = SimParams(
        map_params=map_params,
        weather_input=weather_params,
        t_step_s=30,
        cell_size=30,
        duration_s=300,
        visualize=False,
        num_runs=1,
        write_logs=False,
    )
    sim = FireSim(sim_params, parallel=parallel)

    # Manually ignite the central cell to avoid a shapely dependency.
    cell = sim.get_cell_from_xy(150, 150)
    sim.starting_ignitions.add((cell, 0))
    sim._init_iteration(True)
    return sim


def run_sim(sim: FireSim):
    while not sim.finished:
        sim.iterate()
    arrivals = {cell.id: cell._arrival_time for cell in sim._cell_dict.values()}
    states = {cell.id: cell.state for cell in sim._cell_dict.values()}
    return arrivals, states


def test_parallel_determinism():
    np.random.seed(0)
    seq = build_sim(parallel=False)
    arr_seq, state_seq = run_sim(seq)

    np.random.seed(0)
    par = build_sim(parallel=True)
    arr_par, state_par = run_sim(par)

    assert arr_seq == arr_par
    assert state_seq == state_par
