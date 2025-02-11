import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
import rasterio
from pyproj import Transformer

from datetime import date, datetime

from embrs.utilities.wind_forecast import gen_forecast


def gen_weather_from_open_meteo(elevation_path, vegetation_path, north_angle, params):
    # TODO: need to also process temperature and humidity data

    with rasterio.open(elevation_path, "r") as src:
        bounds = src.bounds  # (left, bottom, right, top)
        
        # Manually set the correct EPSG code for NAD83 / Conus Albers
        epsg_code = "EPSG:5070"

    # Compute midpoint in projected coordinates
    mid_x = (bounds.left + bounds.right) / 2
    mid_y = (bounds.bottom + bounds.top) / 2

    # Define the transformation from raster CRS (NAD83 / Conus Albers) to WGS84 (EPSG:4326)
    transformer = Transformer.from_crs(epsg_code, "EPSG:4326", always_xy=True)

    # Transform the midpoint from projected coordinates to lat/lon
    lon, lat = transformer.transform(mid_x, mid_y)

    seed = extract_wind_data(params, lat, lon)

    mesh_resolution = params["mesh_resolution"]

    # Returns forecast and time step 
    return gen_forecast(elevation_path, vegetation_path, seed, "OpenMeteo", north_angle, mesh_resolution)


def extract_wind_data(params, lat, lon):
    # Use center of the sim region as weather data location
    # TODO: Test to see if it makes a difference to sample across the domain and take average of outputs    
    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    start_datetime = params["start_datetime"]
    end_datetime = params["end_datetime"]

    # TODO: do we want to accept local time as input and convert to utc?
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    api_input = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_datetime.date().strftime("%Y-%m-%d"),
        "end_date": end_datetime.date().strftime("%Y-%m-%d"),
        "hourly": ["wind_speed_10m", "wind_direction_10m"],
        "wind_speed_unit": "ms"
    }
    responses = openmeteo.weather_api(url, params=api_input)
    response = responses[0]

    # TODO: verify units, stored in hourl_units

    hourly = response.Hourly()
    hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(1).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m

    hourly_data = filter_hourly_data(hourly_data, start_datetime, end_datetime)
    
    forecast_seed = ninjaify(hourly_data)

    return forecast_seed

def ninjaify(hourly_data: dict) -> dict:
    # TODO: do we want to pass these options as inputs to generalize this function
    ninja_data = {
        "time_step_min": 60,
        "wind_height": 10,
        "wind_height_units": "m"
    }

    speeds = hourly_data["wind_speed_10m"]
    dirs = hourly_data["wind_direction_10m"]

    wind_vecs = [{"direction": direction, "speed_m_s": speed} for direction, speed in zip(dirs, speeds)]

    ninja_data["data"] = wind_vecs

    print(ninja_data)

    return ninja_data

# TODO: Do we care about half hours and things like that
def filter_hourly_data(hourly_data, start_datetime, end_datetime):
    hourly_data["date"] = pd.to_datetime(hourly_data["date"]).tz_convert(None)

    mask = (hourly_data["date"] >= start_datetime) & (hourly_data["date"] <= end_datetime)
    filtered_data = {key: np.array(value)[mask] for key, value in hourly_data.items()}
    return filtered_data




