import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
import rasterio
from pyproj import Transformer
from timezonefinder import TimezoneFinder
import pytz

from datetime import datetime, timedelta

from embrs.utilities.wind_forecast import run_windninja

def gen_weather_from_open_meteo(elevation_path, vegetation_path, north_angle, params):
    # Get the weather data
    seed = retrieve_weather_data(params)
    mesh_resolution = params["mesh_resolution"]
    timezone = params["timezone"]

    # Returns forecast and time step 
    return run_windninja(elevation_path, vegetation_path, seed, timezone, north_angle, mesh_resolution)

def retrieve_weather_data(params):
    # Use center of the sim region as weather data location
    # TODO: Test to see if it makes a difference to sample across the domain and take average of outputs    
    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    local_tz = pytz.timezone(params["timezone"])

    # Buffer times and format for OpenMeteo
    start_datetime = params["start_datetime"]
    start_datetime = local_tz.localize(start_datetime)
    buffered_start = start_datetime - timedelta(days=1)
    start_datetime_utc = buffered_start.astimezone(pytz.utc)

    end_datetime = params["end_datetime"]
    end_datetime = local_tz.localize(end_datetime)
    buffered_end = start_datetime + timedelta(days=1)
    end_datetime_utc = buffered_end.astimezone(pytz.utc)

    lat, lon = params["center_coords"]

    url = "https://archive-api.open-meteo.com/v1/archive"
    api_input = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_datetime_utc.date().strftime("%Y-%m-%d"),
        "end_date": end_datetime_utc.date().strftime("%Y-%m-%d"),
        "hourly": ["wind_speed_10m", "wind_direction_10m", "temperature_2m", "relative_humidity_2m", "cloud_cover"],
        "wind_speed_unit": "ms",
        "temperature_unit": "fahrenheit",
        "timezone": "auto"
    }
    responses = openmeteo.weather_api(url, params=api_input)
    response = responses[0]

    hourly = response.Hourly()
    hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(1).ValuesAsNumpy()
    hourly_temperature_2m = hourly.Variables(2).ValuesAsNumpy()
    hourly_rel_humidity_2m = hourly.Variables(3).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()

    hourly_data = {}

    hourly_data["date"] = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s"),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ).tz_localize(local_tz)

    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["rel_humidity_2m"] = hourly_rel_humidity_2m
    hourly_data["cloud_cover"] = hourly_cloud_cover

    hourly_data = filter_hourly_data(hourly_data, start_datetime, end_datetime)
    forecast_seed = ninjaify(hourly_data, start_datetime)

    return forecast_seed

def ninjaify(hourly_data: dict, start_datetime: datetime) -> dict:
    # TODO: do we want to pass these options as inputs to generalize this function
    ninja_data = {
        "time_step_min": 60,
        "wind_height": 10,
        "wind_height_units": "m",
        "wind_speed_units": "mps",
        "temperature_units": "F",
        "start_datetime": start_datetime
    }

    wind_speeds = hourly_data["wind_speed_10m"]
    wind_dirs = hourly_data["wind_direction_10m"]
    temps = hourly_data["temperature_2m"]
    rel_humidities = hourly_data["rel_humidity_2m"]
    cloud_covers = hourly_data["cloud_cover"]

    weather_data = zip(wind_speeds, wind_dirs, temps, rel_humidities, cloud_covers)

    weather_vecs = [{"wind_speed": wind_speed,
                    "wind_direction": wind_dir,
                    "temperature": temp,
                    "rel_humidity": rel_humidity,
                    "cloud_cover": cloud_cover} for wind_speed, wind_dir, temp, rel_humidity, cloud_cover in weather_data]

    ninja_data["data"] = weather_vecs

    return ninja_data

def filter_hourly_data(hourly_data, start_datetime, end_datetime):
    hourly_data["date"] = pd.to_datetime(hourly_data["date"])

    mask = (hourly_data["date"] >= start_datetime) & (hourly_data["date"] <= end_datetime)
    filtered_data = {key: np.array(value)[mask] for key, value in hourly_data.items()}
    return filtered_data




