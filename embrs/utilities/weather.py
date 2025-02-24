from retry_requests import retry
from datetime import timedelta
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import pytz
import json

from embrs.utilities.wind_forecast import run_windninja
from embrs.utilities.data_classes import *

# TODO: Document this file
# TODO: Process other data provided and determine how to send it all back to the sim

def generate_weather(sim_params: SimParams):
    weather = sim_params.weather_input
    map = sim_params.map_params

    if weather.input_type == "OpenMeteo":
        forecast, time_step = open_meteo_weather(map, weather)

    elif weather.input_type == "File":
        forecast, time_step = weather_from_file(map, weather)
        
    return forecast, time_step

def open_meteo_weather(map: MapParams, weather: WeatherParams):

    # Get the weather data
    seed = retrieve_openmeteo_data(weather, map.geo_info)
    
    # Returns forecast and time step 
    # TODO: this should return some kind of weather output that returns more than just wind
    return run_windninja(map, seed)

def retrieve_openmeteo_data(weather: WeatherParams, geo: GeoInfo):
    # Use center of the sim region as weather data location
    # TODO: Test to see if it makes a difference to sample across the domain and take average of outputs    
    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    local_tz = pytz.timezone(geo.timezone)

    # Buffer times and format for OpenMeteo
    start_datetime = weather.start_datetime
    start_datetime = local_tz.localize(start_datetime)
    buffered_start = start_datetime - timedelta(days=1)
    start_datetime_utc = buffered_start.astimezone(pytz.utc)

    end_datetime = weather.end_datetime
    end_datetime = local_tz.localize(end_datetime)
    buffered_end = end_datetime + timedelta(days=1)
    end_datetime_utc = buffered_end.astimezone(pytz.utc)

    lat = geo.center_lat
    lon = geo.center_lon

    print(f"input latitude: {lat}, output longitude: {lon}")

    url = "https://archive-api.open-meteo.com/v1/archive"
    api_input = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_datetime_utc.date().strftime("%Y-%m-%d"),
        "end_date": end_datetime_utc.date().strftime("%Y-%m-%d"),
        "hourly": ["wind_speed_10m", "wind_direction_10m", "temperature_2m", "relative_humidity_2m", "cloud_cover", "shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"],
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
    hourly_ghi = hourly.Variables(5).ValuesAsNumpy()
    hourly_dhi = hourly.Variables(6).ValuesAsNumpy()
    hourly_dni = hourly.Variables(7).ValuesAsNumpy()

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
    hourly_data["ghi"] = hourly_ghi
    hourly_data["dhi"] = hourly_dhi
    hourly_data["dni"] = hourly_dni

    hourly_data = filter_hourly_data(hourly_data, start_datetime, end_datetime)
    forecast_seed = openmeteo_ninjaify(hourly_data, weather)

    return forecast_seed

def openmeteo_ninjaify(hourly_data: dict, params: WeatherParams) -> WeatherSeed:
    weather_seed = WeatherSeed(
        params = params,
        time_step = 60,
        input_wind_ht = 10,
        input_wind_ht_units = "m",
        input_wind_vel_units = "mps",
        input_temp_units = "F"
    )

    wind_speeds = hourly_data["wind_speed_10m"]
    wind_dirs = hourly_data["wind_direction_10m"]
    temps = hourly_data["temperature_2m"]
    rel_humidities = hourly_data["rel_humidity_2m"]
    cloud_covers = hourly_data["cloud_cover"]
    weather_data = zip(wind_speeds, wind_dirs, temps, rel_humidities, cloud_covers)
    
    return ninjaify(weather_data, weather_seed)

def file_ninjaify(data: dict, weather: WeatherParams) -> WeatherSeed:
    # Convert iso format to datetime object
    weather.start_datetime = datetime.fromisoformat(data["start_datetime"])

    weather_seed = WeatherSeed(
        params = weather,
        time_step = data["time_step_min"],
        input_wind_ht = data["wind_height"],
        input_wind_ht_units=data["wind_height_units"],
        input_wind_vel_units = data["wind_speed_units"],
        input_temp_units = data["temperature_units"]
    )

    weather_data = [(entry["wind_speed"], entry["wind_direction"], entry["temperature"], entry["rel_humidity"], entry["cloud_cover"]) for entry in data["weather entries"]]

    return ninjaify(weather_data, weather_seed)

def ninjaify(weather_data: list, weather_seed: WeatherSeed) -> WeatherSeed:

    weather_entries = [
        WeatherEntry(
            wind_speed=wind_speed,
            wind_dir_deg=wind_dir,
            temp=temp,
            rel_humidity= rel_humidity,
            cloud_cover=cloud_cover
        ) for wind_speed, wind_dir, temp, rel_humidity, cloud_cover in weather_data]

    weather_seed.weather_entries = weather_entries

    return weather_seed

def filter_hourly_data(hourly_data, start_datetime, end_datetime):
    hourly_data["date"] = pd.to_datetime(hourly_data["date"])

    mask = (hourly_data["date"] >= start_datetime) & (hourly_data["date"] <= end_datetime)
    filtered_data = {key: np.array(value)[mask] for key, value in hourly_data.items()}
    return filtered_data

def weather_from_file(map: MapParams, weather: WeatherParams):
    
    with open(weather.file, 'r') as file:
        seed_data = json.load(file)

    forecast_seed = file_ninjaify(seed_data, weather)

    return run_windninja(map, forecast_seed)

