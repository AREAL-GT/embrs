from retry_requests import retry
from datetime import timedelta
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import pytz
import json
import pvlib
from typing import Iterator

from embrs.utilities.data_classes import *

# TODO: Document this file

class WeatherStream:
    def __init__(self, params: WeatherParams, geo: GeoInfo, input_type = "OpenMeteo"):
        self.params = params

        if input_type == "OpenMeteo":
            self.get_stream_from_openmeteo(params, geo)
        elif input_type == "File":
            self.get_stream_from_file(params, geo)
        else:
            raise ValueError("Invalid weather input_type, must be either 'OpenMeteo' or 'File'")


        print(f"len of weather stream: {len(self.stream)}")


    def get_stream_from_openmeteo(self, params: WeatherParams, geo: GeoInfo):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        local_tz = pytz.timezone(geo.timezone)

        # Buffer times and format for OpenMeteo
        start_datetime = params.start_datetime
        start_datetime = local_tz.localize(start_datetime)
        buffered_start = start_datetime - timedelta(days=1)
        start_datetime_utc = buffered_start.astimezone(pytz.utc)

        end_datetime = params.end_datetime
        end_datetime = local_tz.localize(end_datetime)
        buffered_end = end_datetime + timedelta(days=1)
        end_datetime_utc = buffered_end.astimezone(pytz.utc)

        url = "https://archive-api.open-meteo.com/v1/archive"
        api_input = {
            "latitude": geo.center_lat,
            "longitude": geo.center_lon,
            "start_date": start_datetime_utc.date().strftime("%Y-%m-%d"),
            "end_date": end_datetime_utc.date().strftime("%Y-%m-%d"),
            "hourly": ["wind_speed_10m", "wind_direction_10m", "temperature_2m", "relative_humidity_2m", "cloud_cover", "shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance", "rain"],
            "wind_speed_unit": "ms",
            "temperature_unit": "fahrenheit",
            "timezone": "auto"
        }
        responses = openmeteo.weather_api(url, params=api_input)
        response = responses[0]

        self.ref_elev = response.Elevation()
        hourly = response.Hourly()
        hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(1).ValuesAsNumpy()
        hourly_temperature_2m = hourly.Variables(2).ValuesAsNumpy()
        hourly_rel_humidity_2m = hourly.Variables(3).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
        hourly_ghi = hourly.Variables(5).ValuesAsNumpy()
        hourly_dhi = hourly.Variables(6).ValuesAsNumpy()
        hourly_dni = hourly.Variables(7).ValuesAsNumpy()
        hourly_rain_mm = hourly.Variables(8).ValuesAsNumpy()

        hourly_data = {}

        hourly_data["date"] = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ).tz_localize(local_tz)

        self.times = pd.date_range(hourly_data["date"][0], hourly_data["date"][-1], freq='H', tz=local_tz)

        hourly_data["wind_speed"] = hourly_wind_speed_10m
        hourly_data["wind_direction"] = hourly_wind_direction_10m
        hourly_data["temperature"] = hourly_temperature_2m
        hourly_data["rel_humidity"] = hourly_rel_humidity_2m
        hourly_data["cloud_cover"] = hourly_cloud_cover
        hourly_data["ghi"] = hourly_ghi
        hourly_data["dhi"] = hourly_dhi
        hourly_data["dni"] = hourly_dni
        hourly_data["rain"] = hourly_rain_mm * 0.1 # convert to cm
        
        solpos = pvlib.solarposition.get_solarposition(self.times, geo.center_lat, geo.center_lon)
        
        hourly_data["solar_zenith"] = solpos["zenith"].values
        hourly_data["solar_azimuth"] = solpos["azimuth"].values

        hourly_data = filter_hourly_data(hourly_data, start_datetime, end_datetime)
        self.stream = list(self.generate_stream(hourly_data))

        # Set units and time step based on OpenMeteo params
        self.time_step = 60
        self.input_wind_ht = 10
        self.input_wind_ht_units = "m"
        self.input_wind_vel_units = "mps"
        self.input_temp_units = "F"

    def get_stream_from_file(self, params: WeatherParams, geo: GeoInfo):

        file = params.file
        with open(file) as f:
            data = json.load(f)

        # Get the time step, used to resample if necessary
        time_step_hr = data["time_step_min"] / 60

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        local_tz = pytz.timezone(geo.timezone)

        # Buffer times and format for OpenMeteo
        start_datetime = params.start_datetime
        start_datetime = local_tz.localize(start_datetime)
        buffered_start = start_datetime - timedelta(days=1)
        start_datetime_utc = buffered_start.astimezone(pytz.utc)

        end_datetime = params.end_datetime
        end_datetime = local_tz.localize(end_datetime)
        buffered_end = end_datetime + timedelta(days=1)
        end_datetime_utc = buffered_end.astimezone(pytz.utc)

        url = "https://archive-api.open-meteo.com/v1/archive"
        api_input = {
            "latitude": geo.center_lat,
            "longitude": geo.center_lon,
            "start_date": start_datetime_utc.date().strftime("%Y-%m-%d"),
            "end_date": end_datetime_utc.date().strftime("%Y-%m-%d"),
            "hourly": ["cloud_cover", "shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"],
            "timezone": "auto"
        }
        responses = openmeteo.weather_api(url, params=api_input)
        response = responses[0]

        self.ref_elev = response.Elevation()
        hourly = response.Hourly()
        hourly_cloud_cover = hourly.Variables(0).ValuesAsNumpy()
        hourly_ghi = hourly.Variables(1).ValuesAsNumpy()
        hourly_dhi = hourly.Variables(2).ValuesAsNumpy()
        hourly_dni = hourly.Variables(3).ValuesAsNumpy()

        hourly_data = {}

        hourly_data["date"] = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ).tz_localize(local_tz)

        self.times = pd.date_range(hourly_data["date"][0], hourly_data["date"][-1], freq='h', tz=local_tz)

        hourly_data["cloud_cover"] = hourly_cloud_cover
        hourly_data["ghi"] = hourly_ghi
        hourly_data["dhi"] = hourly_dhi
        hourly_data["dni"] = hourly_dni
        
        solpos = pvlib.solarposition.get_solarposition(self.times, geo.center_lat, geo.center_lon)
        
        hourly_data["solar_zenith"] = solpos["zenith"].values
        hourly_data["solar_azimuth"] = solpos["azimuth"].values

        # Convert Open-Meteo hourly data into DataFrame
        df_hourly = pd.DataFrame({
            "date": hourly_data["date"],
            "cloud_cover": hourly_data["cloud_cover"],
            "ghi": hourly_data["ghi"],
            "dhi": hourly_data["dhi"],
            "dni": hourly_data["dni"],
            "solar_zenith": hourly_data["solar_zenith"],
            "solar_azimuth": hourly_data["solar_azimuth"],
        })

        # Set index for resampling
        df_hourly.set_index("date", inplace=True)

        # Define the target frequency
        target_freq = f"{int(time_step_hr * 60)}T"  # Convert hours to minutes

        # Only resample Open-Meteo data if `time_step_hr` differs from 1 hour
        if time_step_hr < 1:  # Upsampling: interpolate missing values
            df_resampled = df_hourly.resample(target_freq).interpolate(method="linear")
        elif time_step_hr > 1:  # Downsampling: aggregate with mean
            df_resampled = df_hourly.resample(target_freq).mean()
        else:  # time_step_hr == 1: Use the original data
            df_resampled = df_hourly.copy()

        # Convert resampled Open-Meteo data to dictionary
        resampled_hourly_data = {col: df_resampled[col].values for col in df_resampled.columns}
        resampled_hourly_data["date"] = df_resampled.index
        
        resampled_hourly_data = filter_hourly_data(resampled_hourly_data, start_datetime, end_datetime)

        # File-based data is already at the correct time step, so use it directly
        resampled_hourly_data["wind_speed"] = data["weather_entries"]["wind_speed"]
        resampled_hourly_data["wind_direction"] = data["weather_entries"]["wind_direction"]
        resampled_hourly_data["temperature"] = data["weather_entries"]["temperature"]
        resampled_hourly_data["rel_humidity"] = data["weather_entries"]["rel_humidity"]

        # Apply rain unit conversion if necessary
        rain_units = data["rain_units"]
        if rain_units == "mm":
            resampled_hourly_data["rain"] = np.array(data["weather_entries"]["rain"]) * 0.1
        elif rain_units == "in":
            resampled_hourly_data["rain"] = np.array(data["weather_entries"]["rain"]) * 2.54
        elif rain_units == "cm":
            resampled_hourly_data["rain"] = np.array(data["weather_entries"]["rain"])
        else:
            raise ValueError("Rain units invalid, must be one of 'mm', 'in', or 'cm'")

        # Update times after filtering
        self.times = resampled_hourly_data["date"]

        # Generate stream with final data
        self.stream = list(self.generate_stream(resampled_hourly_data))

        # Set units and time step based on OpenMeteo params
        self.time_step = data["time_step_min"]
        self.input_wind_ht = data["wind_height"]
        self.input_wind_ht_units = data["wind_height_units"]
        self.input_wind_vel_units = data["wind_speed_units"]
        self.input_temp_units = data["temperature_units"]

    def generate_stream(self, hourly_data: dict) -> Iterator[WeatherEntry]:
        for wind_speed, wind_dir, temp, rel_humidity, cloud_cover, ghi, dhi, dni, rain, solar_zenith, solar_azimuth in zip(
            hourly_data["wind_speed"],
            hourly_data["wind_direction"],
            hourly_data["temperature"],
            hourly_data["rel_humidity"],
            hourly_data["cloud_cover"],
            hourly_data["ghi"],
            hourly_data["dhi"],
            hourly_data["dni"],
            hourly_data["rain"],
            hourly_data["solar_zenith"],
            hourly_data["solar_azimuth"]
        ):
            yield WeatherEntry(
                wind_speed=wind_speed,
                wind_dir_deg=wind_dir,
                temp=temp,
                rel_humidity=rel_humidity,
                cloud_cover=cloud_cover,
                rain = rain,
                dni=dni,
                dhi=dhi,
                ghi=ghi,
                solar_zenith=solar_zenith,
                solar_azimuth=solar_azimuth
            )

def filter_hourly_data(hourly_data, start_datetime, end_datetime):
    hourly_data["date"] = pd.to_datetime(hourly_data["date"])

    mask = (hourly_data["date"] >= start_datetime) & (hourly_data["date"] <= end_datetime)
    filtered_data = {key: np.array(value)[mask] for key, value in hourly_data.items()}
    return filtered_data

# TODO: need to implement a version of above for file input

def apply_site_specific_correction(cell, elev_ref: float, curr_weather: WeatherEntry):
    ## elev_ref is in meters, temp_air is in Fahrenheit, rh_air is %
    elev_diff = elev_ref - cell.z 
    elev_diff *= 3.2808 # convert to ft

    dewptref = -398.0-7469.0 / (np.log(curr_weather.rel_humidity/100.0)-7469.0/(curr_weather.temp+398.0))

    temp = curr_weather.temp + elev_diff/1000.0*5.5 # Stephenson 1988 found summer adiabat at 3.07 F/1000ft
    dewpt = dewptref + elev_diff/1000.0*1.1 # new humidity, new dewpt, and new humidity
    rh = 7469.0*(1.0/(temp+398.0)-1.0/(dewpt+398.0))
    rh = np.exp(rh)*100.0 #convert from ln units
    if (rh > 99.0):
        rh=99.0

    # Convert temp to celius
    temp = temp-32
    temp /= 1.8 

    # Return humidity as a decimal
    rh /= 100.0

    return temp, rh

def calc_local_solar_radiation(cell, curr_weather: WeatherEntry):
    # Calculate total irradiance using pvlib
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=cell.slope_deg,
        surface_azimuth=cell.aspect,
        solar_zenith=curr_weather.solar_zenith,
        solar_azimuth=curr_weather.solar_azimuth,
        dni=curr_weather.dni,
        ghi=curr_weather.ghi,
        dhi=curr_weather.dhi,
        model='isotropic'
    )

    # Adjust for canopy transmittance (Only thing not modelled in pvlib)
    canopy_transmittance = 1 - (cell.canopy_cover/100)
    I = total_irradiance['poa_global'] * canopy_transmittance

    return I