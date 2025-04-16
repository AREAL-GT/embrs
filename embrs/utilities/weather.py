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
from astropy.time import Time

from embrs.utilities.data_classes import *

# TODO: Document this file

class WeatherStream:
    def __init__(self, params: WeatherParams, geo: GeoInfo, input_type = "OpenMeteo"):
        self.params = params
        self.geo = geo

        if input_type == "OpenMeteo":
            self.get_stream_from_openmeteo()
        elif input_type == "File":
            self.get_stream_from_file()
        else:
            raise ValueError("Invalid weather input_type, must be either 'OpenMeteo' or 'File'")

    def get_stream_from_openmeteo(self):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        local_tz = pytz.timezone(self.geo.timezone)

        # Buffer times and format for OpenMeteo
        start_datetime = self.params.start_datetime
        start_datetime = local_tz.localize(start_datetime)
        buffered_start = start_datetime - timedelta(days=1)
        start_datetime_utc = buffered_start.astimezone(pytz.utc)

        end_datetime = self.params.end_datetime
        end_datetime = local_tz.localize(end_datetime)
        buffered_end = end_datetime + timedelta(days=1)
        end_datetime_utc = buffered_end.astimezone(pytz.utc)

        url = "https://archive-api.open-meteo.com/v1/archive"
        api_input = {
            "latitude": self.geo.center_lat,
            "longitude": self.geo.center_lon,
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
        
        solpos = pvlib.solarposition.get_solarposition(self.times, self.geo.center_lat, self.geo.center_lon)
        
        hourly_data["solar_zenith"] = solpos["zenith"].values
        hourly_data["solar_azimuth"] = solpos["azimuth"].values

        # Convert hourly data into a DataFrame
        hourly_df = pd.DataFrame(hourly_data).set_index("date")

        # Extract morning (08:00) and afternoon (14:00) values
        daily_morning_rh = hourly_df.between_time("08:00", "08:59")["rel_humidity"].resample('D').mean()
        daily_afternoon_rh = hourly_df.between_time("14:00", "14:59")["rel_humidity"].resample('D').mean()

        # Compute daily RH average as per NFDRS convention
        daily_avg_rh = (daily_morning_rh + daily_afternoon_rh) / 2

        # Compute daily temperature mean (can use max/min if needed)
        daily_avg_temp = hourly_df.resample('D')["temperature"].mean()  # Alternative: (Tmax + Tmin) / 2

        # Ensure temperature and rel_humidity have the same length
        min_length = min(len(daily_avg_rh), len(daily_avg_temp))

        # Trim both to the minimum length
        daily_avg_rh = daily_avg_rh.iloc[:min_length]
        daily_avg_temp = daily_avg_temp.iloc[:min_length]

        # Ensure date index also matches
        daily_index = daily_avg_rh.index  # Use RH index (since RH is usually limiting)
        daily_avg_temp = daily_avg_temp.reindex(daily_index)  # Align temp to RH

        # Create final daily DataFrame
        daily_data = pd.DataFrame({
            "date": daily_avg_rh.index,
            "temperature": daily_avg_temp.values,
            "rel_humidity": daily_avg_rh.values
        }).reset_index(drop=True)

        hourly_data = filter_hourly_data(hourly_data, start_datetime, end_datetime)
        self.stream = list(self.generate_stream(hourly_data))

        gsi = self.calc_GSI(daily_data)
        
        # Use GSI information to determine what to set live fuel moistures to
        self.live_h_mf, self.live_w_mf = self.set_live_moistures(gsi)

        # Calculate foliar moisture content
        self.fmc = self.calc_fmc()

        # Set units and time step based on OpenMeteo params
        self.time_step = 60
        self.input_wind_ht = 10
        self.input_wind_ht_units = "m"
        self.input_wind_vel_units = "mps"
        self.input_temp_units = "F"

    def get_stream_from_file(self):

        file = self.params.file
        with open(file) as f:
            data = json.load(f)

        # Get the time step, used to resample if necessary
        time_step_hr = data["time_step_min"] / 60

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        local_tz = pytz.timezone(self.geo.timezone)

        # Buffer times and format for OpenMeteo
        start_datetime = self.params.start_datetime
        start_datetime = local_tz.localize(start_datetime)
        buffered_start = start_datetime - timedelta(days=1)
        start_datetime_utc = buffered_start.astimezone(pytz.utc)

        end_datetime = self.params.end_datetime
        end_datetime = local_tz.localize(end_datetime)
        buffered_end = end_datetime + timedelta(days=1)
        end_datetime_utc = buffered_end.astimezone(pytz.utc)

        url = "https://archive-api.open-meteo.com/v1/archive"
        api_input = {
            "latitude": self.geo.center_lat,
            "longitude": self.geo.center_lon,
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
        
        solpos = pvlib.solarposition.get_solarposition(self.times, self.geo.center_lat, self.geo.center_lon)
        
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

        # Ensure all resampled data arrays have the same length
        min_length = min(len(arr) for arr in resampled_hourly_data.values() if isinstance(arr, np.ndarray))

        # Trim all arrays to the minimum length
        for key in resampled_hourly_data:
            if isinstance(resampled_hourly_data[key], np.ndarray):
               resampled_hourly_data[key] = resampled_hourly_data[key][:min_length]

        # Convert resampled data into a DataFrame
        resampled_df = pd.DataFrame(resampled_hourly_data).set_index("date")

        # Extract morning (08:00) and afternoon (14:00) values
        daily_morning_rh = resampled_df.between_time("08:00", "08:59")["rel_humidity"].resample('D').mean()
        daily_afternoon_rh = resampled_df.between_time("14:00", "14:59")["rel_humidity"].resample('D').mean()

        # Compute daily RH average as per NFDRS convention
        daily_avg_rh = pd.DataFrame({
            "morning": daily_morning_rh,
            "afternoon": daily_afternoon_rh
        }).mean(axis=1)

        # Compute daily temperature mean (can use max/min if needed)
        daily_avg_temp = resampled_df.resample('D')["temperature"].mean()  # Alternative: (Tmax + Tmin) / 2

        # Ensure temperature and rel_humidity have the same length
        min_length = min(len(daily_avg_rh), len(daily_avg_temp))

        # Trim both to the minimum length
        daily_avg_rh = daily_avg_rh.iloc[:min_length]
        daily_avg_temp = daily_avg_temp.iloc[:min_length]

        # Ensure date index also matches
        daily_index = daily_avg_rh.index  # Use RH index (since RH is usually limiting)
        daily_avg_temp = daily_avg_temp.reindex(daily_index)  # Align temp to RH

        # Create final daily DataFrame
        daily_data = pd.DataFrame({
            "date": daily_avg_rh.index,
            "temperature": daily_avg_temp.values,
            "rel_humidity": daily_avg_rh.values
        }).reset_index(drop=True)

        gsi = self.calc_GSI(daily_data)

        # Use GSI information to determine what live fuel moisture is
        self.live_h_mf, self.live_w_mf = self.set_live_moistures(gsi)

        # Calculate foliar moisture content
        self.fmc = self.calc_fmc()

        # Generate stream with final data
        self.stream = list(self.generate_stream(resampled_hourly_data))

        # Set units and time step based on OpenMeteo params
        self.time_step = data["time_step_min"]
        self.input_wind_ht = data["wind_height"]
        self.input_wind_ht_units = data["wind_height_units"]
        self.input_wind_vel_units = data["wind_speed_units"]
        self.input_temp_units = data["temperature_units"]

    def generate_stream(self, hourly_data: dict) -> Iterator[WeatherEntry]:
        cum_rain = 0
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
            cum_rain += rain
            yield WeatherEntry(
                wind_speed=wind_speed,
                wind_dir_deg=wind_dir,
                temp=temp,
                rel_humidity=rel_humidity,
                cloud_cover=cloud_cover,
                rain = cum_rain,
                dni=dni,
                dhi=dhi,
                ghi=ghi,
                solar_zenith=solar_zenith,
                solar_azimuth=solar_azimuth
            )

    def set_live_moistures(self, gsi: float):
        # Set the live moisture for each class based on GSI
        
        # Dormant values
        h_dorm = 0.3
        w_dorm = 0.5
        
        # Max values
        h_max = 2.5
        w_max = 2.0

        # Return dormant if gsi < 0.5
        if gsi < 0.5:
            return h_dorm, w_dorm
        

        # Moisture varies linearly with gsi otherwise
        h_range = h_max - h_dorm
        w_range = w_max - w_dorm

        intrp_pt = gsi - 0.5

        live_h_mf = intrp_pt * (h_range/0.5) + 0.3
        live_w_mf = intrp_pt * (w_range/0.5) + 0.5

        return live_h_mf, live_w_mf

    def calc_GSI(self, daily_data) -> float:
        # Initial GSI
        gsi = 0

        for day in range(len(daily_data["date"])):
            # Calculate the length of the day with pv lib
            date = daily_data["date"][day]
            times = pd.date_range(date, periods=1, freq='D', tz=self.geo.timezone)
            pv_loc = pvlib.location.Location(self.geo.center_lat, self.geo.center_lon, tz=self.geo.timezone)
            solpos = pv_loc.get_sun_rise_set_transit(times) #, self.geo.center_lat, self.geo.center_lon)
            sunrise = solpos['sunrise'].iloc[0]
            sunset = solpos['sunset'].iloc[0]

            # Ensure sunset is after sunrise
            if sunset < sunrise:
                sunset += pd.Timedelta(days=1)

            day_len = (sunset - sunrise).total_seconds()/3600 

            # Calculate the photo indicator function using day length
            iPhoto = (day_len - 10)
            iPhoto = min(max(iPhoto, 0), 1)

            # Get the average temperature and humidities
            temp = daily_data["temperature"][day]
            rel_humidity = daily_data["rel_humidity"][day]

            # Calculate the temperature indicator function
            iTmin = (temp-28)/(41-28)
            iTmin = min(max(iTmin, 0), 1)

            # Calculate vapour pressure deficit
            temp_c = (temp - 32) * (5/9)
            vpd = (1 - rel_humidity / 100) * 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3)) #kPa
            
            # Calculate vapour pressure deficit indicator function
            iVPD = (vpd-4.1)/(0.9-4.1)
            iVPD = min(max(iVPD, 0), 1)

            # Compute the GSI for the day
            gsi += iTmin * iPhoto * iVPD

        # Get the average GSI for the full sim
        gsi /= len(daily_data["date"])
        
        return gsi

    def calc_fmc(self):
        # Calcualte the foliar moisture content
        # Based on Forestry Canada Fire Danger Group 1992
        lat = self.geo.center_lat
        lon = -self.geo.center_lon

        # Convert start date to julian date
        date = self.params.start_datetime
        d_j = date.timetuple().tm_yday


        latn = 43 + 33.7 * np.exp(-0.0351 * (150 - lon))

        d_0 = 151 * (lat / latn) + 0.0172 * self.ref_elev

        nd = min(abs(d_j - d_0), 365 - abs(d_j - d_0))


        if nd < 30:
            fmc = 85 + 0.0189 * nd**2

        elif 30 <= nd < 50:
            fmc = 32.9 + 3.17 * nd - 0.0288 * nd**2

        else:
            fmc = 120

        return fmc

def filter_hourly_data(hourly_data, start_datetime, end_datetime):
    hourly_data["date"] = pd.to_datetime(hourly_data["date"])

    mask = (hourly_data["date"] >= start_datetime) & (hourly_data["date"] <= end_datetime)
    filtered_data = {key: np.array(value)[mask] for key, value in hourly_data.items()}
    return filtered_data

def apply_site_specific_correction(cell, elev_ref: float, curr_weather: WeatherEntry):
    ## elev_ref is in meters, temp_air is in Fahrenheit, rh_air is %
    elev_diff = elev_ref - cell.elevation_m 
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

def datetime_to_julian_date(dt):
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour + dt.minute / 60 + dt.second / 3600

    if month <= 2:
        year -= 1
        month += 12

    A = np.floor(year / 100)
    B = 2 - A + np.floor(A / 4)

    jd_day = np.floor(365.25 * (year + 4716)) + \
            np.floor(30.6001 * (month + 1)) + \
            day + B - 1524.5

    jd = jd_day + hour / 24
    return jd