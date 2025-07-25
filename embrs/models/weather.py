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
from embrs.utilities.unit_conversions import *

# TODO: Document this file

class WeatherStream:
    def __init__(self, params: WeatherParams, geo: GeoInfo):
        self.params = params
        self.geo = geo
        input_type = params.input_type

        if input_type == "OpenMeteo":
            self.get_stream_from_openmeteo()
        elif input_type == "File":
            if geo is not None:
                self.get_stream_from_wxs()
            else:
                self.get_uniform_stream()
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
        buffered_start = start_datetime - timedelta(days=56)
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

        self.times = pd.date_range(hourly_data["date"][0], hourly_data["date"][-1], freq='h', tz=local_tz)

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

        # Compute the GSI
        # Use a 28-day period before the start of the simulation to calculate GSI (56-day for rain)
        gsi = self.calc_GSI(hourly_data, buffered_start, start_datetime)
        
        # Use GSI information to determine what to set live fuel moistures to
        self.live_h_mf, self.live_w_mf = self.set_live_moistures(gsi)

        hourly = filter_hourly_data(hourly_data, start_datetime, end_datetime)
        self.stream = list(self.generate_stream(hourly))
        
        # Calculate foliar moisture content
        self.fmc = self.calc_fmc()

        # Set units and time step based on OpenMeteo params
        self.time_step = 60
        self.input_wind_ht = 10
        self.input_wind_ht_units = "m"
        self.input_wind_vel_units = "mps"
        self.input_temp_units = "F"

    def get_stream_from_wxs(self):
        file = self.params.file

        weather_data = {
            "datetime": [],
            "temperature": [],
            "rel_humidity": [],
            "rain": [],
            "wind_speed": [],
            "wind_direction": [],
            "cloud_cover": [],
        }

        local_tz = pytz.timezone(self.geo.timezone)
        units = "english"

        # ── Step 1: Parse WXS line-by-line ───────────────────────────
        with open(file, "r") as f:
            header_found = False
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("RAWS_UNITS:"):
                    units = line.split(":")[1].strip().lower()
                    continue
                elif line.startswith("RAWS_ELEVATION:"):
                    self.ref_elev = float(line.split(":")[1].strip().lower())
                    continue
                elif line.startswith("RAWS:"):
                    continue
                elif line.startswith("Year") and not header_found:
                    header_found = True
                    continue
                if not header_found:
                    continue

                parts = line.split()
                if len(parts) != 10:
                    continue
                try:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    hour = int(parts[3].zfill(4)[:2])
                    dt = local_tz.localize(datetime(year, month, day, hour))
                    weather_data["datetime"].append(dt)
                    weather_data["temperature"].append(float(parts[4]))
                    weather_data["rel_humidity"].append(float(parts[5]))
                    weather_data["rain"].append(float(parts[6]))
                    weather_data["wind_speed"].append(float(parts[7]))
                    weather_data["wind_direction"].append(float(parts[8]))
                    weather_data["cloud_cover"].append(float(parts[9]))
                except Exception as e:
                    print(f"Skipping malformed line: {line} ({e})")
                    continue

        df = pd.DataFrame(weather_data).set_index("datetime")

        if len(df.index) < 2:
            raise ValueError("WXS file does not contain enough data to determine time step.")

        # ── Step 2: Infer time step and apply unit conversions ───────
        time_step_min = int((df.index[1] - df.index[0]).total_seconds() / 60)
        if units == "english":
            df["rain"] *= 2.54
            df["wind_speed"] *= 0.44704
            self.ref_elev = ft_to_m(self.ref_elev)
        elif units == "metric":
            df["temperature"] = df["temperature"] * 9 / 5 + 32
            df["rain"] /= 10
        else:
            raise ValueError(f"Unknown units: {units}")

        # ── Step 3: Fetch irradiance from Open-Meteo ─────────────────
        # Use buffer around sim dates
        start_datetime = local_tz.localize(self.params.start_datetime)
        end_datetime = local_tz.localize(self.params.end_datetime)

        # Check bounds
        wxs_start = df.index.min()
        wxs_end = df.index.max()

        if start_datetime < wxs_start:
            raise ValueError(f"Start datetime {start_datetime} is before WXS data begins at {wxs_start}.")

        if end_datetime > wxs_end:
            raise ValueError(f"End datetime {end_datetime} is after WXS data ends at {wxs_end}.")

        buffered_start = start_datetime - timedelta(days=56)
        buffered_end = end_datetime + timedelta(days=1)

        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://archive-api.open-meteo.com/v1/archive"
        api_input = {
            "latitude": self.geo.center_lat,
            "longitude": self.geo.center_lon,
            "start_date": buffered_start.astimezone(pytz.utc).date().strftime("%Y-%m-%d"),
            "end_date": buffered_end.astimezone(pytz.utc).date().strftime("%Y-%m-%d"),
            "hourly": ["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"],
            "timezone": "auto"
        }

        responses = openmeteo.weather_api(url, params=api_input)
        response = responses[0]

        hourly = response.Hourly()
        hourly_data = {
            "ghi": hourly.Variables(0).ValuesAsNumpy(),
            "dhi": hourly.Variables(1).ValuesAsNumpy(),
            "dni": hourly.Variables(2).ValuesAsNumpy(),
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ).tz_localize(local_tz)
        }

        irradiance_df = pd.DataFrame(hourly_data).set_index("date")

        # ── Step 4: Resample Open-Meteo irradiance to WXS resolution ─
        target_freq = f"{time_step_min}T"
        if time_step_min < 60:
            irradiance_df = irradiance_df.resample(target_freq).interpolate(method="linear")
        elif time_step_min > 60:
            irradiance_df = irradiance_df.resample(target_freq).mean()

        # ── Step 5: Align WXS and Open-Meteo data on datetime index ──
        df = df.loc[(df.index >= buffered_start) & (df.index <= end_datetime)]
        irradiance_df = irradiance_df.loc[df.index]

        df["ghi"] = irradiance_df["ghi"].values
        df["dhi"] = irradiance_df["dhi"].values
        df["dni"] = irradiance_df["dni"].values

        # ── Step 6: Add solar geometry ───────────────────────────────
        solpos = pvlib.solarposition.get_solarposition(df.index, self.geo.center_lat, self.geo.center_lon)
        df["solar_zenith"] = solpos["zenith"].values
        df["solar_azimuth"] = solpos["azimuth"].values

        # if skip_gis:
        # # TODO: actually set this up
        # # TODO: apply user defined moisture values
        #     pass
        # else:

        # Determine how far back we can go
        min_date_in_wxs = df.index.min()
        desired_start = start_datetime - timedelta(days=56)
        data_start = max(min_date_in_wxs, desired_start)

        # Calculate GSI
        gsi = self.calc_GSI(
            {
                "date": df.index,
                "temperature": df["temperature"].values,
                "rel_humidity": df["rel_humidity"].values,
                "rain": df["rain"].values,
            },
            data_start,
            start_datetime
        )

        if gsi < 0:
            # Set to dormant values
            self.live_h_mf = 0.3
            self.live_w_mf = 0.6
        else:
            self.live_h_mf, self.live_w_mf = self.set_live_moistures(gsi)
        
        # Calculate foliar moisture content
        self.fmc = self.calc_fmc()

        # ── Step 8: Package final stream ─────────────────────────────
        hourly_data["date"] = df.index
        hourly_data = filter_hourly_data(hourly_data, start_datetime, end_datetime)
        self.stream = list(self.generate_stream(hourly_data))

        # ── Step 9: Set metadata attributes ──────────────────────────
        self.time_step = time_step_min
        self.input_wind_ht = 6.1
        self.input_wind_ht_units = "m"
        self.input_wind_vel_units = "mps"
        self.input_temp_units = "F"

    def get_uniform_stream(self):
        file = self.params.file
        weather_data = {
            "datetime": [],
            "wind_speed": [],
            "wind_direction": [],
        }

        units = "english"

        # ── Step 1: Parse WXS line-by-line ───────────────────────────
        with open(file, "r") as f:
            header_found = False
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("RAWS_UNITS:"):
                    units = line.split(":")[1].strip().lower()
                    continue
                elif line.startswith("RAWS_ELEVATION:"):
                    continue
                elif line.startswith("RAWS:"):
                    continue
                elif line.startswith("Year") and not header_found:
                    header_found = True
                    continue
                if not header_found:
                    continue

                parts = line.split()
                if len(parts) != 10:
                    continue
                try:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    hour = int(parts[3].zfill(4)[:2])
                    dt = datetime(year, month, day, hour)
                    weather_data["datetime"].append(dt)
                    weather_data["wind_speed"].append(float(parts[7]))
                    weather_data["wind_direction"].append(float(parts[8]))
                except Exception as e:
                    print(f"Skipping malformed line: {line} ({e})")
                    continue

        df = pd.DataFrame(weather_data).set_index("datetime")

        if len(df.index) < 2:
            raise ValueError("WXS file does not contain enough data to determine time step.")

        # ── Step 2: Infer time step and apply unit conversions ───────
        time_step_min = int((df.index[1] - df.index[0]).total_seconds() / 60)
        if units == "english":
            df["wind_speed"] *= 0.44704
        elif units == "metric":
            df["temperature"] = df["temperature"] * 9 / 5 + 32
        else:
            raise ValueError(f"Unknown units: {units}")

        # Set live moisture values
        self.live_h_mf = 1.4
        self.live_w_mf = 1.25

        # Set Foliar moisture content to 100
        self.fmc = 100

        # Generate stream with final data
        self.stream = list(self.generate_uniform_stream(weather_data))

        # Set units and time step based on OpenMeteo params
        self.time_step = time_step_min
        self.input_wind_ht = 6.1
        self.input_wind_ht_units = 'm'
        self.input_wind_vel_units = 'mps'
        self.input_temp_units = 'F'

    def generate_uniform_stream(self, weather_data: dict) -> Iterator[WeatherEntry]:
        for wind_speed, wind_dir in zip(
            weather_data["wind_speed"],
            weather_data["wind_direction"]
        ):
            yield WeatherEntry(
                wind_speed=wind_speed,
                wind_dir_deg=wind_dir,
                temp=None,
                rel_humidity=None,
                cloud_cover=None,
                rain=None,
                dni=None,
                dhi=None,
                ghi=None,
                solar_zenith=None,
                solar_azimuth=None
            )
 
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
        # TODO: should we incldue max GSI scaling?

        # Dormant values
        h_min = 0.3
        w_min = 0.6
        
        # Green-up threshold
        gu = 0.2
        
        # Max values
        h_max = 2.5
        w_max = 2.0

        if gsi < gu:
            return h_min, w_min

        else:
            m_h = (h_max - h_min) / (1.0 - gu)
            m_w = (w_max - w_min) / (1.0 - gu)

        live_h_mf = m_h * gsi + (h_max - m_h)
        live_w_mf = m_w * gsi + (w_max - m_w)

        return live_h_mf, live_w_mf

    def calc_GSI(self, hourly_data, data_start, sim_start) -> float:
        """
        Calculate the Growing Season Index (GSI) over the period leading up to sim_start.
        Adjusts to use as much data as available if less than 28 or 56 days exists.
        """
        # Determine the maximum available pre-simulation range
        min_available_date = pd.to_datetime(hourly_data["date"]).min()
        desired_non_rain_start = sim_start - timedelta(days=28)
        desired_rain_start = sim_start - timedelta(days=56)

        if desired_non_rain_start < min_available_date:
            # Use as much data as possible for non-rain metrics
            non_rain_data_start = min_available_date
        else:
            non_rain_data_start = desired_non_rain_start

        if desired_rain_start < min_available_date:
            # Use as much data as possible for rain metrics
            data_start = min_available_date
        else:
            data_start = desired_rain_start

        # Filter data
        hourly = filter_hourly_data(hourly_data, non_rain_data_start, sim_start)
        rain_hourly_data = filter_hourly_data(hourly_data, data_start, sim_start)

        # Rain requires a longer lead up period
        rain_df = pd.DataFrame(rain_hourly_data).set_index("date")
        daily_precipitation = rain_df["rain"].resample('D').sum()        

        # Get daily data
        hourly_df = pd.DataFrame(hourly).set_index("date")
        daily_min_temp = hourly_df["temperature"].resample('D').min()
        daily_max_temp = hourly_df["temperature"].resample('D').max()
        daily_min_rh = hourly_df["rel_humidity"].resample('D').min()
        dates = daily_min_temp.index

        if len(dates) < 2:
            print(
                "Warning: Not enough pre-simulation data to compute GSI. "
                "Live moistures will be set to dormant values. "
                "To fix this, ensure the weather data file contains at least 2 days, preferably 28 days, "
                "before the simulation start date, or manually enter live moistures."
            )
            return -1

        gsi = 0.0

        # For each day, calculate iGSI
        for day in range(len(dates)):
            date = dates[day]
            times = pd.date_range(date, periods=1, freq='D', tz=self.geo.timezone)
            pv_loc = pvlib.location.Location(self.geo.center_lat, self.geo.center_lon, tz=self.geo.timezone)
            solpos = pv_loc.get_sun_rise_set_transit(times)
            sunrise = solpos['sunrise'].iloc[0]
            sunset = solpos['sunset'].iloc[0]

            if sunset < sunrise:
                sunset += pd.Timedelta(days=1)
            day_len = (sunset - sunrise).total_seconds() 

            iPhoto = (day_len - 36000) / (39600 - 36000)
            iPhoto = min(max(iPhoto, 0), 1)

            min_temp = F_to_C(daily_min_temp.iloc[day])
            iTmin = (min_temp + 2)/(5 + 2)
            iTmin = min(max(iTmin, 0), 1)

            max_temp = F_to_C(daily_max_temp.iloc[day])
            min_rh = daily_min_rh.iloc[day]
            vpd = (1 - min_rh / 100) * 0.6108 * np.exp((17.27 * max_temp) / (max_temp + 237.3))
            iVPD = (vpd-0.9)/(4.1-0.9)
            iVPD = min(max(iVPD, 0), 1)

            # Adjust precipitation look-back for available data
            rain_idx = min(28, len(daily_precipitation))
            tot_rain = daily_precipitation.iloc[max(0, day - rain_idx):day].sum() * 10  # mm
            iPrcp = (tot_rain - 0)/(10 - 0)
            iPrcp = min(max(iPrcp, 0), 1)

            gsi += iTmin * iPhoto * iVPD * iPrcp

        # Average GSI
        gsi /= len(dates)
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
        
        # Estimate day of the year where the min FMC occurs
        d_0 = 151 * (lat / latn) + 0.0172 * self.ref_elev

        # Estimate the number of days away current date is from d_0
        nd = min(abs(d_j - d_0), 365 - abs(d_j - d_0))

        # Calculate fmc based on nd threshold
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

    # Convert temp to celsius
    temp = F_to_C(temp)

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