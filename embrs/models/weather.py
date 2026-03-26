"""Weather data ingestion and processing for fire simulation.

Fetch, parse, and package hourly weather observations into a stream of
``WeatherEntry`` records for use by the fire simulation. Supports two
input sources:

- **OpenMeteo**: Historical reanalysis data fetched via the Open-Meteo API.
- **File**: RAWS-format weather station files (``.wxs``).

Also computes derived quantities: site-specific temperature/humidity
corrections, local solar radiation, foliar moisture content (FMC), and
the Growing Season Index (GSI) for estimating live fuel moisture.

Classes:
    - WeatherStream: Build a weather stream from config parameters.

Functions:
    - filter_hourly_data: Subset hourly data by datetime range.
    - apply_site_specific_correction: Elevation-lapse adjustment for
        temperature and humidity.
    - calc_local_solar_radiation: Slope- and canopy-adjusted irradiance.
    - datetime_to_julian_date: Convert a datetime to Julian date.
"""

from __future__ import annotations

from retry_requests import retry
from collections import deque
from dataclasses import dataclass
from datetime import timedelta, datetime
import math
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import pytz
import pvlib
from typing import Iterator, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from embrs.fire_simulator.cell import Cell

from embrs.utilities.data_classes import *
from embrs.utilities.unit_conversions import *


@dataclass
class DailySummary:
    """One day of aggregated weather for GSI computation.

    Attributes:
        date: Calendar date for this summary.
        min_temp_F: Daily minimum temperature (Fahrenheit).
        max_temp_F: Daily maximum temperature (Fahrenheit).
        min_rh: Daily minimum relative humidity (percent, 0-100).
        rain_cm: Total rainfall for this day (cm).
    """
    date: datetime
    min_temp_F: float
    max_temp_F: float
    min_rh: float
    rain_cm: float


class GSITracker:
    """Track rolling weather data and recompute GSI during simulation.

    Maintains a 56-day rolling buffer of :class:`DailySummary` records
    (seeded from pre-simulation data) and accumulates hourly weather
    entries into daily summaries as the simulation advances. Call
    :meth:`compute_gsi` after each day boundary to get an updated
    Growing Season Index.

    Args:
        geo: :class:`GeoInfo` with ``center_lat``, ``center_lon``, and
            ``timezone`` for photoperiod calculations.
        pre_sim_summaries: Daily summaries from the pre-simulation window
            (up to 56 days). Oldest first.
        initial_cum_rain: Cumulative rain value (cm) from the last
            pre-simulation weather entry, used to correctly compute
            rain deltas for the first simulation hour.
    """

    def __init__(self, geo: GeoInfo, pre_sim_summaries: List[DailySummary],
                 initial_cum_rain: float = 0.0):
        self._geo = geo
        self._daily_buffer: deque[DailySummary] = deque(maxlen=56)
        for s in pre_sim_summaries:
            self._daily_buffer.append(s)

        # Current-day accumulators
        self._hourly_temps: List[float] = []
        self._hourly_rhs: List[float] = []
        self._hourly_rain_cm: float = 0.0
        self._last_cum_rain: float = initial_cum_rain
        self._current_date: Optional[datetime] = None

    def ingest_hourly(self, entry: WeatherEntry, sim_datetime: datetime) -> bool:
        """Feed one hourly weather observation into the tracker.

        Accumulates temperature, humidity, and rain delta. When the
        calendar day changes, the previous day is finalized into a
        :class:`DailySummary` and appended to the rolling buffer.

        Args:
            entry: Current weather entry (temp in F, rel_humidity in
                percent, rain cumulative in cm).
            sim_datetime: Simulation datetime corresponding to this entry.

        Returns:
            True if a day boundary was crossed (a new daily summary was
            finalized), False otherwise.
        """
        entry_date = sim_datetime.date()
        day_changed = False

        if self._current_date is not None and entry_date != self._current_date:
            # Day boundary — finalize the previous day
            if self._hourly_temps:
                self._finalize_day()
                day_changed = True

            # Reset accumulators for the new day
            self._hourly_temps = []
            self._hourly_rhs = []
            self._hourly_rain_cm = 0.0

        self._current_date = entry_date

        # Accumulate hourly observations
        self._hourly_temps.append(entry.temp)
        self._hourly_rhs.append(entry.rel_humidity)

        # Rain delta from cumulative value
        if self._last_cum_rain is not None:
            delta = entry.rain - self._last_cum_rain
            if delta > 0:
                self._hourly_rain_cm += delta
        self._last_cum_rain = entry.rain

        return day_changed

    def _finalize_day(self):
        """Build a DailySummary from the current-day accumulators and
        append it to the rolling buffer."""
        summary = DailySummary(
            date=self._current_date,
            min_temp_F=min(self._hourly_temps),
            max_temp_F=max(self._hourly_temps),
            min_rh=min(self._hourly_rhs),
            rain_cm=self._hourly_rain_cm,
        )
        self._daily_buffer.append(summary)

    def compute_gsi(self) -> float:
        """Compute GSI from the rolling daily buffer.

        Uses the same sub-index formulas as
        :meth:`WeatherStream.calc_GSI`: photoperiod, minimum temperature,
        vapor pressure deficit, and 28-day accumulated precipitation.

        Returns:
            GSI value in [0, 1], or -1 if fewer than 2 days are available.
        """
        buf = self._daily_buffer
        n = len(buf)
        if n < 2:
            return -1.0

        # Use last 28 entries for non-rain sub-indices
        start = max(0, n - 28)

        pv_loc = pvlib.location.Location(
            self._geo.center_lat, self._geo.center_lon, tz=self._geo.timezone
        )

        gsi = 0.0
        count = 0

        for i in range(start, n):
            day_summary = buf[i]

            # Photoperiod sub-index
            times = pd.date_range(
                pd.Timestamp(day_summary.date), periods=1, freq='D',
                tz=self._geo.timezone
            )
            solpos = pv_loc.get_sun_rise_set_transit(times)
            sunrise = solpos['sunrise'].iloc[0]
            sunset = solpos['sunset'].iloc[0]
            if sunset < sunrise:
                sunset += pd.Timedelta(days=1)
            day_len = (sunset - sunrise).total_seconds()
            iPhoto = max(0.0, min(1.0, (day_len - 36000) / (39600 - 36000)))

            # Min temperature sub-index (convert F → C)
            min_temp_C = F_to_C(day_summary.min_temp_F)
            iTmin = max(0.0, min(1.0, (min_temp_C + 2) / (5 + 2)))

            # VPD sub-index
            max_temp_C = F_to_C(day_summary.max_temp_F)
            min_rh = day_summary.min_rh
            vpd = (1 - min_rh / 100) * 0.6108 * math.exp(
                (17.27 * max_temp_C) / (max_temp_C + 237.3)
            )
            iVPD = max(0.0, min(1.0, (vpd - 0.9) / (4.1 - 0.9)))

            # Precipitation sub-index: 28-day accumulated rain ending at day i
            rain_start = max(0, i - 28)
            tot_rain_mm = sum(buf[j].rain_cm * 10 for j in range(rain_start, i))
            iPrcp = max(0.0, min(1.0, tot_rain_mm / 10))

            gsi += iTmin * iPhoto * iVPD * iPrcp
            count += 1

        return gsi / count


class WeatherStream:
    """Build and manage a weather stream for fire simulation.

    Ingest weather data from either the Open-Meteo API or a RAWS-format
    ``.wxs`` file, compute derived quantities (solar geometry, GSI, live
    fuel moisture, foliar moisture content), and produce a list of
    ``WeatherEntry`` records indexed by time.

    Attributes:
        stream (list[WeatherEntry]): Ordered weather entries for the
            simulation period (including conditioning lead-in).
        stream_times (pd.DatetimeIndex): Timestamps corresponding to
            each entry in ``stream``.
        sim_start_idx (int): Index into ``stream`` where the actual
            simulation begins (after conditioning period).
        fmc (float): Foliar moisture content (percent).
        live_h_mf (float | None): Live herbaceous fuel moisture (fraction),
            or None if GSI is disabled.
        live_w_mf (float | None): Live woody fuel moisture (fraction),
            or None if GSI is disabled.
        time_step (int): Weather observation interval (minutes).
        ref_elev (float): Reference elevation of the weather source (meters).
    """

    def __init__(self, params: WeatherParams, geo: GeoInfo, use_gsi: bool = True):
        """Initialize a weather stream from configuration parameters.

        Args:
            params (WeatherParams): Weather configuration specifying input
                type, date range, and optional file path.
            geo (GeoInfo): Geographic information (lat, lon, timezone,
                center coordinates).
            use_gsi (bool): Whether to compute the Growing Season Index for
                live fuel moisture estimation. Defaults to True.

        Raises:
            ValueError: If ``params.input_type`` is not 'OpenMeteo' or
                'File', or if 'File' is used without ``geo``.
        """
        self.params = params
        self.geo = geo
        self.use_gsi = use_gsi
        input_type = params.input_type

        if input_type == "OpenMeteo":
            self.get_stream_from_openmeteo()
        elif input_type == "File":
            if geo is not None:
                self.get_stream_from_wxs()
            else:
                raise ValueError("GeoInfo must be provided when using 'File' weather input_type")
        else:
            raise ValueError("Invalid weather input_type, must be either 'OpenMeteo' or 'File'")

    def get_stream_from_openmeteo(self):
        """Fetch weather data from the Open-Meteo historical archive API.

        Retrieves hourly wind, temperature, humidity, cloud cover, solar
        radiation, and precipitation. Computes GSI-based live fuel moisture
        if enabled, then builds the weather stream with a conditioning
        lead-in period.

        Side Effects:
            Sets ``self.stream``, ``self.stream_times``, ``self.sim_start_idx``,
            ``self.fmc``, ``self.live_h_mf``, ``self.live_w_mf``,
            ``self.ref_elev``, ``self.time_step``, and input unit attributes.
        """
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        local_tz = pytz.timezone(self.geo.timezone)

        # Buffer times and format for OpenMeteo
        conditioning_start = self.params.conditioning_start
        conditioning_start = local_tz.localize(conditioning_start)
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
            "temperature_unit": "fahrenheit"
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
        ).tz_localize('UTC').tz_convert(local_tz)

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

        if self.use_gsi:
            # Compute the GSI
            # Use a 28-day period before the start of the simulation to calculate GSI (56-day for rain)
            gsi = self.calc_GSI(hourly_data, buffered_start, start_datetime)

            # Use GSI information to determine what to set live fuel moistures to
            self.live_h_mf, self.live_w_mf = self.set_live_moistures(gsi)
        else:
            self.live_h_mf = None
            self.live_w_mf = None

        hourly = filter_hourly_data(hourly_data, conditioning_start, end_datetime)
        self.stream_times = pd.DatetimeIndex(hourly["date"])
        self.stream = list(self.generate_stream(hourly))

        try:
            self.sim_start_idx = self.stream_times.get_loc(start_datetime)
        except KeyError:
            self.sim_start_idx = int(self.stream_times.searchsorted(start_datetime, side="left"))

        if self.use_gsi:
            # Build GSI tracker seeded with pre-simulation daily summaries
            pre_summaries = self._build_pre_sim_summaries(
                hourly_data, buffered_start, start_datetime
            )
            init_rain = self.stream[self.sim_start_idx].rain
            self.gsi_tracker = GSITracker(self.geo, pre_summaries, init_rain)
        else:
            self.gsi_tracker = None

        # Calculate foliar moisture content
        self.fmc = self.calc_fmc()

        # Set units and time step based on OpenMeteo params
        self.time_step = 60
        self.input_wind_ht = 10
        self.input_wind_ht_units = "m"
        self.input_wind_vel_units = "mps"
        self.input_temp_units = "F"

    def get_stream_from_wxs(self):
        """Parse a RAWS-format ``.wxs`` weather file and build the stream.

        Read weather observations from the file, apply unit conversions
        (English or Metric to internal units), fetch solar irradiance from
        Open-Meteo to supplement the file data, compute GSI, and assemble
        the final weather stream.

        Side Effects:
            Sets ``self.stream``, ``self.stream_times``, ``self.sim_start_idx``,
            ``self.fmc``, ``self.live_h_mf``, ``self.live_w_mf``,
            ``self.ref_elev``, ``self.time_step``, and input unit attributes.

        Raises:
            ValueError: If the file has insufficient data or unknown units.
        """
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
        conditioning_start = local_tz.localize(self.params.conditioning_start)
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

        if self.use_gsi:
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
        else:
            self.live_h_mf = None
            self.live_w_mf = None
        
        # Calculate foliar moisture content
        self.fmc = self.calc_fmc()

        # ── Step 8: Package final stream ─────────────────────────────
        df["date"] = df.index
        hourly_data = filter_hourly_data(df, conditioning_start, end_datetime)
        self.stream_times = pd.DatetimeIndex(hourly_data["date"])

        try:
            self.sim_start_idx = self.stream_times.get_loc(start_datetime)
        except KeyError:
            self.sim_start_idx = int(self.stream_times.searchsorted(start_datetime, side="left"))


        self.stream = list(self.generate_stream(hourly_data))

        if self.use_gsi:
            # Build GSI tracker seeded with pre-simulation daily summaries
            wxs_hourly = {
                "date": df.index,
                "temperature": df["temperature"].values,
                "rel_humidity": df["rel_humidity"].values,
                "rain": df["rain"].values,
            }
            pre_summaries = self._build_pre_sim_summaries(
                wxs_hourly, data_start, start_datetime
            )
            init_rain = self.stream[self.sim_start_idx].rain
            self.gsi_tracker = GSITracker(self.geo, pre_summaries, init_rain)
        else:
            self.gsi_tracker = None

        # ── Step 9: Set metadata attributes ──────────────────────────
        self.time_step = time_step_min
        self.input_wind_ht = 6.1
        self.input_wind_ht_units = "m"
        self.input_wind_vel_units = "mps"
        self.input_temp_units = "F"
 
    def generate_stream(self, hourly_data: dict) -> Iterator[WeatherEntry]:
        """Yield full WeatherEntry records from hourly data arrays.

        Rainfall is accumulated across entries (cumulative sum).

        Args:
            hourly_data (dict): Dictionary with keys 'wind_speed',
                'wind_direction', 'temperature', 'rel_humidity',
                'cloud_cover', 'ghi', 'dhi', 'dni', 'rain',
                'solar_zenith', 'solar_azimuth'.

        Yields:
            WeatherEntry: Fully populated weather entry with cumulative
                rainfall.
        """
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

    def set_live_moistures(self, gsi: float) -> tuple:
        """Compute live fuel moisture fractions from the Growing Season Index.

        Map GSI to live herbaceous and woody moisture using linear
        interpolation between dormant and green-up values. Below the
        green-up threshold (GSI < 0.2), dormant values are returned.

        Args:
            gsi (float): Growing Season Index in [0, 1].

        Returns:
            Tuple[float, float]: ``(live_h_mf, live_w_mf)`` live herbaceous
                and live woody moisture content (fractions).
        """
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

    def calc_GSI(self, hourly_data: dict, data_start: datetime, sim_start: datetime) -> float:
        """Compute the Growing Season Index (GSI) for the pre-simulation period.

        Average daily sub-indices for photoperiod, minimum temperature,
        vapor pressure deficit, and precipitation over the available
        pre-simulation window (ideally 28 days for non-rain metrics, 56
        days for rain).

        Args:
            hourly_data (dict): Hourly weather data with keys 'date',
                'temperature' (Fahrenheit), 'rel_humidity' (%), 'rain' (cm).
            data_start: Start of the data window (datetime-like).
            sim_start: Simulation start datetime (datetime-like).

        Returns:
            float: GSI value in [0, 1], or -1 if insufficient data is
                available (fewer than 2 days).
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

    def _build_pre_sim_summaries(self, hourly_data: dict,
                                 data_start, sim_start) -> List[DailySummary]:
        """Convert pre-simulation hourly data into daily summaries for GSI tracking.

        Aggregates hourly observations into per-day min/max temperature,
        min humidity, and total rainfall, matching the aggregation used
        by :meth:`calc_GSI`.

        Args:
            hourly_data: Hourly weather dictionary with keys ``'date'``,
                ``'temperature'`` (Fahrenheit), ``'rel_humidity'`` (percent),
                ``'rain'`` (cm, non-cumulative per-hour increments).
            data_start: Start of the pre-simulation data window.
            sim_start: Simulation start datetime.

        Returns:
            List of :class:`DailySummary`, oldest first, covering up to
            56 days before *sim_start*.
        """
        filtered = filter_hourly_data(hourly_data, data_start, sim_start)
        df = pd.DataFrame(filtered).set_index("date")

        if df.empty:
            return []

        daily_min_temp = df["temperature"].resample('D').min()
        daily_max_temp = df["temperature"].resample('D').max()
        daily_min_rh = df["rel_humidity"].resample('D').min()
        daily_rain = df["rain"].resample('D').sum()

        summaries = []
        for date in daily_min_temp.index:
            summaries.append(DailySummary(
                date=date.date() if hasattr(date, 'date') else date,
                min_temp_F=daily_min_temp[date],
                max_temp_F=daily_max_temp[date],
                min_rh=daily_min_rh[date],
                rain_cm=daily_rain[date],
            ))
        return summaries

    def calc_fmc(self) -> float:
        """Compute foliar moisture content based on latitude and date.

        Uses the Forestry Canada Fire Danger Group (1992) method, which
        estimates the day of minimum FMC from latitude, longitude, and
        elevation, then applies a polynomial fit.

        Returns:
            float: Foliar moisture content (percent).
        """
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

def filter_hourly_data(hourly_data: dict, start_datetime: datetime, end_datetime: datetime) -> dict:
    """Filter hourly weather data to a datetime range (inclusive).

    Args:
        hourly_data (dict): Dictionary with a 'date' key and parallel
            value arrays.
        start_datetime: Start of desired range (datetime-like).
        end_datetime: End of desired range (datetime-like).

    Returns:
        dict: Filtered copy with the same keys, values trimmed to the
            matching datetime window.
    """
    hourly_data["date"] = pd.to_datetime(hourly_data["date"])

    mask = (hourly_data["date"] >= start_datetime) & (hourly_data["date"] <= end_datetime)
    filtered_data = {key: np.array(value)[mask] for key, value in hourly_data.items()}
    return filtered_data

def apply_site_specific_correction(cell: Cell, elev_ref: float,
                                   curr_weather: WeatherEntry) -> Tuple[float, float]:
    """Apply elevation lapse-rate correction for temperature and humidity.

    Adjust the reference-station temperature and relative humidity to the
    cell's elevation using standard lapse rates (Stephenson 1988).

    Args:
        cell: Cell with ``elevation_m`` attribute (meters).
        elev_ref (float): Reference weather station elevation (meters).
        curr_weather (WeatherEntry): Current weather entry with ``temp``
            (Fahrenheit) and ``rel_humidity`` (percent).

    Returns:
        Tuple[float, float]: ``(temp_c, rh)`` where ``temp_c`` is the
            corrected temperature (Celsius) and ``rh`` is the corrected
            relative humidity (fraction, capped at 0.99).
    """
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

def calc_local_solar_radiation(cell: Cell, curr_weather: WeatherEntry) -> float:
    """Compute slope- and canopy-adjusted solar irradiance at a cell.

    Use pvlib to compute plane-of-array irradiance for the cell's slope
    and aspect, then reduce by canopy transmittance.

    Args:
        cell: Cell with ``slope_deg``, ``aspect``, ``canopy_cover`` (percent).
        curr_weather (WeatherEntry): Entry with ``solar_zenith``,
            ``solar_azimuth``, ``dni``, ``ghi``, ``dhi``.

    Returns:
        float: Total irradiance at the cell surface (W/m²).
    """
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

def datetime_to_julian_date(dt: datetime) -> float:
    """Convert a datetime to Julian date.

    Args:
        dt: A datetime-like object with year, month, day, hour, minute,
            and second attributes.

    Returns:
        float: Julian date (fractional day number).
    """
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