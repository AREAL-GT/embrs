"""Component 1b — offline solar synthesis.

Synthesize hourly global horizontal irradiance (GHI) from the ``.wxs``
scenario's own cloud cover so the solar input to the Nelson dead-fuel-moisture
model is physically consistent with the synthetic temperature, humidity, and
wind (plan §2.2, scope §7.3).

Method (OQ-1):
    GHI = clearsky_GHI(lat, lon, time, altitude) * (1 - 0.75 * c**3.4)

where ``c`` is the cloud fraction in ``[0, 1]`` derived from the ``.wxs``
``CloudCov`` column per ``cloud_scale``. The Kasten–Czeplak cloud attenuation
form (``1 - 0.75*c**3.4``) is the standard parameterisation.

Clear-sky GHI is computed with :func:`pvlib.location.Location.get_clearsky`
using the Ineichen model (no extra inputs required).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
import pytz

from embrs.fire_danger.config import HourlyWeather
from embrs.utilities.data_classes import GeoInfo

_VALID_CLOUD_SCALES = {"percent", "fraction", "okta", "tenths"}


def _cloud_to_fraction(values: np.ndarray, cloud_scale: str) -> np.ndarray:
    """Convert raw ``.wxs`` cloud-cover values to a [0, 1] fraction."""
    if cloud_scale == "percent":
        c = values / 100.0
    elif cloud_scale == "fraction":
        c = values.copy()
    elif cloud_scale == "okta":
        c = values / 8.0
    elif cloud_scale == "tenths":
        c = values / 10.0
    else:
        raise ValueError(
            f"cloud_scale must be one of {_VALID_CLOUD_SCALES}, got {cloud_scale!r}"
        )
    return np.clip(c, 0.0, 1.0)


def synthesize_solar(
    weather: HourlyWeather,
    geo: GeoInfo,
    cloud_scale: str = "percent",
) -> HourlyWeather:
    """Add a ``solar_wm2`` column to ``weather`` via offline clear-sky + clouds.

    The input ``weather.df`` index may be tz-naive (as returned by
    :func:`weather_loader.load_wxs`); it is localized to ``geo.timezone``
    in-place so all downstream consumers share a tz-aware index.

    Args:
        weather: Hourly weather table.
        geo: Geographic info (``center_lat``, ``center_lon``, ``timezone``).
        cloud_scale: Units for the ``cloud_cover`` column. Default
            ``"percent"`` matches the ``.wxs`` convention (verified against
            ``long_weather_example.wxs``).

    Returns:
        The same ``HourlyWeather`` (mutated in place) with ``solar_wm2``
        added and the index localized.

    Raises:
        ValueError: For an unsupported ``cloud_scale``.
    """
    if geo.timezone is None or geo.center_lat is None or geo.center_lon is None:
        raise ValueError(
            "synthesize_solar: GeoInfo must have center_lat, center_lon, and "
            "timezone set."
        )

    df = weather.df
    if df.index.tz is None:
        local_tz = pytz.timezone(geo.timezone)
        df.index = df.index.tz_localize(local_tz, nonexistent="shift_forward",
                                        ambiguous="NaT")
        # Drop any NaT rows produced by DST gaps (rare for hourly RAWS data).
        df.dropna(how="all", inplace=True)

    loc = pvlib.location.Location(
        latitude=geo.center_lat,
        longitude=geo.center_lon,
        tz=geo.timezone,
        altitude=weather.ref_elev_m,
    )
    clearsky = loc.get_clearsky(df.index, model="ineichen")
    ghi_clear = clearsky["ghi"].to_numpy()

    c = _cloud_to_fraction(df["cloud_cover"].to_numpy(), cloud_scale)
    ghi = ghi_clear * (1.0 - 0.75 * np.power(c, 3.4))
    ghi = np.clip(ghi, 0.0, None)

    df["solar_wm2"] = ghi
    return weather
