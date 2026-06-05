"""Helpers to synthesise a ``.wxs`` for sim-free scenario_weather tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from embrs.weather_candidate_search.config import WindConversionConfig
from embrs.weather_candidate_search.wxs_writer import WxsWriteSpec, write_wxs


def write_season_wxs(
    path: str,
    start: str,
    days: int,
    *,
    base_temp_c=20.0,
    temp_amp_c=8.0,
    daily_temp_trend_c=0.0,
    base_rh=40.0,
    rh_amp=25.0,
    precip_in_per_day=None,
    wind_mps=3.0,
    wind_dir_deg=180.0,
    elevation_ft=1337,
):
    """Write a diurnally-varying hourly ``.wxs`` with controllable severity.

    Temperature peaks mid-afternoon and RH troughs with it. ``daily_temp_trend_c``
    ramps the daily mean so later windows are hotter/drier (more severe).
    ``precip_in_per_day`` (dict {day_index: inches placed at hour 3}) injects rain
    to exercise the wet guard.
    """
    n = days * 24
    idx = pd.date_range(start, periods=n, freq="h")
    hod = idx.hour.to_numpy().astype(float)
    day_of = (np.arange(n) // 24).astype(float)

    diurnal = np.sin((hod - 9.0) / 24.0 * 2 * np.pi)  # peak ~15:00
    temp_c = base_temp_c + daily_temp_trend_c * day_of + temp_amp_c * diurnal
    rh = np.clip(base_rh - rh_amp * diurnal, 2.0, 100.0)

    rain_mm = np.zeros(n)
    if precip_in_per_day:
        for d, inches in precip_in_per_day.items():
            rain_mm[d * 24 + 3] = inches / 0.0393701

    df = pd.DataFrame(
        {
            "temp_C": temp_c,
            "rh_pct": rh,
            "rain_mm_hr": rain_mm,
            "wind_mps": np.full(n, float(wind_mps)),
            "wind_dir_deg": np.full(n, float(wind_dir_deg)),
            "cloud_pct": np.zeros(n),
        },
        index=idx,
    )
    spec = WxsWriteSpec(
        df=df, elevation_ft=elevation_ft,
        wind_correction=WindConversionConfig(enabled=False),
    )
    write_wxs(spec, path)
    return path
