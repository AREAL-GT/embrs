"""Component 2 — daily live fuel moisture via EMBRS's GSI tracker.

Builds ``DailySummary`` records from the hourly weather table, runs
:class:`embrs.models.weather.GSITracker` per day with a rolling prefix, and
maps the resulting GSI to ``MCHERB`` / ``MCWOOD`` via the same logic EMBRS
uses (:meth:`WeatherStream.set_live_moistures`, lifted to avoid instantiating
a full :class:`WeatherStream`).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from embrs.fire_danger.config import HourlyWeather, LiveMoistureDaily
from embrs.models.weather import DailySummary, GSITracker
from embrs.utilities.data_classes import GeoInfo


# Constants from WeatherStream.set_live_moistures (weather.py ~678-697).
# Multiplied by 100 here for percent semantics at the boundary.
_H_MIN_FRAC = 0.30
_W_MIN_FRAC = 0.60
_H_MAX_FRAC = 2.50
_W_MAX_FRAC = 2.00
_GREENUP_GU = 0.20


def _gsi_to_live_moisture_pct(gsi: float) -> tuple[float, float]:
    """Map GSI in [0, 1] to (MCHERB, MCWOOD) in percent.

    Below ``gu = 0.2`` returns the dormant values (30%, 60%); above, linearly
    interpolates to (250%, 200%) at ``gsi = 1.0``. Matches
    :meth:`WeatherStream.set_live_moistures`.
    """
    if gsi < _GREENUP_GU:
        return _H_MIN_FRAC * 100.0, _W_MIN_FRAC * 100.0
    m_h = (_H_MAX_FRAC - _H_MIN_FRAC) / (1.0 - _GREENUP_GU)
    m_w = (_W_MAX_FRAC - _W_MIN_FRAC) / (1.0 - _GREENUP_GU)
    h_frac = m_h * gsi + (_H_MAX_FRAC - m_h)
    w_frac = m_w * gsi + (_W_MAX_FRAC - m_w)
    return h_frac * 100.0, w_frac * 100.0


def build_daily_summaries(weather: HourlyWeather) -> List[DailySummary]:
    """Resample the hourly weather table into per-day :class:`DailySummary`.

    Mirrors :meth:`WeatherStream._build_pre_sim_summaries` (weather.py:726-739).

    Returns:
        Oldest-first list of :class:`DailySummary`. ``date`` is a naive
        :class:`datetime.date`; the other fields are in NFDRS / GSI units
        (°F, percent, cm).
    """
    df = weather.df
    # Calendar-day boundaries in the local timezone (or naive if no tz yet).
    daily_min_temp = df["temp_F"].resample("D").min()
    daily_max_temp = df["temp_F"].resample("D").max()
    daily_min_rh = df["rh_pct"].resample("D").min()
    daily_rain = df["precip_cm_hr"].resample("D").sum()

    summaries: list[DailySummary] = []
    for ts in daily_min_temp.index:
        d = ts.date() if hasattr(ts, "date") else ts
        # Skip days with no data (resample may emit NaN rows for gaps)
        if pd.isna(daily_min_temp[ts]):
            continue
        summaries.append(
            DailySummary(
                date=d,
                min_temp_F=float(daily_min_temp[ts]),
                max_temp_F=float(daily_max_temp[ts]),
                min_rh=float(daily_min_rh[ts]),
                rain_cm=float(daily_rain[ts]),
            )
        )
    return summaries


def compute_live_moisture(
    weather: HourlyWeather, geo: GeoInfo
) -> LiveMoistureDaily:
    """Compute daily GSI / MCHERB / MCWOOD.

    For each day ``i`` we reconstruct a :class:`GSITracker` over
    ``summaries[:i+1]`` and call :meth:`compute_gsi`. ``compute_gsi`` averages
    the last 28 buffered days internally, so passing a running prefix is
    correct and O(N · 28) — trivial for N ≈ 45.

    The very first day (and any day before two days have been buffered) gets
    ``GSI = NaN`` and dormant live moistures (30%, 60%) — equivalent to the
    NFDRS seed for an uninitialised GSI tracker.

    Returns:
        :class:`LiveMoistureDaily` with a ``date``-indexed DataFrame.
    """
    summaries = build_daily_summaries(weather)

    dates = []
    gsis = []
    mcherbs = []
    mcwoods = []
    for i, _ in enumerate(summaries):
        tracker = GSITracker(geo, summaries[: i + 1])
        gsi = tracker.compute_gsi()
        if gsi is None or gsi < 0:
            gsi_val = float("nan")
            mcherb, mcwood = _gsi_to_live_moisture_pct(-1.0)  # dormant seed
        else:
            gsi_val = float(gsi)
            mcherb, mcwood = _gsi_to_live_moisture_pct(gsi_val)
        dates.append(summaries[i].date)
        gsis.append(gsi_val)
        mcherbs.append(mcherb)
        mcwoods.append(mcwood)

    df = pd.DataFrame(
        {"GSI": gsis, "MCHERB": mcherbs, "MCWOOD": mcwoods},
        index=pd.DatetimeIndex(pd.to_datetime(dates), name="date"),
    )
    return LiveMoistureDaily(df=df)
