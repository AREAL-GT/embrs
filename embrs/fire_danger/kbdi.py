"""Component 4 — KBDI drought index, drought fuel-load transfer, and the
Open-Meteo-backed AvgAnnPrecip climatology fetch.

``calc_kbdi`` is a faithful port of ``NFDRS4py/.../nfdrs4.cpp::iCalcKBDI``
(lines 1092-1133). ``apply_drought_load_transfer`` ports the drought block
inside ``iCalcIndexes`` (lines 806-823) — it is fuel-model-specific and
lives here so the kbdi module owns all drought-related code (plan M0-11).

``fetch_avg_ann_precip_in`` closes OQ-13 by fetching a 30-year-normal annual
precipitation at ``(lat, lon)`` from the Open-Meteo Archive API
(ERA5-backed). Results are cached aggressively so subsequent runs are
offline; failures fall back to ``DEFAULT_AVG_ANN_PRECIP_IN`` (30 in).
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from embrs.fire_danger.config import (
    DEFAULT_AVG_ANN_PRECIP_IN,
    HourlyWeather,
    KBDIDaily,
)
from embrs.fire_danger.nfdrs_fuel_models import (
    CTA,
    KBDI_THRESHOLD_DEFAULT,
    NFDRSFuelModel,
)


# ---------------------------------------------------------------------------
# iCalcKBDI port
# ---------------------------------------------------------------------------


def calc_kbdi(
    precip_in_day: float,
    max_temp_F: float,
    cum_precip_in: float,
    prev_kbdi: float,
    avg_ann_precip_in: float,
) -> tuple[float, float]:
    """One-day KBDI update.

    Faithful port of ``iCalcKBDI`` (``nfdrs4.cpp:1092-1133``). The C++ uses
    integer arithmetic for ``KBDI``, ``net``, and ``idq``; we preserve that
    via explicit ``int(...)`` casts. Returns ``(kbdi, updated_cum_precip)``;
    the C++ mutates a ``CummPrecip`` member instead.

    Args:
        precip_in_day: Today's total precipitation, **inches**.
        max_temp_F: Today's maximum air temperature, **°F**.
        cum_precip_in: Running cumulative-precip tracker, inches. Carried
            over from the previous day's return value.
        prev_kbdi: Yesterday's KBDI (seed: 100 on first day).
        avg_ann_precip_in: 30-yr-normal annual precipitation at site, **inches**.

    Returns:
        ``(kbdi_today, updated_cum_precip_in)``.
    """
    kbdi = int(prev_kbdi)
    cum = float(cum_precip_in)

    if precip_in_day == 0.0:
        cum = 0.0
    else:
        if cum > 0.20:
            pptnet = precip_in_day
            cum = cum + precip_in_day
        else:
            cum = cum + precip_in_day
            pptnet = cum - 0.20 if cum > 0.20 else 0.0
        net = int(100.0 * pptnet + 0.0005)
        kbdi = kbdi - net
        if kbdi < 0:
            kbdi = 0

    if max_temp_F > 50:
        xkbdi = int(kbdi)
        xtemp = float(max_temp_F)
        idq = int(
            (800.0 - xkbdi)
            * (0.9679 * math.exp(0.0486 * xtemp) - 8.299)
            * 0.001
            / (1.0 + 10.88 * math.exp(-0.04409 * avg_ann_precip_in))
            + 0.5
        )
        kbdi = kbdi + idq

    return float(kbdi), cum


# ---------------------------------------------------------------------------
# Drought fuel-load transfer (iCalcIndexes:806-823)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DroughtAdjustedLoads:
    """Fuel loadings after drought transfer.

    All loadings in **lb/ft^2** (already × CTA from tons/acre); ``depth`` in
    **ft**. ``W_herb`` and ``W_wood`` are passed through unmodified — the
    drought block only redistributes dead loadings — but included for the
    caller's convenience.
    """

    W1: float
    W10: float
    W100: float
    W1000: float
    W_herb: float
    W_wood: float
    W_drought: float
    depth: float


def apply_drought_load_transfer(
    fuel: NFDRSFuelModel,
    kbdi: float,
    kbdi_threshold: float = KBDI_THRESHOLD_DEFAULT,
) -> DroughtAdjustedLoads:
    """Apply the iCalcIndexes drought block to a fuel model.

    When ``kbdi <= kbdi_threshold`` this is the identity (loadings are just
    the raw ``L* × CTA`` values). Above threshold, each dead-fuel loading
    grows proportional to its share of the total dead loading × drought
    unit. The fuel-bed depth is adjusted to preserve packing ratio.
    """
    W1 = fuel.L1 * CTA
    W10 = fuel.L10 * CTA
    W100 = fuel.L100 * CTA
    W1000 = fuel.L1000 * CTA
    W_herb = fuel.LHerb * CTA
    W_wood = fuel.LWood * CTA
    W_drought = fuel.LDrought * CTA
    depth = fuel.Depth

    if kbdi > kbdi_threshold:
        WTOTD_excl = W1 + W10 + W100               # excludes 1000-hr initially
        WTOTL = W_herb + W_wood
        WTOT_initial = WTOTD_excl + WTOTL
        packing_ratio = WTOT_initial / depth if depth > 0 else 1.0
        if packing_ratio == 0:
            packing_ratio = 1.0
        WTOTD = WTOTD_excl + W1000                 # now includes 1000-hr
        drought_unit = W_drought / (800.0 - kbdi_threshold)
        delta_total = (kbdi - kbdi_threshold) * drought_unit
        if WTOTD > 0:
            W1 = W1 + (W1 / WTOTD) * delta_total
            W10 = W10 + (W10 / WTOTD) * delta_total
            W100 = W100 + (W100 / WTOTD) * delta_total
            W1000 = W1000 + (W1000 / WTOTD) * delta_total
        WTOT_final = W1 + W10 + W100 + W1000 + WTOTL
        depth = (WTOT_final - W1000) / packing_ratio

    return DroughtAdjustedLoads(
        W1=W1, W10=W10, W100=W100, W1000=W1000,
        W_herb=W_herb, W_wood=W_wood, W_drought=W_drought,
        depth=depth,
    )


# ---------------------------------------------------------------------------
# Daily KBDI driver
# ---------------------------------------------------------------------------


def compute_kbdi_series(
    weather: HourlyWeather,
    avg_ann_precip_in: float,
    reg_obs_hr: int = 13,
) -> KBDIDaily:
    """Compute the daily KBDI series.

    For each day in the weather's index, the trailing-24-h window ending at
    ``reg_obs_hr:00`` provides ``pcp24`` (inches, summed) and ``max_temp_F``
    (max). Windows that are not fully populated (typically the first day)
    are skipped — those hours fall back to the KBDI seed of 100 downstream
    via forward-fill in the trajectory orchestrator.

    Returns:
        :class:`KBDIDaily` indexed by ``date`` (Timestamp at the obs hour);
        column ``KBDI`` (float, [0, 800], no rounding per D4).
    """
    df = weather.df
    if df.empty:
        return KBDIDaily(df=pd.DataFrame({"KBDI": []}, index=pd.DatetimeIndex([], name="date")))

    unique_dates = pd.Index(df.index.normalize().unique()).sort_values()

    prev_kbdi: float = 100.0
    cum_precip: float = 0.0
    out_dates: list = []
    out_kbdi: list[float] = []

    for day_ts in unique_dates:
        # Construct the obs-hour timestamp matching the input index tz.
        obs_dt = pd.Timestamp(year=day_ts.year, month=day_ts.month,
                              day=day_ts.day, hour=reg_obs_hr,
                              tz=df.index.tz)
        window_end = obs_dt
        window_start = obs_dt - pd.Timedelta(hours=23)
        window = df.loc[(df.index >= window_start) & (df.index <= window_end)]
        if len(window) < 24:
            continue
        pcp24 = float(window["precip_in_hr"].sum())
        max_temp_F = float(window["temp_F"].max())
        kbdi, cum_precip = calc_kbdi(
            pcp24, max_temp_F, cum_precip, prev_kbdi, avg_ann_precip_in
        )
        out_dates.append(obs_dt)
        out_kbdi.append(kbdi)
        prev_kbdi = kbdi

    out_df = pd.DataFrame(
        {"KBDI": out_kbdi},
        index=pd.DatetimeIndex(out_dates, name="date"),
    )
    return KBDIDaily(df=out_df)


# ---------------------------------------------------------------------------
# Open-Meteo Archive API: 30-yr-normal annual precipitation (closes OQ-13)
# ---------------------------------------------------------------------------


def _cache_dir() -> str:
    d = os.path.expanduser("~/.cache/embrs")
    os.makedirs(d, exist_ok=True)
    return d


def fetch_avg_ann_precip_in(
    lat: float,
    lon: float,
    year_range: tuple[int, int] = (1991, 2020),
    timeout_s: float = 30.0,
) -> float:
    """Fetch the 30-year-normal annual precipitation at ``(lat, lon)``.

    Hits the Open-Meteo Archive API for ``daily=precipitation_sum`` across
    the requested year range (default 1991-2020 — the WMO 30-yr normal
    window), sums per calendar year, averages across years, then converts
    mm → inches. Uses ``requests_cache`` with an unbounded TTL keyed at a
    sqlite cache under ``~/.cache/embrs/`` so subsequent calls are offline.

    Returns:
        Mean annual precipitation in **inches**. ERA5 is known to
        underestimate precip in mountainous terrain; for the "steep timber"
        region the caller may prefer to override via ``--avg-ann-precip``
        with a PRISM-anchored value (plan §6 item 5).

    Raises:
        Any exception raised by the underlying HTTP / parsing path is
        propagated — callers (the trajectory orchestrator) catch and fall
        back to :data:`DEFAULT_AVG_ANN_PRECIP_IN`.
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache_path = os.path.join(_cache_dir(), "openmeteo_archive")
    cache_session = requests_cache.CachedSession(cache_path, expire_after=-1)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    start = f"{year_range[0]}-01-01"
    end = f"{year_range[1]}-12-31"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": start,
        "end_date": end,
        "daily": "precipitation_sum",
        "timezone": "UTC",
    }
    responses = client.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
    )
    response = responses[0]
    daily = response.Daily()
    precip_mm = np.asarray(daily.Variables(0).ValuesAsNumpy(), dtype=float)
    # Build a daily date index aligned with the API window.
    times = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s"),
        end=pd.to_datetime(daily.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    )
    s = pd.Series(precip_mm, index=times)
    # Replace NaNs with 0 (Open-Meteo occasionally has gaps in early ERA5)
    s = s.fillna(0.0)
    annual_mm = s.groupby(s.index.year).sum()
    if annual_mm.empty:
        raise RuntimeError("Open-Meteo returned no daily values")
    mean_annual_mm = float(annual_mm.mean())
    return mean_annual_mm * 0.0393701   # mm -> in


def resolve_avg_ann_precip_in(
    explicit: Optional[float],
    lat: Optional[float],
    lon: Optional[float],
) -> tuple[float, str]:
    """Resolve ``AvgAnnPrecip`` in inches per the precedence in plan §2.10.

    Order: explicit CLI value > Open-Meteo auto-fetch > 30 in fallback.
    Returns ``(value_in, source)`` where ``source`` is one of
    ``"explicit"``, ``"openmeteo"``, ``"default"``.
    """
    if explicit is not None:
        return float(explicit), "explicit"
    if lat is not None and lon is not None:
        try:
            return fetch_avg_ann_precip_in(lat, lon), "openmeteo"
        except Exception as exc:
            warnings.warn(
                f"fetch_avg_ann_precip_in({lat}, {lon}) failed ({exc!r}); "
                f"falling back to DEFAULT_AVG_ANN_PRECIP_IN={DEFAULT_AVG_ANN_PRECIP_IN}.",
                RuntimeWarning,
            )
    return DEFAULT_AVG_ANN_PRECIP_IN, "default"
