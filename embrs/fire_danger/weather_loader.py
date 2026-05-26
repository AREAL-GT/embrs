"""Component 1 — parse a RAWS-format ``.wxs`` file into an HourlyWeather table.

This loader is intentionally **offline**. It does not fetch solar radiation
from Open-Meteo the way :meth:`embrs.models.weather.WeatherStream.get_stream_from_wxs`
does — solar is synthesized from the scenario's own cloud cover by
:mod:`embrs.fire_danger.solar` so the resulting inputs to the Nelson model are
physically consistent with the synthetic weather (plan §2.1, scope §7.3).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from embrs.fire_danger.config import HourlyWeather
from embrs.utilities.unit_conversions import F_to_C, ft_to_m

_WXS_COLUMNS = (
    "Year", "Mth", "Day", "Time", "Temp", "RH",
    "HrlyPcp", "WindSpd", "WindDir", "CloudCov",
)


def load_wxs(path: str) -> HourlyWeather:
    """Parse a ``.wxs`` file and return an :class:`HourlyWeather` table.

    Args:
        path: Filesystem path to the ``.wxs`` file.

    Returns:
        An :class:`HourlyWeather` with a tz-naive local ``DatetimeIndex`` and
        the canonical columns documented on :class:`HourlyWeather`. The
        timezone is attached later by the orchestrator once geo is resolved.

    Raises:
        ValueError: If the file has fewer than 2 valid data rows, an
            unrecognised ``RAWS_UNITS`` value, or a non-hourly time step.
    """
    raw: dict[str, list[Any]] = {col: [] for col in _WXS_COLUMNS}
    units = "english"
    elevation_raw: float | None = None

    with open(path, "r") as fh:
        header_seen = False
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("RAWS_UNITS:"):
                units = line.split(":", 1)[1].strip().lower()
                continue
            if line.startswith("RAWS_ELEVATION:"):
                elevation_raw = float(line.split(":", 1)[1].strip())
                continue
            if line.startswith("RAWS:"):
                continue
            if line.startswith("Year") and not header_seen:
                header_seen = True
                continue
            if not header_seen:
                continue

            parts = line.split()
            if len(parts) != 10:
                continue
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                hour = int(parts[3].zfill(4)[:2])
                raw["Year"].append(year)
                raw["Mth"].append(month)
                raw["Day"].append(day)
                raw["Time"].append(hour)
                raw["Temp"].append(float(parts[4]))
                raw["RH"].append(float(parts[5]))
                raw["HrlyPcp"].append(float(parts[6]))
                raw["WindSpd"].append(float(parts[7]))
                raw["WindDir"].append(float(parts[8]))
                raw["CloudCov"].append(float(parts[9]))
            except (ValueError, IndexError) as exc:
                # Skip a single malformed row but keep going; only raise if
                # the whole file ends up empty (handled below).
                print(f"weather_loader: skipping malformed row: {line!r} ({exc})")
                continue

    if len(raw["Year"]) < 2:
        raise ValueError(
            f"{path}: parsed fewer than 2 data rows; cannot determine time step"
        )
    if elevation_raw is None:
        raise ValueError(f"{path}: missing RAWS_ELEVATION header")

    datetimes = [
        datetime(y, m, d, h)
        for y, m, d, h in zip(raw["Year"], raw["Mth"], raw["Day"], raw["Time"])
    ]
    index = pd.DatetimeIndex(datetimes)

    time_step_min = int((index[1] - index[0]).total_seconds() / 60)
    if time_step_min != 60:
        raise ValueError(
            f"{path}: inferred time step is {time_step_min} min, but the BI "
            f"trajectory tool requires hourly weather (the Nelson driver "
            f"assumes et=1.0 h)."
        )

    if units == "english":
        temp_F = np.asarray(raw["Temp"], dtype=float)
        temp_C = np.asarray([F_to_C(t) for t in raw["Temp"]], dtype=float)
        precip_in_hr = np.asarray(raw["HrlyPcp"], dtype=float)
        precip_cm_hr = precip_in_hr * 2.54
        wind_mph = np.asarray(raw["WindSpd"], dtype=float)
        ref_elev_m = ft_to_m(elevation_raw)
    elif units == "metric":
        # Metric .wxs: Temp °C, HrlyPcp mm, WindSpd m/s (per EMBRS convention).
        temp_C = np.asarray(raw["Temp"], dtype=float)
        temp_F = temp_C * 9.0 / 5.0 + 32.0
        precip_cm_hr = np.asarray(raw["HrlyPcp"], dtype=float) / 10.0  # mm -> cm
        precip_in_hr = precip_cm_hr / 2.54
        wind_mph = np.asarray(raw["WindSpd"], dtype=float) * 2.23693629  # m/s -> mph
        ref_elev_m = float(elevation_raw)
    else:
        raise ValueError(f"{path}: unknown RAWS_UNITS value {units!r}")

    rh_pct = np.asarray(raw["RH"], dtype=float)
    rh_frac = rh_pct / 100.0

    df = pd.DataFrame(
        {
            "temp_F": temp_F,
            "temp_C": temp_C,
            "rh_pct": rh_pct,
            "rh_frac": rh_frac,
            "wind_mph": wind_mph,
            "wind_dir_deg": np.asarray(raw["WindDir"], dtype=float),
            "precip_in_hr": precip_in_hr,
            "precip_cm_hr": precip_cm_hr,
            "cloud_cover": np.asarray(raw["CloudCov"], dtype=float),
        },
        index=index,
    )
    df.index.name = "datetime"

    return HourlyWeather(
        df=df,
        ref_elev_m=ref_elev_m,
        time_step_min=time_step_min,
        raw_start=index[0].to_pydatetime(),
        raw_end=index[-1].to_pydatetime(),
    )
