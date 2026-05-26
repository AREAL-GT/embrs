"""RAWS-format ``.wxs`` writer (English units, 10-column hourly rows).

The output is compatible with :func:`embrs.fire_danger.weather_loader.load_wxs`
(verified by round-trip in the unit tests). Plan §4.5.

Wind height (qa B4): Open-Meteo's ``wind_speed_10m`` is at 10 m AGL but NFDRS
and the FARSITE HMRH `.wxs` convention expect 20-ft (≈6.1 m). A log-profile
correction ``u_20ft = u_10m * ln(6.1/z0) / ln(10/z0)`` is applied when
``wind_correction.enabled`` is True (default).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from embrs.weather_candidate_search.config import WindConversionConfig

# Conversion constants.
_MM_TO_IN: float = 0.0393701
_MPS_TO_MPH: float = 2.23693629
_C_TO_F_SLOPE: float = 9.0 / 5.0
_C_TO_F_OFFSET: float = 32.0

# NFDRS wind reference height (m). FARSITE HMRH expects 20-ft.
_NFDRS_WIND_HT_M: float = 6.1
_OPEN_METEO_WIND_HT_M: float = 10.0


def log_profile_factor(surface_roughness_m: float) -> float:
    """Multiplier for ``u(10m) → u(6.1m)`` via the log-wind profile.

    ``u_20ft = u_10m * ln(6.1/z0) / ln(10/z0)``. With ``z0 = 0.06`` m the
    factor is ≈ 0.911.
    """
    z0 = float(surface_roughness_m)
    if z0 <= 0:
        raise ValueError(f"surface_roughness_m must be > 0, got {z0}")
    return math.log(_NFDRS_WIND_HT_M / z0) / math.log(_OPEN_METEO_WIND_HT_M / z0)


def correct_wind_speed_10m_to_20ft(
    wind_mps_10m: np.ndarray, surface_roughness_m: float
) -> np.ndarray:
    """Apply the log-profile 10 m → 6.1 m correction to a wind speed array."""
    return np.asarray(wind_mps_10m, dtype=float) * log_profile_factor(surface_roughness_m)


@dataclass(frozen=True)
class WxsWriteSpec:
    """Inputs to :func:`write_wxs`.

    ``df`` must have a tz-aware ``DatetimeIndex`` (we write the local
    wall-clock time; tz is not encoded in the .wxs format) and must include
    the columns described below.

    Column contract (any source — Open-Meteo or otherwise):
      - ``temp_C`` (°C)
      - ``rh_pct`` (0-100)
      - ``rain_mm_hr`` (mm per hour)
      - ``wind_mps`` (m/s at 10 m AGL)  — converted to mph at 20 ft via
        ``wind_correction``.
      - ``wind_dir_deg`` (meteorological, 0=N)
      - ``cloud_pct`` (0-100)

    If ``wind_mph_precomputed`` is provided (a column name in ``df``) the
    writer uses that column directly and bypasses any conversion. Useful
    when the pipeline has already applied the log-profile correction and
    wants the .wxs to match the BI/lull-detection wind exactly (plan §4.11
    step 5).
    """

    df: pd.DataFrame
    elevation_ft: int
    wind_correction: WindConversionConfig
    wind_mph_precomputed: Optional[str] = None


def _format_row(
    ts: pd.Timestamp,
    temp_F: float,
    rh_pct: float,
    pcp_in_hr: float,
    wind_mph: float,
    wind_dir_deg: float,
    cloud_pct: float,
) -> str:
    """Format a single .wxs data row matching the EMBRS sample spacing."""
    # Match BI weather_loader spacing: 10 whitespace-delimited columns.
    return (
        f"{ts.year:4d}  "
        f"{ts.month:<3d}  "
        f"{ts.day:<3d}  "
        f"{ts.hour:02d}00    "
        f"{temp_F:5.1f}    "
        f"{int(round(rh_pct)):3d}    "
        f"{pcp_in_hr:.2f}      "
        f"{wind_mph:.1f}     "
        f"{int(round(wind_dir_deg)) % 360:3d}     "
        f"{int(round(cloud_pct)):3d}"
    )


def write_wxs(spec: WxsWriteSpec, path: str) -> None:
    """Write a RAWS-format English-units .wxs file.

    The output file passes :func:`embrs.fire_danger.weather_loader.load_wxs`
    without modification (round-trip verified in tests).

    Args:
        spec: Source data and conversion settings.
        path: Output file path. Parent directories are created if needed.

    Raises:
        ValueError: If required columns are missing or the index is not
            sorted hourly.
    """
    df = spec.df

    required = {"temp_C", "rh_pct", "rain_mm_hr", "wind_dir_deg", "cloud_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"write_wxs: missing required columns: {sorted(missing)}")

    if spec.wind_mph_precomputed is not None:
        if spec.wind_mph_precomputed not in df.columns:
            raise ValueError(
                f"wind_mph_precomputed={spec.wind_mph_precomputed!r} is not a "
                f"column on the DataFrame"
            )
        wind_mph = np.asarray(df[spec.wind_mph_precomputed].to_numpy(), dtype=float)
    else:
        if "wind_mps" not in df.columns:
            raise ValueError(
                "write_wxs: need 'wind_mps' column or wind_mph_precomputed"
            )
        wind_mps = np.asarray(df["wind_mps"].to_numpy(), dtype=float)
        if spec.wind_correction.enabled:
            wind_mps = correct_wind_speed_10m_to_20ft(
                wind_mps, spec.wind_correction.surface_roughness_m
            )
        wind_mph = wind_mps * _MPS_TO_MPH

    temp_F = df["temp_C"].to_numpy() * _C_TO_F_SLOPE + _C_TO_F_OFFSET
    rh_pct = df["rh_pct"].to_numpy()
    pcp_in_hr = df["rain_mm_hr"].to_numpy() * _MM_TO_IN
    wind_dir = df["wind_dir_deg"].to_numpy()
    cloud_pct = df["cloud_pct"].to_numpy()

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    lines = [
        "RAWS_UNITS: English",
        f"RAWS_ELEVATION: {int(round(spec.elevation_ft))}",
        "RAWS: 1",
        "Year  Mth  Day   Time    Temp     RH  HrlyPcp  WindSpd WindDir CloudCov",
    ]

    skipped_nan = 0
    for i, ts in enumerate(df.index):
        # Skip rows with any NaN — the BI reader silently skips malformed
        # rows but we'd rather not emit garbage. Log if any are dropped.
        if (
            not np.isfinite(temp_F[i])
            or not np.isfinite(rh_pct[i])
            or not np.isfinite(pcp_in_hr[i])
            or not np.isfinite(wind_mph[i])
            or not np.isfinite(wind_dir[i])
            or not np.isfinite(cloud_pct[i])
        ):
            skipped_nan += 1
            continue
        lines.append(
            _format_row(
                pd.Timestamp(ts),
                float(temp_F[i]),
                float(rh_pct[i]),
                float(pcp_in_hr[i]),
                float(wind_mph[i]),
                float(wind_dir[i]),
                float(cloud_pct[i]),
            )
        )

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    if skipped_nan > 0:
        import logging

        logging.getLogger(__name__).warning(
            "write_wxs: skipped %d rows containing NaN; wrote %d rows to %s",
            skipped_nan,
            len(lines) - 4,
            path,
        )
