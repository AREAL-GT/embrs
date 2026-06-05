"""Assemble a ``.wxs`` from a real temp/RH backdrop + synthetic wind (spec §4).

Keep the chosen real slice's temp/RH/cloud/precip **as-is** (real diurnal
structure and day-to-day variation) and **replace only wind speed and wind
direction** with the synthetic profiles from :mod:`wind_model`. Wind is authored
directly at the 20-ft reference height in m/s and converted to mph for the
``.wxs`` via the writer's ``wind_mph_precomputed`` path — so **no** 10 m->20 ft
log-profile correction is applied (spec §4.6).

A realized-vs-intended wind assertion guards against reintroducing the 3.6x
km/h-as-m/s unit bug (spec §4.6, §9).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from embrs.fire_danger.weather_loader import load_wxs
from embrs.scenario_weather.config import GeneratorConfig
from embrs.scenario_weather.period_search import slice_window_df
from embrs.scenario_weather.wind_model import generate_wind
from embrs.utilities.unit_conversions import m_to_ft
from embrs.weather_candidate_search.config import WindConversionConfig
from embrs.weather_candidate_search.wxs_writer import WxsWriteSpec, write_wxs

_MM_TO_IN: float = 0.0393701
_MPS_TO_MPH: float = 2.23693629


@dataclass
class WindStats:
    min_mph: float
    mean_mph: float
    peak_mph: float


@dataclass
class GenerateResult:
    """Outcome of generating a ``.wxs`` (paths + wind diagnostics)."""

    wxs_path: str
    peak_scale_ms: float
    intended: WindStats
    realized: WindStats
    n_rows: int

    def to_dict(self) -> dict:
        return asdict(self)


def _stats(mph: np.ndarray) -> WindStats:
    return WindStats(
        min_mph=float(np.min(mph)),
        mean_mph=float(np.mean(mph)),
        peak_mph=float(np.max(mph)),
    )


def generate_wxs(
    out_path: str,
    backdrop_df: pd.DataFrame,
    elevation_ft: int,
    gen_cfg: GeneratorConfig,
    *,
    peak_scale: Optional[float] = None,
) -> GenerateResult:
    """Write a ``.wxs`` from a backdrop slice + synthetic wind.

    Args:
        out_path: Output ``.wxs`` path.
        backdrop_df: Hourly real slice (from :func:`slice_window_df`) with
            ``temp_C``/``rh_pct``/``precip_in_hr``/``cloud_cover`` columns and a
            ``DatetimeIndex`` (tz-naive local wall-clock).
        elevation_ft: ``.wxs`` header elevation (feet).
        gen_cfg: Generator + wind parameters.
        peak_scale: Override for the wind peak amplitude (the tuning knob).

    Returns:
        A :class:`GenerateResult` with intended vs realized wind statistics.

    Raises:
        AssertionError: If the realized ``.wxs`` wind (re-read) does not match
            the intended 20-ft profile within ``gen_cfg.wind_assert_tol_mph``.
    """
    index = backdrop_df.index
    wind_ms, wind_dir = generate_wind(index, gen_cfg.wind, peak_scale=peak_scale)
    intended_mph = wind_ms * _MPS_TO_MPH

    precip_in = backdrop_df["precip_in_hr"].to_numpy()
    if gen_cfg.zero_precip:
        precip_in = np.zeros_like(precip_in)

    write_df = pd.DataFrame(
        {
            "temp_C": backdrop_df["temp_C"].to_numpy(),
            "rh_pct": backdrop_df["rh_pct"].to_numpy(),
            "rain_mm_hr": precip_in / _MM_TO_IN,
            "wind_mph_native": intended_mph,
            "wind_dir_deg": wind_dir,
            "cloud_pct": backdrop_df["cloud_cover"].to_numpy(),
        },
        index=index,
    )

    spec = WxsWriteSpec(
        df=write_df,
        elevation_ft=int(elevation_ft),
        # Disabled — wind is 20-ft-native; the precomputed-mph path bypasses
        # any conversion entirely (spec §4.6).
        wind_correction=WindConversionConfig(enabled=False),
        wind_mph_precomputed="wind_mph_native",
    )
    write_wxs(spec, out_path)

    # Realized-vs-intended assertion (re-read the file we just wrote).
    realized = load_wxs(out_path).df["wind_mph"].to_numpy()
    n = min(len(realized), len(intended_mph))
    max_abs = float(np.max(np.abs(realized[:n] - intended_mph[:n]))) if n else 0.0
    tol = gen_cfg.wind_assert_tol_mph
    assert max_abs <= tol, (
        f"realized .wxs wind deviates from intended by {max_abs:.3f} mph "
        f"(> {tol} mph) — possible unit/height bug (spec §4.6). "
        f"intended mean={intended_mph.mean():.2f} mph, "
        f"realized mean={realized.mean():.2f} mph"
    )

    used_scale = gen_cfg.wind.peak_scale_ms if peak_scale is None else float(peak_scale)
    return GenerateResult(
        wxs_path=out_path,
        peak_scale_ms=used_scale,
        intended=_stats(intended_mph),
        realized=_stats(realized),
        n_rows=len(write_df),
    )


def generate_from_window(
    out_path: str,
    full_season_wxs: str,
    start: datetime,
    end: datetime,
    gen_cfg: GeneratorConfig,
    *,
    peak_scale: Optional[float] = None,
) -> GenerateResult:
    """Convenience: slice a backdrop window from a season ``.wxs`` and generate.

    Elevation is taken from ``gen_cfg.elevation_ft`` if set, else preserved from
    the source file.
    """
    weather = load_wxs(full_season_wxs)
    backdrop = weather.df.loc[(weather.df.index >= start) & (weather.df.index <= end)].copy()
    elevation_ft = gen_cfg.elevation_ft
    if elevation_ft is None:
        elevation_ft = int(round(m_to_ft(weather.ref_elev_m)))
    return generate_wxs(out_path, backdrop, elevation_ft, gen_cfg, peak_scale=peak_scale)
