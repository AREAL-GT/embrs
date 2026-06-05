"""Temp/RH backdrop period search (spec §4.1.1).

Finds, per region, a ``window_days``-long real window for each class whose
temperature/RH **severity** is appropriate for that class. This is a *backdrop*
selector only — it does **not** define the class (wind tuning + measured flame
length do, spec §2/§5) — so it is deliberately simple: pure pandas over the
``.wxs`` columns, no simulation.

Severity (default) is the **mean over days of the daily-peak vapour-pressure
deficit (VPD)**: for each day take the max hourly VPD (hot/dry afternoon), then
average across the window. VPD combines temp and RH into one physically-
meaningful dryness axis.

Class selection uses **percentile bands of the region's own window-severity
distribution** (over guard-passing windows) so every class is reachable; an
absolute-band override is supported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from embrs.fire_danger.weather_loader import load_wxs
from embrs.scenario_weather.config import SearchConfig


def saturation_vapor_pressure_kpa(temp_c: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure (kPa) via the Magnus/Tetens approximation.

    ``es(T) = 0.6108 * exp(17.27 * T / (T + 237.3))`` — the same form used for
    the GSI VPD sub-index in :mod:`embrs.models.weather`.
    """
    temp_c = np.asarray(temp_c, dtype=float)
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def vapor_pressure_deficit_kpa(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Vapour-pressure deficit (kPa): ``(1 - RH/100) * es(T)``."""
    rh_pct = np.asarray(rh_pct, dtype=float)
    return (1.0 - rh_pct / 100.0) * saturation_vapor_pressure_kpa(temp_c)


@dataclass
class DailySummary:
    """Per-day weather summary used to score windows."""

    date: pd.Timestamp
    peak_vpd_kpa: float
    max_temp_F: float
    min_rh_pct: float
    total_precip_in: float
    max_hourly_precip_in: float
    month: int


@dataclass
class RankedWindow:
    """One candidate backdrop window with diagnostics (spec §4.1.1)."""

    start: str                 # ISO datetime (window's first hour)
    end: str                   # ISO datetime (window's last hour)
    severity_score: float
    mean_daily_max_temp_F: float
    mean_daily_min_rh: float
    total_precip_in: float
    n_days: int
    severity_percentile: float
    band_center_distance: float
    guard_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_daily_summaries(df: pd.DataFrame) -> List[DailySummary]:
    """Resample an hourly ``.wxs`` dataframe into ordered per-day summaries.

    Args:
        df: Hourly frame from :func:`load_wxs` (``temp_C``/``rh_pct``/
            ``temp_F``/``precip_in_hr`` columns, ``DatetimeIndex``).

    Returns:
        Oldest-first list of :class:`DailySummary`.
    """
    vpd = vapor_pressure_deficit_kpa(df["temp_C"].to_numpy(), df["rh_pct"].to_numpy())
    work = pd.DataFrame(
        {
            "vpd": vpd,
            "temp_F": df["temp_F"].to_numpy(),
            "rh_pct": df["rh_pct"].to_numpy(),
            "precip_in": df["precip_in_hr"].to_numpy(),
        },
        index=df.index,
    )
    summaries: List[DailySummary] = []
    for day, g in work.groupby(work.index.normalize()):
        summaries.append(
            DailySummary(
                date=pd.Timestamp(day),
                peak_vpd_kpa=float(np.nanmax(g["vpd"].to_numpy())),
                max_temp_F=float(np.nanmax(g["temp_F"].to_numpy())),
                min_rh_pct=float(np.nanmin(g["rh_pct"].to_numpy())),
                total_precip_in=float(np.nansum(g["precip_in"].to_numpy())),
                max_hourly_precip_in=float(np.nanmax(g["precip_in"].to_numpy())),
                month=int(pd.Timestamp(day).month),
            )
        )
    return summaries


def _window_severity(days: List[DailySummary], metric: str) -> float:
    if metric == "vpd":
        return float(np.mean([d.peak_vpd_kpa for d in days]))
    if metric == "temp_rh":
        # Combine into one axis: hotter + drier = more severe. Normalise RH to a
        # dryness term so both push the same way; VPD is preferred but this is a
        # simple offered alternative (spec §4.1.1).
        mean_max_temp = float(np.mean([d.max_temp_F for d in days]))
        mean_min_rh = float(np.mean([d.min_rh_pct for d in days]))
        return mean_max_temp + (100.0 - mean_min_rh)
    raise ValueError(f"unknown severity_metric {metric!r}")


def _dst_transition(days: List[DailySummary], tz: str) -> bool:
    """True if the window spans a DST change in ``tz`` (offset not constant)."""
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(tz)
    offsets = {
        d.date.to_pydatetime().replace(hour=12).replace(tzinfo=zone).utcoffset()
        for d in days
    }
    return len(offsets) > 1


def _guards(days: List[DailySummary], cfg: SearchConfig) -> List[str]:
    violations: List[str] = []
    if cfg.fire_season_months:
        if any(d.month not in cfg.fire_season_months for d in days):
            violations.append("outside_fire_season")
    total_precip = sum(d.total_precip_in for d in days)
    if total_precip > cfg.max_total_precip_in:
        violations.append(
            f"too_wet_total({total_precip:.2f}>{cfg.max_total_precip_in})"
        )
    if any(d.max_hourly_precip_in > cfg.max_peak_hourly_precip_in for d in days):
        violations.append("too_wet_hourly")
    if cfg.local_tz and _dst_transition(days, cfg.local_tz):
        violations.append("spans_dst_transition")
    return violations


def find_windows(
    wxs_path: str, cfg: Optional[SearchConfig] = None
) -> Dict[str, List[RankedWindow]]:
    """Find per-class backdrop windows in a region's full-season ``.wxs``.

    Args:
        wxs_path: Region's real-season ``.wxs`` (e.g. ``full_season.wxs``).
        cfg: Search parameters; defaults to :class:`SearchConfig`.

    Returns:
        Mapping ``class -> [RankedWindow]`` (best first; up to ``cfg.top_n``).
        Only guard-passing windows are ranked/returned.
    """
    cfg = cfg or SearchConfig()
    weather = load_wxs(wxs_path)
    summaries = build_daily_summaries(weather.df)
    if len(summaries) < cfg.window_days:
        raise ValueError(
            f"{wxs_path}: only {len(summaries)} days; need >= window_days="
            f"{cfg.window_days}"
        )

    # Slide a contiguous-day window; keep only guard-passing placements.
    valid: List[RankedWindow] = []
    n = len(summaries)
    for i in range(0, n - cfg.window_days + 1, cfg.stride_days):
        days = summaries[i : i + cfg.window_days]
        violations = _guards(days, cfg)
        if violations:
            continue
        sev = _window_severity(days, cfg.severity_metric)
        start = days[0].date.to_pydatetime().replace(hour=0)
        end = days[-1].date.to_pydatetime().replace(hour=23)
        valid.append(
            RankedWindow(
                start=start.isoformat(),
                end=end.isoformat(),
                severity_score=sev,
                mean_daily_max_temp_F=float(np.mean([d.max_temp_F for d in days])),
                mean_daily_min_rh=float(np.mean([d.min_rh_pct for d in days])),
                total_precip_in=float(sum(d.total_precip_in for d in days)),
                n_days=cfg.window_days,
                severity_percentile=float("nan"),
                band_center_distance=float("nan"),
            )
        )

    if not valid:
        raise ValueError(
            f"{wxs_path}: no windows passed the guards; relax fire-season/precip "
            "thresholds or shorten window_days"
        )

    # Severity distribution over guard-passing windows -> percentile per window.
    sev_arr = np.array([w.severity_score for w in valid])
    for w in valid:
        # Percentile rank of this window's severity within the distribution.
        w.severity_percentile = float(
            100.0 * (sev_arr <= w.severity_score).mean()
        )

    result: Dict[str, List[RankedWindow]] = {}
    for cls, band in cfg.class_bands.items():
        if cfg.absolute_bands and cls in cfg.absolute_bands:
            lo_v, hi_v = cfg.absolute_bands[cls]
        else:
            lo_v = float(np.percentile(sev_arr, band[0]))
            hi_v = float(np.percentile(sev_arr, band[1]))
        center = 0.5 * (lo_v + hi_v)
        in_band = [
            RankedWindow(**{**w.to_dict(),
                            "band_center_distance": abs(w.severity_score - center)})
            for w in valid
            if lo_v <= w.severity_score <= hi_v
        ]
        in_band.sort(key=lambda w: w.band_center_distance)
        result[cls] = in_band[: cfg.top_n]

    return result


def slice_window_df(wxs_path: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Return the hourly ``.wxs`` dataframe sliced to ``[start, end]`` inclusive.

    Used by the plotter (§4.7) and the generator (§4.1) to pull the chosen real
    temp/RH/cloud/precip backdrop.
    """
    df = load_wxs(wxs_path).df
    return df.loc[(df.index >= start) & (df.index <= end)].copy()
