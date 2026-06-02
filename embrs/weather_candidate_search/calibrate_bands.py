"""Climatology calibration: compute percentile-relative BI bands per region.

For one landscape, pull historical Open-Meteo weather over a multi-year
range, run the EMBRS BI pipeline on each year, build the empirical
distribution of "rolling-fortnight mean of daily-max BI" over the
fire-season months, and emit mild/moderate/extreme bands at user-chosen
percentile breakpoints.

Rationale: NFDRS adjective classes (Low / Moderate / High / Very High /
Extreme) are conventionally defined as percentile breakpoints of the
*local* daily BI distribution — they're already climatology-relative.
This tool lifts that convention to the fortnight-mean of daily-max BI, so
the "moderate" window in one region has the same climatological rarity
as the "moderate" window in another, even if the absolute BI numbers
differ. Measuring at the daily peak (rather than a fixed observation
hour) makes the equivalence insensitive to region-dependent diurnal
phase. See the methodology discussion in README.md.

Usage:
    # Contiguous fire season (e.g. Sierra timber):
    python -m embrs.weather_candidate_search.calibrate_bands \\
        --landscape-tif /path/to/region.tif \\
        --year-start 1994 --year-end 2023 \\
        --fire-season-months 5-10 \\
        --window-length-hours 336 \\
        --region-tag sierra_timber \\
        --output bands_sierra_timber.yaml

    # Non-contiguous (e.g. Appalachian spring + fall):
    python -m embrs.weather_candidate_search.calibrate_bands \\
        --landscape-tif /path/to/region.tif \\
        --year-start 1994 --year-end 2023 \\
        --fire-season-months 3-5,10-11 \\
        --window-length-hours 336 \\
        --region-tag appalachian \\
        --output bands_appalachian.yaml
"""



from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from embrs.weather_candidate_search.bi_search import REG_OBS_HOUR, run_bi
from embrs.weather_candidate_search.config import (
    BISection,
    WindConversionConfig,
)
from embrs.weather_candidate_search.geo import load_landscape_geo
from embrs.weather_candidate_search.openmeteo_client import (
    OpenMeteoPullSpec,
    fetch_history,
)
from embrs.weather_candidate_search.wxs_writer import (
    WxsWriteSpec,
    correct_wind_speed_10m_to_20ft,
    write_wxs,
)

logger = logging.getLogger(__name__)


_MPS_TO_MPH: float = 2.23693629


# Using the below link as a rough guide for fire seasons in different regions
"""https://www.nwcg.gov/publications/pms425-1/12-fire-climate-regions"""

# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------


def _default_breakpoints() -> Dict[str, Tuple[float, float]]:
    """Defaults anchored to recognized fire-management percentile benchmarks.

    Each class is a narrow percentile band centred on a defensible anchor of
    the regional fortnight-mean daily-max BI distribution:

      - mild     ~p50  — the median fire-season fortnight (a typical, real
                         fire week, not a quiet non-event).
      - moderate ~p85  — clearly elevated / "high" fire danger.
      - extreme  ~p97  — the standard "extreme fire weather" benchmark
                         (~a once-per-two-seasons severe fortnight).

    The cross-region equivalence is climatological rarity: the same
    percentile rank = the same recurrence interval in each region's own
    climate. Each entry is the (lo_pct, hi_pct) percentile band; the emitted
    BI band is the values at those two percentiles.
    """
    return {
        "mild": (47.0, 53.0),
        "moderate": (82.0, 88.0),
        "extreme": (95.0, 99.0),
    }


def parse_months_spec(spec: str) -> Tuple[int, ...]:
    """Parse a ``--fire-season-months`` string into a sorted month tuple.

    Accepts comma-separated months and/or hyphen ranges. Examples:

    - ``"5-10"``       → (5, 6, 7, 8, 9, 10)            # Sierra timber
    - ``"3-5,10-11"``  → (3, 4, 5, 10, 11)              # Appalachian
    - ``"2,3,4,5"``    → (2, 3, 4, 5)                   # explicit
    - ``"1-12"``       → (1, 2, ..., 12)                # full year

    Raises ``ValueError`` on duplicates, out-of-range, or unparseable input.
    """
    if not spec or not spec.strip():
        raise ValueError("fire_season_months spec is empty")
    months: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                lo_s, hi_s = part.split("-")
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise ValueError(f"unparseable month range {part!r}: {exc}")
            if lo > hi:
                raise ValueError(f"month range {part!r}: lo > hi")
            for m in range(lo, hi + 1):
                months.add(m)
        else:
            try:
                months.add(int(part))
            except ValueError:
                raise ValueError(f"unparseable month {part!r}")
    if not months:
        raise ValueError(f"no months parsed from {spec!r}")
    if not all(1 <= m <= 12 for m in months):
        raise ValueError(
            f"month(s) out of range 1..12: {sorted(months)}"
        )
    return tuple(sorted(months))


@dataclass(frozen=True)
class CalibrationConfig:
    """Inputs to :func:`calibrate_bands`.

    Args:
        landscape_tif: Path to the region's LANDFIRE .tif.
        year_start, year_end: Inclusive year range (e.g. 1994..2023).
        fire_season_months: Tuple of months (1..12) the climatology
            should include. Supports non-contiguous seasons (e.g.
            ``(3, 4, 5, 10, 11)`` for Appalachian spring + fall). The
            BI pipeline runs over the full span ``min..max`` of months
            so the moisture state is correctly conditioned across the
            gap; only the in-season days enter the rolling-mean
            distribution. Use :func:`parse_months_spec` to build this
            from a CLI-style range string.
        window_length_hours: The window length to calibrate for (must
            match the candidate search's ``scenario_length_hours``).
            Default 336 (14 days).
        conditioning_days: BI spin-up buffer pulled before each year's
            earliest in-season month. Default 30.
        percentile_breakpoints: ``{level: (lo_pct, hi_pct)}`` defining
            each band. Default mild=47-53 (~p50), moderate=82-88 (~p85),
            extreme=95-99 (~p97).
        cache_dir: Where to cache Open-Meteo pulls and per-year BI
            trajectories. Default ``./.openmeteo_cache/``.
        output_yaml: Path for the bands YAML. Default ``bands.yaml``.
        region_tag: Free-form label written into the YAML for traceability.
        wind_conversion, bi: Pass-through to the BI pipeline. Defaults
            match the candidate-search defaults.
    """

    landscape_tif: str
    year_start: int
    year_end: int
    fire_season_months: Tuple[int, ...]
    window_length_hours: int = 336
    conditioning_days: int = 30
    percentile_breakpoints: Dict[str, Tuple[float, float]] = field(
        default_factory=_default_breakpoints
    )
    cache_dir: str = "./.openmeteo_cache/"
    output_yaml: str = "bands.yaml"
    region_tag: str = "region"
    wind_conversion: WindConversionConfig = field(default_factory=WindConversionConfig)
    bi: BISection = field(default_factory=BISection)

    def __post_init__(self) -> None:
        if self.year_start > self.year_end:
            raise ValueError(
                f"year_start ({self.year_start}) must be <= year_end "
                f"({self.year_end})"
            )
        if not self.fire_season_months:
            raise ValueError("fire_season_months must be a non-empty tuple")
        if not all(1 <= m <= 12 for m in self.fire_season_months):
            raise ValueError(
                f"fire_season_months contains values outside 1..12: "
                f"{self.fire_season_months}"
            )
        if len(set(self.fire_season_months)) != len(self.fire_season_months):
            raise ValueError(
                f"fire_season_months contains duplicates: "
                f"{self.fire_season_months}"
            )
        if list(self.fire_season_months) != sorted(self.fire_season_months):
            raise ValueError(
                "fire_season_months must be sorted ascending; use "
                "parse_months_spec() to build it from a CLI range string."
            )
        if self.window_length_hours <= 0 or self.window_length_hours % 24 != 0:
            raise ValueError(
                f"window_length_hours must be a positive multiple of 24, "
                f"got {self.window_length_hours}"
            )
        if self.conditioning_days <= 0:
            raise ValueError("conditioning_days must be > 0")
        for level, (lo, hi) in self.percentile_breakpoints.items():
            if not (0 <= lo <= 100 and 0 <= hi <= 100):
                raise ValueError(f"{level} percentile breakpoints out of [0, 100]")
            if lo > hi:
                raise ValueError(f"{level} lo > hi: {lo} > {hi}")

    @property
    def earliest_in_season_month(self) -> int:
        return self.fire_season_months[0]

    @property
    def latest_in_season_month(self) -> int:
        return self.fire_season_months[-1]


@dataclass
class CalibrationResult:
    """Output of :func:`calibrate_bands`.

    ``band_breakpoints[level] = (lo_bi, hi_bi)`` is the band in BI units
    at the level's configured percentile pair. ``percentile_summary``
    holds the raw percentile lookup table (e.g. 25th, 50th, 75th, ...)
    for diagnostic plotting. ``distribution_stats`` covers n/min/max/etc.
    of the rolling-mean distribution.
    """

    band_breakpoints: Dict[str, Tuple[float, float]]
    percentile_summary: Dict[float, float]
    distribution_stats: Dict[str, float]
    metadata: Dict
    rolling_mean_series: pd.Series


# ---------------------------------------------------------------------------
# Per-year BI trajectory (with cache)
# ---------------------------------------------------------------------------


def _bi_trajectory_cache_path(
    cache_dir: str, lat: float, lon: float, year: int, m_start: int, m_end: int,
    conditioning_days: int,
) -> str:
    d = os.path.join(cache_dir, "bi_climatology_trajectories")
    os.makedirs(d, exist_ok=True)
    name = (
        f"{round(lat, 4):.4f}_{round(lon, 4):.4f}_"
        f"{year}_{m_start:02d}_{m_end:02d}_cd{conditioning_days}.parquet"
    )
    return os.path.join(d, name)


def _wxs_cache_path(
    cache_dir: str, lat: float, lon: float, year: int, m_start: int, m_end: int,
    conditioning_days: int,
) -> str:
    d = os.path.join(cache_dir, "bi_climatology_wxs")
    os.makedirs(d, exist_ok=True)
    name = (
        f"{round(lat, 4):.4f}_{round(lon, 4):.4f}_"
        f"{year}_{m_start:02d}_{m_end:02d}_cd{conditioning_days}.wxs"
    )
    return os.path.join(d, name)


def _pull_span_for_year(year: int, m_start: int, m_end: int, conditioning_days: int):
    """Return ``(pull_start, pull_end, scenario_start)`` for a year.

    For non-contiguous fire seasons (e.g. spring + fall) the pull spans
    the full ``m_start..m_end`` range — including any in-between
    off-season months. BI is computed continuously across the whole
    span so the moisture state stays conditioned across the gap; the
    climatology filter (applied later in ``_extract_daily_max_in_season``)
    drops the off-season days from the percentile distribution.
    """
    scenario_start = dt.date(year, m_start, 1)
    pull_start = scenario_start - dt.timedelta(days=conditioning_days)
    last_day_end = calendar.monthrange(year, m_end)[1]
    pull_end = dt.date(year, m_end, last_day_end)
    return pull_start, pull_end, scenario_start


def _run_year(
    year: int, cfg: CalibrationConfig, geo, elevation_ft: float
) -> pd.DataFrame:
    """Return the year's BI trajectory (BI_area_weighted + phase columns).

    Uses a Parquet cache so repeated invocations skip the BI pipeline.
    The cache key uses the ``(earliest, latest)`` month pair — the
    trajectory depends only on the span, not on which in-between months
    the climatology filter eventually keeps — so re-running the
    calibration with a different ``fire_season_months`` filter on the
    same span hits the same cached trajectory.
    """
    m_start = cfg.earliest_in_season_month
    m_end = cfg.latest_in_season_month

    cache_path = _bi_trajectory_cache_path(
        cfg.cache_dir, geo.center_lat, geo.center_lon,
        year, m_start, m_end, cfg.conditioning_days,
    )
    if os.path.exists(cache_path):
        logger.info("BI trajectory cache hit: %s", cache_path)
        return pd.read_parquet(cache_path)

    pull_start, pull_end, scen_start = _pull_span_for_year(
        year, m_start, m_end, cfg.conditioning_days,
    )
    logger.info(
        "Year %d: pulling Open-Meteo %s..%s, scenario_start=%s "
        "(in-season months: %s)",
        year, pull_start, pull_end, scen_start,
        list(cfg.fire_season_months),
    )
    om = fetch_history(
        OpenMeteoPullSpec(
            lat=float(geo.center_lat), lon=float(geo.center_lon),
            start_date=pull_start, end_date=pull_end, timezone="auto",
        ),
        cache_dir=cfg.cache_dir,
    )

    # Apply wind correction (matching the candidate-search pipeline).
    df = om.df.copy()
    wind_mps = df["wind_mps"].to_numpy(dtype=float)
    if cfg.wind_conversion.enabled:
        wind_mps = correct_wind_speed_10m_to_20ft(
            wind_mps, cfg.wind_conversion.surface_roughness_m
        )
    df["wind_mph"] = wind_mps * _MPS_TO_MPH

    wxs_path = _wxs_cache_path(
        cfg.cache_dir, geo.center_lat, geo.center_lon,
        year, m_start, m_end, cfg.conditioning_days,
    )
    write_wxs(
        WxsWriteSpec(
            df=df, elevation_ft=int(round(elevation_ft)),
            wind_correction=cfg.wind_conversion,
            wind_mph_precomputed="wind_mph",
        ),
        wxs_path,
    )

    bi_result = run_bi(
        full_wxs_path=wxs_path,
        landscape_tif=cfg.landscape_tif,
        scenario_start=dt.datetime(scen_start.year, scen_start.month, scen_start.day),
        bi_section=cfg.bi,
    )
    # Persist only the columns we need; tz info survives parquet via int64 ns.
    out = bi_result.trajectory_df[["BI_area_weighted", "phase"]].copy()
    out.to_parquet(cache_path)
    logger.info("Cached BI trajectory: %s (%d rows)", cache_path, len(out))
    return out


# ---------------------------------------------------------------------------
# Climatology aggregation
# ---------------------------------------------------------------------------


def _extract_daily_max_in_season(
    bi_traj: pd.DataFrame,
    fire_season_months: Tuple[int, ...],
) -> pd.Series:
    """Return daily-max BI for the scenario period, restricted to in-season months.

    For each scenario calendar day the maximum of that day's hourly
    ``BI_area_weighted`` is taken (the daily peak fire behaviour). The
    result is indexed at day boundaries (midnight). For non-contiguous
    seasons (e.g. ``(3, 4, 5, 10, 11)``) off-season days are dropped here;
    the subsequent rolling-mean step (``_rolling_fortnight_mean``) then
    automatically excludes any fortnight that bridges the gap because the
    resampled daily grid contains NaN on the dropped days.
    """
    scen = bi_traj[bi_traj["phase"] == "scenario"]
    bi = scen["BI_area_weighted"].dropna()
    if bi.empty:
        return bi
    daily_max = bi.groupby(bi.index.normalize()).max()
    months = np.asarray(daily_max.index.month)
    in_season = np.isin(months, np.asarray(fire_season_months))
    return daily_max[in_season]


def _rolling_fortnight_mean(
    daily_max: pd.Series, window_days: int
) -> pd.Series:
    """Trailing N-day rolling mean of a daily series (one value per day).

    Resample to a complete daily grid first so .rolling() sees a regular
    cadence (otherwise gaps would let a 14-day window span more than 14
    calendar days). NaN-filled days are excluded from the average via
    ``min_periods=window_days``.
    """
    if daily_max.empty:
        return daily_max
    # Normalize index to date-only so resampling lands on day boundaries.
    s = daily_max.copy()
    s.index = s.index.normalize()
    daily_grid = s.resample("D").mean()
    rolling = daily_grid.rolling(window=window_days, min_periods=window_days).mean()
    return rolling.dropna()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def calibrate_bands(cfg: CalibrationConfig) -> CalibrationResult:
    """Run the full climatology calibration.

    Returns:
        :class:`CalibrationResult` with the band BI ranges, the percentile
        summary, distribution stats, and a copy of the rolling-mean series.
    """
    lgeo = load_landscape_geo(cfg.landscape_tif)
    elevation_ft = (
        float(lgeo.elevation_ft) if np.isfinite(lgeo.elevation_ft) else 0.0
    )
    logger.info(
        "Calibration: centroid lat=%.4f, lon=%.4f, tz=%s, elev=%.0f ft",
        lgeo.geo.center_lat, lgeo.geo.center_lon, lgeo.geo.timezone,
        elevation_ft,
    )

    per_year_daily: List[pd.Series] = []
    for year in range(cfg.year_start, cfg.year_end + 1):
        try:
            bi_traj = _run_year(year, cfg, lgeo.geo, elevation_ft)
        except Exception as exc:
            logger.warning(
                "Year %d failed (%r); skipping that year in the climatology.",
                year, exc,
            )
            continue
        per_year_daily.append(
            _extract_daily_max_in_season(bi_traj, cfg.fire_season_months)
        )

    if not per_year_daily or all(s.empty for s in per_year_daily):
        raise RuntimeError(
            "No daily-max BI values produced across any year in the range. "
            "Check fire_season months, landscape, and Open-Meteo availability."
        )

    # Per-year rolling means concatenated (so a rolling window never spans
    # the off-season gap between years).
    window_days = cfg.window_length_hours // 24
    per_year_rolling = [
        _rolling_fortnight_mean(s, window_days) for s in per_year_daily
    ]
    rolling_mean = pd.concat(per_year_rolling).sort_index()
    rolling_mean = rolling_mean.dropna()

    arr = rolling_mean.to_numpy()
    if arr.size == 0:
        raise RuntimeError(
            f"No valid {window_days}-day rolling means produced. The "
            f"fire-season span may be shorter than the window length."
        )

    # Diagnostic percentiles (always reported)
    diag_pcts = [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    percentile_summary = {
        float(p): float(np.percentile(arr, p)) for p in diag_pcts
    }

    # Bands from the user's breakpoints
    band_breakpoints: Dict[str, Tuple[float, float]] = {}
    for level, (lo_pct, hi_pct) in cfg.percentile_breakpoints.items():
        lo_bi = float(np.percentile(arr, lo_pct))
        hi_bi = float(np.percentile(arr, hi_pct))
        band_breakpoints[level] = (lo_bi, hi_bi)

    stats = {
        "n_windows": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
    }

    metadata = {
        "region_tag": cfg.region_tag,
        "landscape_tif": cfg.landscape_tif,
        "year_range": [cfg.year_start, cfg.year_end],
        "n_years_attempted": cfg.year_end - cfg.year_start + 1,
        "n_years_used": sum(1 for s in per_year_daily if not s.empty),
        "fire_season_months": list(cfg.fire_season_months),
        "window_length_hours": cfg.window_length_hours,
        "window_length_days": window_days,
        "conditioning_days": cfg.conditioning_days,
        "centroid_lat_wgs84": float(lgeo.geo.center_lat),
        "centroid_lon_wgs84": float(lgeo.geo.center_lon),
        "timezone": lgeo.geo.timezone,
        "elevation_ft": elevation_ft,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    write_bands_yaml(
        cfg.output_yaml,
        band_breakpoints, percentile_summary, stats, metadata,
        cfg.percentile_breakpoints,
    )
    logger.info("Bands written to %s", cfg.output_yaml)

    return CalibrationResult(
        band_breakpoints=band_breakpoints,
        percentile_summary=percentile_summary,
        distribution_stats=stats,
        metadata=metadata,
        rolling_mean_series=rolling_mean,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_bands_yaml(
    path: str,
    band_breakpoints: Dict[str, Tuple[float, float]],
    percentile_summary: Dict[float, float],
    stats: Dict[str, float],
    metadata: Dict,
    percentile_breakpoints: Dict[str, Tuple[float, float]],
) -> None:
    """Write a human-readable, paste-ready YAML for per-region search configs.

    The ``bi_target_band`` value under each level is the snippet to paste
    directly into the candidate-search YAML. ``percentile_range`` records
    the corresponding percentile range so the choice stays traceable.
    """
    import yaml

    doc = {
        "climatology_meta": metadata,
        "distribution_stats": stats,
        "percentile_summary": {
            f"p{int(p):02d}": round(v, 2) for p, v in percentile_summary.items()
        },
        "bands": {
            level: {
                "bi_target_band": [round(lo, 2), round(hi, 2)],
                "percentile_range": list(percentile_breakpoints[level]),
            }
            for level, (lo, hi) in band_breakpoints.items()
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        # Keep simple-style YAML for human edits later.
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)


def format_summary(result: CalibrationResult) -> str:
    """Pretty-printed text block for terminal output."""
    md = result.metadata
    stats = result.distribution_stats
    pcts = result.percentile_summary
    lines: List[str] = []
    lines.append(
        f"\n=== Climatology calibration: {md['region_tag']} ==="
    )
    lines.append(
        f"  Years:       {md['year_range'][0]}..{md['year_range'][1]} "
        f"({md['n_years_used']} / {md['n_years_attempted']} years used)"
    )
    lines.append(
        f"  Fire-season months: {md['fire_season_months']}"
    )
    lines.append(
        f"  Window:      {md['window_length_hours']} h "
        f"({md['window_length_days']} days)"
    )
    lines.append(
        f"  Centroid:    lat={md['centroid_lat_wgs84']:.4f}, "
        f"lon={md['centroid_lon_wgs84']:.4f} ({md['timezone']})"
    )
    lines.append(
        f"\n  Distribution of fortnight-mean daily-max BI "
        f"(in-season only):"
    )
    lines.append(
        f"    n={stats['n_windows']}, "
        f"min={stats['min']:.1f}, max={stats['max']:.1f}, "
        f"mean={stats['mean']:.1f}, median={stats['median']:.1f}, "
        f"std={stats['std']:.1f}"
    )
    lines.append("\n  Percentile lookup (BI units):")
    keyline = "    "
    for p in sorted(pcts):
        keyline += f"p{int(p):02d}={pcts[p]:.1f}  "
    lines.append(keyline.rstrip())
    lines.append("\n  Suggested bands (mean_daily_peak_bi):")
    for level, (lo, hi) in result.band_breakpoints.items():
        pct_range = result.metadata.get("percentile_breakpoints_used", {})
        lines.append(f"    {level:<10s}= [{lo:6.2f}, {hi:6.2f}]")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_breakpoint_pair(s: str) -> Tuple[float, float]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected 'lo,hi' (e.g. '20,30'), got {s!r}"
        )
    return float(parts[0]), float(parts[1])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m embrs.weather_candidate_search.calibrate_bands",
        description=(
            "Compute percentile-relative BI bands for a region using "
            "multi-year Open-Meteo ERA5 history. Output is a paste-ready "
            "bands.yaml with mild/moderate/extreme bi_target_band values."
        ),
    )
    p.add_argument("--landscape-tif", required=True)
    p.add_argument("--year-start", type=int, default=1994)
    p.add_argument("--year-end", type=int, default=2023)
    p.add_argument(
        "--fire-season-months",
        required=True,
        type=parse_months_spec,
        metavar="SPEC",
        help=(
            "Comma-separated months and/or hyphen ranges, e.g. "
            "'5-10' (Sierra), '3-5,10-11' (Appalachian spring + fall), "
            "'1-12' (year-round). The BI pipeline runs over the full "
            "min..max span so moisture state stays conditioned across "
            "any off-season gap; only the in-season days enter the "
            "percentile distribution."
        ),
    )
    p.add_argument("--window-length-hours", type=int, default=336,
                   help="Match this to the candidate-search scenario_length_hours.")
    p.add_argument("--conditioning-days", type=int, default=30)
    p.add_argument("--cache-dir", default="./.openmeteo_cache/")
    p.add_argument("--output", default="bands.yaml",
                   help="Output YAML path.")
    p.add_argument("--region-tag", default="region")
    p.add_argument(
        "--mild", type=_parse_breakpoint_pair, default=None,
        metavar="LO,HI",
        help="Mild percentile range, e.g. '47,53'. Default 47,53 (~p50).",
    )
    p.add_argument(
        "--moderate", type=_parse_breakpoint_pair, default=None,
        metavar="LO,HI",
        help="Moderate percentile range, e.g. '82,88'. Default 82,88 (~p85).",
    )
    p.add_argument(
        "--extreme", type=_parse_breakpoint_pair, default=None,
        metavar="LO,HI",
        help="Extreme percentile range, e.g. '95,99'. Default 95,99 (~p97).",
    )
    p.add_argument(
        "--avg-ann-precip-in", type=float, default=None,
        help="Override AvgAnnPrecip; otherwise auto-fetched per BI pipeline.",
    )
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s | %(message)s",
    )

    breakpoints = _default_breakpoints()
    if args.mild is not None:
        breakpoints["mild"] = args.mild
    if args.moderate is not None:
        breakpoints["moderate"] = args.moderate
    if args.extreme is not None:
        breakpoints["extreme"] = args.extreme

    cfg = CalibrationConfig(
        landscape_tif=args.landscape_tif,
        year_start=args.year_start,
        year_end=args.year_end,
        fire_season_months=args.fire_season_months,
        window_length_hours=args.window_length_hours,
        conditioning_days=args.conditioning_days,
        percentile_breakpoints=breakpoints,
        cache_dir=args.cache_dir,
        output_yaml=args.output,
        region_tag=args.region_tag,
        bi=BISection(avg_ann_precip_in=args.avg_ann_precip_in),
    )
    result = calibrate_bands(cfg)
    print(format_summary(result))
    return 0


if __name__ == "__main__":      # pragma: no cover
    sys.exit(main())
