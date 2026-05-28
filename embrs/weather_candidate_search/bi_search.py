"""Drive ``embrs.fire_danger.compute_bi_trajectory`` once and extract per-window peaks.

Per the resolved BI performance strategy (option (c), qa A1): the
candidate-search writes one full-season ``.wxs``, calls the BI pipeline
once, and slices the resulting hourly ``BI_area_weighted`` column by
window.

Plan §4.7.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from embrs.fire_danger import Config as FireDangerConfig
from embrs.fire_danger import compute_bi_trajectory
from embrs.weather_candidate_search.config import BISection

logger = logging.getLogger(__name__)

PEAK_PERCENTILE: int = 97   # qa A3 — fixed at 97, no CLI override
REG_OBS_HOUR: int = 13      # NFDRS regular observation hour (1 PM local)


@dataclass
class BIPipelineResult:
    """Output of :func:`run_bi`.

    ``trajectory_df`` is the hourly BI DataFrame (tz-aware local index)
    with ``BI_area_weighted`` and per-model BI columns. ``full_season_peak``
    is the BI pipeline's own 97th-pct (over its scenario rows) — useful for
    logging.
    """

    trajectory_df: pd.DataFrame
    full_season_peak: float
    metadata: dict


def run_bi(
    full_wxs_path: str,
    landscape_tif: str,
    scenario_start: datetime,
    bi_section: BISection,
) -> BIPipelineResult:
    """Run the BI pipeline on the full-season .wxs and return the trajectory.

    Args:
        full_wxs_path: Path to the .wxs covering conditioning + fire season.
        landscape_tif: Path to the LANDFIRE raster.
        scenario_start: Conditioning↔scenario boundary (qa A2).
        bi_section: Pass-through of BI knobs (qa A4).

    Returns:
        :class:`BIPipelineResult`.
    """
    fd_cfg = FireDangerConfig(
        landscape_path=landscape_tif,
        wxs_path=full_wxs_path,
        scenario_start=scenario_start,
        min_area_frac=bi_section.min_area_frac,
        slope_class=bi_section.slope_class,
        lat_override=bi_section.lat_override,
        reg_obs_hr=bi_section.reg_obs_hr,
        cloud_scale=bi_section.cloud_scale,
        snow_mode=bi_section.snow_mode,
        avg_ann_precip_in=bi_section.avg_ann_precip_in,
    )
    logger.info(
        "Calling compute_bi_trajectory: wxs=%s, landscape=%s, scenario_start=%s",
        full_wxs_path,
        landscape_tif,
        scenario_start.isoformat(),
    )
    result = compute_bi_trajectory(fd_cfg)
    logger.info(
        "BI trajectory complete: %d rows, peak (97th pct scenario)=%.2f, "
        "composition=%s",
        len(result.df),
        result.peak_bi,
        result.fuel_composition.fractions,
    )
    return BIPipelineResult(
        trajectory_df=result.df,
        full_season_peak=float(result.peak_bi),
        metadata=dict(result.metadata),
    )


# ---------------------------------------------------------------------------
# Per-window peak extraction
# ---------------------------------------------------------------------------


def per_window_peaks(
    bi_trajectory: pd.DataFrame,
    windows: Iterable["WindowSlice"],
    percentile: int = PEAK_PERCENTILE,
    reg_obs_hour: int = REG_OBS_HOUR,
) -> pd.DataFrame:
    """Compute hourly peak/mean BI plus the NFDRS daily-1pm mean per window.

    Two BI summaries per window:

    - ``peak_bi`` — the ``percentile``-th of hourly ``BI_area_weighted``
      (default 97th, qa A3). Robust to single-hour spikes; characterizes
      the window's worst hours.
    - ``mean_daily_1pm_bi`` — mean across the daily 1 PM (or
      ``reg_obs_hour``) BI values inside the window. This is the NFDRS
      standard daily summary; it strips the diurnal saw-tooth and is the
      *equivalence* metric used to match a window to a volatility band.

    Args:
        bi_trajectory: Hourly DataFrame containing at least
            ``BI_area_weighted``.
        windows: Iterable of :class:`windowing.Window` (only ``window_id``,
            ``start``, ``end`` are read here — we re-slice the BI series).
        percentile: Percentile for ``peak_bi`` (default 97).
        reg_obs_hour: Local hour to sample for the daily afternoon BI
            (default 13 = 1 PM, NFDRS RegObsHr).

    Returns:
        DataFrame indexed by ``window_id`` with columns ``start``, ``end``,
        ``peak_bi``, ``time_of_peak``, ``mean_bi``, ``mean_daily_1pm_bi``,
        ``n_daily_1pm_samples``, ``n_hours``, ``n_valid_hours``.
    """
    if "BI_area_weighted" not in bi_trajectory.columns:
        raise ValueError(
            "bi_trajectory must contain 'BI_area_weighted' column"
        )

    bi_series = bi_trajectory["BI_area_weighted"]
    # Boolean mask of "this hour is the regular obs hour"; we'll filter per
    # window. Computed once for the whole trajectory.
    afternoon_mask = bi_series.index.hour == int(reg_obs_hour)
    bi_daily_1pm = bi_series[afternoon_mask]

    rows: List[dict] = []
    for w in windows:
        # Slice tells us the window's intended span; index match is on
        # exact local timestamps (the trajectory and the weather are
        # aligned upstream).
        sliced = bi_series.loc[(bi_series.index >= w.start) & (bi_series.index < w.end)]
        valid = sliced.dropna()

        # Daily 1 PM slice (one value per calendar day in the window where
        # a 13:00 row exists and is finite).
        daily_1pm = bi_daily_1pm.loc[
            (bi_daily_1pm.index >= w.start) & (bi_daily_1pm.index < w.end)
        ].dropna()
        mean_daily_1pm_bi = (
            float(daily_1pm.mean()) if not daily_1pm.empty else float("nan")
        )
        n_daily_1pm_samples = int(len(daily_1pm))

        if valid.empty:
            rows.append(
                {
                    "window_id": w.window_id,
                    "start": w.start,
                    "end": w.end,
                    "peak_bi": float("nan"),
                    "time_of_peak": pd.NaT,
                    "mean_bi": float("nan"),
                    "mean_daily_1pm_bi": mean_daily_1pm_bi,
                    "n_daily_1pm_samples": n_daily_1pm_samples,
                    "n_hours": int(len(sliced)),
                    "n_valid_hours": 0,
                }
            )
            continue
        peak = float(np.percentile(valid.to_numpy(), percentile))
        # time_of_peak = the index where BI is closest to the percentile
        # (deterministic; uses first occurrence if tied)
        diff = (valid - peak).abs()
        time_of_peak = pd.Timestamp(diff.idxmin())
        mean = float(valid.mean())
        rows.append(
            {
                "window_id": w.window_id,
                "start": w.start,
                "end": w.end,
                "peak_bi": peak,
                "time_of_peak": time_of_peak,
                "mean_bi": mean,
                "mean_daily_1pm_bi": mean_daily_1pm_bi,
                "n_daily_1pm_samples": n_daily_1pm_samples,
                "n_hours": int(len(sliced)),
                "n_valid_hours": int(len(valid)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "window_id",
                "start",
                "end",
                "peak_bi",
                "time_of_peak",
                "mean_bi",
                "mean_daily_1pm_bi",
                "n_daily_1pm_samples",
                "n_hours",
                "n_valid_hours",
            ]
        ).set_index("window_id")
    df = pd.DataFrame(rows).set_index("window_id")
    return df


# Re-export Window for type hints (avoiding a circular import at module load).
class WindowSlice:        # pragma: no cover — typing-only protocol
    """Minimal protocol — see :class:`windowing.Window`."""

    window_id: str
    start: pd.Timestamp
    end: pd.Timestamp
