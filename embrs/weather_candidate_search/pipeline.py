"""Top-level orchestration: Config in → artifacts on disk out.

Plan §4.11.
"""
from __future__ import annotations

import calendar
import datetime as dt
import logging
import os
import tempfile
from typing import Dict, List

import numpy as np
import pandas as pd

from embrs.weather_candidate_search import artifacts
from embrs.weather_candidate_search.bi_search import (
    BIPipelineResult,
    per_window_peaks,
    run_bi,
)
from embrs.weather_candidate_search.config import (
    Config,
    columns_for_bi_filter_mode,
    score_distance_column_for_mode,
)
from embrs.weather_candidate_search.geo import LandscapeGeo, load_landscape_geo
from embrs.weather_candidate_search.lull_detection import Lull, detect_lulls
from embrs.weather_candidate_search.openmeteo_client import (
    OpenMeteoPullSpec,
    OpenMeteoResult,
    fetch_history,
)
from embrs.weather_candidate_search.ranking import (
    RankedCandidate,
    filter_by_target_band,
    score_windows,
    select_top_n,
)
from embrs.weather_candidate_search.windowing import (
    DEFAULT_REQUIRED_COLUMNS,
    Window,
    iter_windows,
)
from embrs.weather_candidate_search.wxs_writer import (
    WxsWriteSpec,
    correct_wind_speed_10m_to_20ft,
    write_wxs,
)

logger = logging.getLogger(__name__)


_MPS_TO_MPH: float = 2.23693629


def _compute_pull_span(cfg: Config) -> tuple[dt.date, dt.date, dt.date]:
    """Return ``(pull_start, pull_end, scenario_start_date)``.

    ``pull_start = scenario_start_date - conditioning_days``.
    """
    scenario_start = dt.date(cfg.year, cfg.fire_season_start_month, 1)
    pull_start = scenario_start - dt.timedelta(days=cfg.conditioning_days)
    last_day_end = calendar.monthrange(cfg.year, cfg.fire_season_end_month)[1]
    pull_end = dt.date(cfg.year, cfg.fire_season_end_month, last_day_end)
    return pull_start, pull_end, scenario_start


def _apply_wind_correction(om_result: OpenMeteoResult, cfg: Config) -> pd.DataFrame:
    """Add ``wind_mph`` column (post-correction) to a copy of the OM DataFrame."""
    df = om_result.df.copy()
    wind_mps = df["wind_mps"].to_numpy(dtype=float)
    if cfg.wind_conversion.enabled:
        wind_mps_corrected = correct_wind_speed_10m_to_20ft(
            wind_mps, cfg.wind_conversion.surface_roughness_m
        )
    else:
        wind_mps_corrected = wind_mps
    df["wind_mph"] = wind_mps_corrected * _MPS_TO_MPH
    return df


def _write_full_season_wxs(cfg: Config, weather_df: pd.DataFrame, elevation_ft: float) -> str:
    """Write the full-season .wxs into the cell directory."""
    out_path = os.path.join(cfg.cell_dir, "full_season.wxs")
    artifacts.ensure_cell_dir(cfg)
    # Wind correction has already been applied in ``_apply_wind_correction``;
    # tell the writer to use that column directly and bypass its own
    # conversion.
    spec = WxsWriteSpec(
        df=weather_df,
        elevation_ft=int(round(elevation_ft)),
        wind_correction=cfg.wind_conversion,
        wind_mph_precomputed="wind_mph",
    )
    write_wxs(spec, out_path)
    return out_path


def _localize_scenario_start(
    scenario_start_date: dt.date, weather_df: pd.DataFrame
) -> pd.Timestamp:
    """Return a tz-aware Timestamp matching the weather frame's tz."""
    tz = weather_df.index.tz
    naive = pd.Timestamp(scenario_start_date)
    if tz is None:
        return naive
    return naive.tz_localize(tz)


def _localize_pull_end(
    pull_end_date: dt.date, weather_df: pd.DataFrame
) -> pd.Timestamp:
    """Last hour of ``pull_end_date`` in the weather's tz."""
    tz = weather_df.index.tz
    naive = pd.Timestamp(pull_end_date) + pd.Timedelta(hours=23)
    if tz is None:
        return naive
    return naive.tz_localize(tz)


def _merge_bi_into_weather(
    weather_df: pd.DataFrame, bi_trajectory: pd.DataFrame
) -> pd.DataFrame:
    """Left-join ``BI_area_weighted`` from the BI trajectory onto the weather.

    Both indices are tz-aware hourly. The BI pipeline preserves the local
    timeline produced by ``solar.synthesize_solar`` (same tz the weather is
    in). If the two have differing tz objects (e.g. fixed-offset vs IANA),
    convert the BI index to match the weather's.
    """
    bi = bi_trajectory[["BI_area_weighted"]]
    if bi.index.tz is None and weather_df.index.tz is not None:
        bi.index = bi.index.tz_localize(weather_df.index.tz)
    elif (
        bi.index.tz is not None
        and weather_df.index.tz is not None
        and bi.index.tz != weather_df.index.tz
    ):
        bi = bi.copy()
        bi.index = bi.index.tz_convert(weather_df.index.tz)
    return weather_df.join(bi, how="left")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_candidate_search(cfg: Config) -> int:
    """Execute the candidate-search pipeline. Returns an exit code.

    Returns:
        0 on success (≥ 1 candidate selected); 2 if no window passed the BI
        band filter (qa J2). Other errors propagate as exceptions.
    """
    artifacts.ensure_cell_dir(cfg)
    query_ts_utc = dt.datetime.now(dt.timezone.utc)

    # 1. Resolve landscape geo
    logger.info("Resolving landscape geo from %s", cfg.landscape_tif)
    lgeo: LandscapeGeo = load_landscape_geo(cfg.landscape_tif)
    elevation_ft = lgeo.elevation_ft
    if not np.isfinite(elevation_ft):
        # Fall back to the Open-Meteo Elevation() reported in the response
        # below; for now use sea level so the pull can proceed.
        elevation_ft = 0.0

    # 2. Compute pull span
    pull_start, pull_end, scenario_start_date = _compute_pull_span(cfg)
    logger.info(
        "Centroid: lat=%.4f, lon=%.4f, tz=%s; elev=%.0f ft. Pull span: %s..%s",
        lgeo.geo.center_lat,
        lgeo.geo.center_lon,
        lgeo.geo.timezone,
        elevation_ft,
        pull_start.isoformat(),
        pull_end.isoformat(),
    )

    # 3. Open-Meteo pull
    pull_spec = OpenMeteoPullSpec(
        lat=float(lgeo.geo.center_lat),
        lon=float(lgeo.geo.center_lon),
        start_date=pull_start,
        end_date=pull_end,
        timezone="auto",
    )
    om_result = fetch_history(pull_spec, cache_dir=cfg.cache_dir)
    logger.info(
        "Open-Meteo: %d hours (source=%s, NaN=%d)",
        len(om_result.df),
        om_result.source,
        om_result.nan_hour_count,
    )
    # If LANDFIRE elevation was unavailable, use OM's.
    if (
        not np.isfinite(lgeo.elevation_ft)
        and np.isfinite(om_result.elevation_m)
    ):
        from embrs.utilities.unit_conversions import m_to_ft

        elevation_ft = float(m_to_ft(om_result.elevation_m))
        logger.info(
            "LANDFIRE elevation unavailable; using Open-Meteo elevation %.0f ft.",
            elevation_ft,
        )

    # 4. Apply wind correction (once — used everywhere downstream)
    weather_df = _apply_wind_correction(om_result, cfg)

    # 5. Write full-season .wxs
    full_wxs_path = _write_full_season_wxs(cfg, weather_df, elevation_ft)
    logger.info("Wrote full-season .wxs (%d rows): %s", len(weather_df), full_wxs_path)

    # 6. Run the BI pipeline once
    scenario_start_dt = dt.datetime(
        scenario_start_date.year, scenario_start_date.month, scenario_start_date.day
    )
    bi_result: BIPipelineResult = run_bi(
        full_wxs_path=full_wxs_path,
        landscape_tif=cfg.landscape_tif,
        scenario_start=scenario_start_dt,
        bi_section=cfg.bi,
    )

    # 7. Merge BI into the weather frame
    merged = _merge_bi_into_weather(weather_df, bi_result.trajectory_df)

    # 8. Enumerate windows (only within scenario period)
    scenario_start_ts = _localize_scenario_start(scenario_start_date, weather_df)
    pull_end_ts = _localize_pull_end(pull_end, weather_df)
    windows: List[Window] = list(
        iter_windows(
            weather_df=merged,
            scenario_start=scenario_start_ts,
            scenario_end=pull_end_ts + pd.Timedelta(hours=1),
            scenario_length_hours=cfg.scenario_length_hours,
            stride_hours=cfg.window_stride_hours,
            required_columns=DEFAULT_REQUIRED_COLUMNS,
        )
    )
    n_skipped_for_missing = _approx_skipped_count(
        scenario_start_ts,
        pull_end_ts + pd.Timedelta(hours=1),
        cfg.scenario_length_hours,
        cfg.window_stride_hours,
        len(windows),
    )
    logger.info(
        "Windowing: enumerated %d windows (stride=%dh, length=%dh).",
        len(windows),
        cfg.window_stride_hours,
        cfg.scenario_length_hours,
    )

    # 9. Per-window peaks
    per_window = per_window_peaks(merged, windows)

    # 10. Lulls per window
    lulls_by_window: Dict[str, List[Lull]] = {}
    for w in windows:
        lulls_by_window[w.window_id] = detect_lulls(w.df, cfg.lull)

    # 11. Filter + score + select (per cfg.bi_filter_mode)
    filter_columns = columns_for_bi_filter_mode(cfg.bi_filter_mode)
    score_distance_column = score_distance_column_for_mode(cfg.bi_filter_mode)
    logger.info(
        "BI filter mode = %r (filter on %s, score distance on %r).",
        cfg.bi_filter_mode, list(filter_columns), score_distance_column,
    )
    filtered = filter_by_target_band(
        per_window, cfg.bi_target_band, columns=filter_columns
    )
    scored = score_windows(
        filtered, lulls_by_window, cfg.bi_target_band, cfg.scoring,
        bi_distance_column=score_distance_column,
    )
    selected = select_top_n(
        scored,
        lulls_by_window,
        cfg.n_candidates,
        min_separation_hours=cfg.effective_min_separation_hours,
    )
    if len(selected) < cfg.n_candidates and not filtered.empty:
        logger.info(
            "Temporal NMS (min_separation_hours=%d) suppressed %d candidates; "
            "selected %d / requested %d. Lower min_candidate_separation_hours "
            "in the config to soften this.",
            cfg.effective_min_separation_hours,
            len(filtered) - len(selected),
            len(selected),
            cfg.n_candidates,
        )
    n_passing = len(filtered)

    # 12. Artifacts
    selected_by_window = {c.window_id: c for c in selected}
    windows_by_id = {w.window_id: w for w in windows}
    for c in selected:
        w = windows_by_id[c.window_id]
        artifacts.write_candidate_dir(
            cfg=cfg,
            candidate=c,
            window_df=w.df,
            bi_trajectory=bi_result.trajectory_df,
            geo_lat=float(lgeo.geo.center_lat),
            geo_lon=float(lgeo.geo.center_lon),
            geo_tz=str(lgeo.geo.timezone),
            full_season_wxs_path=full_wxs_path,
            query_timestamp_utc=query_ts_utc,
        )

    pull_stats = {
        "n_hours": int(len(om_result.df)),
        "nan_hour_count": int(om_result.nan_hour_count),
        "source": om_result.source,
    }
    artifacts.write_summary_csv(
        cfg=cfg,
        per_window=per_window,
        scored_passing=scored,
        selected_by_window=selected_by_window,
        lulls_by_window=lulls_by_window,
    )
    artifacts.write_report_md(
        cfg=cfg,
        pull_stats=pull_stats,
        n_windows_enumerated=len(windows),
        n_windows_skipped=n_skipped_for_missing,
        n_passing=n_passing,
        selected=selected,
    )

    if not selected:
        logger.warning(
            "No candidates passed the BI band filter [%.1f, %.1f]; "
            "summary.csv and report.md were still written. "
            "Consider another year or a wider band.",
            cfg.bi_target_band[0],
            cfg.bi_target_band[1],
        )
        return 2

    logger.info(
        "Wrote %d candidate directories under %s",
        len(selected),
        artifacts.candidates_dir(cfg),
    )
    return 0


def _approx_skipped_count(
    scenario_start: pd.Timestamp,
    scenario_end: pd.Timestamp,
    scenario_length_hours: int,
    stride_hours: int,
    n_emitted: int,
) -> int:
    """Approximate the number of candidate starts that were skipped."""
    latest_start = scenario_end - pd.Timedelta(hours=scenario_length_hours)
    candidate_count = len(
        pd.date_range(
            start=scenario_start,
            end=latest_start,
            freq=pd.Timedelta(hours=stride_hours),
            inclusive="both",
        )
    )
    return max(0, candidate_count - n_emitted)
