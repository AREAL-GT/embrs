"""Output artifacts: per-candidate dirs, metadata.json, 4-panel plot,
top-level summary.csv, top-level report.md.

Plan §4.10.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from embrs.weather_candidate_search.config import Config
from embrs.weather_candidate_search.lull_detection import Lull
from embrs.weather_candidate_search.ranking import RankedCandidate

logger = logging.getLogger(__name__)


METADATA_SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Directory layout helpers
# ---------------------------------------------------------------------------


def cell_dir(cfg: Config) -> str:
    return cfg.cell_dir


def candidates_dir(cfg: Config) -> str:
    return os.path.join(cell_dir(cfg), "candidates")


def candidate_subdir(cfg: Config, candidate: RankedCandidate) -> str:
    return os.path.join(
        candidates_dir(cfg),
        f"{candidate.rank:02d}_{candidate.window_id}",
    )


def ensure_cell_dir(cfg: Config) -> str:
    """Create the cell directory if missing; return its path."""
    d = cell_dir(cfg)
    os.makedirs(d, exist_ok=True)
    return d


def reset_candidates_dir(cfg: Config) -> str:
    """Remove and recreate the ``candidates/`` directory, returning its path.

    Candidate subdirs are named ``{rank}_{window_id}``; a re-run that selects
    different windows would otherwise leave the previous run's candidate
    directories behind, polluting the cell output with stale results. Wiping
    the directory before writing keeps it in sync with the current run.
    """
    d = candidates_dir(cfg)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Full-season .wxs handling
# ---------------------------------------------------------------------------


def stage_full_season_wxs(cfg: Config, src_wxs_path: str) -> str:
    """Copy ``src_wxs_path`` to ``{cell_dir}/full_season.wxs``; return the path.

    If ``src`` already equals the destination, this is a no-op.
    """
    ensure_cell_dir(cfg)
    dst = os.path.join(cell_dir(cfg), "full_season.wxs")
    if os.path.abspath(src_wxs_path) == os.path.abspath(dst):
        return dst
    shutil.copy2(src_wxs_path, dst)
    logger.info("Staged full-season .wxs at %s", dst)
    return dst


# ---------------------------------------------------------------------------
# Synoptic summary
# ---------------------------------------------------------------------------


def _mean_afternoon_wind(window_df: pd.DataFrame) -> float:
    """Mean wind (mph) over hours 13:00-17:59 local; NaN if none present."""
    if "wind_mph" not in window_df.columns:
        return float("nan")
    hours = window_df.index.hour
    mask = (hours >= 13) & (hours < 18)
    if not mask.any():
        return float("nan")
    return float(window_df["wind_mph"].to_numpy()[mask].mean())


def build_synoptic_summary(
    candidate: RankedCandidate, window_df: pd.DataFrame
) -> str:
    """Short auto-generated description for ``metadata.json``."""
    hour_of_peak = "?"
    try:
        delta = candidate.time_of_peak - candidate.start
        hour_of_peak = int(delta.total_seconds() // 3600)
    except Exception:        # pragma: no cover
        pass
    mean_wind = _mean_afternoon_wind(window_df)
    mean_wind_str = (
        f"{mean_wind:.0f} mph" if np.isfinite(mean_wind) else "n/a"
    )
    return (
        f"Mean daily-peak BI {candidate.mean_daily_peak_bi:.1f}; "
        f"window peak BI {candidate.peak_bi:.1f} (97th pct hourly) at hour "
        f"{hour_of_peak}; mean daily 1 PM BI {candidate.mean_daily_1pm_bi:.1f}; "
        f"{candidate.n_lulls} backburn windows totaling "
        f"{candidate.total_lull_hours} hours; "
        f"mean afternoon wind {mean_wind_str}."
    )


# ---------------------------------------------------------------------------
# metadata.json
# ---------------------------------------------------------------------------


def _serialize_ts(ts: pd.Timestamp) -> str:
    """ISO-8601 with tz suffix if present."""
    if pd.isna(ts):
        return ""
    return pd.Timestamp(ts).isoformat()


def write_metadata_json(
    cfg: Config,
    candidate: RankedCandidate,
    candidate_window_df: pd.DataFrame,
    geo_lat: float,
    geo_lon: float,
    geo_tz: str,
    full_season_wxs_relpath: str,
    out_path: str,
    target_band: Tuple[float, float],
    query_timestamp_utc: dt.datetime,
) -> None:
    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "source": {
            "region_tag": cfg.region_tag,
            "volatility_class": cfg.volatility_class,
            "year": cfg.year,
            "fire_season_start_month": cfg.fire_season_start_month,
            "fire_season_end_month": cfg.fire_season_end_month,
            "centroid_lat_wgs84": float(geo_lat),
            "centroid_lon_wgs84": float(geo_lon),
            "timezone": geo_tz,
            "query_timestamp_utc": query_timestamp_utc.isoformat(),
            "full_season_wxs": full_season_wxs_relpath,
        },
        "window": {
            "window_id": candidate.window_id,
            "start_local": _serialize_ts(candidate.start),
            "end_local": _serialize_ts(candidate.end),
            "duration_hours": int(cfg.scenario_length_hours),
        },
        "bi": {
            "mean_daily_peak_bi": float(candidate.mean_daily_peak_bi),
            "peak_bi": float(candidate.peak_bi),
            "peak_percentile": 97,
            "time_of_peak_local": _serialize_ts(candidate.time_of_peak),
            "mean_bi": float(candidate.mean_bi),
            "mean_daily_1pm_bi": float(candidate.mean_daily_1pm_bi),
            "target_band": [float(target_band[0]), float(target_band[1])],
        },
        "lulls": [
            {
                "start_local": _serialize_ts(l.start),
                "end_local": _serialize_ts(l.end),
                "duration_hours": int(l.duration_hours),
            }
            for l in candidate.lulls
        ],
        "synoptic_summary": build_synoptic_summary(candidate, candidate_window_df),
        "score": float(candidate.score),
        "score_components": {
            "bi_distance_normalized": float(candidate.bi_distance_normalized),
            "n_lulls": int(candidate.n_lulls),
            "total_lull_hours": int(candidate.total_lull_hours),
        },
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=False)


# ---------------------------------------------------------------------------
# 4-panel plot
# ---------------------------------------------------------------------------


def _shade_lulls(ax, lulls: List[Lull]) -> None:
    for l in lulls:
        ax.axvspan(l.start, l.end + pd.Timedelta(hours=1), alpha=0.18, color="green")


def plot_candidate(
    candidate: RankedCandidate,
    window_df: pd.DataFrame,
    target_band: Tuple[float, float],
    out_path: str,
) -> None:
    """4-panel matplotlib figure for one candidate."""
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10), sharex=True)
    ax_bi, ax_th, ax_w, ax_lull = axes

    # Panel 1: BI trajectory
    if "BI_area_weighted" in window_df.columns:
        ax_bi.plot(
            window_df.index,
            window_df["BI_area_weighted"].to_numpy(),
            color="black",
            linewidth=1.8,
            label="BI_area_weighted",
        )
    lo, hi = float(target_band[0]), float(target_band[1])
    ax_bi.axhspan(lo, hi, alpha=0.15, color="orange", label=f"target band [{lo:.0f}, {hi:.0f}]")
    if np.isfinite(candidate.peak_bi):
        ax_bi.axhline(
            candidate.peak_bi,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"peak BI = {candidate.peak_bi:.1f} (97th pct)",
        )
    if pd.notna(candidate.time_of_peak):
        ax_bi.axvline(candidate.time_of_peak, color="red", alpha=0.3, linewidth=0.8)
    _shade_lulls(ax_bi, candidate.lulls)
    ax_bi.set_ylabel("BI")
    ax_bi.set_title(
        f"Candidate rank {candidate.rank} — start {candidate.window_id}"
    )
    ax_bi.legend(loc="upper left", fontsize=8)
    ax_bi.grid(True, alpha=0.3)

    # Panel 2: temperature (°F) and RH (%)
    temp_F = window_df["temp_C"].to_numpy() * 9.0 / 5.0 + 32.0
    ax_th.plot(window_df.index, temp_F, color="firebrick", label="Temp (°F)")
    ax_th.set_ylabel("Temp (°F)", color="firebrick")
    ax_th.tick_params(axis="y", labelcolor="firebrick")
    ax_th_twin = ax_th.twinx()
    ax_th_twin.plot(window_df.index, window_df["rh_pct"].to_numpy(),
                    color="steelblue", linewidth=1.0, label="RH (%)")
    ax_th_twin.set_ylabel("RH (%)", color="steelblue")
    ax_th_twin.tick_params(axis="y", labelcolor="steelblue")
    _shade_lulls(ax_th, candidate.lulls)
    ax_th.grid(True, alpha=0.3)

    # Panel 3: wind speed (mph) and direction (deg)
    if "wind_mph" in window_df.columns:
        ax_w.plot(window_df.index, window_df["wind_mph"].to_numpy(),
                  color="darkgreen", label="WindSpd (mph, 20 ft)")
    ax_w.set_ylabel("Wind speed (mph)", color="darkgreen")
    ax_w.tick_params(axis="y", labelcolor="darkgreen")
    ax_w_twin = ax_w.twinx()
    if "wind_dir_deg" in window_df.columns:
        ax_w_twin.scatter(window_df.index, window_df["wind_dir_deg"].to_numpy(),
                          s=6, color="purple", alpha=0.5, label="WindDir (°)")
    ax_w_twin.set_ylabel("Wind direction (°)", color="purple")
    ax_w_twin.tick_params(axis="y", labelcolor="purple")
    ax_w_twin.set_ylim(0, 360)
    _shade_lulls(ax_w, candidate.lulls)
    ax_w.grid(True, alpha=0.3)

    # Panel 4: lull bars
    ax_lull.set_ylim(0, 1)
    ax_lull.set_yticks([])
    if candidate.lulls:
        for l in candidate.lulls:
            ax_lull.axvspan(l.start, l.end + pd.Timedelta(hours=1), alpha=0.55, color="green")
        for l in candidate.lulls:
            mid = l.start + (l.end - l.start) / 2
            ax_lull.annotate(
                f"{l.duration_hours}h", xy=(mid, 0.5),
                ha="center", va="center", fontsize=9,
            )
    else:
        ax_lull.text(0.5, 0.5, "(no detected backburn windows)",
                     ha="center", va="center", transform=ax_lull.transAxes,
                     fontsize=10, color="gray")
    ax_lull.set_ylabel("Lulls")
    ax_lull.set_xlabel("Local time")

    # Render the date axis in the data's own (local) timezone. matplotlib
    # otherwise labels tz-aware timestamps in UTC (rcParams['timezone']),
    # which shifts an afternoon BI peak ~5-6 h to the right so it appears to
    # fall near midnight. Passing the index tz keeps the axis in local time.
    axis_tz = window_df.index.tz
    locator = mdates.AutoDateLocator(tz=axis_tz)
    ax_lull.xaxis.set_major_locator(locator)
    ax_lull.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator, tz=axis_tz)
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-candidate directory
# ---------------------------------------------------------------------------


def write_candidate_dir(
    cfg: Config,
    candidate: RankedCandidate,
    window_df: pd.DataFrame,
    bi_trajectory: pd.DataFrame,
    geo_lat: float,
    geo_lon: float,
    geo_tz: str,
    full_season_wxs_path: str,
    query_timestamp_utc: dt.datetime,
) -> str:
    """Materialise one candidate's metadata.json + plot.png and return the dir."""
    sub = candidate_subdir(cfg, candidate)
    os.makedirs(sub, exist_ok=True)

    # Build a plotting frame that includes the BI trajectory inside the
    # window. ``window_df`` already has BI joined upstream, but defensively
    # join again in case a future caller passes a weather-only frame.
    plot_df = window_df
    if "BI_area_weighted" not in plot_df.columns:
        sliced_bi = bi_trajectory.loc[
            (bi_trajectory.index >= candidate.start)
            & (bi_trajectory.index < candidate.end),
            ["BI_area_weighted"],
        ]
        plot_df = plot_df.join(sliced_bi, how="left")

    rel_full_season = os.path.relpath(
        full_season_wxs_path, sub
    )
    write_metadata_json(
        cfg=cfg,
        candidate=candidate,
        candidate_window_df=plot_df,
        geo_lat=geo_lat,
        geo_lon=geo_lon,
        geo_tz=geo_tz,
        full_season_wxs_relpath=rel_full_season,
        out_path=os.path.join(sub, "metadata.json"),
        target_band=cfg.bi_target_band,
        query_timestamp_utc=query_timestamp_utc,
    )
    plot_candidate(
        candidate,
        plot_df,
        cfg.bi_target_band,
        out_path=os.path.join(sub, "plot.png"),
    )
    return sub


# ---------------------------------------------------------------------------
# Top-level summary.csv + report.md
# ---------------------------------------------------------------------------


def write_summary_csv(
    cfg: Config,
    per_window: pd.DataFrame,
    scored_passing: pd.DataFrame,
    selected_by_window: dict,
    lulls_by_window: dict,
) -> str:
    """Write one row per evaluated window with all diagnostics.

    Columns: ``window_id, start, end, peak_bi, time_of_peak, mean_bi,
    mean_daily_peak_bi, n_daily_peak_samples, mean_daily_1pm_bi,
    n_daily_1pm_samples, n_lulls, total_lull_hours, score,
    bi_distance_normalized, passed_band_filter, selected, rank``.
    """
    out_path = os.path.join(cell_dir(cfg), "summary.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rows: List[dict] = []
    passing_ids = set(scored_passing.index) if not scored_passing.empty else set()
    for window_id, row in per_window.iterrows():
        lulls = lulls_by_window.get(window_id, [])
        passed = window_id in passing_ids
        n_lulls = len(lulls)
        total_lull_hours = sum(l.duration_hours for l in lulls)
        if passed:
            score = float(scored_passing.loc[window_id, "score"])
            bi_dist = float(scored_passing.loc[window_id, "bi_distance_normalized"])
        else:
            score = float("nan")
            bi_dist = float("nan")
        selection = selected_by_window.get(window_id)
        rows.append(
            {
                "window_id": window_id,
                "start": row["start"],
                "end": row["end"],
                "peak_bi": float(row["peak_bi"]),
                "time_of_peak": row["time_of_peak"],
                "mean_bi": float(row["mean_bi"]),
                "mean_daily_peak_bi": float(row.get("mean_daily_peak_bi", float("nan"))),
                "n_daily_peak_samples": int(row.get("n_daily_peak_samples", 0)),
                "mean_daily_1pm_bi": float(row.get("mean_daily_1pm_bi", float("nan"))),
                "n_daily_1pm_samples": int(row.get("n_daily_1pm_samples", 0)),
                "n_lulls": int(n_lulls),
                "total_lull_hours": int(total_lull_hours),
                "score": score,
                "bi_distance_normalized": bi_dist,
                "passed_band_filter": bool(passed),
                "selected": bool(selection is not None),
                "rank": int(selection.rank) if selection is not None else -1,
            }
        )

    df = pd.DataFrame(rows).sort_values(by="start", kind="mergesort")
    df.to_csv(out_path, index=False)
    return out_path


def write_report_md(
    cfg: Config,
    pull_stats: dict,
    n_windows_enumerated: int,
    n_windows_skipped: int,
    n_passing: int,
    selected: List[RankedCandidate],
) -> str:
    """Write the human-readable per-cell report."""
    out_path = os.path.join(cell_dir(cfg), "report.md")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    lines: List[str] = []
    lines.append(f"# Candidate weather window search — {cfg.region_tag} / {cfg.volatility_class}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Landscape: `{cfg.landscape_tif}`")
    lines.append(f"- Year: {cfg.year}")
    lines.append(
        f"- Fire season: month {cfg.fire_season_start_month} – {cfg.fire_season_end_month}"
    )
    lines.append(f"- Scenario length (h): {cfg.scenario_length_hours}")
    lines.append(
        f"- Target BI band: [{cfg.bi_target_band[0]:.1f}, {cfg.bi_target_band[1]:.1f}]"
    )
    lines.append(f"- Conditioning days: {cfg.conditioning_days}")
    lines.append(f"- Window stride (h): {cfg.window_stride_hours}")
    lines.append(
        "- Wind 10m→20ft correction: "
        + ("on" if cfg.wind_conversion.enabled else "off")
        + f" (z0={cfg.wind_conversion.surface_roughness_m} m)"
    )
    wg = cfg.wetness_guard
    if wg.enabled:
        lines.append(
            f"- Wetness guard: on (antecedent ≤ {wg.max_antecedent_precip_in} in "
            f"over {wg.antecedent_days} d; daily ≤ {wg.max_daily_precip_in} in)"
        )
    else:
        lines.append("- Wetness guard: off")
    lines.append("")
    lines.append("## Search counts")
    lines.append(f"- Hours pulled: {pull_stats.get('n_hours', 'n/a')}")
    lines.append(f"- NaN hours in pull: {pull_stats.get('nan_hour_count', 'n/a')}")
    lines.append(f"- Cache hit: {pull_stats.get('source', 'n/a')}")
    lines.append(f"- Windows enumerated: {n_windows_enumerated}")
    lines.append(f"- Windows skipped (missing data): {n_windows_skipped}")
    lines.append(f"- Windows passing BI band filter: {n_passing}")
    lines.append(f"- Windows selected: {len(selected)}")
    lines.append("")
    lines.append("## Top candidates")
    if not selected:
        lines.append("_No candidates passed the BI band filter (qa J2)._")
    else:
        lines.append(
            "| Rank | Window start | Mean daily-peak BI | Peak BI (97pct) "
            "| Mean 1pm BI | n lulls | Lull hours | Score |"
        )
        lines.append(
            "|------|--------------|--------------------|-----------------"
            "|-------------|---------|------------|-------|"
        )
        for c in selected:
            lines.append(
                f"| {c.rank} | {c.window_id} | {c.mean_daily_peak_bi:.1f} | "
                f"{c.peak_bi:.1f} | {c.mean_daily_1pm_bi:.1f} | {c.n_lulls} | "
                f"{c.total_lull_hours} | {c.score:+.3f} |"
            )

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_path
