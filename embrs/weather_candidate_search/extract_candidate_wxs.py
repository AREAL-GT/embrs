"""Slice the full-season ``.wxs`` for a selected candidate into a clean
``[window.start - conditioning_days, window.end]`` ``.wxs`` that EMBRS
``WeatherStream.get_stream_from_wxs`` will accept directly.

Plan §4.12, qa G2.

Usage:
    python -m embrs.weather_candidate_search.extract_candidate_wxs \\
        path/to/01_2025-07-22T0600 --conditioning-days 30 --out forecast.wxs
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

from embrs.fire_danger.weather_loader import load_wxs
from embrs.utilities.unit_conversions import m_to_ft, ft_to_m
from embrs.weather_candidate_search.config import WindConversionConfig
from embrs.weather_candidate_search.wxs_writer import WxsWriteSpec, write_wxs

logger = logging.getLogger(__name__)

DEFAULT_CONDITIONING_DAYS: int = 30


def _read_metadata(candidate_dir: str) -> dict:
    path = os.path.join(candidate_dir, "metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metadata.json not found in {candidate_dir}")
    with open(path) as fh:
        return json.load(fh)


def _resolve_full_season_path(candidate_dir: str, metadata: dict) -> str:
    rel = metadata["source"]["full_season_wxs"]
    resolved = os.path.normpath(os.path.join(candidate_dir, rel))
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"Full-season .wxs referenced by metadata not found: {resolved}"
        )
    return resolved


def _slice_to_candidate_window(
    full_season_df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp,
    conditioning_days: int,
) -> pd.DataFrame:
    """Return the ``[start - conditioning_days, end)`` slice."""
    lower = window_start - pd.Timedelta(days=conditioning_days)
    sliced = full_season_df.loc[
        (full_season_df.index >= lower) & (full_season_df.index < window_end)
    ]
    if sliced.empty:
        raise ValueError(
            "Sliced candidate .wxs is empty — check window timestamps and the "
            "full-season .wxs span."
        )
    return sliced


def _to_writer_columns(loaded_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the BI loader's `HourlyWeather` schema to the WxsWriteSpec one.

    The BI loader exposes: temp_F/C, rh_pct/frac, wind_mph (already 20 ft),
    wind_dir_deg, precip_in_hr/cm_hr, cloud_cover. We map these into the
    writer's canonical columns and pass ``wind_mph_precomputed='wind_mph'``
    so no second log-profile correction is applied.
    """
    out = pd.DataFrame(
        {
            "temp_C": loaded_df["temp_C"],
            "rh_pct": loaded_df["rh_pct"],
            # Convert in/hr -> mm/hr (writer multiplies by mm_to_in again).
            "rain_mm_hr": loaded_df["precip_in_hr"] * 25.4,
            "wind_mph": loaded_df["wind_mph"],
            "wind_dir_deg": loaded_df["wind_dir_deg"],
            "cloud_pct": loaded_df["cloud_cover"],
        },
        index=loaded_df.index,
    )
    return out


def extract_candidate_wxs(
    candidate_dir: str,
    conditioning_days: int = DEFAULT_CONDITIONING_DAYS,
    out_path: Optional[str] = None,
) -> str:
    """Materialise an EMBRS-runnable ``.wxs`` for one candidate.

    Args:
        candidate_dir: Path to a ``{rank}_{window_id}/`` directory.
        conditioning_days: How many days of pre-window history to include.
        out_path: Output path. Default: ``{candidate_dir}/forecast.wxs``.

    Returns:
        The path to the written .wxs.
    """
    metadata = _read_metadata(candidate_dir)
    full_season_path = _resolve_full_season_path(candidate_dir, metadata)

    full = load_wxs(full_season_path)
    full_df = full.df.copy()

    # Parse the window bounds (tz-aware ISO strings).
    window_start = pd.Timestamp(metadata["window"]["start_local"])
    window_end = pd.Timestamp(metadata["window"]["end_local"])
    if window_start.tz is not None and full_df.index.tz is None:
        # The original .wxs was written naive-local; assume window
        # timestamps' tz matches the source timezone in metadata.
        # Drop the tz for slicing convenience.
        window_start = window_start.tz_localize(None)
        window_end = window_end.tz_localize(None)
    elif window_start.tz is None and full_df.index.tz is not None:
        # Should not happen, but tolerate it.
        full_df.index = full_df.index.tz_localize(None)

    sliced_loader_df = _slice_to_candidate_window(
        full_df, window_start, window_end, conditioning_days
    )

    writer_df = _to_writer_columns(sliced_loader_df)
    out_path = out_path or os.path.join(candidate_dir, "forecast.wxs")

    spec = WxsWriteSpec(
        df=writer_df,
        elevation_ft=int(round(m_to_ft(full.ref_elev_m))),
        wind_correction=WindConversionConfig(enabled=False),  # already corrected
        wind_mph_precomputed="wind_mph",
    )
    write_wxs(spec, out_path)
    logger.info(
        "Wrote %d-row candidate .wxs to %s (window=%s..%s, conditioning=%d days)",
        len(writer_df),
        out_path,
        window_start,
        window_end,
        conditioning_days,
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m embrs.weather_candidate_search.extract_candidate_wxs",
        description=(
            "Slice the shared full-season .wxs for a candidate into a clean "
            "[window.start - conditioning_days, window.end] .wxs."
        ),
    )
    p.add_argument(
        "candidate_dir",
        help="Path to a candidate directory (e.g. .../01_2025-07-22T0600)",
    )
    p.add_argument(
        "--conditioning-days",
        type=int,
        default=DEFAULT_CONDITIONING_DAYS,
        help=f"Days of pre-window history (default: {DEFAULT_CONDITIONING_DAYS})",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .wxs path (default: <candidate_dir>/forecast.wxs)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    args = _build_parser().parse_args(argv)
    extract_candidate_wxs(
        candidate_dir=args.candidate_dir,
        conditioning_days=args.conditioning_days,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":      # pragma: no cover
    sys.exit(main())
