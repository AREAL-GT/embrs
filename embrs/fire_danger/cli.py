"""Component 9 — command-line interface.

Run ``python -m embrs.fire_danger --landscape <.tif> --wxs <.wxs>
--scenario-start <YYYY-MM-DDTHH:MM>``. See ``--help`` for all options.
"""
from __future__ import annotations

import argparse
import configparser
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from embrs.fire_danger.config import Config
from embrs.fire_danger.output import plot_trajectory, write_csv
from embrs.fire_danger.trajectory import compute_bi_trajectory


def _parse_dt(s: str) -> datetime:
    # Accept "YYYY-MM-DDTHH:MM" or "YYYY-MM-DD HH:MM" or with seconds.
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"--scenario-start: could not parse {s!r} (try YYYY-MM-DDTHH:MM)"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m embrs.fire_danger",
        description="Area-weighted NFDRS Burning Index trajectory tool.",
    )
    p.add_argument("--config", help="INI config file (CLI flags override)")
    p.add_argument("--landscape", help="LANDFIRE .tif/.lcp path")
    p.add_argument("--wxs", help=".wxs weather file path")
    p.add_argument("--scenario-start", type=_parse_dt,
                   help="Conditioning <-> scenario boundary (e.g. 2025-07-22T06:00)")
    p.add_argument("--out-csv", default="bi_trajectory.csv",
                   help="Output CSV path")
    p.add_argument("--out-plot", default=None,
                   help="Optional output PNG plot path")
    p.add_argument("--avg-ann-precip", type=float, default=None,
                   dest="avg_ann_precip_in",
                   help="Avg annual precip (in); auto-fetched if omitted")
    p.add_argument("--slope-class", type=int, default=None,
                   help="NFDRS slope class 1-5 (override LANDFIRE-derived)")
    p.add_argument("--lat", type=float, default=None, dest="lat_override",
                   help="Override landscape-centroid latitude")
    p.add_argument("--min-area-frac", type=float, default=0.05,
                   help="Drop NFDRS models with <this fraction (OQ-8)")
    p.add_argument("--reg-obs-hr", type=int, default=13,
                   help="NFDRS regular observation hour (default 13 = 1 PM)")
    p.add_argument("--cloud-scale", default="percent",
                   choices=("percent", "fraction", "okta", "tenths"),
                   help="Units of .wxs CloudCov column (default percent)")
    p.add_argument("--snow-mode", default="none",
                   choices=("none", "temp-derived"),
                   help="Snow handling (default no snow)")
    return p


def _config_from_args(ns: argparse.Namespace) -> Config:
    """Build a Config from parsed args + optional INI overrides.

    INI file (loaded first when --config is given) has section
    ``[fire_danger]`` whose keys mirror the CLI flag names (with underscores).
    CLI flags then override anything from the INI.
    """
    defaults = {
        "landscape_path": ns.landscape,
        "wxs_path": ns.wxs,
        "scenario_start": ns.scenario_start,
        "out_csv": ns.out_csv,
        "out_plot": ns.out_plot,
        "avg_ann_precip_in": ns.avg_ann_precip_in,
        "slope_class": ns.slope_class,
        "lat_override": ns.lat_override,
        "min_area_frac": ns.min_area_frac,
        "reg_obs_hr": ns.reg_obs_hr,
        "cloud_scale": ns.cloud_scale,
        "snow_mode": ns.snow_mode,
    }
    if ns.config:
        parser = configparser.ConfigParser()
        parser.read(ns.config)
        if parser.has_section("fire_danger"):
            section = parser["fire_danger"]
            # Only override the fields the user did NOT pass on CLI.
            # Detect "user passed" by comparing against the argparse defaults.
            cli_defaults = {
                "out_csv": "bi_trajectory.csv",
                "min_area_frac": 0.05,
                "reg_obs_hr": 13,
                "cloud_scale": "percent",
                "snow_mode": "none",
            }
            for key in defaults:
                ini_val = section.get(key)
                if ini_val is None:
                    continue
                cli_val = defaults[key]
                # If CLI value is at its default (or None), accept INI value.
                if cli_val is None or cli_val == cli_defaults.get(key):
                    if key == "scenario_start":
                        defaults[key] = _parse_dt(ini_val)
                    elif key in ("avg_ann_precip_in", "min_area_frac", "lat_override"):
                        defaults[key] = float(ini_val)
                    elif key in ("slope_class", "reg_obs_hr"):
                        defaults[key] = int(ini_val)
                    else:
                        defaults[key] = ini_val

    # Required fields validation
    missing = [k for k in ("landscape_path", "wxs_path", "scenario_start")
               if defaults[k] is None]
    if missing:
        raise SystemExit(
            f"missing required: {', '.join(missing)} "
            f"(supply via CLI or --config)"
        )

    return Config(**defaults)


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _config_from_args(args)
    result = compute_bi_trajectory(cfg)
    write_csv(result, cfg.out_csv)
    if cfg.out_plot:
        plot_trajectory(result, cfg.out_plot)
    print(
        f"wrote {cfg.out_csv} ({len(result.df)} rows) — "
        f"peak BI (97th pct scenario) = {result.peak_bi:.2f} — "
        f"composition: "
        + ", ".join(f"{m}={f:.0%}"
                    for m, f in result.fuel_composition.fractions.items())
        + f"; slope class {result.metadata['slope_class_used']}, "
        + f"AvgAnnPrecip={result.metadata['avg_ann_precip_in']:.2f} in "
          f"({result.metadata['avg_ann_precip_source']})"
    )
    return 0
