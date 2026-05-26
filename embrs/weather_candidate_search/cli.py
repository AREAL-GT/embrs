"""Command-line interface — ``python -m embrs.weather_candidate_search``.

YAML config + CLI overrides. CLI flags override YAML; YAML overrides defaults.
Plan §4.2.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Optional

from embrs.weather_candidate_search.config import (
    BISection,
    Config,
    LullConfig,
    ScoringConfig,
    WindConversionConfig,
)
from embrs.weather_candidate_search.pipeline import run_candidate_search


_SENTINEL = object()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m embrs.weather_candidate_search",
        description=(
            "Search Open-Meteo ERA5 history for fire-weather windows whose peak "
            "BI lands in a target volatility band, with detected lulls as a "
            "soft ranking signal."
        ),
    )
    p.add_argument("--config", help="YAML config file (CLI overrides)", default=None)

    # Required top-level fields (also acceptable in YAML).
    p.add_argument("--landscape-tif", default=_SENTINEL)
    p.add_argument("--year", type=int, default=_SENTINEL)
    p.add_argument("--fire-season-start-month", type=int, default=_SENTINEL)
    p.add_argument("--fire-season-end-month", type=int, default=_SENTINEL)
    p.add_argument(
        "--scenario-length-hours",
        type=int,
        default=_SENTINEL,
        help="REQUIRED — window length in hours (qa H3, no default).",
    )
    p.add_argument(
        "--bi-target-band",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=_SENTINEL,
        help="Target BI band, e.g. --bi-target-band 60 80",
    )
    p.add_argument("--output-dir", default=_SENTINEL)
    p.add_argument("--region-tag", default=_SENTINEL)
    p.add_argument("--volatility-class", default=_SENTINEL)

    # Optional knobs
    p.add_argument("--n-candidates", type=int, default=_SENTINEL)
    p.add_argument("--cache-dir", default=_SENTINEL)
    p.add_argument("--conditioning-days", type=int, default=_SENTINEL)
    p.add_argument("--window-stride-hours", type=int, default=_SENTINEL)
    p.add_argument(
        "--min-candidate-separation-hours",
        type=int,
        default=_SENTINEL,
        help=(
            "Greedy NMS: forbid two selected candidates whose starts are "
            "closer than this. Default = scenario_length_hours (selected "
            "windows never overlap). Set 0 to disable."
        ),
    )

    # Most-frequently-tuned nested fields surfaced as flags (plan §4.2).
    p.add_argument("--wind-threshold-mph", type=float, default=_SENTINEL)
    p.add_argument("--rh-threshold-pct", type=float, default=_SENTINEL)
    p.add_argument("--min-lull-hours", type=int, default=_SENTINEL,
                   help="Equivalent to lull.min_consecutive_hours")
    p.add_argument("--lull-tolerance-hours", type=int, default=_SENTINEL)
    p.add_argument(
        "--no-wind-height-correction",
        action="store_true",
        help="Disable the 10 m → 20 ft log-profile wind correction (qa B4).",
    )

    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _coalesce(cli_value, yaml_value, default):
    """CLI > YAML > default."""
    if cli_value is not _SENTINEL:
        return cli_value
    if yaml_value is not None:
        return yaml_value
    return default


def _build_lull(yaml_section: dict, ns) -> LullConfig:
    defaults = LullConfig()
    return LullConfig(
        wind_threshold_mph=_coalesce(
            ns.wind_threshold_mph,
            yaml_section.get("wind_threshold_mph"),
            defaults.wind_threshold_mph,
        ),
        rh_threshold_pct=_coalesce(
            ns.rh_threshold_pct,
            yaml_section.get("rh_threshold_pct"),
            defaults.rh_threshold_pct,
        ),
        min_consecutive_hours=_coalesce(
            ns.min_lull_hours,
            yaml_section.get("min_consecutive_hours"),
            defaults.min_consecutive_hours,
        ),
        tolerance_hours=_coalesce(
            ns.lull_tolerance_hours,
            yaml_section.get("tolerance_hours"),
            defaults.tolerance_hours,
        ),
    )


def _build_scoring(yaml_section: dict) -> ScoringConfig:
    defaults = ScoringConfig()
    return ScoringConfig(
        bi_distance_weight=float(
            yaml_section.get("bi_distance_weight", defaults.bi_distance_weight)
        ),
        lulls_weight=float(
            yaml_section.get("lulls_weight", defaults.lulls_weight)
        ),
        lull_hours_weight=float(
            yaml_section.get("lull_hours_weight", defaults.lull_hours_weight)
        ),
    )


def _build_wind_conversion(yaml_section: dict, ns) -> WindConversionConfig:
    defaults = WindConversionConfig()
    # CLI flag is an unconditional off-switch; otherwise honor YAML.
    if ns.no_wind_height_correction:
        enabled = False
    else:
        enabled = bool(yaml_section.get("enabled", defaults.enabled))
    return WindConversionConfig(
        enabled=enabled,
        surface_roughness_m=float(
            yaml_section.get("surface_roughness_m", defaults.surface_roughness_m)
        ),
    )


def _build_bi_section(yaml_section: dict) -> BISection:
    defaults = BISection()
    return BISection(
        min_area_frac=float(yaml_section.get("min_area_frac", defaults.min_area_frac)),
        slope_class=yaml_section.get("slope_class"),
        lat_override=yaml_section.get("lat_override"),
        reg_obs_hr=int(yaml_section.get("reg_obs_hr", defaults.reg_obs_hr)),
        cloud_scale=str(yaml_section.get("cloud_scale", defaults.cloud_scale)),
        snow_mode=str(yaml_section.get("snow_mode", defaults.snow_mode)),
        avg_ann_precip_in=yaml_section.get("avg_ann_precip_in"),
    )


def config_from_namespace(ns: argparse.Namespace) -> Config:
    """Build a :class:`Config` from parsed args + optional YAML."""
    yaml_cfg: dict = {}
    if ns.config:
        if not os.path.exists(ns.config):
            raise SystemExit(f"--config: file not found: {ns.config}")
        yaml_cfg = _load_yaml(ns.config)

    def y(key, default=None):
        return yaml_cfg.get(key, default)

    landscape_tif = _coalesce(ns.landscape_tif, y("landscape_tif"), None)
    year = _coalesce(ns.year, y("year"), None)
    start_m = _coalesce(ns.fire_season_start_month, y("fire_season_start_month"), None)
    end_m = _coalesce(ns.fire_season_end_month, y("fire_season_end_month"), None)
    scenario_len = _coalesce(ns.scenario_length_hours, y("scenario_length_hours"), None)
    target_band = _coalesce(ns.bi_target_band, y("bi_target_band"), None)
    output_dir = _coalesce(ns.output_dir, y("output_dir"), None)
    region_tag = _coalesce(ns.region_tag, y("region_tag"), None)
    volatility_class = _coalesce(ns.volatility_class, y("volatility_class"), None)

    missing = [
        name
        for name, val in [
            ("landscape_tif", landscape_tif),
            ("year", year),
            ("fire_season_start_month", start_m),
            ("fire_season_end_month", end_m),
            ("scenario_length_hours", scenario_len),
            ("bi_target_band", target_band),
            ("output_dir", output_dir),
            ("region_tag", region_tag),
            ("volatility_class", volatility_class),
        ]
        if val is None
    ]
    if missing:
        raise SystemExit(
            "Missing required field(s): "
            + ", ".join(missing)
            + " — supply via CLI or --config."
        )

    if isinstance(target_band, list):
        target_band = tuple(target_band)

    cfg = Config(
        landscape_tif=str(landscape_tif),
        year=int(year),
        fire_season_start_month=int(start_m),
        fire_season_end_month=int(end_m),
        scenario_length_hours=int(scenario_len),
        bi_target_band=(float(target_band[0]), float(target_band[1])),
        output_dir=str(output_dir),
        region_tag=str(region_tag),
        volatility_class=str(volatility_class),
        n_candidates=int(_coalesce(ns.n_candidates, y("n_candidates"), 5)),
        cache_dir=str(_coalesce(ns.cache_dir, y("cache_dir"), "./.openmeteo_cache/")),
        conditioning_days=int(
            _coalesce(ns.conditioning_days, y("conditioning_days"), 30)
        ),
        window_stride_hours=int(
            _coalesce(ns.window_stride_hours, y("window_stride_hours"), 1)
        ),
        min_candidate_separation_hours=_coalesce(
            ns.min_candidate_separation_hours,
            y("min_candidate_separation_hours"),
            None,
        ),
        lull=_build_lull(yaml_cfg.get("lull", {}) or {}, ns),
        scoring=_build_scoring(yaml_cfg.get("scoring", {}) or {}),
        wind_conversion=_build_wind_conversion(
            yaml_cfg.get("wind_conversion", {}) or {}, ns
        ),
        bi=_build_bi_section(yaml_cfg.get("bi", {}) or {}),
    )
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s | %(message)s",
    )
    cfg = config_from_namespace(args)
    return run_candidate_search(cfg)


if __name__ == "__main__":      # pragma: no cover
    sys.exit(main())
