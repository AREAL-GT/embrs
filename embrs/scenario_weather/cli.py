"""Command-line interface for the controlled scenario weather system.

Subcommands (spec §11 deliverables):

    search          temp/RH backdrop period finder (§4.1.1)
    generate        assemble a .wxs from a backdrop window + synthetic wind (§4)
    classify        run the class metric on a .wxs (§2)
    reclassify      re-verify a (hand-edited) .wxs against its class (§7)
    tune            auto-tune the wind peak to a target flame length (§5)
    variance        cross-seed variance of the class metric (§5.1)
    backburn-check  validate backburn windows in a .wxs (§6)
    plot-temprh     temp/RH-profile plot of a backdrop window (§4.7)
    plot-wxs        full multi-panel plot of a .wxs (§4.7)

Each subcommand prints a JSON result to stdout. The heavy, sim-running
subcommands (classify/reclassify/tune/variance) construct ``FireSim`` in-process;
run them from this module (which has the required ``if __name__ == '__main__'``
guard, spec §9) so spotting workers can re-import safely.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import List, Optional

from embrs.scenario_weather.config import (
    BackburnProxyConfig,
    ClassifierConfig,
    DEFAULT_FLAME_TARGETS_FT,
    GeneratorConfig,
    RunConfig,
    SearchConfig,
    TuningConfig,
    WindModelConfig,
)


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _init_mf(s: str):
    return tuple(float(x.strip()) for x in s.split(","))


def _run_cfg(a) -> RunConfig:
    return RunConfig(
        live_herb_mf=a.live_herb,
        live_woody_mf=a.live_woody,
        init_mf=_init_mf(a.init_mf),
        seed=a.seed,
        model_spotting=not a.no_spotting,
        cell_size_m=a.cell_size,
        t_step_s=a.t_step,
    )


def _wind_cfg(a) -> WindModelConfig:
    return WindModelConfig(
        w_min_ms=a.w_min,
        peak_scale_ms=a.peak_scale,
        prevailing_dir_deg=a.prevailing_dir,
        noise_seed=a.noise_seed,
    )


def _add_run_args(p):
    p.add_argument("--live-herb", type=float, default=0.30)
    p.add_argument("--live-woody", type=float, default=0.60)
    p.add_argument("--init-mf", default="0.06,0.07,0.08")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-spotting", action="store_true")
    p.add_argument("--cell-size", type=int, default=30)
    p.add_argument("--t-step", type=int, default=30)


def _add_wind_args(p):
    p.add_argument("--w-min", type=float, default=1.5)
    p.add_argument("--peak-scale", type=float, default=4.0)
    p.add_argument("--prevailing-dir", type=float, default=180.0)
    p.add_argument("--noise-seed", type=int, default=7)


def _emit(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #


def cmd_search(a) -> None:
    from embrs.scenario_weather.period_search import find_windows

    cfg = SearchConfig(
        window_days=a.window_days,
        local_tz=a.local_tz,
        fire_season_months=tuple(a.season_months) if a.season_months else (),
        top_n=a.top_n,
    )
    res = find_windows(a.wxs, cfg)
    _emit({cls: [w.to_dict() for w in ws] for cls, ws in res.items()})


def cmd_generate(a) -> None:
    from embrs.scenario_weather.generator import generate_from_window

    gcfg = GeneratorConfig(wind=_wind_cfg(a), elevation_ft=a.elevation_ft)
    res = generate_from_window(a.out, a.season_wxs, _dt(a.start), _dt(a.end), gcfg,
                               peak_scale=a.peak_scale)
    _emit(res.to_dict())


def cmd_classify(a) -> None:
    from embrs.scenario_weather.classifier import classify

    workdir = a.workdir or tempfile.mkdtemp(prefix="sw_classify_")
    report = classify(
        a.wxs, a.map, _dt(a.start), _dt(a.end), _run_cfg(a),
        cfg_path=os.path.join(workdir, "classify.cfg"),
        classifier_cfg=ClassifierConfig(flame_percentile=a.flame_percentile),
        region=a.region, fire_class=a.fire_class,
    )
    _emit(report.to_dict())


def cmd_reclassify(a) -> None:
    from embrs.scenario_weather.reclassify import reclassify

    workdir = a.workdir or tempfile.mkdtemp(prefix="sw_reclassify_")
    res = reclassify(
        a.wxs, a.map, a.fire_class, _dt(a.start), _dt(a.end), _run_cfg(a),
        cfg_path=os.path.join(workdir, "reclassify.cfg"),
        target_ft=a.target_ft, tolerance_ft=a.tolerance, region=a.region,
    )
    _emit(asdict(res))
    sys.exit(0 if res.passed else 1)


def cmd_tune(a) -> None:
    from embrs.scenario_weather.tuning import tune

    target = a.target_ft if a.target_ft is not None else DEFAULT_FLAME_TARGETS_FT[a.fire_class]
    gcfg = GeneratorConfig(wind=_wind_cfg(a), elevation_ft=a.elevation_ft)
    tcfg = TuningConfig(
        target_ft=target, tolerance_ft=a.tolerance, max_iter=a.max_iter,
        bracket_lo_ms=a.bracket_lo, bracket_hi_ms=a.bracket_hi,
        tuning_days=a.tuning_days,
    )
    res = tune(
        a.out_dir, full_season_wxs=a.season_wxs,
        backdrop_start=_dt(a.backdrop_start), backdrop_end=_dt(a.backdrop_end),
        map_dir=a.map, run_cfg=_run_cfg(a), gen_cfg=gcfg, tuning_cfg=tcfg,
        classify_start=_dt(a.classify_start), region=a.region, fire_class=a.fire_class,
    )
    with open(os.path.join(a.out_dir, "tuning_result.json"), "w") as fh:
        fh.write(res.to_json())
    _emit(res.to_dict())


def cmd_variance(a) -> None:
    from embrs.scenario_weather.variance import characterize

    res = characterize(
        a.wxs, a.map, _dt(a.start), _dt(a.end), _run_cfg(a), list(a.seeds),
        out_dir=a.out_dir, region=a.region, fire_class=a.fire_class,
        target_ft=a.target_ft, tolerance_ft=a.tolerance,
    )
    _emit(res.to_dict())


def cmd_backburn(a) -> None:
    from embrs.scenario_weather.backburn_check import validate

    cfg = BackburnProxyConfig(
        hi_wind_speed_thresh_m_s=a.wind_thresh,
        wind_angle_tol_deg=a.angle_tol,
        fireline_bearing_deg=a.fireline_bearing,
        min_window_hours=a.min_window_hours,
    )
    _emit(validate(a.wxs, cfg).to_dict())


def cmd_plot_temprh(a) -> None:
    from embrs.scenario_weather.plotting import plot_temprh

    out = plot_temprh(a.wxs, a.out, local_tz=a.local_tz, title=a.title)
    _emit({"wrote": out})


def cmd_plot_wxs(a) -> None:
    from embrs.scenario_weather.plotting import plot_wxs

    wcfg = WindModelConfig(prevailing_dir_deg=a.prevailing_dir) if a.show_mean else None
    out = plot_wxs(
        a.wxs, a.out, wind_cfg=wcfg, peak_scale=a.peak_scale,
        backburn_threshold_m_s=a.wind_thresh, local_tz=a.local_tz, title=a.title,
    )
    _emit({"wrote": out})


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m embrs.scenario_weather",
                                description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="temp/RH backdrop period finder")
    s.add_argument("wxs", help="full-season .wxs")
    s.add_argument("--window-days", type=int, default=14)
    s.add_argument("--local-tz", default=None)
    s.add_argument("--season-months", type=int, nargs="*", default=None)
    s.add_argument("--top-n", type=int, default=5)
    s.set_defaults(func=cmd_search)

    g = sub.add_parser("generate", help="backdrop + synthetic wind -> .wxs")
    g.add_argument("--season-wxs", required=True)
    g.add_argument("--start", required=True)
    g.add_argument("--end", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--elevation-ft", type=int, default=None)
    _add_wind_args(g)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("classify", help="class metric on a .wxs")
    c.add_argument("--wxs", required=True)
    c.add_argument("--map", required=True)
    c.add_argument("--start", required=True)
    c.add_argument("--end", required=True)
    c.add_argument("--region", default=None)
    c.add_argument("--fire-class", default=None)
    c.add_argument("--flame-percentile", type=float, default=97.0)
    c.add_argument("--workdir", default=None)
    _add_run_args(c)
    c.set_defaults(func=cmd_classify)

    r = sub.add_parser("reclassify", help="re-verify a .wxs against its class")
    r.add_argument("--wxs", required=True)
    r.add_argument("--map", required=True)
    r.add_argument("--fire-class", required=True, choices=list(DEFAULT_FLAME_TARGETS_FT))
    r.add_argument("--start", required=True)
    r.add_argument("--end", required=True)
    r.add_argument("--target-ft", type=float, default=None)
    r.add_argument("--tolerance", type=float, default=0.3)
    r.add_argument("--region", default=None)
    r.add_argument("--workdir", default=None)
    _add_run_args(r)
    r.set_defaults(func=cmd_reclassify)

    t = sub.add_parser("tune", help="auto-tune wind peak to a target flame length")
    t.add_argument("--season-wxs", required=True)
    t.add_argument("--backdrop-start", required=True)
    t.add_argument("--backdrop-end", required=True)
    t.add_argument("--classify-start", required=True)
    t.add_argument("--map", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--fire-class", required=True, choices=list(DEFAULT_FLAME_TARGETS_FT))
    t.add_argument("--target-ft", type=float, default=None)
    t.add_argument("--tolerance", type=float, default=0.3)
    t.add_argument("--max-iter", type=int, default=12)
    t.add_argument("--bracket-lo", type=float, default=1.0)
    t.add_argument("--bracket-hi", type=float, default=12.0)
    t.add_argument("--tuning-days", type=int, default=6)
    t.add_argument("--region", default=None)
    t.add_argument("--elevation-ft", type=int, default=None)
    _add_wind_args(t)
    _add_run_args(t)
    t.set_defaults(func=cmd_tune)

    v = sub.add_parser("variance", help="cross-seed variance of the class metric")
    v.add_argument("--wxs", required=True)
    v.add_argument("--map", required=True)
    v.add_argument("--start", required=True)
    v.add_argument("--end", required=True)
    v.add_argument("--seeds", type=int, nargs="+", required=True)
    v.add_argument("--out-dir", required=True)
    v.add_argument("--region", default=None)
    v.add_argument("--fire-class", default=None)
    v.add_argument("--target-ft", type=float, default=None)
    v.add_argument("--tolerance", type=float, default=0.3)
    _add_run_args(v)
    v.set_defaults(func=cmd_variance)

    b = sub.add_parser("backburn-check", help="validate backburn windows in a .wxs")
    b.add_argument("--wxs", required=True)
    b.add_argument("--fireline-bearing", type=float, required=True,
                   help="outward-normal bearing (met deg) the backburn is pushed")
    b.add_argument("--wind-thresh", type=float, default=10.0)
    b.add_argument("--angle-tol", type=float, default=45.0)
    b.add_argument("--min-window-hours", type=float, default=2.0)
    b.set_defaults(func=cmd_backburn)

    pt = sub.add_parser("plot-temprh", help="temp/RH-profile plot of a backdrop window")
    pt.add_argument("--wxs", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--local-tz", default=None)
    pt.add_argument("--title", default=None)
    pt.set_defaults(func=cmd_plot_temprh)

    pw = sub.add_parser("plot-wxs", help="full multi-panel plot of a .wxs")
    pw.add_argument("--wxs", required=True)
    pw.add_argument("--out", required=True)
    pw.add_argument("--show-mean", action="store_true")
    pw.add_argument("--peak-scale", type=float, default=None)
    pw.add_argument("--prevailing-dir", type=float, default=180.0)
    pw.add_argument("--wind-thresh", type=float, default=10.0)
    pw.add_argument("--local-tz", default=None)
    pw.add_argument("--title", default=None)
    pw.set_defaults(func=cmd_plot_wxs)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
