"""Tuning harness — auto-tune the wind peak to a target flame length (spec §5).

Find the wind ``peak_scale`` such that the classifier's
``mean_daily_peak_flame_ft`` equals the class target (4/6/8 ft) within
tolerance. ``f(peak_scale) = measured_flame_ft(peak_scale) - target`` is
monotonic-increasing in peak wind (modulo the shrub-transition nonlinearity of
spec §8), so we use a **bracketed secant with bisection fallback**: the secant
step is taken when it lands inside the current sign-change bracket, otherwise we
bisect. This converges fast on the smooth part and stays robust across the
nonlinearity.

Each evaluation generates a ``.wxs`` at the candidate ``peak_scale`` and runs the
classifier on the real map; the temp/RH slice, moisture and seeds are held fixed
across iterations so **only wind changes**.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from embrs.scenario_weather.classifier import ClassifierReport, classify
from embrs.scenario_weather.config import (
    ClassifierConfig,
    GeneratorConfig,
    RunConfig,
    TuningConfig,
)
from embrs.scenario_weather.generator import generate_from_window


@dataclass
class TuningIteration:
    iteration: int
    peak_scale_ms: float
    flame_ft: float
    residual_ft: float
    wxs_path: str


@dataclass
class TuningResult:
    """Outcome of a tuning run (spec §5)."""

    target_ft: float
    tolerance_ft: float
    converged: bool
    final_peak_scale_ms: float
    final_flame_ft: float
    final_wxs_path: str
    n_iterations: int
    trace: List[TuningIteration] = field(default_factory=list)
    final_report: Optional[dict] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def tune(
    out_dir: str,
    *,
    full_season_wxs: str,
    backdrop_start: datetime,
    backdrop_end: datetime,
    map_dir: str,
    run_cfg: RunConfig,
    gen_cfg: GeneratorConfig,
    tuning_cfg: TuningConfig,
    classify_start: datetime,
    classifier_cfg: Optional[ClassifierConfig] = None,
    region: Optional[str] = None,
    fire_class: Optional[str] = None,
) -> TuningResult:
    """Tune ``peak_scale`` so the measured flame length hits the target.

    Args:
        out_dir: Directory for per-iteration ``.wxs``/``.cfg`` and the trace.
        full_season_wxs: Region's full-season ``.wxs`` (temp/RH backdrop source).
        backdrop_start, backdrop_end: Real window providing the temp/RH backdrop
            (must cover the classify window).
        map_dir: Real scenario map folder.
        run_cfg: Fixed run settings (moisture, seed, spotting).
        gen_cfg: Generator/wind settings (its ``peak_scale_ms`` is overridden
            each iteration).
        tuning_cfg: Target, tolerance, bracket and ``tuning_days``.
        classify_start: Start of the short classify window; the end is
            ``classify_start + tuning_days``.
        classifier_cfg: Metric parameters.
        region, fire_class: Labels for artifacts.

    Returns:
        A :class:`TuningResult` with the full trace and the tuned file.
    """
    classifier_cfg = classifier_cfg or ClassifierConfig()
    os.makedirs(out_dir, exist_ok=True)
    classify_end = classify_start + timedelta(days=tuning_cfg.tuning_days)

    trace: List[TuningIteration] = []

    def evaluate(peak_scale: float) -> TuningIteration:
        idx = len(trace)
        tag = f"iter{idx:02d}_ps{peak_scale:.3f}"
        wxs = os.path.join(out_dir, f"{tag}.wxs")
        generate_from_window(
            wxs, full_season_wxs, backdrop_start, backdrop_end, gen_cfg,
            peak_scale=peak_scale,
        )
        report: ClassifierReport = classify(
            wxs, map_dir, classify_start, classify_end, run_cfg,
            cfg_path=os.path.join(out_dir, f"{tag}.cfg"),
            classifier_cfg=classifier_cfg, region=region, fire_class=fire_class,
        )
        flame = report.mean_daily_peak_flame_ft
        it = TuningIteration(
            iteration=idx,
            peak_scale_ms=peak_scale,
            flame_ft=flame,
            residual_ft=flame - tuning_cfg.target_ft,
            wxs_path=wxs,
        )
        trace.append(it)
        # The full report for the winning iteration is captured by the caller
        # via re-classify; keep the trace light here.
        it._report = report  # type: ignore[attr-defined]
        return it

    notes: List[str] = []
    lo, hi = tuning_cfg.bracket_lo_ms, tuning_cfg.bracket_hi_ms
    it_lo = evaluate(lo)
    if abs(it_lo.residual_ft) <= tuning_cfg.tolerance_ft:
        return _finalize(tuning_cfg, trace, it_lo, converged=True, notes=notes)
    it_hi = evaluate(hi)
    if abs(it_hi.residual_ft) <= tuning_cfg.tolerance_ft:
        return _finalize(tuning_cfg, trace, it_hi, converged=True, notes=notes)

    if it_lo.residual_ft * it_hi.residual_ft > 0:
        # Target not bracketed: both endpoints on the same side.
        notes.append(
            f"target {tuning_cfg.target_ft} ft not bracketed by peak_scale "
            f"[{lo}, {hi}] m/s (flame {it_lo.flame_ft:.2f}..{it_hi.flame_ft:.2f} "
            f"ft); returning the closest endpoint — widen the bracket"
        )
        best = min(trace, key=lambda t: abs(t.residual_ft))
        return _finalize(tuning_cfg, trace, best, converged=False, notes=notes)

    # Bracketed: [a, b] straddle the root (fa<0<fb or vice-versa). Each step
    # takes a secant through the two most recent evaluations; if it lands
    # outside the current bracket we bisect instead. ``fa`` carries the sign at
    # endpoint ``a`` so the bracket update stays correct across the shrub jump.
    a, fa = it_lo.peak_scale_ms, it_lo.residual_ft
    b, fb = it_hi.peak_scale_ms, it_hi.residual_ft
    x0, f0 = a, fa          # two most recent points for the secant
    x1, f1 = b, fb
    best = min(trace, key=lambda t: abs(t.residual_ft))

    for _ in range(len(trace), tuning_cfg.max_iter):
        if f1 != f0:
            cand = x1 - f1 * (x1 - x0) / (f1 - f0)
        else:
            cand = 0.5 * (a + b)
        # Fall back to bisection if the secant leaves the bracket.
        if not (min(a, b) < cand < max(a, b)):
            cand = 0.5 * (a + b)

        it = evaluate(cand)
        fc = it.residual_ft
        if abs(fc) < abs(best.residual_ft):
            best = it
        if abs(fc) <= tuning_cfg.tolerance_ft:
            return _finalize(tuning_cfg, trace, it, converged=True, notes=notes)

        # Shrink the sign-change bracket, then shift the most-recent pair.
        if fa * fc < 0:
            b, fb = cand, fc
        else:
            a, fa = cand, fc
        x0, f0 = x1, f1
        x1, f1 = cand, fc

    notes.append(
        f"hit max_iter={tuning_cfg.max_iter} without converging to "
        f"±{tuning_cfg.tolerance_ft} ft; returning best (residual "
        f"{best.residual_ft:+.2f} ft)"
    )
    return _finalize(tuning_cfg, trace, best, converged=False, notes=notes)


def _finalize(
    tuning_cfg: TuningConfig,
    trace: List[TuningIteration],
    best: TuningIteration,
    *,
    converged: bool,
    notes: List[str],
) -> TuningResult:
    report = getattr(best, "_report", None)
    result = TuningResult(
        target_ft=tuning_cfg.target_ft,
        tolerance_ft=tuning_cfg.tolerance_ft,
        converged=converged,
        final_peak_scale_ms=best.peak_scale_ms,
        final_flame_ft=best.flame_ft,
        final_wxs_path=best.wxs_path,
        n_iterations=len(trace),
        trace=[TuningIteration(t.iteration, t.peak_scale_ms, t.flame_ft,
                               t.residual_ft, t.wxs_path) for t in trace],
        final_report=report.to_dict() if report is not None else None,
        notes=notes,
    )
    return result
