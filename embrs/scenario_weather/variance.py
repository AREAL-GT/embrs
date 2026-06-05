"""Cross-seed variance characterizer (spec §5.1).

After a scenario is tuned, characterise the stochastic spread of its class
metric. Re-run the classifier (§2) on the tuned ``.wxs`` + real map across a set
of EMBRS **spotting** seeds, holding the weather-noise realisation fixed (it is
baked into the generated ``.wxs``), so only spotting stochasticity is probed.

Spotting is the dominant stochastic element, so expect modest spread; the point
is to **quantify and report** it, not to re-tune.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import List, Optional

import numpy as np

from embrs.scenario_weather.classifier import classify
from embrs.scenario_weather.config import ClassifierConfig, RunConfig


@dataclass
class SeedResult:
    seed: int
    mean_daily_peak_flame_ft: float
    mean_head_ros_m_min: float
    n_days: int


@dataclass
class VarianceReport:
    """Cross-seed spread of the class metric (spec §5.1)."""

    seeds: List[int]
    per_seed: List[SeedResult]
    mean_ft: float
    std_ft: float
    min_ft: float
    max_ft: float
    target_ft: Optional[float] = None
    tolerance_ft: Optional[float] = None
    all_in_band: Optional[bool] = None
    out_of_band_seeds: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def characterize(
    wxs_path: str,
    map_dir: str,
    start: datetime,
    end: datetime,
    base_run_cfg: RunConfig,
    seeds: List[int],
    *,
    out_dir: str,
    classifier_cfg: Optional[ClassifierConfig] = None,
    region: Optional[str] = None,
    fire_class: Optional[str] = None,
    target_ft: Optional[float] = None,
    tolerance_ft: Optional[float] = None,
) -> VarianceReport:
    """Run the classifier across spotting seeds and summarise the spread.

    Args:
        wxs_path: The tuned ``.wxs``.
        map_dir: Real scenario map folder.
        start, end: Classify window.
        base_run_cfg: Run settings; ``seed`` is overridden per run.
        seeds: EMBRS spotting seeds to sweep.
        out_dir: Directory for per-seed ``.cfg`` files.
        classifier_cfg: Metric parameters.
        region, fire_class: Labels echoed into the report.
        target_ft, tolerance_ft: If given, flag seeds that leave the band.

    Returns:
        A :class:`VarianceReport`.
    """
    classifier_cfg = classifier_cfg or ClassifierConfig()
    os.makedirs(out_dir, exist_ok=True)

    per_seed: List[SeedResult] = []
    for seed in seeds:
        run_cfg = replace(base_run_cfg, seed=seed)
        report = classify(
            wxs_path, map_dir, start, end, run_cfg,
            cfg_path=os.path.join(out_dir, f"seed{seed}.cfg"),
            classifier_cfg=classifier_cfg, region=region, fire_class=fire_class,
        )
        per_seed.append(
            SeedResult(
                seed=seed,
                mean_daily_peak_flame_ft=report.mean_daily_peak_flame_ft,
                mean_head_ros_m_min=report.mean_head_ros_m_min,
                n_days=report.n_days,
            )
        )

    flame = np.array([r.mean_daily_peak_flame_ft for r in per_seed], dtype=float)
    all_in_band: Optional[bool] = None
    out_of_band: List[int] = []
    if target_ft is not None and tolerance_ft is not None:
        out_of_band = [
            r.seed for r in per_seed
            if abs(r.mean_daily_peak_flame_ft - target_ft) > tolerance_ft
        ]
        all_in_band = len(out_of_band) == 0

    return VarianceReport(
        seeds=list(seeds),
        per_seed=per_seed,
        mean_ft=float(flame.mean()) if flame.size else 0.0,
        std_ft=float(flame.std(ddof=0)) if flame.size else 0.0,
        min_ft=float(flame.min()) if flame.size else 0.0,
        max_ft=float(flame.max()) if flame.size else 0.0,
        target_ft=target_ft,
        tolerance_ft=tolerance_ft,
        all_in_band=all_in_band,
        out_of_band_seeds=out_of_band,
    )
