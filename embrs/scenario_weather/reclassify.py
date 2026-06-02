"""Re-classification workflow (spec §7).

When the firefighting app is applied, the user may hand-edit a ``.wxs`` (tweak a
lull, shift a wind event) to make a scenario tactically viable. After any such
edit the scenario must be re-verified to still fall in its class. This is just
the §2 classifier exposed standalone with a PASS/FAIL against the class target.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from embrs.scenario_weather.classifier import ClassifierReport, classify
from embrs.scenario_weather.config import ClassifierConfig, RunConfig
from embrs.scenario_weather.config import DEFAULT_FLAME_TARGETS_FT


@dataclass
class ReclassifyResult:
    fire_class: str
    target_ft: float
    tolerance_ft: float
    measured_ft: float
    passed: bool
    report: dict


def reclassify(
    wxs_path: str,
    map_dir: str,
    fire_class: str,
    start: datetime,
    end: datetime,
    run_cfg: RunConfig,
    *,
    cfg_path: str,
    target_ft: Optional[float] = None,
    tolerance_ft: float = 0.3,
    classifier_cfg: Optional[ClassifierConfig] = None,
    region: Optional[str] = None,
) -> ReclassifyResult:
    """Re-run the classifier on a (possibly hand-edited) ``.wxs`` and PASS/FAIL.

    Args:
        wxs_path: The weather file to verify.
        map_dir: Real scenario map folder.
        fire_class: ``"mild"``/``"moderate"``/``"extreme"`` (selects the default
            target if ``target_ft`` is not given).
        start, end: Classify window.
        run_cfg: Fixed run settings.
        cfg_path: Where to write the temporary ``.cfg``.
        target_ft: Override the class target flame length (ft).
        tolerance_ft: Acceptance half-band.
        classifier_cfg: Metric parameters.
        region: Label echoed into the report.

    Returns:
        A :class:`ReclassifyResult` with PASS/FAIL.
    """
    if target_ft is None:
        if fire_class not in DEFAULT_FLAME_TARGETS_FT:
            raise ValueError(
                f"unknown fire_class {fire_class!r}; pass target_ft explicitly"
            )
        target_ft = DEFAULT_FLAME_TARGETS_FT[fire_class]

    report: ClassifierReport = classify(
        wxs_path, map_dir, start, end, run_cfg,
        cfg_path=cfg_path, classifier_cfg=classifier_cfg,
        region=region, fire_class=fire_class,
    )
    measured = report.mean_daily_peak_flame_ft
    passed = abs(measured - target_ft) <= tolerance_ft
    return ReclassifyResult(
        fire_class=fire_class,
        target_ft=target_ft,
        tolerance_ft=tolerance_ft,
        measured_ft=measured,
        passed=passed,
        report=report.to_dict(),
    )
