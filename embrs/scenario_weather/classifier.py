"""The class metric — run EMBRS and measure the average daily-peak flame length.

This is the single most important component (spec §2); everything tunes to it.

Flame length comes from EMBRS's per-cell steady-state fireline intensity
``I_ss`` (BTU/ft/min; head value = ``np.max(cell.I_ss)``) via EMBRS's own
converter then Byram (1959):

    I_kW_m  = BTU_ft_min_to_kW_m(I_ss_btu)
    flame_m = 0.0775 * I_kW_m ** 0.46
    flame_ft = flame_m * 3.28084

The class metric is robust, NOT max-over-region: for each simulation **day**,
pool the head flame-length samples of all burning cells across all timesteps and
take the **97th percentile** (``daily_peak_flame_ft``); the metric is the
**mean of those daily peaks** over the run's full days (dropping partial days).

Because a multi-day 30 s run produces hundreds of millions of (cell, timestep)
samples, we accumulate each day's distribution in a fixed-bin histogram rather
than storing raw samples — memory- and time-cheap, and exact enough for a
percentile at 0.05 ft resolution.

Reads ``cell.I_ss`` / ``cell.r_t`` from ``fire.burning_cells`` in-process; does
**not** rely on parquet logs or stdout (spec §2, §9).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional

import numpy as np

from embrs.scenario_weather.config import ClassifierConfig, RunConfig
from embrs.scenario_weather.run_config import build_fire, write_cfg
from embrs.utilities.unit_conversions import BTU_ft_min_to_kW_m, m_to_ft

# Precompute scalar conversion factors so the per-step hot path is pure numpy.
_KWM_PER_BTU: float = BTU_ft_min_to_kW_m(1.0)   # BTU/ft/min -> kW/m
_FT_PER_M: float = m_to_ft(1.0)                 # m -> ft
_BYRAM_A: float = 0.0775
_BYRAM_B: float = 0.46

# EMBRS cell states (see embrs/fire_simulator/cell.py :: CellStates).
_STATE_BURNT = 0
_STATE_FIRE = 2


def i_ss_btu_to_flame_ft(i_ss_btu: np.ndarray) -> np.ndarray:
    """Vectorised head fireline-intensity (BTU/ft/min) -> flame length (ft)."""
    i_ss_btu = np.asarray(i_ss_btu, dtype=float)
    i_kw_m = np.maximum(i_ss_btu, 0.0) * _KWM_PER_BTU
    return _FT_PER_M * _BYRAM_A * np.power(i_kw_m, _BYRAM_B)


class _DayHistogram:
    """Streaming histogram of flame-length samples plus a running max ROS."""

    __slots__ = ("_counts", "_bin_ft", "_nbins", "_overflow", "max_ros_m_min")

    def __init__(self, max_ft: float, bin_ft: float):
        self._bin_ft = float(bin_ft)
        self._nbins = int(np.ceil(max_ft / bin_ft))
        self._counts = np.zeros(self._nbins, dtype=np.int64)
        self._overflow = 0  # samples >= max_ft, folded into the top bin
        self.max_ros_m_min = 0.0

    def add(self, flame_ft: np.ndarray, ros_m_min: np.ndarray) -> None:
        if flame_ft.size:
            idx = (flame_ft / self._bin_ft).astype(np.int64)
            over = idx >= self._nbins
            self._overflow += int(over.sum())
            np.clip(idx, 0, self._nbins - 1, out=idx)
            self._counts += np.bincount(idx, minlength=self._nbins)
        if ros_m_min.size:
            m = float(ros_m_min.max())
            if m > self.max_ros_m_min:
                self.max_ros_m_min = m

    @property
    def n_samples(self) -> int:
        return int(self._counts.sum())

    @property
    def overflow(self) -> int:
        return self._overflow

    def percentile(self, q: float) -> float:
        """Linear-interpolated ``q``-th percentile (q in [0, 100])."""
        total = self._counts.sum()
        if total == 0:
            return 0.0
        # Target rank using the same convention as numpy's default ('linear'):
        # position = q/100 * (N - 1) over the sorted samples.
        target = (q / 100.0) * (total - 1)
        cum = np.cumsum(self._counts)
        # First bin whose cumulative count exceeds the target rank.
        b = int(np.searchsorted(cum, target, side="right"))
        b = min(b, self._nbins - 1)
        prev_cum = cum[b - 1] if b > 0 else 0
        # Fraction into bin b toward its samples; interpolate across the bin's
        # width. With fine bins this term is sub-bin (<= bin_ft) so the exact
        # within-bin distribution is immaterial.
        in_bin = self._counts[b]
        frac = 0.0 if in_bin <= 0 else min(1.0, (target - prev_cum + 1) / in_bin)
        return (b + frac) * self._bin_ft


@dataclass
class DayResult:
    """One simulation day's metric (spec §2)."""

    date: str                  # ISO calendar date
    daily_peak_flame_ft: float
    head_ros_m_min: float
    n_samples: int
    eligible: bool             # counted toward the class metric (full day)


@dataclass
class ClassifierReport:
    """Structured classifier output (spec §2)."""

    region: Optional[str]
    fire_class: Optional[str]
    mean_daily_peak_flame_ft: float
    per_day_flame_ft: List[float]            # eligible days only
    mean_head_ros_m_min: float
    n_days: int                              # number of eligible days
    burned_area_m2: float
    seed: int
    per_day: List[DayResult] = field(default_factory=list)
    flame_percentile: float = 97.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def classify(
    wxs_path: str,
    map_dir: str,
    start_datetime: datetime,
    end_datetime: datetime,
    run_cfg: RunConfig,
    *,
    cfg_path: str,
    classifier_cfg: Optional[ClassifierConfig] = None,
    region: Optional[str] = None,
    fire_class: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> ClassifierReport:
    """Run a ``FireSim`` on the real map under ``wxs_path`` and measure the metric.

    Args:
        wxs_path: Candidate weather file.
        map_dir: Real scenario map folder (with its real ignition region).
        start_datetime: Scenario start (used to map sim seconds -> civil days).
        end_datetime: Scenario end.
        run_cfg: Fixed run settings (moisture, seed, spotting, grid).
        cfg_path: Where to write the temporary ``.cfg`` (caller owns the dir so
            artifacts can be inspected/cleaned).
        classifier_cfg: Metric parameters; defaults to :class:`ClassifierConfig`.
        region, fire_class: Labels echoed into the report.
        max_steps: Optional cap on iterations (smoke tests / safety).

    Returns:
        A :class:`ClassifierReport`.
    """
    classifier_cfg = classifier_cfg or ClassifierConfig()

    write_cfg(
        cfg_path,
        map_dir=map_dir,
        wxs_path=wxs_path,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        run_cfg=run_cfg,
    )
    fire = build_fire(cfg_path)

    days: dict[date, _DayHistogram] = {}
    last_time_s = 0
    steps = 0
    while not fire.finished:
        fire.iterate()
        steps += 1
        t_s = int(fire.curr_time_s)
        last_time_s = t_s
        bc = fire.burning_cells
        if bc:
            iss = np.fromiter(
                (float(np.max(c.I_ss)) for c in bc), dtype=float, count=len(bc)
            )
            rt = np.fromiter(
                (float(np.max(c.r_t)) for c in bc), dtype=float, count=len(bc)
            )
            flame_ft = i_ss_btu_to_flame_ft(iss)
            ros_m_min = rt * 60.0
            d = (start_datetime + timedelta(seconds=t_s)).date()
            hist = days.get(d)
            if hist is None:
                hist = _DayHistogram(classifier_cfg.hist_max_ft, classifier_cfg.hist_bin_ft)
                days[d] = hist
            hist.add(flame_ft, ros_m_min)
        if max_steps is not None and steps >= max_steps:
            break

    warnings: List[str] = []
    last_dt = start_datetime + timedelta(seconds=last_time_s)

    per_day: List[DayResult] = []
    for d in sorted(days):
        hist = days[d]
        day_start = datetime(d.year, d.month, d.day)
        day_end = day_start + timedelta(days=1)
        eligible = (not classifier_cfg.drop_partial_days) or (
            day_start >= start_datetime and day_end <= last_dt
        )
        per_day.append(
            DayResult(
                date=d.isoformat(),
                daily_peak_flame_ft=hist.percentile(classifier_cfg.flame_percentile),
                head_ros_m_min=hist.max_ros_m_min,
                n_samples=hist.n_samples,
                eligible=eligible,
            )
        )
        if hist.overflow:
            warnings.append(
                f"{d.isoformat()}: {hist.overflow} flame samples exceeded "
                f"hist_max_ft={classifier_cfg.hist_max_ft} ft (clipped)"
            )

    eligible_days = [r for r in per_day if r.eligible]
    if not eligible_days and per_day:
        warnings.append(
            "no full days after dropping partial days; falling back to all "
            "days with fire (metric may be biased by establishment/end)"
        )
        eligible_days = per_day

    flame_list = [r.daily_peak_flame_ft for r in eligible_days]
    ros_list = [r.head_ros_m_min for r in eligible_days]
    mean_flame = float(np.mean(flame_list)) if flame_list else 0.0
    mean_ros = float(np.mean(ros_list)) if ros_list else 0.0

    burned = sum(
        1 for c in fire.cell_dict.values() if c.state in (_STATE_BURNT, _STATE_FIRE)
    )
    burned_area_m2 = float(burned) * (run_cfg.cell_size_m ** 2)

    return ClassifierReport(
        region=region,
        fire_class=fire_class,
        mean_daily_peak_flame_ft=mean_flame,
        per_day_flame_ft=flame_list,
        mean_head_ros_m_min=mean_ros,
        n_days=len(eligible_days),
        burned_area_m2=burned_area_m2,
        seed=run_cfg.seed,
        per_day=per_day,
        flame_percentile=classifier_cfg.flame_percentile,
        warnings=warnings,
    )
