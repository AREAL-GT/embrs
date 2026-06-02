"""Wetness guard — drop candidate windows whose fuels are too wet to carry fire.

Two precipitation checks per window (see :class:`config.WetnessGuard`):

- antecedent precip over the N days before the window start (end-of-conditioning
  dryness), and
- the wettest single in-window calendar day (no soaking mid-scenario day).

Both are wind-independent and read the ``rain_mm_hr`` weather column directly,
so the guard does not depend on the BI pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd

from embrs.weather_candidate_search.config import WetnessGuard

logger = logging.getLogger(__name__)

_MM_TO_IN: float = 0.0393701
_RAIN_COL: str = "rain_mm_hr"


@dataclass
class WetnessResult:
    """Per-window wetness diagnostics and the pass/fail verdict."""

    window_id: str
    antecedent_precip_in: float
    max_daily_precip_in: float
    passed: bool
    reason: str          # "" if passed, else why it failed


def evaluate_wetness(
    weather_df: pd.DataFrame,
    windows: Iterable["object"],
    guard: WetnessGuard,
) -> Dict[str, WetnessResult]:
    """Evaluate the wetness guard for each window.

    Args:
        weather_df: Full hourly weather frame (tz-aware index) containing a
            ``rain_mm_hr`` column — used to look up antecedent precip before
            each window's start.
        windows: Iterable of windows exposing ``window_id``, ``start``,
            ``end`` (and ``df`` for the in-window slice).
        guard: Threshold configuration.

    Returns:
        ``{window_id: WetnessResult}``. If ``guard.enabled`` is False every
        window passes (diagnostics still computed).
    """
    if _RAIN_COL not in weather_df.columns:
        raise ValueError(
            f"evaluate_wetness: weather_df missing required column {_RAIN_COL!r}"
        )
    rain_in = weather_df[_RAIN_COL] * _MM_TO_IN
    out: Dict[str, WetnessResult] = {}
    for w in windows:
        ante_start = w.start - pd.Timedelta(days=guard.antecedent_days)
        antecedent = float(
            rain_in.loc[(rain_in.index >= ante_start) & (rain_in.index < w.start)].sum()
        )
        in_window = rain_in.loc[(rain_in.index >= w.start) & (rain_in.index < w.end)]
        if in_window.empty:
            max_daily = 0.0
        else:
            max_daily = float(in_window.groupby(in_window.index.normalize()).sum().max())

        reason = ""
        if guard.enabled:
            if antecedent > guard.max_antecedent_precip_in:
                reason = (
                    f"antecedent {antecedent:.2f}in > "
                    f"{guard.max_antecedent_precip_in:.2f}in"
                )
            elif max_daily > guard.max_daily_precip_in:
                reason = (
                    f"max daily {max_daily:.2f}in > "
                    f"{guard.max_daily_precip_in:.2f}in"
                )
        out[w.window_id] = WetnessResult(
            window_id=w.window_id,
            antecedent_precip_in=antecedent,
            max_daily_precip_in=max_daily,
            passed=(reason == ""),
            reason=reason,
        )
    return out
