"""Synthetic wind: two-sine diurnal mean + OU noise + OU direction drift.

Speed (spec §4.2-4.3): the diurnal **mean** ``W_mean(t)`` is an afternoon-peaked,
asymmetric two-sine shape in the family of Ephrath, Goudriaan & Marani (1996),
*"Modelling Diurnal Patterns of Air Temperature, Radiation, Wind Speed and
Relative Humidity by Equations from Daily Characteristics,"* Agricultural
Systems 51(4):377-393 (PDF: ``modelling_diurnal_patterns.pdf``). The mean is a
night floor ``W_min`` plus a peak amplitude scaled by ``peak_scale`` (the single
tuning degree of freedom), with the peak placed in the afternoon. A
mean-reverting AR(1)/Ornstein-Uhlenbeck perturbation is layered on top.

Direction (spec §4.4) does not cycle diurnally; it drifts as a mean-reverting
random walk around an adjustable prevailing bearing.

All wind is produced at the **20-ft reference height in m/s** (spec §4.6) — the
generator converts to mph for the ``.wxs`` and applies **no** log-profile
correction.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from embrs.scenario_weather.config import WindModelConfig


def diurnal_mean_ms(
    hour_of_day: np.ndarray, cfg: WindModelConfig, peak_scale: Optional[float] = None
) -> np.ndarray:
    """Two-sine afternoon-peaked diurnal **mean** wind speed (m/s).

    The daytime envelope spans ``[rise_start_hr, rise_start_hr + daytime_span_hr]``
    with its peak at ``rise_start_hr + peak_frac * daytime_span_hr``. The rise
    and fall are quarter-sine arcs (two sine pieces, generally asymmetric); the
    night floor is ``w_min_ms``.

    Args:
        hour_of_day: Local hour(s) in [0, 24) (fractional allowed).
        cfg: Wind model parameters.
        peak_scale: Override for ``cfg.peak_scale_ms`` (the tuning knob).

    Returns:
        ``W_mean`` at each hour (m/s), shape matching ``hour_of_day``.
    """
    h = np.asarray(hour_of_day, dtype=float)
    amp = cfg.peak_scale_ms if peak_scale is None else float(peak_scale)
    ts = cfg.rise_start_hr
    span = cfg.daytime_span_hr
    t_peak = ts + cfg.peak_frac * span
    t_end = ts + span

    shape = np.zeros_like(h)
    rising = (h >= ts) & (h <= t_peak)
    falling = (h > t_peak) & (h <= t_end)
    # Rise: quarter sine 0 -> 1 over [ts, t_peak].
    shape[rising] = np.sin(0.5 * np.pi * (h[rising] - ts) / (t_peak - ts))
    # Fall: quarter sine 1 -> 0 over [t_peak, t_end].
    shape[falling] = np.sin(
        0.5 * np.pi * (1.0 + (h[falling] - t_peak) / (t_end - t_peak))
    )
    return cfg.w_min_ms + amp * shape


def _ou_series(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1)/OU perturbation ``e[k] = phi*e[k-1] + sigma*z[k]`` (e[0] = 0)."""
    e = np.zeros(n)
    if sigma <= 0:
        return e
    z = rng.standard_normal(n)
    for k in range(1, n):
        e[k] = phi * e[k - 1] + sigma * z[k]
    return e


def _ou_direction(
    n: int, prevailing: float, reversion: float, sigma_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mean-reverting random walk of bearing around ``prevailing`` (deg, wrapped).

    ``d[k] = d[k-1] + reversion*(prevailing - d[k-1]) + sigma*z[k]`` with the
    reversion computed on the *signed shortest angular* difference so the walk
    pulls back across the 0/360 wrap correctly.
    """
    d = np.empty(n)
    d[0] = prevailing % 360.0
    if sigma_deg <= 0 and reversion <= 0:
        d[:] = prevailing % 360.0
        return d
    z = rng.standard_normal(n)
    for k in range(1, n):
        # Signed shortest difference from current bearing to prevailing.
        diff = ((prevailing - d[k - 1] + 180.0) % 360.0) - 180.0
        d[k] = (d[k - 1] + reversion * diff + sigma_deg * z[k]) % 360.0
    return d


def generate_wind(
    index: pd.DatetimeIndex,
    cfg: WindModelConfig,
    peak_scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 20-ft wind speed (m/s) and direction (deg) over ``index``.

    Speed and direction draw from **independent** RNG substreams derived from
    ``cfg.noise_seed`` so changing one process does not perturb the other
    (spec §4.4). The realisation is fully determined by ``noise_seed``.

    Args:
        index: Hourly (or finer) tz-aware/naive datetimes for the window.
        cfg: Wind model parameters.
        peak_scale: Override for the tuning knob.

    Returns:
        ``(wind_ms, wind_dir_deg)`` arrays aligned with ``index``.
    """
    n = len(index)
    hod = index.hour + index.minute / 60.0
    w_mean = diurnal_mean_ms(np.asarray(hod, dtype=float), cfg, peak_scale=peak_scale)

    speed_ss, dir_ss = np.random.SeedSequence(cfg.noise_seed).spawn(2)
    speed_rng = np.random.default_rng(speed_ss)
    dir_rng = np.random.default_rng(dir_ss)

    e = _ou_series(n, cfg.ou_phi, cfg.ou_sigma_ms, speed_rng)
    wind_ms = np.maximum(0.0, w_mean + e)
    wind_dir = _ou_direction(
        n, cfg.prevailing_dir_deg, cfg.dir_reversion, cfg.dir_sigma_deg, dir_rng
    )
    return wind_ms, wind_dir
