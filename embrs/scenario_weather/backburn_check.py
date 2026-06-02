"""Lightweight, self-contained backburn-feasibility proxy (spec §6).

After tuning, confirm the scenario actually affords backburn windows for the
firefighting app — don't assume. Rather than wiring up the firefighting
``TimeWindowFinder`` (which needs WeatherStream scenarios, a containment
geometry and a prediction manager), this proxy replicates its core suitability
test against the generated ``.wxs`` directly:

    suitable(hour) = wind_speed <= hi_wind_speed_thresh_m_s
                     AND angle(wind, fireline_outward_normal) <= wind_angle_tol_deg

The wind-direction math mirrors
``ra-cbba-core/applications/firefighting/backburn/time_windows.py ::
backburn_wind_check`` (wind should blow from the interior toward the segment,
i.e. align with the segment's *outward* normal). Thresholds default to the
firefighting ``BackburnTaskCfg`` values (copied, not imported).

The ``fireline_bearing_deg`` the user provides is the bearing of the outward
normal — the direction the backburn should be carried — in the same
meteorological "degrees FROM which the wind blows" convention as the ``.wxs``
``WindDir`` column, so a wind blowing along that bearing is perfectly aligned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from embrs.fire_danger.weather_loader import load_wxs
from embrs.scenario_weather.config import BackburnProxyConfig

_MPH_TO_MPS: float = 1.0 / 2.23693629


@dataclass
class BackburnWindow:
    start: str          # ISO datetime of first suitable hour
    end: str            # ISO datetime of last suitable hour
    duration_hours: float


@dataclass
class BackburnReport:
    """Backburn-feasibility diagnostics for a ``.wxs`` (spec §6)."""

    n_hours: int
    suitable_hours: int
    suitable_fraction: float
    wind_under_thresh_fraction: float   # ignoring direction (pure lull coverage)
    directional_coverage: float         # of under-threshold hours, fraction aligned
    n_windows: int
    windows: List[BackburnWindow] = field(default_factory=list)
    flagged: bool = False               # too few/short windows
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _angular_distance_deg(a: np.ndarray, b: float) -> np.ndarray:
    """Smallest absolute angle (deg) between bearings ``a`` and bearing ``b``."""
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def validate(
    wxs_path: str,
    cfg: Optional[BackburnProxyConfig] = None,
) -> BackburnReport:
    """Validate backburn feasibility of a ``.wxs`` against a fireline bearing.

    Args:
        wxs_path: The (tuned or hand-edited) weather file.
        cfg: Proxy thresholds + the representative fireline outward-normal
            bearing; defaults to :class:`BackburnProxyConfig`.

    Returns:
        A :class:`BackburnReport`.
    """
    cfg = cfg or BackburnProxyConfig()
    df = load_wxs(wxs_path).df
    wind_mps = df["wind_mph"].to_numpy() * _MPH_TO_MPS
    wind_dir = df["wind_dir_deg"].to_numpy()
    index = df.index

    under_thresh = wind_mps <= cfg.hi_wind_speed_thresh_m_s
    aligned = _angular_distance_deg(wind_dir, cfg.fireline_bearing_deg) <= cfg.wind_angle_tol_deg
    suitable = under_thresh & aligned

    n = len(df)
    n_under = int(under_thresh.sum())
    directional_coverage = (
        float((under_thresh & aligned).sum() / n_under) if n_under else 0.0
    )

    # Contiguous suitable runs -> windows. Assumes hourly rows (the loader
    # enforces a 60-min step), so run length in steps == hours.
    windows: List[BackburnWindow] = []
    i = 0
    while i < n:
        if not suitable[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and suitable[j + 1]:
            j += 1
        start = index[i]
        end = index[j]
        duration = (end - start).total_seconds() / 3600.0 + 1.0  # inclusive hours
        windows.append(
            BackburnWindow(start=start.isoformat(), end=end.isoformat(),
                           duration_hours=float(duration))
        )
        i = j + 1

    usable = [w for w in windows if w.duration_hours >= cfg.min_window_hours]
    notes: List[str] = []
    flagged = False
    if not usable:
        flagged = True
        notes.append(
            f"no backburn window >= {cfg.min_window_hours} h "
            f"(found {len(windows)} shorter runs)"
        )
    if n_under == 0:
        notes.append("wind never drops below threshold — no lulls at all")
    if directional_coverage == 0.0 and n_under:
        notes.append(
            "lulls exist but wind never aligns with the fireline bearing — "
            "consider a different prevailing_dir_deg or fireline geometry"
        )

    return BackburnReport(
        n_hours=n,
        suitable_hours=int(suitable.sum()),
        suitable_fraction=float(suitable.mean()) if n else 0.0,
        wind_under_thresh_fraction=float(under_thresh.mean()) if n else 0.0,
        directional_coverage=directional_coverage,
        n_windows=len(usable),
        windows=usable,
        flagged=flagged,
        notes=notes,
    )
