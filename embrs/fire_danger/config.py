"""Configuration and shared data types for the BI trajectory tool.

Every inter-component data structure lives here so the unit contract has one
authoritative location. Module-level docstrings on each dataclass enumerate
column units explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Inter-component data structures
# ---------------------------------------------------------------------------


@dataclass
class HourlyWeather:
    """Hourly weather table indexed by tz-aware (or tz-naive) local datetime.

    The ``df`` is a :class:`pandas.DataFrame` whose columns follow this
    contract (all columns are present after :func:`solar.synthesize_solar`
    and :func:`trajectory._add_phase_and_snow`):

    =================  =========================  ============================
    column             unit                       source
    =================  =========================  ============================
    ``temp_F``         degrees Fahrenheit         raw ``.wxs``
    ``temp_C``         degrees Celsius            ``F_to_C(temp_F)``
    ``rh_pct``         percent (0-100)            raw ``.wxs``
    ``rh_frac``        fraction (0-1)             ``rh_pct / 100``
    ``wind_mph``       miles per hour             raw ``.wxs`` (kept as-is)
    ``wind_dir_deg``   degrees                    raw ``.wxs``
    ``precip_in_hr``   inches, per-hour amount    raw ``.wxs`` ``HrlyPcp``
    ``precip_cm_hr``   centimetres, per-hour      ``precip_in_hr * 2.54``
    ``cloud_cover``    percent (0-100), per .wxs  raw ``.wxs``
    ``solar_wm2``      watts / m^2                ``solar.synthesize_solar``
    ``snow``           bool                       ``trajectory`` (OQ-3)
    ``phase``          ``"conditioning"``/``"scenario"`` (OQ-12)
    =================  =========================  ============================
    """

    df: pd.DataFrame
    ref_elev_m: float
    time_step_min: int
    raw_start: datetime
    raw_end: datetime


@dataclass
class DeadMoistureHourly:
    """Hourly dead fuel moisture series.

    DataFrame indexed by the weather datetime; columns ``MC1``, ``MC10``,
    ``MC100``, ``MC1000`` are all in **percent moisture content**.
    """

    df: pd.DataFrame


@dataclass
class LiveMoistureDaily:
    """Daily live fuel moisture series.

    DataFrame indexed by ``date`` (the calendar day boundary used by
    :class:`embrs.models.weather.GSITracker`); columns:

    - ``GSI``: float in ``[0, 1]`` or ``NaN`` when fewer than two days have
      been buffered.
    - ``MCHERB``: live herbaceous fuel moisture, **percent**.
    - ``MCWOOD``: live woody fuel moisture, **percent**.
    """

    df: pd.DataFrame


@dataclass
class KBDIDaily:
    """Daily KBDI drought-index series.

    DataFrame indexed by ``date`` (the trailing-24-h window ends at
    ``reg_obs_hr``); column ``KBDI`` is a float in ``[0, 800]`` (no rounding,
    per scope §6.2 delta D4).
    """

    df: pd.DataFrame


@dataclass
class FuelComposition:
    """Area composition of a landscape over NFDRS V/W/X/Y/Z fuel models.

    ``fractions`` maps the NFDRS model character (``'V'`` ... ``'Z'``) to the
    fraction of the burnable area assigned to it. Values sum to ``1.0`` over
    burnable pixels (non-burnable codes ``{91, 92, 93, 98, 99}`` and the
    ``-9999`` NoData sentinel are excluded from the denominator, per
    OQ-9 / scope §10).
    """

    fractions: dict[str, float]
    slope_class: int           # 1..5
    fbfm_type: str             # 'ScottBurgan' | 'Anderson'
    n_burnable_pixels: int
    n_total_pixels: int
    pixel_area_m2: float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Run parameters for :func:`compute_bi_trajectory`.

    ``landscape_path`` accepts any LANDFIRE ``.tif`` or ``.lcp`` — full tile
    or pre-cropped. The BI tool does no clipping of its own; the user passes
    a raster at whatever extent they want.

    ``scenario_start`` is the boundary between the conditioning period and
    the scenario period (OQ-12). It must lie within the ``.wxs`` timespan.

    ``avg_ann_precip_in`` is optional: when omitted, the trajectory
    orchestrator fetches the 30-year-normal annual precip from the
    Open-Meteo Archive API at the landscape centroid (closes OQ-13).

    See plan §2.10 for full field semantics.
    """

    landscape_path: str
    wxs_path: str
    scenario_start: datetime
    out_csv: str = "bi_trajectory.csv"
    out_plot: Optional[str] = None
    avg_ann_precip_in: Optional[float] = None
    slope_class: Optional[int] = None
    lat_override: Optional[float] = None
    min_area_frac: float = 0.05
    reg_obs_hr: int = 13
    cloud_scale: str = "percent"
    snow_mode: str = "none"


# Hard-coded fallback annual precip (inches) used only when both the explicit
# CLI value is absent and the Open-Meteo fetch fails. Documented in §2.10.
DEFAULT_AVG_ANN_PRECIP_IN: float = 30.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryResult:
    """Output of :func:`compute_bi_trajectory`.

    ``df`` is the hourly trajectory DataFrame with the columns described in
    plan §2.8 (``phase``, ``BI_<model>``, ``SC_<model>``, ``ERC_<model>``,
    ``BI_area_weighted``, plus debug ``MC1``, ``MC10``, ``MC100``, ``MC1000``,
    ``MCHERB``, ``MCWOOD``, ``GSI``, ``KBDI``).

    ``peak_bi`` is the 97th percentile of the scenario-period
    ``BI_area_weighted`` (OQ-15).
    """

    df: pd.DataFrame
    peak_bi: float
    fuel_composition: FuelComposition
    config: Config
    metadata: dict[str, Any] = field(default_factory=dict)
