"""Configuration dataclasses for the controlled scenario weather system.

These are deliberately plain dataclasses with sensible defaults so they can be
constructed in code, from a CLI, or from a small JSON/dict per region. Every
value the spec calls a "lever" or "parameter" lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Default per-class percentile bands over the region's own window-severity
# distribution (spec §4.1.1). These pick the *backdrop* only; the class is still
# defined by the measured flame length (§2), so percentile selection here is
# benign (unlike the abandoned BI-percentile class definition).
DEFAULT_CLASS_BANDS: Dict[str, Tuple[float, float]] = {
    "mild": (40.0, 60.0),
    "moderate": (70.0, 85.0),
    "extreme": (90.0, 98.0),
}

# Default per-class target average daily-peak flame length, feet (spec §0).
DEFAULT_FLAME_TARGETS_FT: Dict[str, float] = {
    "mild": 4.0,
    "moderate": 6.0,
    "extreme": 8.0,
}


@dataclass
class RunConfig:
    """Fixed EMBRS run settings shared by the classifier and scenario runs.

    Mirrors the proven ``.cfg`` from ``tuning/flame_sweep/sweep_one.py`` (spec
    §3). Spotting is the only stochastic element and is pinned via ``seed``.
    Fuel state is set directly with no conditioning period: ``use_gsi=False``
    with explicit live moisture (the persistent lever), plus an initial dead
    moisture triplet that evolves toward equilibrium with the real RH.

    Attributes:
        live_herb_mf: Live herbaceous moisture fraction (e.g. ``0.30`` for
            cured grass). Values > 1 are treated as percent by the cfg loader.
        live_woody_mf: Live woody moisture fraction.
        init_mf: Initial dead fuel moisture ``[1hr, 10hr, 100hr]`` (fractions).
        seed: Master/​spotting seed, pinned for reproducibility.
        model_spotting: Spotting on (``True``) per spec §3.
        cell_size_m: Hex cell side length (m). Matches the real scenarios.
        t_step_s: Simulation time step (s).
        solar_source: ``"offline"`` — irradiance synthesised from cloud, no
            network.
        mesh_resolution: WindNinja mesh resolution (m).
    """

    live_herb_mf: float
    live_woody_mf: float
    init_mf: Tuple[float, float, float] = (0.06, 0.07, 0.08)
    seed: int = 42
    model_spotting: bool = True
    cell_size_m: int = 30
    t_step_s: int = 30
    solar_source: str = "offline"
    mesh_resolution: int = 250


@dataclass
class ClassifierConfig:
    """Parameters for the class metric (spec §2).

    Attributes:
        flame_percentile: Per-day percentile of the pooled head flame-length
            samples (default 97, consistent with EMBRS's ``peak_bi``).
        hist_max_ft: Upper bound of the streaming flame-length histogram. Head
            flame length effectively never exceeds this; samples above are
            clipped into the top bin (and counted as an overflow diagnostic).
        hist_bin_ft: Histogram bin width (ft). 0.05 ft over [0, hist_max_ft]
            gives sub-tenth-foot percentile resolution at negligible memory.
        drop_partial_days: Drop days not fully contained in the simulated time
            span (the first day when the run starts mid-day, and the final day
            when it ends mid-day). Spec §2.
    """

    flame_percentile: float = 97.0
    hist_max_ft: float = 40.0
    hist_bin_ft: float = 0.05
    drop_partial_days: bool = True


@dataclass
class SearchConfig:
    """Parameters for the temp/RH backdrop period search (spec §4.1.1).

    A *backdrop* selector only — it does not define the class. It finds, per
    class, a ``window_days``-long real window whose temperature/RH severity fits
    that class, keeping the search cheap (pure pandas, no simulation).

    Attributes:
        window_days: Window length in calendar days (default 14).
        stride_days: Day stride between candidate placements.
        severity_metric: ``"vpd"`` (mean of daily-peak vapour-pressure deficit,
            default) or ``"temp_rh"`` (severity from mean daily-max temp and
            mean daily-min RH).
        fire_season_months: Months (1-12) inside the region's fire season; a
            window touching any other month is rejected. Empty/None = no
            season guard.
        max_total_precip_in: Reject a window whose total precip exceeds this
            (a soaking window is not a fire scenario).
        max_peak_hourly_precip_in: Reject a window with any hour wetter than
            this.
        class_bands: Per-class percentile band over the region's own
            window-severity distribution.
        absolute_bands: Optional override: per-class absolute severity
            ``(lo, hi)`` ranges (in the metric's units). Takes precedence over
            ``class_bands`` when provided.
        local_tz: IANA zone (e.g. ``"America/Chicago"``) used only to reject
            windows that span a DST transition (the ``.wxs`` localisation
            hazard, spec §9). None disables the DST guard.
        top_n: Number of ranked windows to return per class.
    """

    window_days: int = 14
    stride_days: int = 1
    severity_metric: str = "vpd"
    fire_season_months: Tuple[int, ...] = ()
    max_total_precip_in: float = 1.0
    max_peak_hourly_precip_in: float = 0.25
    class_bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_BANDS)
    )
    absolute_bands: Optional[Dict[str, Tuple[float, float]]] = None
    local_tz: Optional[str] = None
    top_n: int = 5


@dataclass
class WindModelConfig:
    """Synthetic wind profile (spec §4.2-4.5).

    Speed = a two-sine afternoon-peaked diurnal **mean** (Ephrath, Goudriaan &
    Marani 1996) plus a mean-reverting AR(1)/OU perturbation. Direction = an
    OU random walk around an adjustable prevailing bearing. Wind is authored
    directly at the 20-ft reference height in m/s (spec §4.6) — no log-profile
    correction is applied downstream.

    The **peak amplitude** is the single tuning degree of freedom the harness
    adjusts; ``W_max = w_min_ms + peak_scale_ms``.

    Attributes:
        w_min_ms: Night wind floor (m/s). Kept comfortably below the backburn
            wind threshold so night lulls are clean backburn windows.
        peak_scale_ms: Peak wind above the floor (m/s) — the tuning knob.
        rise_start_hr: Local hour the daytime rise begins.
        daytime_span_hr: Duration (h) of the daytime rise+fall envelope.
        peak_frac: Fraction of ``daytime_span_hr`` at which the peak occurs
            (>0.5 = longer rise, peak later) — placed to align with the
            afternoon RH minimum.
        ou_phi: AR(1) step-to-step correlation of the speed perturbation.
        ou_sigma_ms: Per-step volatility of the speed perturbation (m/s).
        prevailing_dir_deg: Prevailing wind bearing (meteorological, deg FROM).
        dir_reversion: Mean-reversion rate of the direction walk (per step).
        dir_sigma_deg: Per-step direction volatility (deg).
        noise_seed: RNG seed for the weather-noise realisation, pinned
            **separately** from the EMBRS spotting seed.
    """

    w_min_ms: float = 1.5
    peak_scale_ms: float = 4.0
    rise_start_hr: float = 7.0
    daytime_span_hr: float = 14.0
    peak_frac: float = 0.55
    ou_phi: float = 0.8
    ou_sigma_ms: float = 0.5
    prevailing_dir_deg: float = 180.0
    dir_reversion: float = 0.05
    dir_sigma_deg: float = 8.0
    noise_seed: int = 7


@dataclass
class GeneratorConfig:
    """Assemble a ``.wxs`` from a real temp/RH backdrop + synthetic wind (§4).

    Attributes:
        wind: Synthetic wind parameters.
        elevation_ft: ``.wxs`` header elevation; if None, the source backdrop's
            elevation is preserved.
        wind_assert_tol_mph: Tolerance for the realized-vs-intended wind
            statistics assertion (spec §4.6).
        zero_precip: Zero out the precipitation column in the generated ``.wxs``
            (default True). The backdrop window is already chosen to be dry (the
            period search rejects wet windows), and a controlled scenario should
            not have rain confounding the fire behaviour we tune to.
    """

    wind: WindModelConfig = field(default_factory=WindModelConfig)
    elevation_ft: Optional[int] = None
    wind_assert_tol_mph: float = 0.5
    zero_precip: bool = True


@dataclass
class TuningConfig:
    """Auto-tune the wind peak to a target flame length (spec §5).

    Attributes:
        target_ft: Target ``mean_daily_peak_flame_ft`` (e.g. 4/6/8).
        tolerance_ft: Acceptance half-band around the target.
        max_iter: Cap on classifier evaluations.
        bracket_lo_ms: Lower bound of the initial ``peak_scale`` bracket.
        bracket_hi_ms: Upper bound of the initial ``peak_scale`` bracket.
        tuning_days: Length (days) of the short baseline classify window —
            enough to average the daily-peak since drivers are ~stationary.
    """

    target_ft: float
    tolerance_ft: float = 0.3
    max_iter: int = 12
    bracket_lo_ms: float = 1.0
    bracket_hi_ms: float = 12.0
    tuning_days: int = 6


# Default backburn-feasibility thresholds. Copied (not imported) from
# ``ra-cbba-core/applications/firefighting/config/schema.py`` so this package has
# no cross-repo dependency; keep in sync with BackburnTaskCfg there (spec §6).
BACKBURN_HI_WIND_THRESH_M_S: float = 10.0
BACKBURN_WIND_ANGLE_TOL_DEG: float = 45.0


@dataclass
class BackburnProxyConfig:
    """Lightweight self-contained backburn-feasibility proxy (spec §6).

    A timestep is backburn-suitable when wind speed is below the threshold AND
    the wind aligns (within tolerance) with a representative fireline segment's
    outward normal. This replicates ``backburn_wind_check`` without importing
    the firefighting app.

    Attributes:
        hi_wind_speed_thresh_m_s: Max wind speed (m/s) for a suitable timestep.
        wind_angle_tol_deg: Max angle between the wind and the segment's outward
            normal.
        fireline_bearing_deg: Bearing (meteorological deg) of the representative
            fireline segment's *outward normal* — the direction the backburn
            should be pushed. The user supplies this for the region/geometry.
        min_window_hours: Flag windows shorter than this as not useful.
    """

    hi_wind_speed_thresh_m_s: float = BACKBURN_HI_WIND_THRESH_M_S
    wind_angle_tol_deg: float = BACKBURN_WIND_ANGLE_TOL_DEG
    fireline_bearing_deg: float = 0.0
    min_window_hours: float = 2.0
