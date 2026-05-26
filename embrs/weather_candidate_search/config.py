"""Configuration dataclasses for the weather candidate search tool.

The :class:`Config` here is **separate** from :class:`embrs.fire_danger.Config`.
The ``bi`` sub-section is a pass-through to the BI pipeline (resolved in
``bi_search.run_bi``).

See ``implementation_plan.md`` §4.1 for the field-by-field contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class LullConfig:
    """Lull-detection thresholds (plan §4.1; qa E1/E2/E3)."""

    wind_threshold_mph: float = 8.0
    rh_threshold_pct: float = 40.0
    min_consecutive_hours: int = 4
    tolerance_hours: int = 0


@dataclass(frozen=True)
class ScoringConfig:
    """Composite-score weights (plan §4.9, qa F1)."""

    bi_distance_weight: float = 1.0
    lulls_weight: float = 0.5
    lull_hours_weight: float = 0.05


@dataclass(frozen=True)
class WindConversionConfig:
    """Open-Meteo 10 m → NFDRS 20 ft log-profile correction (qa B4).

    ``u_20ft = u_10m * ln(6.1 / z0) / ln(10 / z0)``. With ``z0 = 0.06`` m
    (grass-shrub default) the multiplier is ≈ 0.911.
    """

    enabled: bool = True
    surface_roughness_m: float = 0.06


@dataclass(frozen=True)
class BISection:
    """Pass-through to :class:`embrs.fire_danger.Config` (plan §4.1, qa A4).

    All fields default to the BI pipeline's own defaults; ``None`` means
    "let the BI pipeline decide" (e.g., auto-derive slope class from the
    landscape, auto-fetch ``AvgAnnPrecip`` from Open-Meteo).
    """

    min_area_frac: float = 0.05
    slope_class: Optional[int] = None
    lat_override: Optional[float] = None
    reg_obs_hr: int = 13
    cloud_scale: str = "percent"
    snow_mode: str = "none"
    avg_ann_precip_in: Optional[float] = None


@dataclass(frozen=True)
class Config:
    """Run parameters for :func:`pipeline.run_candidate_search`.

    Required inputs (spec §Inputs, plan §4.1):
        landscape_tif: Path to a LANDFIRE ``.tif`` (band 4 = fuel,
            band 2 = slope, band 1 = elevation).
        year: The fire-season year to pull historical weather for.
        fire_season_start_month: 1..12 (inclusive). Must be <= end_month
            (northern hemisphere only — qa J1).
        fire_season_end_month: 1..12 (inclusive).
        scenario_length_hours: Window length in hours. No default
            (qa H3 — fail loud).
        bi_target_band: ``(min, max)`` BI band defining this volatility cell.
        output_dir: Root output directory.
        region_tag, volatility_class: Free-form labels controlling the
            ``{output_dir}/{region}_{volatility}/`` sub-directory (qa G1).

    Optional knobs follow the plan's defaults.
    """

    landscape_tif: str
    year: int
    fire_season_start_month: int
    fire_season_end_month: int
    scenario_length_hours: int
    bi_target_band: Tuple[float, float]
    output_dir: str
    region_tag: str
    volatility_class: str

    n_candidates: int = 5
    cache_dir: str = "./.openmeteo_cache/"
    conditioning_days: int = 30
    window_stride_hours: int = 1

    # Temporal non-maximum suppression: forbid two selected candidates whose
    # starts are closer than this many hours apart. ``None`` ⇒ use
    # ``scenario_length_hours`` (i.e. selected windows never overlap), which
    # is the natural default for a sliding-window search where adjacent
    # candidates share ≥ 99% of their data. Set to 0 to disable NMS entirely.
    min_candidate_separation_hours: Optional[int] = None

    lull: LullConfig = field(default_factory=LullConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    wind_conversion: WindConversionConfig = field(default_factory=WindConversionConfig)
    bi: BISection = field(default_factory=BISection)

    def __post_init__(self) -> None:
        if not isinstance(self.fire_season_start_month, int) or not (
            1 <= self.fire_season_start_month <= 12
        ):
            raise ValueError(
                f"fire_season_start_month must be int in [1, 12], "
                f"got {self.fire_season_start_month!r}"
            )
        if not isinstance(self.fire_season_end_month, int) or not (
            1 <= self.fire_season_end_month <= 12
        ):
            raise ValueError(
                f"fire_season_end_month must be int in [1, 12], "
                f"got {self.fire_season_end_month!r}"
            )
        if self.fire_season_start_month > self.fire_season_end_month:
            raise ValueError(
                "Northern hemisphere only (qa J1): "
                "fire_season_start_month must be <= fire_season_end_month, got "
                f"start={self.fire_season_start_month}, "
                f"end={self.fire_season_end_month}"
            )
        if self.scenario_length_hours <= 0:
            raise ValueError(
                f"scenario_length_hours must be > 0, got {self.scenario_length_hours}"
            )
        if len(self.bi_target_band) != 2 or self.bi_target_band[0] > self.bi_target_band[1]:
            raise ValueError(
                f"bi_target_band must be (min, max) with min <= max, "
                f"got {self.bi_target_band!r}"
            )
        if self.n_candidates <= 0:
            raise ValueError(f"n_candidates must be > 0, got {self.n_candidates}")
        if self.conditioning_days <= 0:
            raise ValueError(
                f"conditioning_days must be > 0, got {self.conditioning_days}"
            )
        if self.window_stride_hours <= 0:
            raise ValueError(
                f"window_stride_hours must be > 0, got {self.window_stride_hours}"
            )
        if (
            self.min_candidate_separation_hours is not None
            and self.min_candidate_separation_hours < 0
        ):
            raise ValueError(
                f"min_candidate_separation_hours must be >= 0 or None, "
                f"got {self.min_candidate_separation_hours}"
            )
        if not self.region_tag:
            raise ValueError("region_tag must be a non-empty string")
        if not self.volatility_class:
            raise ValueError("volatility_class must be a non-empty string")

    @property
    def cell_dir(self) -> str:
        """``{output_dir}/{region_tag}_{volatility_class}`` — the cell root."""
        return f"{self.output_dir.rstrip('/')}/{self.region_tag}_{self.volatility_class}"

    @property
    def effective_min_separation_hours(self) -> int:
        """Resolve ``min_candidate_separation_hours`` to a concrete int."""
        if self.min_candidate_separation_hours is None:
            return int(self.scenario_length_hours)
        return int(self.min_candidate_separation_hours)
