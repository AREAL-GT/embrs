"""Unit tests for the Config dataclass + sub-configs."""
from __future__ import annotations

import pytest

from embrs.weather_candidate_search.config import (
    BISection,
    Config,
    LullConfig,
    ScoringConfig,
    WindConversionConfig,
)


def _minimal_kwargs(**overrides):
    base = dict(
        landscape_tif="/tmp/foo.tif",
        year=2024,
        fire_season_start_month=5,
        fire_season_end_month=10,
        scenario_length_hours=72,
        bi_target_band=(60.0, 80.0),
        output_dir="/tmp/out",
        region_tag="shrub_grass",
        volatility_class="moderate",
    )
    base.update(overrides)
    return base


def test_config_defaults_apply():
    cfg = Config(**_minimal_kwargs())
    assert cfg.n_candidates == 5
    assert cfg.conditioning_days == 30
    assert cfg.window_stride_hours == 1
    assert isinstance(cfg.lull, LullConfig)
    assert isinstance(cfg.scoring, ScoringConfig)
    assert isinstance(cfg.wind_conversion, WindConversionConfig)
    assert isinstance(cfg.bi, BISection)


def test_cell_dir_format():
    cfg = Config(**_minimal_kwargs(output_dir="/tmp/x/"))
    assert cfg.cell_dir == "/tmp/x/shrub_grass_moderate"


def test_rejects_southern_hemisphere():
    with pytest.raises(ValueError, match="Northern hemisphere"):
        Config(**_minimal_kwargs(fire_season_start_month=11, fire_season_end_month=2))


def test_rejects_invalid_months():
    with pytest.raises(ValueError):
        Config(**_minimal_kwargs(fire_season_start_month=0))
    with pytest.raises(ValueError):
        Config(**_minimal_kwargs(fire_season_end_month=13))


def test_requires_scenario_length():
    with pytest.raises(ValueError):
        Config(**_minimal_kwargs(scenario_length_hours=0))
    with pytest.raises(ValueError):
        Config(**_minimal_kwargs(scenario_length_hours=-1))


def test_rejects_inverted_band():
    with pytest.raises(ValueError, match="bi_target_band"):
        Config(**_minimal_kwargs(bi_target_band=(80.0, 60.0)))


def test_requires_region_and_volatility_strings():
    with pytest.raises(ValueError):
        Config(**_minimal_kwargs(region_tag=""))
    with pytest.raises(ValueError):
        Config(**_minimal_kwargs(volatility_class=""))


def test_other_validation_paths():
    with pytest.raises(ValueError):
        Config(**_minimal_kwargs(n_candidates=0))
    with pytest.raises(ValueError):
        Config(**_minimal_kwargs(conditioning_days=0))
    with pytest.raises(ValueError):
        Config(**_minimal_kwargs(window_stride_hours=0))
