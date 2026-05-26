"""Tests for the CLI / YAML config loader."""
from __future__ import annotations

import pytest

from embrs.weather_candidate_search.cli import _build_parser, config_from_namespace


def _parse(argv):
    return _build_parser().parse_args(argv)


def test_cli_minimal_required_fields(tmp_path):
    argv = [
        "--landscape-tif", "/tmp/foo.tif",
        "--year", "2024",
        "--fire-season-start-month", "5",
        "--fire-season-end-month", "10",
        "--scenario-length-hours", "72",
        "--bi-target-band", "60", "80",
        "--output-dir", str(tmp_path),
        "--region-tag", "r1",
        "--volatility-class", "moderate",
    ]
    cfg = config_from_namespace(_parse(argv))
    assert cfg.landscape_tif == "/tmp/foo.tif"
    assert cfg.year == 2024
    assert cfg.scenario_length_hours == 72
    assert cfg.bi_target_band == (60.0, 80.0)
    assert cfg.lull.wind_threshold_mph == 8.0
    assert cfg.wind_conversion.enabled is True


def test_cli_missing_required_field_fails():
    argv = ["--year", "2024"]
    with pytest.raises(SystemExit, match="Missing required"):
        config_from_namespace(_parse(argv))


def test_cli_wind_correction_off_switch(tmp_path):
    argv = [
        "--landscape-tif", "/tmp/foo.tif",
        "--year", "2024",
        "--fire-season-start-month", "5",
        "--fire-season-end-month", "10",
        "--scenario-length-hours", "72",
        "--bi-target-band", "60", "80",
        "--output-dir", str(tmp_path),
        "--region-tag", "r1",
        "--volatility-class", "moderate",
        "--no-wind-height-correction",
    ]
    cfg = config_from_namespace(_parse(argv))
    assert cfg.wind_conversion.enabled is False


def test_yaml_supplies_required_fields(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
landscape_tif: /tmp/foo.tif
year: 2024
fire_season_start_month: 5
fire_season_end_month: 10
scenario_length_hours: 72
bi_target_band: [60, 80]
output_dir: /tmp/out
region_tag: r1
volatility_class: moderate
lull:
  wind_threshold_mph: 6.0
  min_consecutive_hours: 3
scoring:
  bi_distance_weight: 2.0
bi:
  min_area_frac: 0.10
""".strip()
    )
    cfg = config_from_namespace(_parse(["--config", str(yaml_path)]))
    assert cfg.year == 2024
    assert cfg.lull.wind_threshold_mph == 6.0
    assert cfg.lull.min_consecutive_hours == 3
    assert cfg.scoring.bi_distance_weight == 2.0
    assert cfg.bi.min_area_frac == 0.10


def test_cli_min_candidate_separation_hours_round_trip(tmp_path):
    argv = [
        "--landscape-tif", "/tmp/foo.tif",
        "--year", "2024",
        "--fire-season-start-month", "5",
        "--fire-season-end-month", "10",
        "--scenario-length-hours", "168",
        "--bi-target-band", "60", "80",
        "--output-dir", str(tmp_path),
        "--region-tag", "r1",
        "--volatility-class", "moderate",
        "--min-candidate-separation-hours", "24",
    ]
    cfg = config_from_namespace(_parse(argv))
    assert cfg.min_candidate_separation_hours == 24
    assert cfg.effective_min_separation_hours == 24


def test_cli_min_separation_default_is_window_length(tmp_path):
    argv = [
        "--landscape-tif", "/tmp/foo.tif",
        "--year", "2024",
        "--fire-season-start-month", "5",
        "--fire-season-end-month", "10",
        "--scenario-length-hours", "168",
        "--bi-target-band", "60", "80",
        "--output-dir", str(tmp_path),
        "--region-tag", "r1",
        "--volatility-class", "moderate",
    ]
    cfg = config_from_namespace(_parse(argv))
    assert cfg.min_candidate_separation_hours is None
    assert cfg.effective_min_separation_hours == 168


def test_cli_overrides_yaml(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
landscape_tif: /tmp/foo.tif
year: 2024
fire_season_start_month: 5
fire_season_end_month: 10
scenario_length_hours: 72
bi_target_band: [60, 80]
output_dir: /tmp/out
region_tag: r1
volatility_class: moderate
lull:
  wind_threshold_mph: 6.0
""".strip()
    )
    cfg = config_from_namespace(
        _parse(["--config", str(yaml_path), "--wind-threshold-mph", "10"])
    )
    assert cfg.lull.wind_threshold_mph == 10.0
