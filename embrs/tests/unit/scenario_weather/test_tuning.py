"""Tests for the tuning harness root-find logic (spec §5).

The classifier and generator are monkeypatched with a cheap monotonic surrogate
``flame(peak_scale)`` so we exercise the bracketed-secant/bisection control flow
without running any simulation.
"""
from datetime import datetime
from types import SimpleNamespace

import pytest

import embrs.scenario_weather.tuning as tuning
from embrs.scenario_weather.config import GeneratorConfig, RunConfig, TuningConfig


def _patch_surrogate(monkeypatch, flame_of_scale):
    """Patch generate + classify so each eval returns flame_of_scale(peak)."""
    def fake_generate(out, season, bstart, bend, gcfg, *, peak_scale):
        # Record the scale on the path so fake_classify can recover it.
        return SimpleNamespace(wxs_path=out, peak_scale_ms=peak_scale)

    def fake_classify(wxs, map_dir, start, end, run_cfg, *, cfg_path,
                      classifier_cfg=None, region=None, fire_class=None):
        # Recover the peak scale from the filename tag iterNN_psX.XXX.wxs.
        tag = wxs.split("ps")[-1].replace(".wxs", "")
        scale = float(tag)
        flame = flame_of_scale(scale)
        return SimpleNamespace(
            mean_daily_peak_flame_ft=flame,
            to_dict=lambda: {"mean_daily_peak_flame_ft": flame},
        )

    monkeypatch.setattr(tuning, "generate_from_window", fake_generate)
    monkeypatch.setattr(tuning, "classify", fake_classify)


def _tune(tmp_path, tcfg):
    return tuning.tune(
        str(tmp_path),
        full_season_wxs="season.wxs",
        backdrop_start=datetime(2022, 7, 1),
        backdrop_end=datetime(2022, 7, 14),
        map_dir="map",
        run_cfg=RunConfig(live_herb_mf=0.3, live_woody_mf=0.6),
        gen_cfg=GeneratorConfig(),
        tuning_cfg=tcfg,
        classify_start=datetime(2022, 7, 2),
    )


def test_converges_on_monotonic_surrogate(tmp_path, monkeypatch):
    # flame = 2 + 0.8 * peak_scale ; target 6 -> peak_scale = 5.
    _patch_surrogate(monkeypatch, lambda ps: 2.0 + 0.8 * ps)
    res = _tune(tmp_path, TuningConfig(target_ft=6.0, tolerance_ft=0.3,
                                       bracket_lo_ms=1.0, bracket_hi_ms=12.0))
    assert res.converged
    assert res.final_flame_ft == pytest.approx(6.0, abs=0.3)
    assert res.final_peak_scale_ms == pytest.approx(5.0, abs=0.5)


def test_unbracketed_returns_closest_with_note(tmp_path, monkeypatch):
    # Target above everything reachable in the bracket.
    _patch_surrogate(monkeypatch, lambda ps: 2.0 + 0.1 * ps)
    res = _tune(tmp_path, TuningConfig(target_ft=20.0, tolerance_ft=0.3,
                                       bracket_lo_ms=1.0, bracket_hi_ms=12.0))
    assert not res.converged
    assert any("not bracketed" in n for n in res.notes)
    # Closest endpoint is the high bracket (largest flame).
    assert res.final_peak_scale_ms == 12.0


def test_handles_shrub_jump_nonmonotonicity(tmp_path, monkeypatch):
    # Smooth then a jump at ps>=10 (shrub transition, spec §8). Target sits on
    # the smooth part; bracketed search should still find it.
    def flame(ps):
        base = 2.0 + 0.6 * ps
        return base + (5.0 if ps >= 10.0 else 0.0)
    _patch_surrogate(monkeypatch, flame)
    res = _tune(tmp_path, TuningConfig(target_ft=5.0, tolerance_ft=0.3,
                                       bracket_lo_ms=1.0, bracket_hi_ms=8.0))
    assert res.converged
    assert res.final_flame_ft == pytest.approx(5.0, abs=0.3)
