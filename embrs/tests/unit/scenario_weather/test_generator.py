"""Tests for the .wxs generator: unit/height round-trips and the realized-wind
assertion (spec §4.6). No simulation is run."""
import numpy as np
import pytest

from embrs.fire_danger.weather_loader import load_wxs
from embrs.scenario_weather.config import GeneratorConfig, WindModelConfig
from embrs.scenario_weather.generator import generate_from_window
from embrs.tests.unit.scenario_weather._synth import write_season_wxs

_MPS_TO_MPH = 2.23693629


def _backdrop(tmp_path):
    return write_season_wxs(str(tmp_path / "season.wxs"), "2022-07-01", 20,
                            wind_mps=99.0)  # real wind is overwritten by the generator


def test_generate_roundtrips_and_preserves_backdrop(tmp_path):
    season = _backdrop(tmp_path)
    gcfg = GeneratorConfig(wind=WindModelConfig(w_min_ms=1.5, peak_scale_ms=4.0))
    out = str(tmp_path / "gen.wxs")
    res = generate_from_window(out, season, _dt("2022-07-02T00:00:00"),
                               _dt("2022-07-09T23:00:00"), gcfg, peak_scale=5.0)
    assert res.n_rows == 8 * 24

    gen = load_wxs(out).df
    src = load_wxs(season).df
    sl = src.loc[(src.index >= _dt("2022-07-02")) & (src.index <= _dt("2022-07-09T23:00:00"))]
    # temp/RH/cloud/precip preserved from the backdrop; only wind replaced.
    assert np.allclose(gen["temp_F"].to_numpy(), sl["temp_F"].to_numpy(), atol=0.1)
    assert np.allclose(gen["rh_pct"].to_numpy(), sl["rh_pct"].to_numpy(), atol=1.0)
    # The 99 m/s placeholder wind is gone -> realized wind is the synthetic one.
    assert gen["wind_mph"].max() < 99.0 * _MPS_TO_MPH


def test_precip_zeroed_by_default(tmp_path):
    # Backdrop with real rain on day 3 -> generated .wxs must be bone dry.
    season = write_season_wxs(str(tmp_path / "wet.wxs"), "2022-07-01", 20,
                              precip_in_per_day={3: 0.5})
    gcfg = GeneratorConfig(wind=WindModelConfig())
    out = str(tmp_path / "gen.wxs")
    generate_from_window(out, season, _dt("2022-07-02T00:00:00"),
                         _dt("2022-07-09T23:00:00"), gcfg)
    assert load_wxs(out).df["precip_in_hr"].to_numpy().max() == 0.0

    # Opt back in -> the backdrop rain is preserved.
    out2 = str(tmp_path / "gen_wet.wxs")
    generate_from_window(out2, season, _dt("2022-07-02T00:00:00"),
                         _dt("2022-07-09T23:00:00"),
                         GeneratorConfig(wind=WindModelConfig(), zero_precip=False))
    assert load_wxs(out2).df["precip_in_hr"].to_numpy().max() > 0.0


def test_realized_wind_matches_intended_no_height_correction(tmp_path):
    season = _backdrop(tmp_path)
    gcfg = GeneratorConfig(wind=WindModelConfig(w_min_ms=1.5, peak_scale_ms=4.0))
    out = str(tmp_path / "gen.wxs")
    res = generate_from_window(out, season, _dt("2022-07-02T00:00:00"),
                               _dt("2022-07-05T23:00:00"), gcfg, peak_scale=4.0)
    # 20-ft-native: no log-profile (~0.911) correction is applied. The realized
    # mean must equal the intended mean within rounding, not be ~9% lower.
    assert res.realized.mean_mph == pytest.approx(res.intended.mean_mph, abs=0.1)
    assert res.realized.peak_mph == pytest.approx(res.intended.peak_mph, abs=0.2)


def test_assertion_catches_unit_bug(tmp_path, monkeypatch):
    season = _backdrop(tmp_path)
    gcfg = GeneratorConfig(wind=WindModelConfig(peak_scale_ms=4.0),
                           wind_assert_tol_mph=0.5)
    out = str(tmp_path / "gen.wxs")

    # Simulate a 3.6x km/h-as-m/s style bug by corrupting the writer output the
    # generator reads back: monkeypatch load_wxs *inside the generator module* to
    # return tripled wind, which must trip the realized-vs-intended assertion.
    import embrs.scenario_weather.generator as genmod
    real_load = genmod.load_wxs

    def fake_load(path):
        hw = real_load(path)
        hw.df["wind_mph"] = hw.df["wind_mph"] * 3.0
        return hw

    monkeypatch.setattr(genmod, "load_wxs", fake_load)
    with pytest.raises(AssertionError):
        generate_from_window(out, season, _dt("2022-07-02T00:00:00"),
                             _dt("2022-07-04T23:00:00"), gcfg, peak_scale=4.0)


def _dt(s):
    from datetime import datetime
    return datetime.fromisoformat(s)
