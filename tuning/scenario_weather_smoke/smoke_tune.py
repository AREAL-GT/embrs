"""Smoke test: run the real tuning harness for a few iterations on region A.

Exercises generate->classify per iteration, the bracketed root-find, and trace
persistence (spec §5). Heavy (each iteration runs WindNinja + a sim); kept to a
1-day classify window and a tight bracket. Real file + __main__ guard (§9).
"""
import datetime as dt
import os
import tempfile

from embrs.scenario_weather.config import GeneratorConfig, RunConfig, TuningConfig, WindModelConfig
from embrs.scenario_weather.tuning import tune

MAP_DIR = "/Users/rjdp3/Documents/Research/embrs_map/thesis_eval_maps/a_flint_hills"
SEASON = "embrs/weather_candidate_search/search_outputs/region_a_flint_hills_extreme/full_season.wxs"


def main():
    out_dir = tempfile.mkdtemp(prefix="sw_tune_")
    gcfg = GeneratorConfig(wind=WindModelConfig(w_min_ms=1.5, prevailing_dir_deg=180.0))
    tcfg = TuningConfig(target_ft=6.0, tolerance_ft=0.3, max_iter=4,
                        bracket_lo_ms=3.0, bracket_hi_ms=7.0, tuning_days=1)
    res = tune(
        out_dir,
        full_season_wxs=SEASON,
        backdrop_start=dt.datetime(2022, 7, 8, 0, 0),
        backdrop_end=dt.datetime(2022, 7, 21, 23, 0),
        map_dir=MAP_DIR,
        run_cfg=RunConfig(live_herb_mf=0.30, live_woody_mf=0.60, seed=42),
        gen_cfg=gcfg,
        tuning_cfg=tcfg,
        classify_start=dt.datetime(2022, 7, 8, 0, 0),
        region="region_a_flint_hills",
        fire_class="moderate",
    )
    print("TUNE_JSON_BEGIN")
    print(res.to_json())
    print("TUNE_JSON_END")
    print("artifacts in", out_dir)


if __name__ == "__main__":
    main()
