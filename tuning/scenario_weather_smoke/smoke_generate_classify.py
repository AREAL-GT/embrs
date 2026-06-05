"""Smoke test: generate a .wxs from a real backdrop + synthetic wind, then
classify it on the real region-A map (spec §12 step 3 acceptance: a generated
.wxs runs and classifies). Real file + __main__ guard for spawn safety (§9).
"""
import datetime as dt
import os
import tempfile

from embrs.scenario_weather.classifier import classify
from embrs.scenario_weather.config import ClassifierConfig, GeneratorConfig, RunConfig, WindModelConfig
from embrs.scenario_weather.generator import generate_from_window

MAP_DIR = "/Users/rjdp3/Documents/Research/embrs_map/thesis_eval_maps/a_flint_hills"
SEASON = "embrs/weather_candidate_search/search_outputs/region_a_flint_hills_extreme/full_season.wxs"


def main():
    workdir = tempfile.mkdtemp(prefix="sw_gen_")
    wxs = os.path.join(workdir, "gen_extreme.wxs")
    # Extreme backdrop window (from the period search), short classify span.
    backdrop_start = dt.datetime(2022, 7, 8, 0, 0)
    backdrop_end = dt.datetime(2022, 7, 21, 23, 0)
    gcfg = GeneratorConfig(wind=WindModelConfig(w_min_ms=1.5, peak_scale_ms=5.0,
                                                prevailing_dir_deg=180.0))
    gen = generate_from_window(wxs, SEASON, backdrop_start, backdrop_end, gcfg,
                               peak_scale=5.0)
    print("GEN", gen.to_dict())

    start = dt.datetime(2022, 7, 8, 0, 0)
    end = dt.datetime(2022, 7, 9, 6, 0)  # ~30h -> one eligible day
    run_cfg = RunConfig(live_herb_mf=0.30, live_woody_mf=0.60, seed=42)
    report = classify(wxs, MAP_DIR, start, end, run_cfg,
                      cfg_path=os.path.join(workdir, "gen.cfg"),
                      classifier_cfg=ClassifierConfig(),
                      region="region_a_flint_hills", fire_class="extreme")
    print("REPORT_JSON_BEGIN")
    print(report.to_json())
    print("REPORT_JSON_END")


if __name__ == "__main__":
    main()
