"""Smoke test: run the scenario_weather classifier on the real region-A map.

Real file with an ``if __name__ == "__main__"`` guard so FireSim's spotting
workers can re-import it safely (spec §9). Short window → one eligible day.
"""
import datetime as dt
import os
import tempfile

from embrs.scenario_weather.classifier import classify
from embrs.scenario_weather.config import ClassifierConfig, RunConfig

MAP_DIR = "/Users/rjdp3/Documents/Research/embrs_map/thesis_eval_maps/a_flint_hills"
WXS = "/Users/rjdp3/Documents/Research/embrs_weather/thesis_forecasts/a_flint_hills_extreme.wxs"


def main():
    start = dt.datetime(2022, 9, 14, 0, 0, 0)
    end = dt.datetime(2022, 9, 15, 6, 0, 0)  # ~30h → 1 full day + partial
    run_cfg = RunConfig(live_herb_mf=0.30, live_woody_mf=0.60, seed=42)
    workdir = tempfile.mkdtemp(prefix="sw_smoke_")
    report = classify(
        WXS,
        MAP_DIR,
        start,
        end,
        run_cfg,
        cfg_path=os.path.join(workdir, "smoke.cfg"),
        classifier_cfg=ClassifierConfig(),
        region="region_a_flint_hills",
        fire_class="extreme",
    )
    print("REPORT_JSON_BEGIN")
    print(report.to_json())
    print("REPORT_JSON_END")


if __name__ == "__main__":
    main()
