"""Controlled scenario weather generation.

Produces, for each ``(region, intensity-class)`` pair, an EMBRS ``.wxs`` weather
file that is time-varying and realistic, contains feasible backburn windows,
and — when run on that region's real scenario map — yields a target *average
daily-peak flame length* that **defines** the intensity class.

This replaces the abandoned approach of selecting real weather windows by
Burning Index. See ``embrs/controlled_scenario_weather_spec.md`` for the full
design and rationale.

The package is organised into:

- :mod:`~embrs.scenario_weather.run_config`  — write a temporary ``.cfg`` and
  build a :class:`~embrs.fire_simulator.fire.FireSim` in-process.
- :mod:`~embrs.scenario_weather.classifier` — run the sim and measure the
  class metric (mean daily-peak flame length) plus head ROS.
- :mod:`~embrs.scenario_weather.period_search` — find per-class real temp/RH
  backdrop windows from a region's full-season ``.wxs``.
- :mod:`~embrs.scenario_weather.wind_model` / ``generator`` — synthesise wind
  and assemble a ``.wxs``.
- :mod:`~embrs.scenario_weather.tuning` / ``variance`` / ``backburn_check`` /
  ``reclassify`` — tune to a target, characterise seed spread, validate
  backburn windows, and re-verify hand-edited files.
- :mod:`~embrs.scenario_weather.plotting` — temp/RH-profile and full-``.wxs``
  diagnostic plots.
"""
from embrs.scenario_weather.config import (
    ClassifierConfig,
    RunConfig,
)

__all__ = ["RunConfig", "ClassifierConfig"]
