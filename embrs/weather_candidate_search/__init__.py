"""Weather candidate-window search for the wildfire backburning study.

Pulls Open-Meteo ERA5 history for a LANDFIRE region, runs the EMBRS
NFDRS BI pipeline once over the fire season, slides a fixed-length window
across the resulting BI trajectory, and writes the top-N candidate windows
whose peak BI hits a target volatility band — plus detected backburn
("lull") windows as a soft ranking signal.

Public entry points:

- :func:`run_candidate_search` (programmatic).
- ``python -m embrs.weather_candidate_search`` (CLI).
- ``python -m embrs.weather_candidate_search.extract_candidate_wxs``
  (standalone helper to materialise an EMBRS-runnable ``.wxs`` for one
  selected candidate).
"""
from embrs.weather_candidate_search.config import (
    BISection,
    Config,
    LullConfig,
    ScoringConfig,
    WindConversionConfig,
)
from embrs.weather_candidate_search.pipeline import run_candidate_search

__all__ = [
    "BISection",
    "Config",
    "LullConfig",
    "ScoringConfig",
    "WindConversionConfig",
    "run_candidate_search",
]
