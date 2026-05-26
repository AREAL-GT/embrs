"""Area-weighted NFDRS Burning Index trajectory tool.

Compute an hourly area-weighted NFDRS Burning Index trajectory for a LANDFIRE
landscape under a synthetic ``.wxs`` weather scenario. Used iteratively to tune
weather forecasts to peak-BI volatility targets per region.

The public entry points are :func:`compute_bi_trajectory` (programmatic) and
``python -m embrs.fire_danger`` (CLI).
"""
from embrs.fire_danger.config import Config, TrajectoryResult
from embrs.fire_danger.trajectory import compute_bi_trajectory

__all__ = ["Config", "TrajectoryResult", "compute_bi_trajectory"]
