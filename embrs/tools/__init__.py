"""Fire prediction tools for EMBRS.

This package provides forward fire spread prediction capabilities with
uncertainty modeling for ensemble forecasting.

Classes:
    - FirePredictor: Ensemble fire prediction with wind uncertainty.
    - ForecastPool: Collection of pre-computed wind forecasts for ensemble use.
    - ForecastData: Container for a single wind forecast with metadata.
    - ForecastPoolManager: Global manager for active forecast pools.
    - PredictorSerializer: Handles FirePredictor serialization for multiprocessing.

.. autoclass:: FirePredictor
    :members:
.. autoclass:: ForecastPool
    :members:
.. autoclass:: PredictorSerializer
    :members:
"""

from embrs.tools.fire_predictor import FirePredictor
from embrs.tools.forecast_pool import (
    ForecastPool,
    ForecastData,
    ForecastPoolManager,
)
from embrs.tools.predictor_serializer import PredictorSerializer

__all__ = [
    "FirePredictor",
    "ForecastPool",
    "ForecastData",
    "ForecastPoolManager",
    "PredictorSerializer",
]
