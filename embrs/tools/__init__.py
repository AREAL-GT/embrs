"""Fire prediction tools for EMBRS.

This package provides forward fire spread prediction capabilities with
uncertainty modeling for ensemble forecasting.

Classes:
    - FirePredictor: Ensemble fire prediction with wind uncertainty.

.. autoclass:: FirePredictor
    :members:
"""

from embrs.tools.fire_predictor import FirePredictor

__all__ = ["FirePredictor"]
