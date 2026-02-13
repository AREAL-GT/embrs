# Tools Reference

The `embrs.tools` package provides standalone tools that build on top of the core simulation engine. The primary tool is `FirePredictor`, which runs forward fire spread predictions with uncertainty modeling and ensemble support. Supporting classes handle forecast pool management for reusing pre-computed wind forecasts across predictions, and serialization for efficient parallel execution.

The `ensemble_video` utility generates video visualizations of ensemble prediction output.

## Fire Predictor

::: embrs.tools.fire_predictor

## Forecast Pool

::: embrs.tools.forecast_pool

## Predictor Serializer

::: embrs.tools.predictor_serializer

## Ensemble Video

::: embrs.utilities.ensemble_video
