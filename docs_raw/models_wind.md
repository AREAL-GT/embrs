# Wind Forecast

This module generates spatially-resolved wind fields by running the WindNinja CLI with domain-average initialization. It parallelizes WindNinja execution across weather time steps and loads the outputs into structured NumPy arrays for use in fire simulations. Users interact with this module when configuring wind forecasts for ensemble predictions or when WindNinja-based spatially-varying wind is needed.

::: embrs.models.wind_forecast
