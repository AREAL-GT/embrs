"""Fire behavior and environmental models for EMBRS.

This package provides the physical and statistical models that drive fire
spread, fuel moisture, weather processing, and firebrand transport in the
EMBRS simulation framework.

Modules:
    - rothermel: Rothermel (1972) surface fire spread model.
    - fuel_models: Anderson 13 and Scott-Burgan 40 fuel model definitions.
    - crown_model: Crown fire initiation and spread (Van Wagner / Rothermel).
    - weather: Weather data ingestion from Open-Meteo and RAWS files.
    - wind_forecast: Spatially-resolved wind fields via WindNinja.
    - dead_fuel_moisture: Nelson dead fuel moisture diffusion model.
    - irpg_moisture_model: IRPG fine dead fuel moisture estimation.
    - embers: Albini (1979) firebrand lofting and ballistic transport.
    - perryman_spot: Perryman et al. (2013) statistical spotting model.
"""
