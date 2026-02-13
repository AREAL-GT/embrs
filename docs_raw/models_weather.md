# Weather

This module handles weather data ingestion and processing for fire simulations. It supports two input sources: the Open-Meteo historical reanalysis API and RAWS-format weather station files (`.wxs`). The `WeatherStream` class builds a time-indexed sequence of weather observations, computes derived quantities (solar radiation, Growing Season Index, foliar moisture content, live fuel moisture), and provides the weather data that drives the simulation's fire behavior calculations.

::: embrs.models.weather
