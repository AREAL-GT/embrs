# Data Classes

This module defines the dataclasses that carry configuration and results through the
EMBRS pipeline. Users encounter them in two main contexts: setting up a simulation
(via [`SimParams`](#embrs.utilities.data_classes.SimParams),
[`MapParams`](#embrs.utilities.data_classes.MapParams), and
[`WeatherParams`](#embrs.utilities.data_classes.WeatherParams) loaded from `.cfg` files)
and working with fire prediction outputs
(via [`PredictionOutput`](#embrs.utilities.data_classes.PredictionOutput),
[`EnsemblePredictionOutput`](#embrs.utilities.data_classes.EnsemblePredictionOutput), and
[`StateEstimate`](#embrs.utilities.data_classes.StateEstimate) returned by
[`FirePredictor`](tools_documentation.md)). Most fields mirror the configuration file
options documented in the [Tutorials](running_sim.md) section.

::: embrs.utilities.data_classes
