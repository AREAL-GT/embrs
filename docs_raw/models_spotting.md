# Spotting Models

These modules model firebrand generation and spot fire ignition. The `Embers` class implements the Albini (1979) physics-based model for firebrand lofting and ballistic transport from torching trees. The `PerrymanSpotting` class implements the Perryman et al. (2013) statistical model that samples firebrand landing locations from probability distributions parameterized by fireline intensity and wind speed.

The core simulation (`FireSim`) uses the physics-based `Embers` model, which tracks individual firebrand trajectories through the wind field. The [prediction model](fire_prediction.md) (`FirePredictor`) uses the statistical `PerrymanSpotting` model, which is faster and better suited to the simplified prediction environment where fuel moisture is fixed and fire acceleration is not modeled.

::: embrs.models.embers

::: embrs.models.perryman_spot
