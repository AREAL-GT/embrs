# Fuel Moisture Models

These modules estimate dead fuel moisture content, which is a key driver of fire behavior. The `DeadFuelMoisture` class implements the Nelson model, a physics-based 1D finite-difference solver that simulates moisture diffusion through cylindrical fuel sticks based on weather conditions. The `IRPGMoistureModel` class implements the Incident Response Pocket Guide (IRPG) method for quick fine dead fuel moisture estimation from temperature, humidity, and site condition correction factors.

The `DeadFuelMoisture` model is used by the core simulation to dynamically update fuel moisture at each time step. The `IRPGMoistureModel` can be used to sample realistic dead fuel moisture values for the [prediction model](fire_prediction.md), where fuel moisture is held constant — by sampling from known weather conditions for the prediction region, the fixed `dead_mf` parameter in [`PredictorParams`](data_classes_documentation.md#embrs.utilities.data_classes.PredictorParams) can be set to a value informed by the IRPG tables rather than an arbitrary default.

::: embrs.models.dead_fuel_moisture

::: embrs.models.irpg_moisture_model
