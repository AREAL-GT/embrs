# Fire Prediction

EMBRS includes a fire prediction tool that runs a simplified version of the core fire model to predict future fire propagation over a fixed time horizon. The prediction tool supports both single predictions and ensemble predictions with parallel execution to quantify forecast uncertainty.

The prediction model differs from the core simulation in the following ways:

- **Wind uncertainty**: AR(1) autoregressive noise is added to the wind forecast to model erroneous wind forecasting. Separate bias terms control wind speed, wind direction, and rate of spread (ROS).
- **Fixed fuel moisture**: Dead and live fuel moisture values are held constant throughout the prediction rather than being dynamically updated. These values can optionally be sampled using the [`IRPGMoistureModel`](models_moisture.md#embrs.models.irpg_moisture_model.IRPGMoistureModel), which estimates fine dead fuel moisture from temperature, humidity, and stochastic site condition correction factors. This allows the fixed moisture to reflect known conditions across the prediction region rather than using an arbitrary default.
- **Statistical spotting model**: The prediction model uses the [`PerrymanSpotting`](models_spotting.md#embrs.models.perryman_spot.PerrymanSpotting) model, which samples firebrand landing locations from probability distributions. The core simulation instead uses the physics-based [`Embers`](models_spotting.md#embrs.models.embers.Embers) model (Albini 1979), which tracks individual firebrand trajectories through the wind field.
- **No fire acceleration modeled**: Cells immediately burn at steady-state ROS rather than accelerating over time.

For a full working example, see [the prediction example](example_index.md#iv_fire_prediction_modelpy) and the ensemble example in `examples/ix_ensemble_prediction_test.py`.

## Configuration

Before creating a predictor, configure it with a [`PredictorParams`](data_classes_documentation.md#embrs.utilities.data_classes.PredictorParams) dataclass. Key fields include:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_horizon_hr` | float | 2.0 | How far ahead to predict (hours) |
| `time_step_s` | int | 30 | Prediction time step (seconds) |
| `cell_size_m` | float | 30 | Prediction cell size (meters) |
| `dead_mf` | float | 0.08 | Dead fuel moisture fraction |
| `live_mf` | float | 0.30 | Live fuel moisture fraction |
| `wind_speed_bias` | float | 0 | Wind speed bias (-1 to 1, scaled by `max_wind_speed_bias`) |
| `wind_dir_bias` | float | 0 | Wind direction bias (-1 to 1, scaled by `max_wind_dir_bias`) |
| `ros_bias` | float | 0 | ROS bias (-0.5 to 0.5, applied as multiplicative factor) |
| `wind_uncertainty_factor` | float | 0 | AR(1) noise scaling (0 = no noise, 1 = full noise) |
| `model_spotting` | bool | False | Whether to model ember spotting |

```python
from embrs.utilities.data_classes import PredictorParams

# Default parameters — 2-hour prediction with no bias
params = PredictorParams(time_horizon_hr=2)

# Custom parameters with larger cells and higher moisture
params = PredictorParams(
    time_horizon_hr=3,
    time_step_s=20,
    cell_size_m=45,
    dead_mf=0.10,
    live_mf=0.30,
    wind_speed_bias=0.0,
    wind_dir_bias=0.0,
    ros_bias=0.0,
    wind_uncertainty_factor=0.5,  # moderate wind noise
)
```

## Creating a FirePredictor

The [`FirePredictor`](tools_documentation.md#embrs.tools.fire_predictor.FirePredictor) constructor takes a `PredictorParams` and a reference to the running `FireSim`:

```python
from embrs.tools.fire_predictor import FirePredictor

predictor = FirePredictor(params, fire)
```

The predictor synchronizes with the `FireSim` state before each prediction run. Changing `cell_size_m` triggers an expensive grid regeneration, so avoid changing it between runs when possible.

## Running a Single Prediction

Call `run()` to execute a single forward prediction from the current fire state:

```python
result = predictor.run()

# Or visualize the prediction on the fire display
result = predictor.run(visualize=True)
```

`run()` returns a [`PredictionOutput`](data_classes_documentation.md#embrs.utilities.data_classes.PredictionOutput) containing the predicted fire spread and fire behavior metrics.

You can also pass a [`StateEstimate`](data_classes_documentation.md#embrs.utilities.data_classes.StateEstimate) to predict from a different initial state (see [Ensemble Predictions](#ensemble-predictions) below).

!!! note
    Each call to `run()` re-synchronizes with the parent `FireSim`, so successive calls on the same predictor will reflect the fire's progression. You do not need to create a new `FirePredictor` between calls.

## Interpreting Single Prediction Results

`PredictionOutput` contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `spread` | `dict[float, list[tuple]]` | Maps time (seconds) to list of (x, y) positions where fire arrives |
| `flame_len_m` | `dict[tuple, float]` | Maps (x, y) to flame length in meters |
| `fli_kw_m` | `dict[tuple, float]` | Maps (x, y) to fireline intensity in kW/m |
| `ros_ms` | `dict[tuple, float]` | Maps (x, y) to rate of spread in m/s |
| `spread_dir` | `dict[tuple, float]` | Maps (x, y) to spread direction in radians |
| `crown_fire` | `dict[tuple, CrownStatus]` | Maps (x, y) to crown fire status |
| `hold_probs` | `dict[tuple, float]` | Maps (x, y) to fireline hold probability (0-1) |
| `breaches` | `dict[tuple, bool]` | Maps (x, y) to whether fire breached the fireline |
| `active_fire_front` | `dict[float, list[tuple]]` | Maps time to (x, y) positions of cells currently burning |
| `burnt_spread` | `dict[float, list[tuple]]` | Maps time to (x, y) positions of fully burnt cells |

**Example** — iterating over predicted fire locations by time step:

```python
result = predictor.run()

for time_s, positions in result.spread.items():
    print(f"Time {time_s:.0f}s: {len(positions)} new ignitions")
```

## Ensemble Predictions

Ensemble predictions run multiple predictions in parallel with different initial state estimates and aggregate the results into probabilistic burn maps and fire behavior statistics.

### StateEstimate

Each ensemble member starts from a [`StateEstimate`](data_classes_documentation.md#embrs.utilities.data_classes.StateEstimate) describing the assumed fire state:

| Field | Type | Description |
|-------|------|-------------|
| `burnt_polys` | `list[Polygon]` | Polygons of burnt area |
| `burning_polys` | `list[Polygon]` | Polygons of actively burning area |
| `start_time_s` | `float` (optional) | Start time in seconds from simulation start. If `None`, uses current fire time. |

```python
from embrs.utilities.data_classes import StateEstimate
from embrs.utilities.fire_util import UtilFuncs

# Create a state estimate from the current fire state
burnt = UtilFuncs.get_cell_polygons(fire._burnt_cells) if fire._burnt_cells else []
burning = UtilFuncs.get_cell_polygons(fire.burning_cells)

estimate = StateEstimate(
    burnt_polys=burnt,
    burning_polys=burning
)
```

To create variation across ensemble members, perturb the burning region polygons (e.g., by buffering them inward or outward).

### Running an Ensemble

Call `run_ensemble()` with a list of `StateEstimate` objects:

```python
from embrs.utilities.data_classes import StateEstimate

# Create multiple state estimates (e.g., exact + perturbed)
state_estimates = [estimate_0, estimate_1, estimate_2, ...]

result = predictor.run_ensemble(
    state_estimates=state_estimates,
    num_workers=4,              # parallel workers
    visualize=True,             # show burn probability on display
    return_individual=True,     # include per-member results
)
```

Key `run_ensemble()` parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state_estimates` | `list[StateEstimate]` | (required) | One per ensemble member |
| `num_workers` | `int` | `cpu_count()` | Number of parallel workers |
| `random_seeds` | `list[int]` | `None` | Per-member seeds for reproducibility |
| `return_individual` | `bool` | `False` | Include individual `PredictionOutput` objects |
| `predictor_params_list` | `list[PredictorParams]` | `None` | Per-member parameters |
| `vary_wind_per_member` | `bool` | `False` | Generate separate wind forecasts per member |
| `forecast_pool` | `ForecastPool` | `None` | Pre-computed wind forecasts (see below) |

### Interpreting Ensemble Results

`run_ensemble()` returns an [`EnsemblePredictionOutput`](data_classes_documentation.md#embrs.utilities.data_classes.EnsemblePredictionOutput) with aggregated statistics:

| Field | Type | Description |
|-------|------|-------------|
| `n_ensemble` | `int` | Number of successful ensemble members |
| `burn_probability` | `dict[float, dict[tuple, float]]` | Cumulative burn probability by time and (x, y) |
| `flame_len_m_stats` | `dict[tuple, CellStatistics]` | Flame length statistics per cell |
| `fli_kw_m_stats` | `dict[tuple, CellStatistics]` | Fireline intensity statistics per cell |
| `ros_ms_stats` | `dict[tuple, CellStatistics]` | Rate of spread statistics per cell |
| `spread_dir_stats` | `dict[tuple, dict]` | Circular mean direction and dispersion per cell |
| `crown_fire_frequency` | `dict[tuple, float]` | Fraction of members with crown fire per cell |
| `hold_prob_stats` | `dict[tuple, CellStatistics]` | Fireline hold probability statistics |
| `breach_frequency` | `dict[tuple, float]` | Fraction of members breaching each fireline cell |
| `individual_predictions` | `list[PredictionOutput]` | Per-member results (if `return_individual=True`) |

Each [`CellStatistics`](data_classes_documentation.md#embrs.utilities.data_classes.CellStatistics) contains `mean`, `std`, `min`, `max`, and `count`.

**Example** — finding high-probability burn cells:

```python
result = predictor.run_ensemble(state_estimates=estimates)

# Get burn probabilities at the final time step
final_time = max(result.burn_probability.keys())
probs = result.burn_probability[final_time]

high_risk = {loc: p for loc, p in probs.items() if p >= 0.8}
print(f"{len(high_risk)} cells with >=80% burn probability")
```

## Forecast Pools

For scenarios that require running multiple ensembles (e.g., rollout planning), you can pre-compute a pool of perturbed wind forecasts and reuse them across ensemble runs. This avoids redundant WindNinja calls.

```python
# Generate a pool of 30 perturbed wind forecasts
pool = predictor.generate_forecast_pool(
    n_forecasts=30,
    num_workers=4,
    random_seed=42  # for reproducibility
)

# Use the pool in an ensemble prediction
result = predictor.run_ensemble(
    state_estimates=estimates,
    forecast_pool=pool
)

# The pool can be reused across multiple ensemble runs
result2 = predictor.run_ensemble(
    state_estimates=other_estimates,
    forecast_pool=pool,
    forecast_indices=[0, 1, 2, 3, 4]  # explicit forecast assignment
)

# Release pool memory when done
pool.close()
```

See [`ForecastPool`](tools_documentation.md#embrs.tools.forecast_pool.ForecastPool) and [`ForecastPoolManager`](tools_documentation.md#embrs.tools.forecast_pool.ForecastPoolManager) for full API details.

## Cleanup

Call `cleanup()` when the predictor is no longer needed to release memory, including any active forecast pools:

```python
predictor.cleanup()
```

For full API documentation of all prediction classes and methods, see the [Tools Reference](tools_documentation.md) and [Data Classes](data_classes_documentation.md) pages.
