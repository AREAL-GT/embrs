# Testing Ensemble Prediction - Quick Start Guide

## Overview

The test control class `examples/ix_ensemble_prediction_test.py` provides a complete example of using the ensemble prediction feature in a live EMBRS simulation.

## How to Use

### Method 1: GUI-Based Simulation (Recommended)

1. **Start EMBRS simulation with GUI**:
   ```bash
   python -m embrs.main
   ```

2. **Configure simulation**:
   - Select a map
   - Select weather data
   - Set simulation parameters
   - **Important**: Select `examples/ix_ensemble_prediction_test.py` as the "User Module"

3. **Run simulation**:
   - Start the simulation
   - After 30 minutes of simulation time, the ensemble prediction will automatically trigger
   - Watch the console output for detailed results

### Method 2: Configuration File

1. **Create/modify a .cfg file** to include the user code:
   ```ini
   [Simulation]
   t_step_s = 5
   cell_size_m = 30
   visualize = True
   # ... other params ...

   # Add user code reference
   user_code_path = examples/ix_ensemble_prediction_test.py
   user_code_class = EnsemblePredictionTest
   ```

2. **Run with config**:
   ```bash
   python -m embrs.main --config path/to/config.cfg
   ```

## What the Test Does

### 1. Initialization
- Creates predictor with 2-hour forecast horizon
- Configures 5 ensemble members
- Sets up wind uncertainty parameters

### 2. State Estimate Generation (at 30 min mark)
The test creates 5 different initial states:
- **Member 0**: Exact current fire state (baseline)
- **Member 1**: Slightly expanded burning boundary (+50% cell size)
- **Member 2**: More expanded boundary (+100% cell size)
- **Member 3**: Slightly contracted boundary (-30% cell size)
- **Member 4**: More contracted boundary (-60% cell size)

This simulates uncertainty in the actual fire boundary.

### 3. Parallel Execution
- Runs all 5 predictions in parallel (up to 4 workers)
- Uses random seeds for reproducibility
- Each member gets slightly different wind forecasts

### 4. Results Analysis
Displays comprehensive statistics:
- **Burn probability maps**: Fraction of members predicting fire at each location
- **Flame length statistics**: Mean, std, min, max across ensemble
- **Rate of spread statistics**: Distribution of ROS values
- **Crown fire frequency**: How often crown fire occurs
- **High-risk areas**: Cells with ≥90% burn probability

### 5. Output
- Console output with detailed analysis
- Saved summary file: `ensemble_results_YYYYMMDD_HHMMSS.txt`

## Customization

You can modify the test behavior by editing these attributes in `__init__`:

```python
self.trigger_time_hr = 0.5       # When to run (hours)
self.n_ensemble = 5              # Number of members
self.prediction_horizon_hr = 2.0  # Forecast horizon
self.use_seeds = True            # Reproducibility
```

Or modify the predictor parameters:

```python
params = PredictorParams(
    time_horizon_hr=3.0,          # Longer forecast
    wind_uncertainty_factor=0.8,  # More uncertainty
    max_wind_speed_bias=5.0,      # Larger wind variations
    # ... etc
)
```

## Example Output

```
============================================================
TRIGGER: Fire time = 0.50 hours
Starting ensemble prediction...
============================================================

Creating FirePredictor...
✓ FirePredictor created

Generating 5 state estimates...
✓ Generated 5 state estimates

Using random seeds: [42, 43, 44, 45, 46]

Running ensemble prediction in parallel...
Preparing predictor for serialization...
Running ensemble prediction:
  - 5 ensemble members
  - 4 parallel workers
Ensemble predictions: 100%|███████████| 5/5 [02:30<00:00, 30.1s/member]
Aggregating ensemble predictions...
✓ Ensemble prediction complete!

============================================================
ENSEMBLE PREDICTION RESULTS
============================================================

Ensemble size: 5 members
Successful members: 5
Prediction time steps: 1440
  First: 1800s (0.50h)
  Last:  9000s (2.50h)

--- Burn Probability Analysis ---

Final time (2.50h): 2453 cells with fire
  High probability (≥80%): 1876 cells
  Medium probability (50-80%): 342 cells
  Low probability (<50%): 235 cells

  Probability statistics:
    Mean: 82.34%
    Median: 100.00%
    Std: 28.45%

--- Flame Length Statistics ---
Cells with flame data: 2453
  Mean flame length across cells:
    Average: 2.34 m
    Range: 0.45 - 8.92 m
  Uncertainty (std) across cells:
    Average: 0.67 m
    Max: 2.31 m

--- Rate of Spread Statistics ---
  Mean ROS across cells:
    Average: 0.0023 m/s
    Range: 0.0001 - 0.0156 m/s

--- Crown Fire Analysis ---
Cells with crown fire in any member: 342
  Crown fire frequency:
    Mean: 64.23%
    Max: 100.00%

--- Individual Member Comparison ---
  Member 0: 2401 cells burning at final time
  Member 1: 2508 cells burning at final time
  Member 2: 2587 cells burning at final time
  Member 3: 2298 cells burning at final time
  Member 4: 2189 cells burning at final time

--- High-Risk Areas (≥90% probability) ---

  Time 2.33h: 1654 high-risk cells
    (12450, 8730): 100.0%
    (12480, 8745): 100.0%
    (12510, 8730): 100.0%
    (12540, 8745): 100.0%
    (12570, 8730): 100.0%

============================================================
Analysis complete!
============================================================

Results summary saved to: ensemble_results_20260122_143052.txt
```

## Interpreting Results

### Burn Probability
- **100%**: All ensemble members predicted fire → Very confident
- **80-99%**: Most members predicted fire → High confidence
- **50-79%**: Mixed predictions → Moderate confidence
- **<50%**: Most members didn't predict fire → Low confidence

### Flame Length Uncertainty
- **Low std (<0.5m)**: Consistent predictions across members
- **High std (>1.0m)**: Large variation, high uncertainty

### Crown Fire Frequency
- **>80%**: Crown fire very likely
- **50-80%**: Crown fire possible
- **<50%**: Crown fire unlikely

## Troubleshooting

### "All ensemble members failed"
- Check that the fire is actively burning when triggered
- Verify map data is loaded correctly
- Check console for detailed error messages

### "ModuleNotFoundError: No module named 'rasterio'"
- Install EMBRS dependencies: `pip install -e .`

### Ensemble runs but produces no results
- Increase `trigger_time_hr` (fire may not have spread enough)
- Check that `fire.burning_cells` is not empty at trigger time

### Out of memory errors
- Reduce `n_ensemble` (try 3 instead of 5)
- Reduce `prediction_horizon_hr` (try 1.0 instead of 2.0)
- Use fewer workers (try 2 instead of 4)

## Performance Tips

### For Faster Testing
```python
self.n_ensemble = 3              # Fewer members
self.prediction_horizon_hr = 1.0  # Shorter horizon
# Use up to 2 workers in run_ensemble()
```

### For Production Runs
```python
self.n_ensemble = 20             # More members for better statistics
self.prediction_horizon_hr = 4.0  # Longer horizon
# Use up to 8 workers for large ensembles
```

## Next Steps

1. **Modify state estimate generation**: Create different perturbation strategies
2. **Add custom analysis**: Extract specific metrics from results
3. **Visualize results**: Plot burn probability heat maps
4. **Compare with actual fire**: Validate predictions against continued simulation
5. **Tune uncertainty parameters**: Adjust wind/ROS bias to match observed variation

## Support

For issues or questions:
- Check `ENSEMBLE_PREDICTION_README.md` for API details
- Review implementation plans in the repository
- File an issue on GitHub
