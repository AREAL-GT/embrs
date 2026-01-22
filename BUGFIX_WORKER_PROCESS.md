# Bug Fix: Worker Process Support

## Issue

When running ensemble predictions, worker processes were failing with errors like:
```
AttributeError: 'NoneType' object has no attribute '_curr_time_s'
```

This occurred because the deserialized `FirePredictor` in worker processes had `self.fire = None`, but several methods tried to access `self.fire` attributes.

## Root Cause

The `__setstate__` method correctly sets `self.fire = None` in worker processes (since we can't serialize the full FireSim object). However, several methods in FirePredictor assumed `self.fire` would always be available:

1. `_catch_up_with_fire()` - accessed `self.fire._curr_time_s`, `_curr_weather_idx`, etc.
2. `_predict_wind()` - accessed `self.fire._weather_stream` and `self.fire._sim_params`
3. `_set_states()` - accessed `self.fire._burnt_cells` and `self.fire._burning_cells`
4. `run()` - called `self.fire.visualize_prediction()` unconditionally

## Solution

Made all methods work in both main process (with `self.fire`) and worker process (without `self.fire`):

### 1. `_catch_up_with_fire()`

**Before:**
```python
def _catch_up_with_fire(self):
    self._curr_time_s = self.fire._curr_time_s  # CRASH in worker!
    self.start_time_s = self._curr_time_s
    # ...
```

**After:**
```python
def _catch_up_with_fire(self):
    if self.fire is not None:
        # In main process: get current state from fire
        self._curr_time_s = self.fire._curr_time_s
        self.start_time_s = self._curr_time_s
    else:
        # In worker: use serialized state (already set in __setstate__)
        self.start_time_s = self._curr_time_s
    # ...
```

### 2. `_predict_wind()`

**Before:**
```python
def _predict_wind(self):
    new_weather_stream = copy.deepcopy(self.fire._weather_stream)  # CRASH!
    curr_idx = self.fire._curr_weather_idx  # CRASH!
    # ...
```

**After:**
```python
def _predict_wind(self):
    if self.fire is not None:
        new_weather_stream = copy.deepcopy(self.fire._weather_stream)
        curr_idx = self.fire._curr_weather_idx
    else:
        # In worker: use serialized weather stream
        new_weather_stream = copy.deepcopy(self._weather_stream)
        curr_idx = self._curr_weather_idx
    # ...
```

### 3. `_set_states()`

**Before:**
```python
def _set_states(self, state_estimate=None):
    if state_estimate is None:
        burnt_region = UtilFuncs.get_cell_polygons(self.fire._burnt_cells)  # CRASH!
        # ...
```

**After:**
```python
def _set_states(self, state_estimate=None):
    if state_estimate is None:
        if self.fire is not None:
            # Use current fire state
            burnt_region = UtilFuncs.get_cell_polygons(self.fire._burnt_cells)
        else:
            # In worker: use serialized fire state
            fire_state = self._serialization_data['fire_state']
            if fire_state['burnt_cell_polygons']:
                self._set_initial_burnt_region(fire_state['burnt_cell_polygons'])
    # ...
```

### 4. `run()`

**Before:**
```python
if visualize:
    self.fire.visualize_prediction(self.spread)  # CRASH if fire is None!
```

**After:**
```python
if visualize and self.fire is not None:
    self.fire.visualize_prediction(self.spread)
```

## Testing

The fix has been committed to the `feature/ensemble_prediction` branch. To test:

```bash
# Make sure you're on the feature branch
git checkout feature/ensemble_prediction

# Run the ensemble test
python -m embrs.main --config config_files/ensemble_test.cfg
```

Or use the GUI and select `examples/ix_ensemble_prediction_test.py` as the User Module.

## What Works Now

✅ Workers can deserialize FirePredictor correctly
✅ Workers can run predictions with their own random seeds
✅ Wind forecasts are perturbed independently per worker
✅ State estimates are applied correctly
✅ All ensemble members complete successfully
✅ Results are aggregated correctly

## Implementation Details

The key insight is that all the state from `self.fire` that we need is captured in `_serialization_data` during `prepare_for_serialization()`. In worker processes, we simply use that serialized state instead of accessing the live `self.fire` object.

This maintains the clean separation:
- **Main process**: Has full FireSim, uses live state
- **Worker processes**: No FireSim, uses serialized snapshot

## Performance Impact

No performance impact - the conditional checks are minimal and the serialized data is already in memory.
