# Bug Fix Summary - Ensemble Prediction Feature

This document summarizes all bugs discovered and fixed during the implementation of the ensemble prediction feature.

## Bug #1: Worker Process NoneType Attribute Access

**Commit:** b4e2274
**Documentation:** BUGFIX_WORKER_PROCESS.md

### Problem
Worker processes were failing with `AttributeError: 'NoneType' object has no attribute '_curr_time_s'` because deserialized `FirePredictor` instances had `self.fire = None`, but methods tried to access fire attributes.

### Root Cause
`__setstate__` correctly set `self.fire = None` in workers (can't serialize full FireSim), but these methods assumed `self.fire` would always be available:
- `_catch_up_with_fire()` - accessed `self.fire._curr_time_s`, `_curr_weather_idx`
- `_predict_wind()` - accessed `self.fire._weather_stream`, `self.fire._sim_params`
- `_set_states()` - accessed `self.fire._burnt_cells`, `self.fire._burning_cells`
- `run()` - called `self.fire.visualize_prediction()` unconditionally

### Solution
Added conditional checks: if `self.fire is not None`, use live fire object; else use serialized state from `_serialization_data`.

---

## Bug #2: WindNinja Temp File Race Condition

**Commit:** 8962bd7
**Documentation:** BUGFIX_WINDNINJA_RACE_CONDITION.md

### Problem
Multiple ensemble workers calling `run_windninja()` simultaneously would fail with file access errors because all workers used the same `temp_file_path` directory.

### Root Cause
All workers used global `temp_file_path` to store WindNinja outputs. When Worker B started, it would clear the temp directory and delete Worker A's files, causing Worker A to crash when trying to read them.

### Solution
1. Added `custom_temp_dir` parameter to `run_windninja()`
2. Each worker generates unique temp directory using UUID: `worker_{uuid}`
3. Updated all references to use `work_temp_path` variable
4. Added cleanup logic for worker-specific directories

---

## Bug #3: run_windninja_single() Ignored Custom Temp Path

**Commit:** 0cfd552

### Problem
Even after adding `custom_temp_dir` support, workers were still conflicting because `run_windninja_single()` ignored the task's temp path.

### Root Cause
```python
def run_windninja_single(task: WindNinjaTask):
    output_path = os.path.join(temp_file_path, f"{task.index}")  # BUG!
```

The function used global `temp_file_path` instead of `task.temp_file_path`, so all workers still wrote to the same directory.

### Solution
Changed to use the task's temp path:
```python
def run_windninja_single(task: WindNinjaTask):
    output_path = os.path.join(task.temp_file_path, f"{task.index}")  # FIXED
```

---

## Bug #4: create_forecast_array() UnboundLocalError

**Commit:** 0cfd552

### Problem
Function would crash with `UnboundLocalError: local variable 'forecast' referenced before assignment` if first file didn't exist or if no files existed at all.

### Root Cause
```python
for i in range(num_files):
    if os.path.exists(speed_file) and os.path.exists(direction_file):
        speed_data = np.loadtxt(file, skiprows=6)

        if i == 0:  # Only initializes if i==0 AND files exist
            forecast = np.zeros((num_files, *speed_data.shape, 2))

        forecast[i, :, :, 0] = speed_data  # Crashes if forecast not yet defined

return forecast  # Crashes if forecast never defined
```

If the first file (i=0) didn't exist, `forecast` was never initialized. Even if later files existed, accessing `forecast[1, :, :, 0]` would crash.

### Solution
1. Initialize `forecast = None` before the loop
2. Initialize array on **first valid file** (not just i=0)
3. Raise `FileNotFoundError` if no files were found

```python
forecast = None  # Initialize before loop

for i in range(num_files):
    if os.path.exists(speed_file) and os.path.exists(direction_file):
        speed_data = np.loadtxt(file, skiprows=6)

        if forecast is None:  # Initialize on first valid file
            forecast = np.zeros((num_files, *speed_data.shape, 2))

        forecast[i, :, :, 0] = speed_data

if forecast is None:
    raise FileNotFoundError("No WindNinja output files found...")

return forecast
```

---

## Impact on Ensemble Predictions

### Before Fixes
- ❌ Worker processes crashed on deserialization
- ❌ Workers couldn't access fire attributes
- ❌ Workers conflicted on temp file access
- ❌ WindNinja outputs were deleted by other workers
- ❌ Missing forecast array initialization
- **Result:** Ensemble predictions completely non-functional

### After Fixes
- ✅ Workers deserialize correctly
- ✅ Workers use serialized state when `self.fire is None`
- ✅ Each worker has isolated temp directory
- ✅ No file access conflicts
- ✅ Robust forecast array initialization
- ✅ Clear error messages when WindNinja fails
- **Result:** Ensemble predictions fully functional

---

## Testing

To test all fixes:

```bash
git checkout feature/ensemble_prediction
python -m embrs.main --config config_files/ensemble_test.cfg
```

Or use GUI with `examples/ix_ensemble_prediction_test.py` as User Module.

---

## Files Modified

### Core Implementation
- `embrs/fire_simulator/cell.py` - Cell serialization
- `embrs/tools/fire_predictor.py` - FirePredictor serialization, ensemble execution
- `embrs/models/wind_forecast.py` - WindNinja parallel execution fixes
- `embrs/utilities/data_classes.py` - New data classes

### Test & Documentation
- `examples/ix_ensemble_prediction_test.py` - Test control class
- `BUGFIX_WORKER_PROCESS.md` - Bug #1 documentation
- `BUGFIX_WINDNINJA_RACE_CONDITION.md` - Bug #2 documentation
- `TESTING_ENSEMBLE_PREDICTION.md` - Testing guide
- `BUGFIX_SUMMARY.md` - This file

---

## Lessons Learned

1. **Parallel file I/O requires isolation**: Always use unique directories for parallel workers
2. **Serialization is tricky**: Objects with weak references or parent pointers need special handling
3. **Initialize before loops**: Variables used in loops should be initialized before the loop
4. **Test edge cases**: Missing first file, missing all files, single worker vs multiple workers
5. **Clear error messages**: Raise informative exceptions when operations fail

---

## Performance Impact

All fixes have minimal performance impact:
- Conditional checks are negligible
- UUID generation is ~microseconds
- Directory creation is fast
- Isolated directories actually **improve** performance by eliminating retries/conflicts

---

## Next Steps

The ensemble prediction feature is now fully functional. Future enhancements could include:

1. **Automatic cleanup**: Remove old worker directories on startup
2. **Better error handling**: More specific exceptions for different failure modes
3. **Progress tracking**: Report which ensemble members have completed
4. **Validation**: Check that forecast arrays are complete (no missing time steps)
5. **Logging**: Detailed logs for debugging parallel execution issues

---

## Credits

All bugs were discovered and fixed through systematic testing and code review during the implementation of the ensemble prediction feature on the `feature/ensemble_prediction` branch.
