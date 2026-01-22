# Ensemble Prediction Serialization: Problem & Solution Summary

## The Problem

You correctly identified that reconstructing FireSim for each ensemble member would be too slow. My investigation confirmed:

**FireSim initialization is expensive:**
- Loads and processes map data (elevation, fuel, slope, aspect)
- Initializes thousands of Cell objects
- Sets up wind forecasts via WindNinja
- Estimated cost: **10-30 seconds per FireSim**

**With 50 ensemble members:**
- Serial: 50 × 20s = **16.7 minutes** just for initialization
- Parallel (8 workers): 6.25 batches × 20s = **125 seconds** of overhead

This is unacceptable overhead when predictions themselves take 30-60 seconds.

---

## The Root Cause

FirePredictor contains `self.fire` reference to FireSim, which has non-serializable components:

```python
class FireSim:
    _visualizer: RealTimeVisualizer  # matplotlib figures, tkinter GUI
    logger: Logger                    # ParquetWriter with file handles
    _cell_dict: dict[Cell]           # Cells with weak refs to parent
    embers: Embers                   # Contains callback functions
```

Attempting to pickle this directly fails or produces corrupt objects.

---

## Key Insight

**FirePredictor only needs FireSim during initialization, then becomes independent.**

After `set_params()` and `_catch_up_with_fire()`:
- FirePredictor has its own `_cell_grid` and `_cell_dict` (deep copied)
- FirePredictor has its own `_weather_stream` (deep copied)
- FirePredictor has its own `wind_forecast` (regenerated with uncertainty)
- The `self.fire` reference is only used for:
  - Extracting initial state data (done once)
  - Optional visualization callback (not needed in workers)

**FirePredictor is self-contained after initialization.**

---

## The Solution: Custom Serialization

Instead of serializing the whole FireSim, we:

### 1. Extract Minimal Data (Before Parallelization)

```python
predictor.prepare_for_serialization()
```

This captures:
- `sim_params` (map paths, settings)
- Current fire state (burning/burnt cell polygons)
- Current time and weather state
- Predictor parameters (uncertainty, bias, etc.)

### 2. Implement Custom Pickle Methods

```python
class FirePredictor:
    def __getstate__(self):
        """Pickle only the essential data."""
        return {
            'serialization_data': self._serialization_data,
            'orig_grid': self.orig_grid,
            'orig_dict': self.orig_dict,
        }
        # Excludes: self.fire, visualizer, logger

    def __setstate__(self, state):
        """Reconstruct predictor in worker process."""
        # Rebuild cell grid from scratch
        BaseFireSim.__init__(self, sim_params, burnt_region)

        # Fix weak references
        for cell in self._cell_dict.values():
            cell.set_parent(self)  # Point to self, not original fire

        # Rebuild callbacks
        self.embers = Embers(self, self.get_cell_from_xy)

        # Skip visualizer and logger
        self._visualizer = None
        self.logger = None
```

### 3. Worker Receives Clean Predictor

```python
def _run_ensemble_member_worker(predictor, state_estimate, seed):
    # predictor is fully reconstructed via __setstate__
    # No parent FireSim reference needed
    return predictor.run(fire_estimate=state_estimate)
```

---

## Performance Comparison

### Approach A: Reconstruct FireSim (v1)
```
Per member: Construct FireSim (20s) + Run prediction (45s) = 65s
50 members ÷ 8 workers = 6.25 batches × 65s = 406s total
```

### Approach B: Custom Serialization (v2)
```
One-time: prepare_for_serialization (1s)
Per member: Unpickle & fix refs (3s) + Run prediction (45s) = 48s
50 members ÷ 8 workers = 6.25 batches × 48s = 300s total

Speedup: 26% faster
```

### Approach C: With Shared Memory (v3, future)
```
One-time: prepare + setup shared memory (3s)
Per member: Unpickle light (1s) + Map shared mem (0.5s) + Run (45s) = 46.5s
50 members ÷ 8 workers = 6.25 batches × 46.5s = 290s total

Additional speedup: 3% (more for larger maps)
```

---

## What Gets Serialized

### Included (Serializable):
✅ `sim_params` - Simulation parameters and settings
✅ `map_params` - Map metadata and file paths
✅ `weather_stream` - Weather data (numpy arrays, datetimes)
✅ `predictor_params` - Uncertainty parameters
✅ Fire state polygons - Shapely geometries
✅ Time and index values - Scalars
✅ Cell grid template - Will be reconstructed

### Excluded (Non-Serializable):
❌ `self.fire` - Full FireSim reference
❌ `_visualizer` - matplotlib/tkinter objects
❌ `logger` - File handles and writers
❌ Weak references - Will be rebuilt
❌ Callback functions - Will be recreated

### Size Estimate:
- Serialized predictor: ~5-50 MB (depending on map size)
- Original FireSim: Would fail to pickle entirely

---

## Implementation Complexity

### Required Changes:

1. **Cell class** (`embrs/fire_simulator/cell.py`):
   ```python
   def __getstate__(self): ...  # ~5 lines
   def __setstate__(self): ...  # ~5 lines
   ```

2. **FirePredictor class** (`embrs/tools/fire_predictor.py`):
   ```python
   def prepare_for_serialization(self): ...  # ~40 lines
   def __getstate__(self): ...              # ~10 lines
   def __setstate__(self): ...              # ~50 lines
   ```

3. **Worker function** (module-level):
   ```python
   def _run_ensemble_member_worker(...): ... # ~15 lines
   ```

4. **run_ensemble() method**:
   ```python
   def run_ensemble(self, ...): ...  # ~80 lines
   ```

Total new code: ~200 lines

---

## Risks and Mitigations

### Risk 1: Pickle Protocol Version Incompatibility
**Mitigation**: Use pickle protocol 4+ (Python 3.4+), which is stable

### Risk 2: Deep Copy Overhead
**Mitigation**: `prepare_for_serialization()` does deep copy once, not per member

### Risk 3: Weak Reference Bugs After Deserialization
**Mitigation**: Explicit `cell.set_parent(self)` in `__setstate__`; unit test this

### Risk 4: Missing State Variable
**Mitigation**: Comprehensive testing; include all accessed variables in `_serialization_data`

### Risk 5: Memory Usage (All Predictors in Memory)
**Mitigation**: Process in batches if needed; or streaming aggregation

---

## Alternative Approaches Considered

### Alternative 1: Threading Instead of Multiprocessing
**Pros**: No serialization needed (shared memory)
**Cons**: Python GIL limits parallelism
**Verdict**: Fire simulation is compute-heavy with numpy (releases GIL), so threading could work but multiprocessing is faster

### Alternative 2: Ray or Dask
**Pros**: Better handling of complex objects
**Cons**: Adds major dependency; overkill for this use case
**Verdict**: Not worth the dependency

### Alternative 3: Manual State Extraction
**Pros**: Full control over what's serialized
**Cons**: Fragile; easy to miss variables
**Verdict**: `__getstate__` IS manual extraction, but more Pythonic

### Alternative 4: Separate Initialization Function
**Pros**: Simpler than custom pickle
**Cons**: Still requires expensive FireSim construction
**Verdict**: Defeats purpose of avoiding reconstruction

---

## Testing Strategy

### Unit Tests:
1. Test Cell serialization (pickle → unpickle → verify state)
2. Test FirePredictor serialization without `prepare_for_serialization()` (should fail)
3. Test FirePredictor serialization with preparation (should succeed)
4. Test weak reference reconstruction
5. Test callback reconstruction (Embers)

### Integration Tests:
1. Small ensemble (3 members, verify results match)
2. Large ensemble (50 members, verify completion)
3. Failed member handling (inject error, verify graceful degradation)
4. Reproducibility (same seeds → same results)

### Performance Tests:
1. Profile `prepare_for_serialization()` time
2. Profile pickle/unpickle time per member
3. Profile total ensemble time vs. baseline
4. Memory profiling (ensure no leaks)

---

## Summary

**The custom serialization approach (v2) is the right solution because:**

1. ✅ **25% faster** than reconstructing FireSim
2. ✅ **Pythonic** using standard pickle protocol
3. ✅ **Minimal code changes** (~200 lines)
4. ✅ **Backward compatible** (doesn't affect existing code)
5. ✅ **Extensible** (can add shared memory optimization later)
6. ✅ **Testable** (clear test strategy)

**Recommended implementation order:**
1. Cell serialization (simple, validates approach)
2. FirePredictor serialization (core of solution)
3. Worker function (uses deserialized predictor)
4. run_ensemble() (orchestrates everything)
5. Aggregation (same as v1)
6. Testing and validation

**Estimated time: 1 week of focused development**
