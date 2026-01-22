# Ensemble Fire Prediction Implementation Plan v2
## Revised Serialization Strategy

## Overview
This document outlines the revised implementation plan for adding ensemble prediction capability to EMBRS FirePredictor. This version addresses the critical serialization challenges identified in v1, avoiding the slow approach of reconstructing FireSim for each prediction while enabling efficient parallel execution.

## Key Insight from Investigation

**FirePredictor only needs FireSim during initialization.** After calling `set_params()` and `_catch_up_with_fire()`, the predictor operates completely independently with its own cell grid, weather stream, and wind forecast. The reference to `self.fire` is only used for:

1. **Initialization** (in `set_params()`):
   - Extract `_sim_params`, `_burning_cells`, `_burnt_cells`
   - Copy current fire state

2. **Synchronization** (in `_catch_up_with_fire()`):
   - Get current time, weather index, weather stream
   - These are copied, not referenced

3. **Visualization** (in `run()` if `visualize=True`):
   - Call `self.fire.visualize_prediction()`
   - Not needed in ensemble members

**After initialization, FirePredictor is self-contained.**

---

## Revised Design: Custom Serialization

### Solution Overview

Instead of reconstructing FireSim (slow) or trying to serialize it with all its non-serializable components, we'll:

1. **Extract initialization data** from FireSim once before parallelization
2. **Implement custom pickle methods** (`__getstate__`, `__setstate__`) for FirePredictor
3. **Serialize only what's needed** for independent prediction runs
4. **Reconstruct non-serializable components** in worker processes
5. **Use shared memory** for large read-only map data (optional optimization)

### Architecture

```
Main Process:
├── FireSim (original, with visualizer/logger)
├── FirePredictor (initialized from FireSim)
└── Extract serializable snapshot
    ├── sim_params (deep copy)
    ├── predictor_params
    ├── Fire state data (time, weather, burning/burnt cells)
    └── Map data (can be shared via shared_memory)

Worker Process (each ensemble member):
├── Receive serializable snapshot
├── Reconstruct FirePredictor from snapshot
│   ├── Rebuild cell grid (fresh cells with new parent refs)
│   ├── Apply perturbed weather/uncertainty
│   └── Skip visualizer/logger creation
└── Run prediction independently
```

---

## Implementation Changes from v1

### Change 1: Add Serialization Support to FirePredictor

**File**: `embrs/tools/fire_predictor.py`

#### Add custom pickle methods:

```python
class FirePredictor(BaseFireSim):
    """Fire prediction model with ensemble capability."""

    def __init__(self, params: PredictorParams, fire: FireSim):
        self.fire = fire
        self.c_size = -1
        self._params = params  # Store original params
        self._serialization_data = None  # Will hold snapshot for pickling
        self.set_params(params)

    def prepare_for_serialization(self):
        """
        Extract all data needed for serialization before parallel execution.

        This method should be called once before spawning workers to capture
        the current state of the parent FireSim in a serializable format.
        """
        # Extract fire state data
        fire_state = {
            'curr_time_s': self.fire._curr_time_s,
            'curr_weather_idx': self.fire._curr_weather_idx,
            'last_weather_update': self.fire._last_weather_update,
            'burning_cell_polygons': UtilFuncs.get_cell_polygons(self.fire._burning_cells),
            'burnt_cell_polygons': UtilFuncs.get_cell_polygons(self.fire._burnt_cells) if self.fire._burnt_cells else None,
        }

        # Deep copy sim_params and weather stream (these are serializable)
        sim_params_copy = copy.deepcopy(self.fire._sim_params)
        weather_stream_copy = copy.deepcopy(self.fire._weather_stream)

        # Store serializable snapshot
        self._serialization_data = {
            'predictor_params': copy.deepcopy(self._params),
            'sim_params': sim_params_copy,
            'weather_stream': weather_stream_copy,
            'fire_state': fire_state,
            'time_horizon_hr': self.time_horizon_hr,
            'wind_uncertainty_factor': self.wind_uncertainty_factor,
            'wind_speed_bias': self.wind_speed_bias,
            'wind_dir_bias': self.wind_dir_bias,
            'ros_bias_factor': self.ros_bias_factor,
            'beta': self.beta,
            'wnd_spd_std': self.wnd_spd_std,
            'wnd_dir_std': self.wnd_dir_std,
            'dead_mf': self.dead_mf,
            'live_mf': self.live_mf,
            'nom_ign_prob': self.nom_ign_prob,
        }

        # Map data arrays (large, read-only)
        # Option 1: Include in serialization (slower but simpler)
        # Option 2: Use shared memory (faster but more complex)
        self._serialization_data['map_arrays'] = {
            'elevation': self.fire._sim_params.map_params.map_data.elevation,
            'slope': self.fire._sim_params.map_params.map_data.slope,
            'aspect': self.fire._sim_params.map_params.map_data.aspect,
            'fuel': self.fire._sim_params.map_params.map_data.fuel,
            # ... other maps
        }

    def __getstate__(self):
        """
        Custom pickle method to serialize only necessary data.

        Excludes non-serializable components like visualizers, loggers,
        and the parent FireSim reference.
        """
        if self._serialization_data is None:
            raise RuntimeError(
                "Must call prepare_for_serialization() before pickling FirePredictor"
            )

        # Return only the serializable snapshot
        # Exclude: self.fire, visualizer, logger, weak references
        state = {
            'serialization_data': self._serialization_data,
            'orig_grid': self.orig_grid,  # Template cell grid
            'orig_dict': self.orig_dict,  # Template cell dict
        }

        return state

    def __setstate__(self, state):
        """
        Custom unpickle method to reconstruct FirePredictor in worker process.

        Rebuilds the predictor without needing the original FireSim reference.
        """
        # Extract serialization data
        data = state['serialization_data']

        # Reconstruct without parent fire reference
        self.fire = None  # No parent fire in worker
        self.c_size = data['predictor_params'].cell_size_m
        self._params = data['predictor_params']
        self._serialization_data = data

        # Restore predictor attributes
        self.time_horizon_hr = data['time_horizon_hr']
        self.wind_uncertainty_factor = data['wind_uncertainty_factor']
        self.wind_speed_bias = data['wind_speed_bias']
        self.wind_dir_bias = data['wind_dir_bias']
        self.ros_bias_factor = data['ros_bias_factor']
        self.beta = data['beta']
        self.wnd_spd_std = data['wnd_spd_std']
        self.wnd_dir_std = data['wnd_dir_std']
        self.dead_mf = data['dead_mf']
        self.live_mf = data['live_mf']
        self.nom_ign_prob = data['nom_ign_prob']

        # Restore cell grids
        self.orig_grid = state['orig_grid']
        self.orig_dict = state['orig_dict']

        # Initialize as BaseFireSim
        # This sets up the grid, cells, and other infrastructure
        sim_params = data['sim_params']
        burnt_region = data['fire_state'].get('burnt_cell_polygons')

        # Call parent init to set up cell grid
        BaseFireSim.__init__(self, sim_params, burnt_region=burnt_region)

        # Fix weak references in cells - point to self instead of original fire
        for cell in self._cell_dict.values():
            cell.set_parent(self)

        # Rebuild Embers with correct callback
        if hasattr(self, 'embers') and self.embers is not None:
            self.embers = Embers(self, self.get_cell_from_xy)

        # Restore time and weather state
        fire_state = data['fire_state']
        self._curr_time_s = fire_state['curr_time_s']
        self._curr_weather_idx = fire_state['curr_weather_idx']
        self._last_weather_update = fire_state['last_weather_update']
        self._weather_stream = data['weather_stream']

        # Note: Don't restore visualizer or logger
        self._visualizer = None
        self.logger = None

    def _create_minimal_fire_reference(self):
        """
        Create a minimal object that satisfies FirePredictor's need for self.fire.

        Used in worker processes where we don't have the full FireSim.
        Only provides the attributes that FirePredictor actually accesses.
        """
        class MinimalFireRef:
            """Minimal fire reference for deserialized predictors."""
            def __init__(self, serialization_data):
                data = serialization_data
                self._sim_params = data['sim_params']
                self._weather_stream = data['weather_stream']
                self._curr_time_s = data['fire_state']['curr_time_s']
                self._curr_weather_idx = data['fire_state']['curr_weather_idx']
                self._last_weather_update = data['fire_state']['last_weather_update']
                self._burning_cells = []  # Empty, will be set from state estimate
                self._burnt_cells = []

            def visualize_prediction(self, spread):
                """No-op visualization in worker."""
                pass

        self.fire = MinimalFireRef(self._serialization_data)
```

### Change 2: Modify run_ensemble() to Use Custom Serialization

**File**: `embrs/tools/fire_predictor.py`

```python
def run_ensemble(
    self,
    state_estimates: List[StateEstimate],
    visualize: bool = False,
    num_workers: Optional[int] = None,
    random_seeds: Optional[List[int]] = None,
    return_individual: bool = False,
    use_shared_memory: bool = False  # New option for optimization
) -> EnsemblePredictionOutput:
    """
    Run ensemble predictions using multiple initial state estimates.

    This implementation uses custom serialization to avoid reconstructing
    FireSim in each worker, significantly improving performance.

    Args:
        state_estimates: List of StateEstimate objects representing
                        different possible initial fire states
        visualize: If True, visualize the mean burn probability
        num_workers: Number of parallel workers. If None, uses cpu_count()
        random_seeds: Optional list of random seeds for reproducibility.
                     Must match length of state_estimates if provided.
        return_individual: If True, include individual predictions in output
        use_shared_memory: If True, use shared memory for map data (faster
                          for large maps, requires Python 3.8+)

    Returns:
        EnsemblePredictionOutput with aggregated probabilistic predictions

    Raises:
        ValueError: If state_estimates is empty or random_seeds length mismatch
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    # Validation
    if not state_estimates:
        raise ValueError("state_estimates cannot be empty")

    if random_seeds is not None and len(random_seeds) != len(state_estimates):
        raise ValueError(
            f"random_seeds length ({len(random_seeds)}) must match "
            f"state_estimates length ({len(state_estimates)})"
        )

    n_ensemble = len(state_estimates)
    num_workers = num_workers or mp.cpu_count()

    print(f"Running ensemble prediction with {n_ensemble} members "
          f"using {num_workers} workers...")

    # CRITICAL: Prepare for serialization
    print("Preparing predictor for serialization...")
    self.prepare_for_serialization()

    # Optional: Set up shared memory for map data
    shared_mem_manager = None
    if use_shared_memory:
        shared_mem_manager = self._setup_shared_memory()

    try:
        # Run predictions in parallel
        predictions = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all jobs
            futures = []
            for i, state_est in enumerate(state_estimates):
                seed = random_seeds[i] if random_seeds else None

                # Submit predictor (will be pickled using __getstate__)
                future = executor.submit(
                    _run_ensemble_member_worker,
                    self,  # Will be serialized via __getstate__
                    state_est,
                    seed
                )
                futures.append(future)

            # Collect results with progress bar
            for future in tqdm(as_completed(futures),
                              total=n_ensemble,
                              desc="Ensemble predictions",
                              unit="member"):
                try:
                    result = future.result()
                    predictions.append(result)
                except Exception as e:
                    print(f"Warning: Ensemble member failed: {e}")
                    # Continue with other members

        if len(predictions) == 0:
            raise RuntimeError("All ensemble members failed")

        if len(predictions) < n_ensemble:
            print(f"Warning: Only {len(predictions)}/{n_ensemble} members completed")

        # Aggregate results
        print("Aggregating ensemble predictions...")
        ensemble_output = _aggregate_ensemble_predictions(predictions)

        # Update with actual ensemble size
        ensemble_output.n_ensemble = len(predictions)

        # Optionally include individual predictions
        if return_individual:
            ensemble_output.individual_predictions = predictions

        # Optionally visualize
        if visualize:
            self._visualize_ensemble(ensemble_output)

        return ensemble_output

    finally:
        # Clean up shared memory if used
        if shared_mem_manager is not None:
            self._cleanup_shared_memory(shared_mem_manager)

def _setup_shared_memory(self):
    """
    Set up shared memory for large map arrays (optional optimization).

    This is more complex but can significantly reduce memory usage and
    serialization overhead for large maps.

    Returns:
        Dictionary with shared memory handles
    """
    try:
        from multiprocessing import shared_memory
    except ImportError:
        print("Warning: shared_memory not available (requires Python 3.8+)")
        return None

    # TODO: Implementation for shared memory
    # This is an advanced optimization and can be added later
    return None

def _cleanup_shared_memory(self, manager):
    """Clean up shared memory resources."""
    if manager is None:
        return
    # TODO: Cleanup implementation
    pass
```

### Change 3: Worker Function for Ensemble Members

**File**: `embrs/tools/fire_predictor.py` (module level)

```python
def _run_ensemble_member_worker(
    predictor: FirePredictor,
    state_estimate: StateEstimate,
    seed: Optional[int] = None
) -> PredictionOutput:
    """
    Worker function for parallel ensemble prediction.

    This function receives a serialized FirePredictor (via __setstate__),
    which has been reconstructed in the worker process without the original
    FireSim reference.

    Args:
        predictor: FirePredictor instance (deserialized in worker)
        state_estimate: Initial state for this ensemble member
        seed: Random seed for reproducibility

    Returns:
        PredictionOutput for this ensemble member
    """
    import numpy as np

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Run prediction
    # Note: visualize=False always in workers
    try:
        output = predictor.run(fire_estimate=state_estimate, visualize=False)
        return output
    except Exception as e:
        # Return error information instead of crashing
        import traceback
        print(f"Prediction failed: {e}")
        print(traceback.format_exc())
        raise
```

---

## Changes to BaseFireSim

### Optional: Add Serialization Support to Cell

**File**: `embrs/fire_simulator/cell.py`

The Cell class contains a weak reference to its parent, which can cause issues during pickling. We can handle this in two ways:

**Option 1: Keep weak references, fix in __setstate__**
- Current approach: FirePredictor.__setstate__ calls `cell.set_parent(self)`
- Pro: Minimal changes to Cell
- Con: Requires explicit parent fixing after deserialization

**Option 2: Add __getstate__/__setstate__ to Cell** (Recommended)

```python
# Add to Cell class in embrs/fire_simulator/cell.py

def __getstate__(self):
    """Custom pickle to handle weak reference to parent."""
    state = self.__dict__.copy()
    # Remove weak reference before pickling
    state['_parent'] = None
    return state

def __setstate__(self, state):
    """Custom unpickle to restore state without parent reference."""
    self.__dict__.update(state)
    # Parent will be set by FirePredictor.__setstate__
    # via cell.set_parent(predictor)
```

This makes Cell serialization automatic and safer.

---

## Performance Comparison

### Original v1 Approach (Reconstruct FireSim each time):
```
Per ensemble member:
├── Construct FireSim: ~10-30 seconds (expensive!)
├── Initialize cell grid: ~5-15 seconds
├── Run prediction: ~30-60 seconds
└── Total: ~45-105 seconds per member

10 members × 60s = 600s = 10 minutes (serial)
10 members ÷ 4 workers = 2.5 batches × 60s = 150s (parallel)
```

### Revised v2 Approach (Custom serialization):
```
One-time setup:
├── prepare_for_serialization(): ~1 second
└── Pickle predictor: ~2-5 seconds per worker

Per ensemble member:
├── Unpickle predictor: ~2-5 seconds
├── Fix parent references: ~1 second
├── Run prediction: ~30-60 seconds
└── Total: ~33-66 seconds per member

10 members × 45s = 450s = 7.5 minutes (serial)
10 members ÷ 4 workers = 2.5 batches × 45s = 112s (parallel)

Speedup: ~25% faster than v1
```

### With Shared Memory Optimization (v3, future):
```
One-time setup:
├── prepare_for_serialization(): ~1 second
├── Setup shared memory: ~2 seconds
└── Pickle predictor (without map data): ~1 second per worker

Per ensemble member:
├── Unpickle predictor: ~1 second (lighter payload)
├── Map to shared memory: ~0.5 seconds
├── Fix parent references: ~1 second
├── Run prediction: ~30-60 seconds
└── Total: ~32.5-62.5 seconds per member

10 members ÷ 4 workers = 2.5 batches × 46s = 115s (parallel)

Additional speedup: ~3% faster (more significant for larger maps)
```

---

## Implementation Steps (Revised)

### Step 1: Add Serialization Support to Cell
- Add `__getstate__` and `__setstate__` methods to Cell class
- Test pickling individual cells

### Step 2: Add Serialization Support to FirePredictor
- Implement `prepare_for_serialization()` method
- Implement `__getstate__` method
- Implement `__setstate__` method
- Test pickling/unpickling predictor

### Step 3: Implement Worker Function
- Create `_run_ensemble_member_worker()` module-level function
- Test running prediction from deserialized predictor

### Step 4: Implement run_ensemble() Method
- Implement main ensemble execution logic
- Add progress reporting with tqdm
- Add error handling for failed members

### Step 5: Implement Aggregation (Same as v1)
- Implement `_aggregate_ensemble_predictions()` function
- Add circular statistics for direction

### Step 6: Add Data Classes (Same as v1)
- Create `EnsemblePredictionOutput` dataclass
- Create `CellStatistics` dataclass

### Step 7: Testing
- Unit test Cell serialization
- Unit test FirePredictor serialization
- Integration test with small ensemble (3 members)
- Integration test with larger ensemble (20 members)
- Performance profiling

### Step 8: Documentation and Examples (Same as v1)
- Add docstrings
- Create example usage
- Update user documentation

### Step 9: Optional: Shared Memory Optimization
- Implement `_setup_shared_memory()` and `_cleanup_shared_memory()`
- Test with large maps
- Profile memory usage improvement

---

## Testing Strategy

### Test Serialization Specifically

**File**: `embrs/test_code/test_serialization.py`

```python
import unittest
import pickle
import copy
from embrs.fire_simulator.fire import FireSim
from embrs.fire_simulator.cell import Cell
from embrs.tools.fire_predictor import FirePredictor
from embrs.utilities.data_classes import PredictorParams, SimParams

class TestSerialization(unittest.TestCase):

    def test_cell_pickle(self):
        """Test that Cell can be pickled and unpickled."""
        # Create a minimal cell
        # ... setup code ...

        # Pickle and unpickle
        pickled = pickle.dumps(cell)
        restored = pickle.loads(pickled)

        # Verify state preserved
        self.assertEqual(cell.x_pos, restored.x_pos)
        self.assertEqual(cell.state, restored.state)

    def test_predictor_pickle_without_prepare(self):
        """Test that pickling without prepare_for_serialization fails."""
        fire = FireSim(sim_params)
        predictor = FirePredictor(pred_params, fire)

        with self.assertRaises(RuntimeError):
            pickle.dumps(predictor)

    def test_predictor_pickle_with_prepare(self):
        """Test that predictor can be pickled after prepare_for_serialization."""
        fire = FireSim(sim_params)
        predictor = FirePredictor(pred_params, fire)

        # Prepare for serialization
        predictor.prepare_for_serialization()

        # Pickle and unpickle
        pickled = pickle.dumps(predictor)
        restored = pickle.loads(pickled)

        # Verify restored predictor is functional
        self.assertIsNotNone(restored)
        self.assertEqual(restored.time_horizon_hr, predictor.time_horizon_hr)

        # Verify can run prediction
        output = restored.run(fire_estimate=None)
        self.assertIsInstance(output, PredictionOutput)

    def test_deserialized_predictor_independence(self):
        """Test that deserialized predictors are independent."""
        fire = FireSim(sim_params)
        predictor = FirePredictor(pred_params, fire)
        predictor.prepare_for_serialization()

        # Create two copies
        restored1 = pickle.loads(pickle.dumps(predictor))
        restored2 = pickle.loads(pickle.dumps(predictor))

        # Run predictions
        output1 = restored1.run()
        output2 = restored2.run()

        # Verify they produce same results (same seed)
        self.assertEqual(len(output1.spread), len(output2.spread))

        # Modify one and verify other unchanged
        # ... verification code ...
```

---

## Migration from v1

This revised plan is **fully backward compatible** with v1. The only changes are:

1. **Added methods** to FirePredictor (doesn't break existing code)
2. **Optional methods** to Cell (backward compatible)
3. **Same API** for `run_ensemble()` (users won't notice difference)

Users of the existing `run()` method are completely unaffected.

---

## Summary of Key Improvements over v1

| Aspect | v1 (Reconstruct FireSim) | v2 (Custom Serialization) |
|--------|--------------------------|---------------------------|
| **Speed** | ~60s per member | ~45s per member (25% faster) |
| **Memory** | High (full FireSim × workers) | Medium (predictor only) |
| **Complexity** | Low (simple but slow) | Medium (custom pickle) |
| **Flexibility** | Limited | High (can add shared memory) |
| **Robustness** | Depends on FireSim init | Independent after prep |

---

## Implementation Checklist

- [ ] Add `__getstate__`/`__setstate__` to Cell class
- [ ] Add `prepare_for_serialization()` to FirePredictor
- [ ] Add `__getstate__`/`__setstate__` to FirePredictor
- [ ] Add `_create_minimal_fire_reference()` to FirePredictor
- [ ] Implement `_run_ensemble_member_worker()` function
- [ ] Implement `run_ensemble()` method
- [ ] Implement `_aggregate_ensemble_predictions()` (same as v1)
- [ ] Create `EnsemblePredictionOutput` and `CellStatistics` dataclasses
- [ ] Write serialization unit tests
- [ ] Write ensemble integration tests
- [ ] Profile performance vs v1 approach
- [ ] Add comprehensive docstrings
- [ ] Create example usage code
- [ ] Update documentation
- [ ] Optional: Add shared memory optimization

## Estimated Implementation Time

- **Cell serialization**: 2 hours
- **FirePredictor serialization**: 6 hours (complex, needs testing)
- **Worker function**: 2 hours
- **run_ensemble() method**: 4 hours
- **Aggregation logic**: 6 hours (same as v1)
- **Data structures**: 2 hours
- **Serialization testing**: 4 hours
- **Integration testing**: 4 hours
- **Example code**: 2 hours
- **Documentation**: 3 hours
- **Debugging/refinement**: 6 hours

**Total**: ~41 hours (approximately 1 week)

**Slightly longer than v1 but much better performance in production.**
