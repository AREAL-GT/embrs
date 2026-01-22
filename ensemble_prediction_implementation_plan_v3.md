# Ensemble Fire Prediction Implementation Plan v3
## Correct Serialization Without BaseFireSim Reconstruction

## Critical Fix from v2

**Problem in v2**: Called `BaseFireSim.__init__()` in `__setstate__`, which reconstructs the entire cell grid from scratch (10-30 seconds per member).

**Solution in v3**: Serialize the already-constructed cell templates (`orig_grid`, `orig_dict`) and manually restore all BaseFireSim attributes without calling `__init__`.

---

## Key Insights

### 1. FirePredictor Already Has Cell Templates

```python
# In FirePredictor.set_params() lines 72-75:
if generate_cell_grid:
    super().__init__(sim_params, burnt_region=burnt_region)  # Creates cells
    self.orig_grid = copy.deepcopy(self._cell_grid)  # Save template
    self.orig_dict = copy.deepcopy(self._cell_dict)  # Save template
```

These templates (`orig_grid`, `orig_dict`) contain fully initialized cells with:
- Terrain data (elevation, slope, aspect)
- Fuel properties
- Neighbor references
- Wind forecasts
- All cell data pre-loaded

### 2. Deep Copy is Already Fast Enough

```python
# In FirePredictor._set_states() lines 114-115:
self._cell_grid = copy.deepcopy(self.orig_grid)  # Already happens!
self._cell_dict = copy.deepcopy(self.orig_dict)
```

**Cost**: ~2-5 seconds (vs. 10-30s for reconstruction)

This is ALREADY part of every prediction run, so it's acceptable overhead.

### 3. BaseFireSim.__init__() Sets ~70 Attributes

We need to manually restore all attributes that `__init__()` would set, broken down by cost:

| Operation | Lines | Cost | Strategy |
|-----------|-------|------|----------|
| Simple assignments | 45-88 | <0.01s | Serialize and restore |
| _parse_sim_params | 241-353 | <0.01s | Serialize sim_params, restore attributes |
| Cell creation loop | 121-217 | 10-30s | **SKIP - use serialized templates** |
| _add_cell_neighbors | 220 | 1-3s | **SKIP - templates already have neighbors** |
| Initial state setup | 222-237 | <1s | Will be done in _set_states() |

---

## Complete Serialization Strategy

### What to Serialize

```python
def prepare_for_serialization(self):
    """Extract all data for serialization."""

    # 1. Already-built cell templates (CRITICAL - avoids reconstruction)
    # These are already stored in self.orig_grid and self.orig_dict

    # 2. All BaseFireSim attributes from __init__ and _parse_sim_params
    self._serialization_data = {
        # From sim_params
        'sim_params': copy.deepcopy(self.fire._sim_params),

        # From predictor params
        'predictor_params': copy.deepcopy(self._params),

        # Fire state at prediction start
        'fire_state': {
            'curr_time_s': self.fire._curr_time_s,
            'curr_weather_idx': self.fire._curr_weather_idx,
            'last_weather_update': self.fire._last_weather_update,
            'burning_cell_polygons': UtilFuncs.get_cell_polygons(self.fire._burning_cells),
            'burnt_cell_polygons': UtilFuncs.get_cell_polygons(self.fire._burnt_cells) if self.fire._burnt_cells else None,
        },

        # Weather stream (deep copied)
        'weather_stream': copy.deepcopy(self.fire._weather_stream),

        # Predictor-specific attributes
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

        # BaseFireSim attributes that won't be in orig_grid/orig_dict
        # (these are class-level, not per-cell)
        'wind_forecast': self.wind_forecast,  # Large array, but necessary
        'flipud_forecast': self.flipud_forecast,
        'wind_xpad': self.wind_xpad,
        'wind_ypad': self.wind_ypad,
        'coarse_elevation': self.coarse_elevation,
    }
```

### What Gets Excluded

```python
# Non-serializable (will be None or rebuilt):
- self.fire                    # Parent reference (replaced with minimal stub)
- self._visualizer             # matplotlib/tkinter (set to None)
- self.logger                  # File handles (set to None)
- self.embers.get_cell_from_xy # Callback function (rebuilt)
```

### Size Estimate

- `sim_params`: ~1 KB (mostly metadata and file paths)
- `orig_grid`: ~5-50 MB (depends on grid size and cell data)
- `orig_dict`: ~5-50 MB (duplicate of grid data)
- `wind_forecast`: ~10-100 MB (depends on time steps and resolution)
- Other attributes: ~1 KB

**Total: ~20-200 MB per ensemble member**

This is large but acceptable for parallelization (modern systems have GB of RAM).

---

## Implementation

### Step 1: Add __getstate__ and __setstate__ to Cell

**File**: `embrs/fire_simulator/cell.py`

```python
def __getstate__(self):
    """
    Custom pickle to handle weak reference to parent.

    The _parent weak reference cannot be pickled, so we exclude it.
    It will be restored by FirePredictor.__setstate__ calling set_parent().
    """
    state = self.__dict__.copy()
    # Remove weak reference - will be restored later
    state['_parent'] = None
    return state

def __setstate__(self, state):
    """
    Restore cell state after unpickling.

    Parent reference is set to None and will be fixed by
    FirePredictor.__setstate__().
    """
    self.__dict__.update(state)
    # Parent will be set later via cell.set_parent(predictor)
```

**Estimated time**: 30 minutes
**Testing**: Unit test pickle/unpickle single cell

---

### Step 2: Add Serialization Methods to FirePredictor

**File**: `embrs/tools/fire_predictor.py`

#### 2a. Modify __init__ to store params

```python
def __init__(self, params: PredictorParams, fire: FireSim):
    self.fire = fire
    self.c_size = -1
    self._params = params  # NEW: Store original params for serialization
    self._serialization_data = None  # Will hold snapshot
    self.set_params(params)
```

#### 2b. Add prepare_for_serialization

```python
def prepare_for_serialization(self):
    """
    Prepare predictor for parallel execution by extracting serializable data.

    Must be called once before pickling the predictor. Captures the current
    state of the parent FireSim and stores it in a serializable format.

    This method should be called in the main process before spawning workers.
    """
    if self.fire is None:
        raise RuntimeError("Cannot prepare predictor without fire reference")

    from embrs.utilities.fire_util import UtilFuncs

    # Extract fire state at prediction start
    fire_state = {
        'curr_time_s': self.fire._curr_time_s,
        'curr_weather_idx': self.fire._curr_weather_idx,
        'last_weather_update': self.fire._last_weather_update,
        'burning_cell_polygons': UtilFuncs.get_cell_polygons(self.fire._burning_cells),
        'burnt_cell_polygons': (UtilFuncs.get_cell_polygons(self.fire._burnt_cells)
                               if self.fire._burnt_cells else None),
    }

    # Deep copy simulation parameters
    sim_params_copy = copy.deepcopy(self.fire._sim_params)
    weather_stream_copy = copy.deepcopy(self.fire._weather_stream)

    # Store all serializable data
    self._serialization_data = {
        'sim_params': sim_params_copy,
        'predictor_params': copy.deepcopy(self._params),
        'fire_state': fire_state,
        'weather_stream': weather_stream_copy,

        # Predictor-specific attributes
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

        # Wind and elevation data
        'wind_forecast': self.wind_forecast,
        'flipud_forecast': self.flipud_forecast,
        'wind_xpad': self.wind_xpad,
        'wind_ypad': self.wind_ypad,
        'coarse_elevation': self.coarse_elevation,
    }
```

#### 2c. Add __getstate__

```python
def __getstate__(self):
    """
    Serialize predictor for parallel execution.

    Returns only the essential data needed to reconstruct the predictor
    in a worker process. Excludes non-serializable components like the
    parent FireSim reference, visualizer, and logger.

    Raises:
        RuntimeError: If prepare_for_serialization() was not called first
    """
    if self._serialization_data is None:
        raise RuntimeError(
            "Must call prepare_for_serialization() before pickling. "
            "This ensures all necessary state is captured from the parent FireSim."
        )

    # Return minimal state
    state = {
        'serialization_data': self._serialization_data,
        'orig_grid': self.orig_grid,  # Template cells (pre-built)
        'orig_dict': self.orig_dict,  # Template cells (pre-built)
        'c_size': self.c_size,
    }

    return state
```

#### 2d. Add __setstate__ (THE CRITICAL PART)

```python
def __setstate__(self, state):
    """
    Reconstruct predictor in worker process WITHOUT calling BaseFireSim.__init__.

    This method manually restores all attributes that BaseFireSim.__init__()
    would have set, but WITHOUT the expensive cell creation loop.

    The key optimization: use pre-built cell templates (orig_grid, orig_dict)
    instead of reconstructing cells from map data.
    """
    import numpy as np
    from embrs.models.perryman_spot import PerrymanSpotting
    from embrs.models.fuel_models import Anderson13, ScottBurgan40

    # Extract serialization data
    data = state['serialization_data']
    sim_params = data['sim_params']

    # =====================================================================
    # Phase 1: Restore FirePredictor-specific attributes
    # =====================================================================
    self.fire = None  # No parent fire in worker
    self.c_size = state['c_size']
    self._params = data['predictor_params']
    self._serialization_data = data

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

    # =====================================================================
    # Phase 2: Restore BaseFireSim attributes (manually, without __init__)
    # =====================================================================

    # From BaseFireSim.__init__ lines 45-88
    self.display_frequency = 300
    self._sim_params = sim_params
    self.burnout_thresh = 0.01
    self.sim_start_w_idx = 0
    self._curr_weather_idx = data['fire_state']['curr_weather_idx']
    self._last_weather_update = data['fire_state']['last_weather_update']
    self.weather_changed = True
    self._curr_time_s = data['fire_state']['curr_time_s']
    self._iters = 0
    self.logger = None  # No logger in worker
    self._visualizer = None  # No visualizer in worker
    self._finished = False

    # Empty containers (will be populated by _set_states)
    self._updated_cells = {}
    self._cell_dict = {}
    self._long_term_retardants = set()
    self._active_water_drops = []
    self._burning_cells = []
    self._new_ignitions = []
    self._burnt_cells = set()
    self._frontier = set()
    self._fire_break_cells = []
    self._active_firelines = {}
    self._new_fire_break_cache = []
    self.starting_ignitions = set()
    self._urban_cells = []
    self._scheduled_spot_fires = {}

    # From _parse_sim_params (lines 252-353)
    map_params = sim_params.map_params
    self._cell_size = sim_params.cell_size
    self._sim_duration = sim_params.duration_s
    self._time_step = sim_params.t_step_s
    self._init_mf = sim_params.init_mf
    self._fuel_moisture_map = getattr(sim_params, 'fuel_moisture_map', {})
    self._fms_has_live = getattr(sim_params, 'fms_has_live', False)
    self._init_live_h_mf = getattr(sim_params, 'live_h_mf', 0.0)
    self._init_live_w_mf = getattr(sim_params, 'live_w_mf', 0.0)
    self._size = map_params.size()
    self._shape = map_params.shape(self._cell_size)
    self._roads = map_params.roads
    self.coarse_elevation = data['coarse_elevation']

    # Fuel class selection
    fbfm_type = map_params.fbfm_type
    if fbfm_type == "Anderson":
        self.FuelClass = Anderson13
    elif fbfm_type == "ScottBurgan":
        self.FuelClass = ScottBurgan40
    else:
        raise ValueError(f"FBFM Type {fbfm_type} not supported")

    # Map data (from lcp_data, but already in sim_params)
    lcp_data = map_params.lcp_data
    self._elevation_map = np.flipud(lcp_data.elevation_map)
    self._slope_map = np.flipud(lcp_data.slope_map)
    self._aspect_map = np.flipud(lcp_data.aspect_map)
    self._fuel_map = np.flipud(lcp_data.fuel_map)
    self._cc_map = np.flipud(lcp_data.canopy_cover_map)
    self._ch_map = np.flipud(lcp_data.canopy_height_map)
    self._cbh_map = np.flipud(lcp_data.canopy_base_height_map)
    self._cbd_map = np.flipud(lcp_data.canopy_bulk_density_map)
    self._data_res = lcp_data.resolution

    # Scenario data
    scenario = map_params.scenario_data
    self._fire_breaks = list(zip(scenario.fire_breaks, scenario.break_widths, scenario.break_ids))
    self.fire_break_dict = {
        id: (fire_break, break_width)
        for fire_break, break_width, id in self._fire_breaks
    }
    self._initial_ignition = scenario.initial_ign

    # Datetime and orientation
    self._start_datetime = sim_params.weather_input.start_datetime
    self._north_dir_deg = map_params.geo_info.north_angle_deg

    # Wind forecast (already computed, just restore)
    self.wind_forecast = data['wind_forecast']
    self.flipud_forecast = data['flipud_forecast']
    self._wind_res = sim_params.weather_input.mesh_resolution
    self.wind_xpad = data['wind_xpad']
    self.wind_ypad = data['wind_ypad']

    # Weather stream
    self._weather_stream = data['weather_stream']
    self.weather_t_step = self._weather_stream.time_step * 60

    # Spotting parameters
    self.model_spotting = sim_params.model_spotting
    self._spot_ign_prob = 0.0
    if self.model_spotting:
        self._canopy_species = sim_params.canopy_species
        self._dbh_cm = sim_params.dbh_cm
        self._spot_ign_prob = sim_params.spot_ign_prob
        self._min_spot_distance = sim_params.min_spot_dist
        self._spot_delay_s = sim_params.spot_delay_s

    # Moisture (prediction model specific)
    self.fmc = 100  # Prediction model default

    # =====================================================================
    # Phase 3: Restore cell templates (CRITICAL - uses pre-built cells)
    # =====================================================================

    # Use the serialized templates instead of reconstructing
    self.orig_grid = state['orig_grid']
    self.orig_dict = state['orig_dict']

    # Initialize cell_grid to the template shape
    self._cell_grid = np.empty(self._shape, dtype=Cell)
    self._grid_width = self._cell_grid.shape[1] - 1
    self._grid_height = self._cell_grid.shape[0] - 1

    # Fix weak references in cells (point to self instead of original fire)
    for cell in self.orig_dict.values():
        cell.set_parent(self)

    # =====================================================================
    # Phase 4: Rebuild lightweight components
    # =====================================================================

    # Rebuild spotting model (PerrymanSpotting for prediction)
    if self.model_spotting:
        limits = (map_params.geo_info.x_max - map_params.geo_info.x_min,
                 map_params.geo_info.y_max - map_params.geo_info.y_min)
        self.embers = PerrymanSpotting(self._spot_delay_s, limits)

    # Calculate x_lim and y_lim if needed
    self.x_lim = map_params.geo_info.x_max - map_params.geo_info.x_min
    self.y_lim = map_params.geo_info.y_max - map_params.geo_info.y_min

    # Note: _set_states() will be called by run() to deep copy the cells
    # and set up the initial burning/burnt regions
```

**Estimated time**: 4 hours (complex, needs careful attribute matching)
**Testing**: Unit test pickle/unpickle predictor

---

### Step 3: Worker Function

**File**: `embrs/tools/fire_predictor.py` (module-level)

```python
def _run_ensemble_member_worker(
    predictor: FirePredictor,
    state_estimate: StateEstimate,
    seed: Optional[int] = None
) -> PredictionOutput:
    """
    Worker function for parallel ensemble prediction.

    Receives a deserialized FirePredictor (via __setstate__) and runs
    a single prediction. The predictor has been reconstructed in this
    worker process without the original FireSim reference.

    Args:
        predictor: Deserialized FirePredictor instance
        state_estimate: Initial fire state for this ensemble member
        seed: Random seed for reproducibility

    Returns:
        PredictionOutput for this ensemble member

    Raises:
        Exception: Any errors during prediction (will be caught by executor)
    """
    import numpy as np

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Run prediction (visualize=False always in workers)
    try:
        output = predictor.run(fire_estimate=state_estimate, visualize=False)
        return output
    except Exception as e:
        # Log error and re-raise (executor will handle)
        import traceback
        print(f"ERROR in ensemble member: {e}")
        print(traceback.format_exc())
        raise
```

**Estimated time**: 30 minutes
**Testing**: Run worker with deserialized predictor

---

### Step 4: run_ensemble() Method

**File**: `embrs/tools/fire_predictor.py`

```python
def run_ensemble(
    self,
    state_estimates: List[StateEstimate],
    visualize: bool = False,
    num_workers: Optional[int] = None,
    random_seeds: Optional[List[int]] = None,
    return_individual: bool = False
) -> EnsemblePredictionOutput:
    """
    Run ensemble predictions using multiple initial state estimates.

    Executes predictions in parallel, each starting from a different
    StateEstimate. Results are aggregated into probabilistic predictions.

    This method uses custom serialization to avoid reconstructing FireSim
    for each ensemble member, providing ~25% speedup over naive approaches.

    Args:
        state_estimates: List of StateEstimate objects representing
                        different possible initial fire states
        visualize: If True, visualize aggregated burn probability
        num_workers: Number of parallel workers (default: cpu_count)
        random_seeds: Optional list of random seeds for reproducibility
        return_individual: If True, include individual predictions in output

    Returns:
        EnsemblePredictionOutput with aggregated predictions

    Raises:
        ValueError: If state_estimates is empty or seeds length mismatch
        RuntimeError: If more than 50% of members fail
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

    print(f"Running ensemble prediction:")
    print(f"  - {n_ensemble} ensemble members")
    print(f"  - {num_workers} parallel workers")

    # CRITICAL: Prepare for serialization
    print("Preparing predictor for serialization...")
    self.prepare_for_serialization()

    # Run predictions in parallel
    predictions = []
    failed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        futures = {}
        for i, state_est in enumerate(state_estimates):
            seed = random_seeds[i] if random_seeds else None

            # Submit (predictor will be pickled via __getstate__)
            future = executor.submit(
                _run_ensemble_member_worker,
                self,
                state_est,
                seed
            )
            futures[future] = i  # Track member index

        # Collect results with progress bar
        for future in tqdm(as_completed(futures),
                          total=n_ensemble,
                          desc="Ensemble predictions",
                          unit="member"):
            member_idx = futures[future]
            try:
                result = future.result()
                predictions.append(result)
            except Exception as e:
                failed_count += 1
                print(f"Warning: Member {member_idx} failed: {e}")

    # Check failure rate
    if len(predictions) == 0:
        raise RuntimeError("All ensemble members failed")

    if failed_count > n_ensemble * 0.5:
        raise RuntimeError(
            f"More than 50% of ensemble members failed "
            f"({failed_count}/{n_ensemble})"
        )

    if failed_count > 0:
        print(f"Completed with {failed_count} failures, "
              f"{len(predictions)} successful members")

    # Aggregate results
    print("Aggregating ensemble predictions...")
    ensemble_output = _aggregate_ensemble_predictions(predictions)
    ensemble_output.n_ensemble = len(predictions)

    # Optionally include individual predictions
    if return_individual:
        ensemble_output.individual_predictions = predictions

    # Optionally visualize
    if visualize:
        self._visualize_ensemble(ensemble_output)

    return ensemble_output
```

**Estimated time**: 2 hours
**Testing**: Integration test with small ensemble

---

## Performance Analysis

### Timing Breakdown (50 members, 8 workers)

**v1 (Reconstruct FireSim):**
```
Per member:
  - Construct FireSim: 20s
  - Run prediction: 45s
  Total: 65s

50 members ÷ 8 workers = 6.25 batches
6.25 × 65s = 406 seconds
```

**v2 (Flawed - calls BaseFireSim.__init__):**
```
Per member:
  - Unpickle: 2s
  - BaseFireSim.__init__: 20s  # STILL EXPENSIVE!
  - Run prediction: 45s
  Total: 67s

50 members ÷ 8 workers = 6.25 batches
6.25 × 67s = 419 seconds  # WORSE than v1!
```

**v3 (Correct - use templates):**
```
One-time setup:
  - prepare_for_serialization: 1s

Per member:
  - Unpickle predictor: 2s
  - Fix parent references: 0.5s
  - Run prediction (includes deep copy in _set_states): 47s
  Total: 49.5s

50 members ÷ 8 workers = 6.25 batches
6.25 × 49.5s = 309 seconds

Speedup vs v1: 24%
Speedup vs v2: 26%
```

### Memory Usage

**Per worker:**
- Pickled predictor: ~50 MB (templates + wind data)
- Deep copied cells during run: ~100 MB
- Prediction outputs: ~10 MB
- Total: ~160 MB per worker

**8 workers = ~1.3 GB**

Acceptable on modern systems (4-16 GB RAM typical).

---

## Testing Strategy

### Phase 1: Unit Tests

```python
# Test 1: Cell serialization
def test_cell_pickle():
    cell = create_test_cell()
    restored = pickle.loads(pickle.dumps(cell))
    assert restored.x_pos == cell.x_pos
    assert restored._parent is None  # Weak ref excluded

# Test 2: Predictor serialization requires prepare
def test_predictor_pickle_without_prepare():
    predictor = create_test_predictor()
    with pytest.raises(RuntimeError):
        pickle.dumps(predictor)

# Test 3: Predictor serialization with prepare
def test_predictor_pickle_with_prepare():
    predictor = create_test_predictor()
    predictor.prepare_for_serialization()
    restored = pickle.loads(pickle.dumps(predictor))
    assert restored.time_horizon_hr == predictor.time_horizon_hr
    assert restored.fire is None  # No parent in worker

# Test 4: Restored predictor can run
def test_restored_predictor_runs():
    predictor = create_test_predictor()
    predictor.prepare_for_serialization()
    restored = pickle.loads(pickle.dumps(predictor))
    output = restored.run()
    assert isinstance(output, PredictionOutput)
    assert len(output.spread) > 0
```

### Phase 2: Integration Tests

```python
# Test 5: Small ensemble
def test_small_ensemble():
    predictor = create_test_predictor()
    states = [create_state_estimate() for _ in range(3)]
    result = predictor.run_ensemble(states, num_workers=2)
    assert result.n_ensemble == 3
    assert len(result.burn_probability) > 0

# Test 6: Reproducibility with seeds
def test_ensemble_reproducibility():
    predictor = create_test_predictor()
    states = [create_state_estimate() for _ in range(5)]
    seeds = [42, 43, 44, 45, 46]

    result1 = predictor.run_ensemble(states, random_seeds=seeds)
    result2 = predictor.run_ensemble(states, random_seeds=seeds)

    # Should get identical results
    assert result1.burn_probability == result2.burn_probability

# Test 7: Failed member handling
def test_ensemble_with_failures():
    predictor = create_test_predictor()
    # Create one invalid state estimate
    states = [create_valid_state() for _ in range(5)]
    states[2] = create_invalid_state()

    result = predictor.run_ensemble(states)
    assert result.n_ensemble == 4  # One failed
```

### Phase 3: Performance Tests

```python
# Test 8: Profile serialization time
def test_serialization_speed():
    predictor = create_large_predictor()

    start = time.time()
    predictor.prepare_for_serialization()
    prep_time = time.time() - start

    start = time.time()
    pickled = pickle.dumps(predictor)
    pickle_time = time.time() - start

    start = time.time()
    restored = pickle.loads(pickled)
    unpickle_time = time.time() - start

    print(f"Prepare: {prep_time:.2f}s")
    print(f"Pickle: {pickle_time:.2f}s")
    print(f"Unpickle: {unpickle_time:.2f}s")

    assert prep_time < 2.0  # Should be fast
    assert pickle_time < 5.0
    assert unpickle_time < 5.0

# Test 9: Profile ensemble execution
def test_ensemble_performance():
    predictor = create_test_predictor()
    states = [create_state_estimate() for _ in range(10)]

    start = time.time()
    result = predictor.run_ensemble(states, num_workers=4)
    total_time = time.time() - start

    per_member = total_time / 10
    print(f"Total: {total_time:.1f}s, Per member: {per_member:.1f}s")

    # Should be faster than serial
    assert total_time < 10 * 60  # Less than 10min for 10 members
```

---

## Implementation Checklist

- [ ] Add `__getstate__`/`__setstate__` to Cell class
- [ ] Test Cell serialization (unit test)
- [ ] Modify FirePredictor.__init__ to store `_params`
- [ ] Add `prepare_for_serialization()` to FirePredictor
- [ ] Add `__getstate__` to FirePredictor
- [ ] Add `__setstate__` to FirePredictor (CRITICAL - verify all attributes)
- [ ] Test predictor serialization without prepare (should fail)
- [ ] Test predictor serialization with prepare (should succeed)
- [ ] Test deserialized predictor can run predictions
- [ ] Implement `_run_ensemble_member_worker()` function
- [ ] Test worker function with deserialized predictor
- [ ] Implement `run_ensemble()` method
- [ ] Test small ensemble (3-5 members)
- [ ] Implement `_aggregate_ensemble_predictions()` (from v1)
- [ ] Implement circular statistics for directions
- [ ] Create `EnsemblePredictionOutput` and `CellStatistics` dataclasses
- [ ] Test ensemble aggregation logic
- [ ] Test reproducibility with random seeds
- [ ] Test failed member handling
- [ ] Profile serialization performance
- [ ] Profile ensemble execution performance
- [ ] Add comprehensive docstrings
- [ ] Create example usage code
- [ ] Update user documentation

---

## Estimated Implementation Time

- **Cell serialization**: 1 hour
- **FirePredictor serialization**: 6 hours (critical, needs careful verification)
- **Worker function**: 1 hour
- **run_ensemble() method**: 3 hours
- **Aggregation logic**: 6 hours (from v1, unchanged)
- **Data structures**: 2 hours
- **Unit testing**: 4 hours
- **Integration testing**: 4 hours
- **Performance testing**: 3 hours
- **Example code**: 2 hours
- **Documentation**: 3 hours
- **Debugging/refinement**: 5 hours

**Total**: ~40 hours (approximately 1 week)

---

## Critical Success Factors

1. **Verify all BaseFireSim attributes** are restored in `__setstate__`
   - Use `attribute_analysis.md` as checklist
   - Compare with BaseFireSim.__init__() line-by-line

2. **Test parent references** are fixed correctly
   - Every cell must have `cell._parent` pointing to predictor
   - Embers callback must use predictor's `get_cell_from_xy`

3. **Confirm deep copy happens in _set_states()**, not __setstate__
   - __setstate__ just restores templates
   - run() calls _set_states() which does deep copy

4. **Profile actual performance** with realistic grid sizes
   - Small grid (100×100): ~3s unpickle
   - Medium grid (500×500): ~15s unpickle
   - Large grid (1000×1000): ~60s unpickle

---

## Summary

**v3 is the correct approach because:**

✅ Avoids calling BaseFireSim.__init__() entirely
✅ Uses pre-built cell templates (orig_grid, orig_dict)
✅ Deep copy already happens in _set_states() - no extra overhead
✅ 24% faster than reconstructing FireSim
✅ Pythonic using standard pickle protocol
✅ ~200 lines of code for complete solution

**Next step: Implement and test Cell serialization first (lowest risk, validates approach)**
