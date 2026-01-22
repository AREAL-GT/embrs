# Ensemble Fire Prediction Implementation Plan

## Overview
This document outlines the implementation plan for adding ensemble prediction capability to the EMBRS FirePredictor. The new feature will run multiple predictions in parallel using different StateEstimates, then aggregate results to produce probabilistic fire spread predictions.

## Feature Requirements

### Input
- List of `StateEstimate` objects, each representing a possible initial fire state
- Same `PredictorParams` applied to all predictions
- Ability to run predictions in parallel for performance

### Output
- Aggregated prediction showing for each time step and cell location:
  - **Burn probability**: Fraction of ensemble members predicting fire at that cell
  - **Statistical metrics**: Mean, std, min, max for flame length, ROS, fireline intensity
  - **Crown fire frequency**: Fraction of predictions with crown fire
  - **Fireline statistics**: Mean hold probability and breach frequency

### Performance
- Parallel execution to reduce wall-clock time
- Memory-efficient aggregation to handle large ensembles

---

## Design Decisions

### 1. API Design

**Option A: New method on FirePredictor class** (Recommended)
```python
# Add to FirePredictor class
def run_ensemble(
    self,
    state_estimates: List[StateEstimate],
    visualize: bool = False,
    num_workers: Optional[int] = None
) -> EnsemblePredictionOutput:
    """
    Run ensemble predictions using multiple initial state estimates.

    Args:
        state_estimates: List of StateEstimate objects for ensemble members
        visualize: Whether to visualize the ensemble mean prediction
        num_workers: Number of parallel workers (None = cpu_count)

    Returns:
        EnsemblePredictionOutput containing probabilistic predictions
    """
```

**Rationale**:
- Reuses existing FirePredictor initialization and parameters
- Natural extension of the existing `run()` method
- Avoids duplicating predictor setup logic

**Option B: Separate EnsemblePredictor class**
- More complex, requires duplicating predictor setup
- Better separation of concerns but unnecessary for this use case
- Rejected in favor of Option A

### 2. Parallel Execution Strategy

**Recommended: `concurrent.futures.ProcessPoolExecutor`**

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# In run_ensemble():
num_workers = num_workers or mp.cpu_count()

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = []
    for state_est in state_estimates:
        future = executor.submit(_run_single_prediction, self, state_est)
        futures.append(future)

    results = []
    for future in as_completed(futures):
        results.append(future.result())
```

**Challenges**:
- FirePredictor contains a reference to FireSim which may not be easily serializable
- Need to ensure all predictor state is serializable or recreated in worker processes

**Alternative: Threading** (Fallback)
- Use `ThreadPoolExecutor` if serialization issues are severe
- May be limited by Python GIL, but predictions are compute-intensive so should release GIL often
- Simpler to implement since shared memory

**Serialization Solution**:
Create a helper function that reconstructs the predictor in each worker:

```python
def _run_single_prediction(params_dict: dict, state_est: StateEstimate) -> PredictionOutput:
    """
    Worker function for parallel prediction execution.
    Reconstructs predictor from serializable params.
    """
    # Reconstruct FireSim and FirePredictor from params
    fire = FireSim(params_dict['sim_params'])
    predictor = FirePredictor(params_dict['predictor_params'], fire)

    # Run prediction
    return predictor.run(fire_estimate=state_est, visualize=False)
```

### 3. Aggregation Strategy

**Data Structure for Aggregation**:
```python
# During aggregation, build these structures:
burn_probability = {}  # {time_s: {(x, y): probability}}
flame_stats = {}       # {(x, y): {'mean': float, 'std': float, 'min': float, 'max': float}}
ros_stats = {}         # {(x, y): {'mean': float, 'std': float, 'min': float, 'max': float}}
fli_stats = {}         # {(x, y): {'mean': float, 'std': float, 'min': float, 'max': float}}
crown_frequency = {}   # {(x, y): fraction}
hold_prob_stats = {}   # {(x, y): {'mean': float, 'std': float}}
breach_frequency = {}  # {(x, y): fraction}
```

**Aggregation Algorithm**:
```python
def _aggregate_predictions(
    predictions: List[PredictionOutput]
) -> EnsemblePredictionOutput:
    """
    Aggregate multiple prediction outputs into ensemble statistics.
    """
    n_ensemble = len(predictions)

    # 1. Build burn probability map
    burn_counts = {}  # {time_s: {(x,y): count}}
    for pred in predictions:
        for time_s, locations in pred.spread.items():
            if time_s not in burn_counts:
                burn_counts[time_s] = {}
            for loc in locations:
                burn_counts[time_s][loc] = burn_counts[time_s].get(loc, 0) + 1

    burn_probability = {
        time_s: {loc: count / n_ensemble for loc, count in counts.items()}
        for time_s, counts in burn_counts.items()
    }

    # 2. Collect all unique cell locations that burned in any prediction
    all_burned_cells = set()
    for pred in predictions:
        all_burned_cells.update(pred.flame_len_m.keys())

    # 3. Aggregate statistics for each cell
    flame_stats = {}
    ros_stats = {}
    fli_stats = {}

    for cell_loc in all_burned_cells:
        # Collect values from all predictions where this cell burned
        flame_values = [p.flame_len_m.get(cell_loc) for p in predictions
                       if cell_loc in p.flame_len_m]
        ros_values = [p.ros_ms.get(cell_loc) for p in predictions
                     if cell_loc in p.ros_ms]
        fli_values = [p.fli_kw_m.get(cell_loc) for p in predictions
                     if cell_loc in p.fli_kw_m]

        if flame_values:
            flame_stats[cell_loc] = {
                'mean': np.mean(flame_values),
                'std': np.std(flame_values),
                'min': np.min(flame_values),
                'max': np.max(flame_values),
                'count': len(flame_values)  # How many predictions had fire here
            }

        # Similar for ROS and FLI...

    # 4. Crown fire frequency
    crown_frequency = {}
    for cell_loc in all_burned_cells:
        crown_count = sum(1 for p in predictions if cell_loc in p.crown_fire)
        crown_frequency[cell_loc] = crown_count / n_ensemble

    # 5. Fireline statistics
    all_fireline_cells = set()
    for pred in predictions:
        all_fireline_cells.update(pred.hold_probs.keys())

    hold_prob_stats = {}
    breach_frequency = {}

    for cell_loc in all_fireline_cells:
        hold_probs = [p.hold_probs.get(cell_loc) for p in predictions
                     if cell_loc in p.hold_probs]
        breaches = [p.breaches.get(cell_loc) for p in predictions
                   if cell_loc in p.breaches]

        if hold_probs:
            hold_prob_stats[cell_loc] = {
                'mean': np.mean(hold_probs),
                'std': np.std(hold_probs)
            }

        if breaches:
            breach_frequency[cell_loc] = sum(breaches) / len(breaches)

    return EnsemblePredictionOutput(...)
```

---

## Implementation Steps

### Step 1: Create New Data Class for Ensemble Output

**File**: `embrs/utilities/data_classes.py`

```python
@dataclass
class CellStatistics:
    """Statistics for a single metric across ensemble members."""
    mean: float
    std: float
    min: float
    max: float
    count: int  # Number of ensemble members with data for this cell

@dataclass
class EnsemblePredictionOutput:
    """Output from ensemble prediction runs."""

    # Number of ensemble members
    n_ensemble: int

    # Burn probability: {time_s: {(x,y): probability [0-1]}}
    burn_probability: dict

    # Cell-level statistics: {(x,y): CellStatistics}
    flame_len_m_stats: dict
    fli_kw_m_stats: dict
    ros_ms_stats: dict

    # Spread direction: {(x,y): {'mean_x': float, 'mean_y': float}}
    # Note: direction is circular, need special handling
    spread_dir_stats: dict

    # Crown fire frequency: {(x,y): probability [0-1]}
    crown_fire_frequency: dict

    # Fireline statistics: {(x,y): CellStatistics}
    hold_prob_stats: dict

    # Breach frequency: {(x,y): probability [0-1]}
    breach_frequency: dict

    # Optional: Individual predictions for inspection
    individual_predictions: Optional[List[PredictionOutput]] = None
```

### Step 2: Implement Helper Function for Parallel Execution

**File**: `embrs/tools/fire_predictor.py`

Add module-level function (must be at module level for pickling):

```python
def _run_single_prediction_worker(
    predictor_params: PredictorParams,
    sim_params: Any,  # SimParams type
    state_estimate: StateEstimate,
    seed: Optional[int] = None
) -> PredictionOutput:
    """
    Worker function for parallel ensemble prediction.

    Must be module-level for multiprocessing serialization.
    Creates a fresh FireSim and FirePredictor instance.

    Args:
        predictor_params: Parameters for FirePredictor
        sim_params: Parameters for FireSim (from fire._sim_params)
        state_estimate: Initial state for this ensemble member
        seed: Random seed for reproducibility

    Returns:
        PredictionOutput for this ensemble member
    """
    if seed is not None:
        np.random.seed(seed)

    # Create fresh instances
    fire = FireSim(sim_params)
    predictor = FirePredictor(predictor_params, fire)

    # Run prediction
    return predictor.run(fire_estimate=state_estimate, visualize=False)
```

### Step 3: Implement Aggregation Function

**File**: `embrs/tools/fire_predictor.py`

```python
def _aggregate_ensemble_predictions(
    predictions: List[PredictionOutput]
) -> EnsemblePredictionOutput:
    """
    Aggregate multiple prediction outputs into ensemble statistics.

    Args:
        predictions: List of PredictionOutput from ensemble members

    Returns:
        EnsemblePredictionOutput with probabilistic predictions
    """
    # Implementation as outlined in "Aggregation Strategy" section above
    # ... (full implementation details)

    return EnsemblePredictionOutput(...)
```

**Special Handling for Spread Direction**:
Spread direction is a circular variable (0° = 360°). Need circular statistics:

```python
def _aggregate_circular_direction(directions: List[float]) -> dict:
    """
    Compute mean direction using circular statistics.

    Args:
        directions: List of directions in radians

    Returns:
        {'mean_dir': float, 'circular_std': float, 'mean_x': float, 'mean_y': float}
    """
    # Convert to unit vectors
    x_components = [np.cos(d) for d in directions]
    y_components = [np.sin(d) for d in directions]

    # Mean direction
    mean_x = np.mean(x_components)
    mean_y = np.mean(y_components)
    mean_dir = np.arctan2(mean_y, mean_x)

    # Circular standard deviation
    R = np.sqrt(mean_x**2 + mean_y**2)
    circular_std = np.sqrt(-2 * np.log(R))

    return {
        'mean_dir': mean_dir,
        'circular_std': circular_std,
        'mean_x': mean_x,
        'mean_y': mean_y
    }
```

### Step 4: Implement run_ensemble() Method

**File**: `embrs/tools/fire_predictor.py`

Add to `FirePredictor` class:

```python
def run_ensemble(
    self,
    state_estimates: List[StateEstimate],
    visualize: bool = False,
    num_workers: Optional[int] = None,
    use_multiprocessing: bool = True,
    random_seeds: Optional[List[int]] = None,
    return_individual: bool = False
) -> EnsemblePredictionOutput:
    """
    Run ensemble predictions using multiple initial state estimates.

    Executes predictions in parallel and aggregates results into
    probabilistic fire spread predictions.

    Args:
        state_estimates: List of StateEstimate objects representing
                        different possible initial fire states
        visualize: If True, visualize the mean burn probability
        num_workers: Number of parallel workers. If None, uses cpu_count()
        use_multiprocessing: If True, use ProcessPoolExecutor.
                            If False, use ThreadPoolExecutor (slower but
                            avoids serialization issues)
        random_seeds: Optional list of random seeds for reproducibility.
                     Must match length of state_estimates if provided.
        return_individual: If True, include individual predictions in output

    Returns:
        EnsemblePredictionOutput with aggregated probabilistic predictions

    Raises:
        ValueError: If state_estimates is empty or random_seeds length mismatch
    """
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

    # Prepare parameters for workers
    predictor_params = copy.deepcopy(self._get_predictor_params())
    sim_params = copy.deepcopy(self.fire._sim_params)

    # Run predictions in parallel
    predictions = []

    if use_multiprocessing:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all jobs
            futures = []
            for i, state_est in enumerate(state_estimates):
                seed = random_seeds[i] if random_seeds else None
                future = executor.submit(
                    _run_single_prediction_worker,
                    predictor_params,
                    sim_params,
                    state_est,
                    seed
                )
                futures.append(future)

            # Collect results with progress bar
            from tqdm import tqdm
            for future in tqdm(as_completed(futures),
                              total=n_ensemble,
                              desc="Ensemble predictions"):
                predictions.append(future.result())

    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Threading version (shares memory, avoids serialization)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, state_est in enumerate(state_estimates):
                seed = random_seeds[i] if random_seeds else None
                future = executor.submit(
                    self._run_ensemble_member,
                    state_est,
                    seed
                )
                futures.append(future)

            from tqdm import tqdm
            for future in tqdm(as_completed(futures),
                              total=n_ensemble,
                              desc="Ensemble predictions"):
                predictions.append(future.result())

    # Aggregate results
    print("Aggregating ensemble predictions...")
    ensemble_output = _aggregate_ensemble_predictions(predictions)

    # Optionally include individual predictions
    if return_individual:
        ensemble_output.individual_predictions = predictions

    # Optionally visualize
    if visualize:
        self._visualize_ensemble(ensemble_output)

    return ensemble_output

def _run_ensemble_member(
    self,
    state_estimate: StateEstimate,
    seed: Optional[int] = None
) -> PredictionOutput:
    """
    Run a single ensemble member (for threading mode).

    This method is used when use_multiprocessing=False to avoid
    serialization issues.
    """
    if seed is not None:
        np.random.seed(seed)

    return self.run(fire_estimate=state_estimate, visualize=False)

def _get_predictor_params(self) -> PredictorParams:
    """
    Extract current predictor parameters for serialization.

    Returns:
        PredictorParams object with current settings
    """
    # Reconstruct PredictorParams from current state
    # This needs to be implemented based on how params are stored
    # May need to add this to __init__ to preserve original params
    return self._params  # Assuming we store this in __init__
```

**Note**: Need to modify `FirePredictor.__init__()` to store original params:

```python
def __init__(self, params: PredictorParams, fire: FireSim):
    self.fire = fire
    self.c_size = -1
    self._params = params  # Store for ensemble execution
    self.set_params(params)
```

### Step 5: Implement Visualization for Ensemble Output

**File**: `embrs/tools/fire_predictor.py`

```python
def _visualize_ensemble(self, ensemble_output: EnsemblePredictionOutput):
    """
    Visualize ensemble prediction results.

    Shows burn probability map at final time step.

    Args:
        ensemble_output: Aggregated ensemble predictions
    """
    # Get final time step
    if not ensemble_output.burn_probability:
        print("No burn probability data to visualize")
        return

    final_time = max(ensemble_output.burn_probability.keys())
    burn_probs = ensemble_output.burn_probability[final_time]

    # Convert to visualization format expected by fire.visualize_prediction
    # This may need adjustment based on actual visualization API

    # For now, threshold at different probability levels
    prob_levels = {
        'high': [],      # > 0.75
        'medium': [],    # 0.25 - 0.75
        'low': []        # < 0.25
    }

    for loc, prob in burn_probs.items():
        if prob > 0.75:
            prob_levels['high'].append(loc)
        elif prob > 0.25:
            prob_levels['medium'].append(loc)
        else:
            prob_levels['low'].append(loc)

    # Use existing visualization infrastructure
    # May need to extend fire.visualize_prediction to handle probabilities
    spread_dict = {final_time: list(burn_probs.keys())}
    self.fire.visualize_prediction(spread_dict)
```

### Step 6: Add Example Usage

**File**: `examples/ix_ensemble_prediction.py`

```python
"""
Example demonstrating ensemble fire prediction.

This example shows how to run multiple predictions with different
initial state estimates and aggregate them into probabilistic predictions.
"""

from shapely.geometry import Polygon
from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim
from embrs.tools.fire_predictor import FirePredictor
from embrs.utilities.data_classes import StateEstimate

class EnsemblePredictionDemo(ControlClass):
    def __init__(self, fire: FireSim):
        self.first_prediction = True
        self.predictor = None

    def process_state(self, fire: FireSim):
        # Run ensemble prediction after 30 minutes
        if fire.curr_time_h > 0.5 and self.first_prediction:
            self.first_prediction = False

            # Create predictor
            if self.predictor is None:
                from embrs.utilities.data_classes import PredictorParams
                params = PredictorParams(
                    time_horizon_hr=2.0,
                    cell_size_m=30,
                    time_step_s=5,
                    dead_mf=0.08,
                    live_mf=0.3
                )
                self.predictor = FirePredictor(params, fire)

            # Create ensemble of state estimates
            # Example: Perturb the burning area slightly for each member
            state_estimates = self._create_ensemble_states(fire)

            # Run ensemble prediction
            print("Running ensemble prediction...")
            ensemble_output = self.predictor.run_ensemble(
                state_estimates=state_estimates,
                visualize=True,
                num_workers=4
            )

            # Analyze results
            self._analyze_ensemble_output(ensemble_output)

    def _create_ensemble_states(self, fire: FireSim, n_members: int = 10):
        """
        Create ensemble of state estimates by perturbing the fire boundary.
        """
        from embrs.utilities.fire_util import UtilFuncs

        state_estimates = []

        # Get current fire polygons
        base_burnt = UtilFuncs.get_cell_polygons(fire.burnt_cells)
        base_burning = UtilFuncs.get_cell_polygons(fire.burning_cells)

        for i in range(n_members):
            # For this example, use same state (in practice, perturb boundaries)
            state_est = StateEstimate(
                burnt_polys=base_burnt,
                burning_polys=base_burning
            )
            state_estimates.append(state_est)

        return state_estimates

    def _analyze_ensemble_output(self, output):
        """Print summary statistics from ensemble output."""
        print(f"\nEnsemble Prediction Results (n={output.n_ensemble}):")

        # Find cells with high burn probability
        high_prob_cells = []
        for time_s, probs in output.burn_probability.items():
            for loc, prob in probs.items():
                if prob > 0.8:
                    high_prob_cells.append((loc, prob))

        print(f"  Cells with >80% burn probability: {len(high_prob_cells)}")

        # Average flame length
        if output.flame_len_m_stats:
            mean_flames = [stats['mean'] for stats in
                          output.flame_len_m_stats.values()]
            print(f"  Mean flame length: {np.mean(mean_flames):.2f} m")
```

---

## Testing Strategy

### Unit Tests

**File**: `embrs/test_code/test_ensemble_prediction.py`

```python
import unittest
import numpy as np
from embrs.tools.fire_predictor import (
    FirePredictor,
    _aggregate_ensemble_predictions,
    _run_single_prediction_worker
)
from embrs.utilities.data_classes import (
    PredictorParams,
    StateEstimate,
    PredictionOutput
)

class TestEnsemblePrediction(unittest.TestCase):

    def test_aggregate_empty_list(self):
        """Test aggregation handles empty prediction list."""
        with self.assertRaises(ValueError):
            _aggregate_ensemble_predictions([])

    def test_aggregate_single_prediction(self):
        """Test aggregation with single prediction."""
        # Create mock prediction
        pred = PredictionOutput(
            spread={0: [(0, 0)]},
            flame_len_m={(0, 0): 2.0},
            # ... other fields
        )

        result = _aggregate_ensemble_predictions([pred])

        self.assertEqual(result.n_ensemble, 1)
        self.assertEqual(result.burn_probability[0][(0, 0)], 1.0)

    def test_burn_probability_calculation(self):
        """Test burn probability computed correctly."""
        # Create 3 predictions where 2/3 burn same cell
        predictions = [
            PredictionOutput(spread={0: [(0, 0)]}, ...),
            PredictionOutput(spread={0: [(0, 0)]}, ...),
            PredictionOutput(spread={0: [(1, 1)]}, ...)
        ]

        result = _aggregate_ensemble_predictions(predictions)

        self.assertAlmostEqual(
            result.burn_probability[0][(0, 0)],
            2/3,
            places=5
        )

    def test_statistics_calculation(self):
        """Test mean/std computed correctly."""
        predictions = [
            PredictionOutput(
                spread={0: [(0, 0)]},
                flame_len_m={(0, 0): 2.0},
                ...
            ),
            PredictionOutput(
                spread={0: [(0, 0)]},
                flame_len_m={(0, 0): 4.0},
                ...
            ),
        ]

        result = _aggregate_ensemble_predictions(predictions)

        stats = result.flame_len_m_stats[(0, 0)]
        self.assertAlmostEqual(stats['mean'], 3.0)
        self.assertAlmostEqual(stats['std'], 1.0)
```

### Integration Tests

Create test config file and run full ensemble:

```python
def test_full_ensemble_run(self):
    """Test complete ensemble run with small fire."""
    # Load minimal test configuration
    sim_params = load_sim_params('test_configs/minimal.cfg')
    fire = FireSim(sim_params)

    # Create predictor
    pred_params = PredictorParams(...)
    predictor = FirePredictor(pred_params, fire)

    # Create simple ensemble
    state_ests = [StateEstimate(...) for _ in range(3)]

    # Run ensemble
    result = predictor.run_ensemble(
        state_estimates=state_ests,
        num_workers=2
    )

    # Verify output structure
    self.assertIsInstance(result, EnsemblePredictionOutput)
    self.assertEqual(result.n_ensemble, 3)
    self.assertGreater(len(result.burn_probability), 0)
```

---

## Performance Considerations

### Memory Usage

**Challenge**: Running N predictions in parallel can consume N × memory per prediction

**Solutions**:
1. **Batch processing**: If N is large, process in batches
   ```python
   batch_size = min(num_workers * 2, n_ensemble)
   for i in range(0, n_ensemble, batch_size):
       batch = state_estimates[i:i+batch_size]
       batch_results = _run_batch(batch)
       all_results.extend(batch_results)
   ```

2. **Streaming aggregation**: Aggregate results as they complete rather than storing all
   ```python
   # Initialize accumulators
   burn_counts = {}
   flame_accumulators = {}

   # Update as results arrive
   for future in as_completed(futures):
       result = future.result()
       _update_accumulators(burn_counts, flame_accumulators, result)
       del result  # Free memory immediately
   ```

3. **Reduced resolution predictions**: Use coarser grid for ensemble members

### Computation Time

**Estimation**: If single prediction takes T seconds:
- Serial: N × T seconds
- Parallel with P workers: (N / P) × T seconds (ideally)
- With overhead: (N / P) × T + setup + aggregation

**Optimization**:
- Pre-serialize common data (map, weather) once
- Use shared memory for read-only data if possible
- Profile aggregation step for bottlenecks

---

## Error Handling

### Worker Failures

```python
# Wrap worker execution in try-except
def _run_single_prediction_worker(...):
    try:
        # ... prediction code
        return predictor.run(...)
    except Exception as e:
        # Return error instead of crashing worker
        return PredictionError(
            state_estimate=state_estimate,
            error=str(e),
            traceback=traceback.format_exc()
        )

# In run_ensemble(), handle errors
failed_predictions = []
successful_predictions = []

for future in as_completed(futures):
    result = future.result()
    if isinstance(result, PredictionError):
        failed_predictions.append(result)
        print(f"Warning: Prediction failed: {result.error}")
    else:
        successful_predictions.append(result)

if len(successful_predictions) < len(state_estimates) * 0.5:
    raise RuntimeError(
        f"More than 50% of predictions failed "
        f"({len(failed_predictions)}/{len(state_estimates)})"
    )
```

### Input Validation

```python
# At start of run_ensemble()
if not state_estimates:
    raise ValueError("state_estimates cannot be empty")

if not all(isinstance(s, StateEstimate) for s in state_estimates):
    raise TypeError("All elements must be StateEstimate objects")

if num_workers is not None and num_workers < 1:
    raise ValueError("num_workers must be >= 1")
```

---

## Documentation

### Docstrings

All new functions should have comprehensive docstrings following Google style:

```python
def run_ensemble(self, state_estimates, ...):
    """Run ensemble predictions using multiple initial state estimates.

    This method executes multiple fire predictions in parallel, each starting
    from a different initial state estimate. The results are aggregated to
    produce probabilistic fire spread predictions showing the likelihood of
    fire spread to each cell.

    Args:
        state_estimates: List of StateEstimate objects representing different
            possible initial fire states. Each estimate contains polygons
            defining burnt and burning regions.
        visualize: If True, visualize the aggregated burn probability map
            using the existing fire visualization system.
        num_workers: Number of parallel worker processes. Defaults to the
            number of CPU cores if not specified.
        use_multiprocessing: If True, use process-based parallelism. If False,
            use thread-based parallelism (slower but avoids serialization).
        random_seeds: Optional list of random seeds for reproducible results.
            Length must match state_estimates if provided.
        return_individual: If True, include individual prediction outputs in
            the returned EnsemblePredictionOutput object.

    Returns:
        EnsemblePredictionOutput containing:
            - burn_probability: For each time step, probability of fire at
              each cell location (fraction of ensemble members predicting fire)
            - flame_len_m_stats: Mean, std, min, max flame length statistics
            - fli_kw_m_stats: Fireline intensity statistics
            - ros_ms_stats: Rate of spread statistics
            - crown_fire_frequency: Fraction of predictions with crown fire
            - hold_prob_stats: Fireline hold probability statistics
            - breach_frequency: Fraction of predictions breaching firelines

    Raises:
        ValueError: If state_estimates is empty or random_seeds length doesn't
            match state_estimates length.
        RuntimeError: If more than 50% of ensemble members fail to execute.

    Example:
        >>> # Create predictor
        >>> predictor = FirePredictor(params, fire)
        >>>
        >>> # Create ensemble of state estimates
        >>> estimates = [StateEstimate(...) for _ in range(10)]
        >>>
        >>> # Run ensemble
        >>> result = predictor.run_ensemble(estimates, num_workers=4)
        >>>
        >>> # Access burn probability at final time
        >>> final_time = max(result.burn_probability.keys())
        >>> probs = result.burn_probability[final_time]
        >>> high_risk_cells = {loc: p for loc, p in probs.items() if p > 0.8}

    Note:
        - Each ensemble member uses the same PredictorParams but different
          initial state estimates
        - Parallel execution may require significant memory (N predictions
          running simultaneously)
        - For large ensembles (>100), consider using batching or reduced
          resolution to manage memory
    """
```

### User Documentation

Add section to documentation (e.g., `docs_raw/fire_prediction.md`):

```markdown
## Ensemble Prediction

Ensemble prediction runs multiple fire predictions with different initial
state estimates to quantify uncertainty in fire spread forecasts.

### Basic Usage

```python
from embrs.tools.fire_predictor import FirePredictor
from embrs.utilities.data_classes import StateEstimate, PredictorParams

# Create predictor
params = PredictorParams(...)
predictor = FirePredictor(params, fire)

# Create ensemble of state estimates
state_estimates = [
    StateEstimate(burnt_polys=[...], burning_polys=[...]),
    StateEstimate(burnt_polys=[...], burning_polys=[...]),
    # ... more estimates
]

# Run ensemble prediction
result = predictor.run_ensemble(
    state_estimates=state_estimates,
    visualize=True,
    num_workers=4
)

# Access burn probability
for time_s, cell_probs in result.burn_probability.items():
    for (x, y), probability in cell_probs.items():
        if probability > 0.8:
            print(f"High fire risk at ({x}, {y}): {probability:.2%}")
```

### Output Interpretation

- **burn_probability**: Fraction of ensemble members predicting fire at each
  cell. Values near 1.0 indicate high confidence in fire spread.

- **flame_len_m_stats**: Statistics on flame length. Use `mean` for expected
  value, `std` for uncertainty, `count` for how many predictions had fire.

- **crown_fire_frequency**: Fraction of predictions with crown fire activity.
  Values > 0.5 suggest crown fire is likely.
```

---

## Migration Path

### Backward Compatibility

The new `run_ensemble()` method is additive and does not modify the existing
`run()` method, ensuring full backward compatibility.

### Deprecation (None Required)

No existing functionality needs to be deprecated.

---

## Future Enhancements

### Potential Extensions

1. **Adaptive ensemble generation**: Automatically generate state estimates
   from uncertainty in fire detection

2. **Ensemble visualization improvements**: Heat maps, contours, confidence
   intervals on fire perimeter

3. **Targeted ensemble thinning**: Reduce ensemble size while preserving
   variance using clustering

4. **Ensemble Kalman filtering**: Update ensemble members based on new
   observations

5. **Risk metrics**: Compute probability of fire reaching specific assets
   or crossing control lines

---

## Summary Checklist

- [ ] Create `EnsemblePredictionOutput` dataclass
- [ ] Create `CellStatistics` dataclass
- [ ] Implement `_run_single_prediction_worker()` function
- [ ] Implement `_aggregate_ensemble_predictions()` function
- [ ] Implement circular statistics for direction aggregation
- [ ] Add `run_ensemble()` method to `FirePredictor`
- [ ] Add `_run_ensemble_member()` helper method
- [ ] Store original `PredictorParams` in `__init__`
- [ ] Implement `_get_predictor_params()` method
- [ ] Implement `_visualize_ensemble()` method
- [ ] Create example usage file
- [ ] Write unit tests for aggregation
- [ ] Write integration tests for full ensemble
- [ ] Add comprehensive docstrings
- [ ] Update user documentation
- [ ] Test with small ensemble (3-5 members)
- [ ] Test with large ensemble (50-100 members)
- [ ] Profile memory usage
- [ ] Profile computation time
- [ ] Test error handling (failed predictions)
- [ ] Test serialization (multiprocessing mode)
- [ ] Test threading mode (fallback)

## Estimated Implementation Time

- **Data structures**: 2 hours
- **Worker function**: 2 hours
- **Aggregation logic**: 6 hours (complex, needs testing)
- **Main run_ensemble() method**: 4 hours
- **Visualization**: 3 hours
- **Example code**: 2 hours
- **Testing**: 6 hours
- **Documentation**: 3 hours
- **Debugging/refinement**: 6 hours

**Total**: ~34 hours (approximately 1 week)
