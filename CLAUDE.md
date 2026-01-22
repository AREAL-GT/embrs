# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EMBRS (Empirical Modular Fire Response Simulation) is a Python-based wildfire simulation framework using hexagonal grids and Rothermel's fire spread equations. It supports real-time visualization, fire prediction, and user-defined control strategies.

Full documentation: https://areal-gt.github.io/embrs/

## Development Commands

### Installation
```bash
# Install in editable mode for development
pip install -e .

# For documentation building
pip install -r docs_raw/mkdocs-requirements.txt
```

### Running Simulations
```bash
# With GUI for parameter selection
python -m embrs.main

# With configuration file
python -m embrs.main --config path/to/config.cfg

# With profiling enabled
python -m embrs.main --config path/to/config.cfg --profile
```

### Documentation
```bash
# Build documentation locally
mkdocs build --clean

# Serve documentation with live reload
mkdocs serve
```

### Testing
There is no formal test suite with pytest/unittest runner. Test files in `embrs/test_code/` are standalone scripts that validate specific components (e.g., `test_rothermel.py` for fire spread calculations). Run them directly:
```bash
python embrs/test_code/test_rothermel.py
```

## Architecture

### Core Simulation Flow

```
main.py (entry point)
  └─> initialize() creates FireSim + optional RealTimeVisualizer
      └─> run_sim() loops:
          ├─> fire.iterate() - propagates fire one time step
          ├─> user_code.process_state(fire) - custom control logic
          └─> viz.update() - real-time visualization
```

### Key Components

**1. FireSim (embrs/fire_simulator/fire.py)**
- Main simulation class extending BaseFireSim
- `iterate()` is the core method called each time step:
  - Updates weather if changed
  - Processes new ignitions
  - Calculates fire spread via Rothermel equations
  - Applies fire acceleration model
  - Handles spotting (ember transport)
  - Updates visualization and logging
- Manages hexagonal grid of Cell objects
- Tracks burning_cells, burnt_cells, frontier

**2. BaseFireSim (embrs/base_classes/base_fire.py)**
- Abstract base providing the core API for fire manipulation
- Key methods for control actions:
  - `set_ignition_at_xy/indices/cell()` - start fires
  - `construct_fireline(line, width, rate)` - build firebreaks
  - `add_retardant_at_xy/indices/cell()` - apply retardant
  - `water_drop_at_xy/indices/cell_as_rain()` - water drops
  - `water_drop_at_xy/indices/cell_as_moisture_bump()` - alternate water model
- Query methods:
  - `get_cell_from_xy(x, y)` - get cell at coordinates
  - `get_cell_from_indices(row, col)` - get cell by grid indices
  - `get_cells_at_geometry(geom)` - get cells intersecting shapely geometry

**3. Cell (embrs/fire_simulator/cell.py)**
- Hexagonal grid cell (point-up orientation)
- States: FUEL (0), FIRE (1), BURNT (2)
- Contains terrain (elevation, slope, aspect), fuel properties, weather data
- Six neighbors stored in `cell.neighbors` dict
- Fire behavior calculated per-cell based on local conditions

**4. Fire Behavior Models (embrs/models/)**
- `rothermel.py`: Surface fire spread calculations (ROS, fireline intensity)
- `crown_model.py`: Crown fire initiation and active crowning
- `fuel_models.py`: Anderson 13 fuel model definitions
- `weather.py`: WeatherStream for temporal weather data
- `wind_forecast.py`: WindNinja integration for terrain-adjusted winds
- `dead_fuel_moisture.py`: Fuel moisture calculations from weather
- `embers.py`: Spotting model (Albini-based ember transport)

**5. Fire Prediction (embrs/tools/fire_predictor.py)**
- `FirePredictor` class runs forward predictions with uncertainty
- Accepts `PredictorParams` for configuration:
  - Time horizon, cell size, time step
  - Wind bias/uncertainty, ROS bias
  - Fuel moisture overrides
- Can initialize from current fire state (perfect) or StateEstimate (uncertain)
- Returns `PredictionOutput` with spread timeline, flame length, hold probabilities

**6. User Code Integration (embrs/base_classes/control_base.py)**
- Extend `ControlClass` and implement `process_state(fire: FireSim)`
- Called after each iteration for decision-making
- See `examples/` directory for patterns (especially `ii_embrs_interface.py`)

### Hexagonal Grid System

- **Grid type**: Point-up hexagonal grid (6 neighbors per cell)
- **Indexing**: Row increases upward (bottom-to-top), column increases left-to-right
- **Cell center calculation**:
  - Even rows: x = col × cell_size × √3
  - Odd rows: x = (col + 0.5) × cell_size × √3
  - All rows: y = row × cell_size × 1.5
- **Neighbor access**: `cell.neighbors` dict, `cell.burnable_neighbors` for fuel-filtered
- Reduces angular bias vs. square grids

### State Management Pattern

- Cell state changes tracked in `_updated_cells` dict during iteration
- Batch flushed to logger/visualizer at end of iteration
- Logger writes to Parquet format (columnar storage) for efficiency
- Visualizer caches changes for rendering

### Configuration Files

Configuration files (.cfg) use INI format with three sections:

```ini
[Simulation]
t_step_s = 5
cell_size_m = 30
visualize = True
num_runs = 1
write_logs = False
model_spotting = True

[Weather]
input_type = File
file = /path/to/weather.json
start_datetime = 2024-05-04T12:00:00
end_datetime = 2024-05-05T00:00:00

[Map]
folder = /path/to/embrs_map
```

Examples in `embrs/config_files/` and `examples/example_configs/`

## Important Patterns

### Taking Control Actions in User Code

```python
from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim

class MyController(ControlClass):
    def process_state(self, fire: FireSim):
        # Query fire state
        if fire.iters % 10 == 0:
            cell = fire.get_cell_from_xy(x, y)
            if cell and cell.state == CellStates.FIRE:
                # Take action
                fire.construct_fireline(line_geom, width_m=10)
```

### Running Fire Predictions

```python
from embrs.tools.fire_predictor import FirePredictor

# In your ControlClass.process_state():
predictor = FirePredictor(fire, prediction_hours=3, bias=1)
output = predictor.run_prediction()

# Access results
for time_s, positions in output.spread.items():
    print(f"At {time_s}s, fire at: {positions}")
```

### Cell State Queries

```python
# Single cell access
cell = fire.get_cell_from_xy(x, y)
cell = fire.get_cell_from_indices(row, col)

# Geometric queries
from shapely.geometry import LineString, Polygon
cells = fire.get_cells_at_geometry(LineString([(x1,y1), (x2,y2)]))
cells = fire.get_cells_at_geometry(Polygon([(x1,y1), (x2,y2), (x3,y3)]))

# Iterate burning cells
for cell in fire.burning_cells:
    print(f"Cell {cell.id} burning at ROS={cell.ros_steady_m_s}")
```

### Multi-Run Simulations

When `num_runs > 1` in config, FireSim is deep-copied for each run to ensure independence. This supports Monte Carlo analysis or parameter sweeps.

## Design Decisions

1. **Weak references**: Cell holds weak reference to parent FireSim to prevent memory leaks during deep copies
2. **Hexagonal grid**: Chosen over square grid to reduce directional bias in fire spread
3. **Per-cell weather**: Wind and moisture vary spatially; wind from WindNinja terrain adjustment
4. **Action queuing**: Firelines and water drops applied incrementally over time, not instantaneously
5. **Modular visualization**: BaseVisualizer, RealTimeVisualizer, PlaybackVisualizer separated for flexibility
6. **Parquet logging**: Columnar format with caching reduces I/O overhead for large simulations

## File Organization

```
embrs/
├── main.py                       # Entry point
├── fire_simulator/
│   ├── fire.py                   # FireSim class
│   ├── cell.py                   # Cell class
│   └── visualizer.py             # RealTimeVisualizer
├── base_classes/
│   ├── base_fire.py              # BaseFireSim (core API)
│   ├── control_base.py           # ControlClass for user code
│   └── agent_base.py             # AgentBase for tracked entities
├── models/
│   ├── rothermel.py              # Fire spread equations
│   ├── crown_model.py            # Crown fire model
│   ├── fuel_models.py            # Anderson 13 fuel types
│   ├── weather.py                # Weather data streams
│   └── embers.py                 # Spotting/ember transport
├── tools/
│   └── fire_predictor.py         # FirePredictor for forward prediction
├── utilities/
│   ├── logger.py                 # Parquet-based logging
│   ├── action.py                 # Action classes (retardant, water, etc.)
│   ├── data_classes.py           # Dataclass definitions
│   └── fire_util.py              # Enums, hex math, utilities
├── map_generator.py              # Create maps from GIS data
├── visualization_tool.py         # Playback visualization tool
└── examples/                     # Example user code implementations
```

## Common Issues

- **Import errors on first run**: Install with `pip install -e .` to ensure all dependencies are available
- **Cell state confusion**: Remember FUEL=0, FIRE=1, BURNT=2 (use CellStates enum)
- **Coordinate vs. index**: `get_cell_from_xy()` uses meters, `get_cell_from_indices()` uses row/col
- **Fireline not blocking**: Firebreaks work probabilistically based on flame length vs. break width
- **Deep copy issues**: If working with multi-run sims, be aware each run gets an independent FireSim copy
