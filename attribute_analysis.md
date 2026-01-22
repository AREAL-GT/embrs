# BaseFireSim.__init__() Attribute Analysis

## What __init__() Does

### Phase 1: Simple Attribute Assignment (Lines 45-88)
**Cost: Negligible (~0.001s)**

Direct assignments from parameters or constants:
- `self.display_frequency = 300`
- `self._sim_params = sim_params`
- `self.burnout_thresh = 0.01`
- `self.sim_start_w_idx = 0`
- `self._curr_weather_idx = None`
- `self._last_weather_update = 0`
- `self.weather_changed = True`
- `self._curr_time_s = 0`
- `self._iters = 0`
- `self.logger = None`
- `self._visualizer = None`
- `self._finished = False`
- `self._updated_cells = {}`
- `self._cell_dict = {}`
- `self._long_term_retardants = set()`
- `self._active_water_drops = []`
- `self._burning_cells = []`
- `self._new_ignitions = []`
- `self._burnt_cells = set()`
- `self._frontier = set()`
- `self._fire_break_cells = []`
- `self._active_firelines = {}`
- `self._new_fire_break_cache = []`
- `self.starting_ignitions = set()`
- `self._urban_cells = []`
- `self._scheduled_spot_fires = {}`
- `self._cell_grid = np.empty(self._shape, dtype=Cell)`
- `self._grid_width = self._cell_grid.shape[1] - 1`
- `self._grid_height = self._cell_grid.shape[0] - 1`

### Phase 2: Parse Sim Params (Lines 56, calls _parse_sim_params)
**Cost: Negligible (~0.01s)**

Extracts and stores attributes from sim_params:
- `self._cell_size = sim_params.cell_size`
- `self._sim_duration = sim_params.duration_s`
- `self._time_step = sim_params.t_step_s`
- `self._init_mf = sim_params.init_mf`
- `self._fuel_moisture_map = getattr(sim_params, 'fuel_moisture_map', {})`
- `self._fms_has_live = getattr(sim_params, 'fms_has_live', False)`
- `self._init_live_h_mf = getattr(sim_params, 'live_h_mf', 0.0)`
- `self._init_live_w_mf = getattr(sim_params, 'live_w_mf', 0.0)`
- `self._size = map_params.size()`
- `self._shape = map_params.shape(self._cell_size)`
- `self._roads = map_params.roads`
- `self.coarse_elevation = np.empty(self._shape)`
- `self.FuelClass = Anderson13 or ScottBurgan40` (based on fbfm_type)
- `self._elevation_map = np.flipud(lcp_data.elevation_map)`
- `self._slope_map = np.flipud(lcp_data.slope_map)`
- `self._aspect_map = np.flipud(lcp_data.aspect_map)`
- `self._fuel_map = np.flipud(lcp_data.fuel_map)`
- `self._cc_map = np.flipud(lcp_data.canopy_cover_map)`
- `self._ch_map = np.flipud(lcp_data.canopy_height_map)`
- `self._cbh_map = np.flipud(lcp_data.canopy_base_height_map)`
- `self._cbd_map = np.flipud(lcp_data.canopy_bulk_density_map)`
- `self._data_res = lcp_data.resolution`
- `self._fire_breaks = list(zip(...))`
- `self.fire_break_dict = {...}`
- `self._initial_ignition = scenario.initial_ign`
- `self._start_datetime = sim_params.weather_input.start_datetime`
- `self._north_dir_deg = map_params.geo_info.north_angle_deg`
- `self._aspect_map = (180 + self._aspect_map) % 360`
- `self.wind_forecast = ...` (zeros for prediction, or from run_windninja)
- `self.flipud_forecast = ...`
- `self._wind_res = ...`
- `self.wind_xpad, self.wind_ypad = ...`
- `self._weather_stream = ...` (for non-prediction)
- `self.weather_t_step = ...`
- `self.model_spotting = sim_params.model_spotting`
- `self._spot_ign_prob = ...`
- `self._canopy_species = ...`
- `self._dbh_cm = ...`
- `self._min_spot_distance = ...`
- `self._spot_delay_s = ...`

### Phase 3: Moisture and Spotting Setup (Lines 94-118)
**Cost: Negligible (~0.001s)**

Conditional assignments based on prediction mode:
- `live_h_mf`, `live_w_mf` (from weather or attributes)
- `self.fmc` (foliar moisture content)
- `self.embers = Embers(...) or PerrymanSpotting(...)`

### Phase 4: Cell Creation Loop (Lines 121-217)
**Cost: EXPENSIVE (~10-30 seconds for typical grid)**

Creates thousands of Cell objects:
```python
for i in range(self._shape[1]):
    for j in range(self._shape[0]):
        new_cell = Cell(id, i, j, self._cell_size)
        new_cell.set_parent(self)
        # Extract data from maps
        # Set cell_data
        new_cell._set_cell_data(cell_data)
        # Set wind forecast
        new_cell._set_wind_forecast(wind_speed, wind_dir)
        self._cell_grid[j,i] = new_cell
        self._cell_dict[id] = new_cell
```

Also populates:
- `self._urban_cells` (cells with fuel type 91)
- `self.coarse_elevation[j, i]` (elevation array)

### Phase 5: Neighbor Assignment (Line 220)
**Cost: Moderate (~1-3 seconds)**

Calls `_add_cell_neighbors()` which iterates through grid:
```python
for each cell:
    determine neighbors based on hex grid geometry
    cell._neighbors = neighbors dict
    cell._burnable_neighbors = dict(neighbors)
```

### Phase 6: Initial State Setup (Lines 222-237)
**Cost: Light (~0.1-1 second)**

- `_set_initial_ignition(self.initial_ignition)`
- `_set_initial_burnt_region(burnt_region)`
- `_overwrite_urban_fuel(cell)` for urban cells
- `_set_firebreaks()`
- `_set_roads()`

## Summary

### Expensive Operations to Avoid:
1. **Cell creation loop** (lines 121-217): 10-30 seconds
2. **run_windninja()** (in _parse_sim_params for non-prediction): 5-15 seconds
3. **Neighbor assignment** (line 220): 1-3 seconds

### Cheap Operations (Can Repeat):
1. Simple attribute assignments: <0.01s
2. Extracting from sim_params: <0.01s
3. Initial state setup: <1s

### Already Serializable in FirePredictor:
- `self.orig_grid` - Template cell grid (already created)
- `self.orig_dict` - Template cell dict (already created)
- Both contain fully initialized cells with:
  - Terrain data
  - Fuel properties
  - Neighbor references
  - Wind forecasts

### Deep Copy Cost:
FirePredictor already deep copies cells in `_set_states()`:
```python
self._cell_grid = copy.deepcopy(self.orig_grid)  # Current approach
self._cell_dict = copy.deepcopy(self.orig_dict)
```
**Measured cost: ~2-5 seconds for typical grid**
This is MUCH faster than creating from scratch (10-30s)
