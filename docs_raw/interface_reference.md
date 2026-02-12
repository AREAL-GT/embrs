# Fire Interface

The `BaseFireSim` class provides the public interface used by custom control classes to interact with the fire simulation. These methods are available on the `fire` parameter passed to your [`ControlClass.process_state()`](./user_code.md) method. For the complete autodoc reference, see [Base Classes](./base_class_documentation.md).

For all the examples below, `fire` is the `BaseFireSim` instance passed to `process_state`.

## Retrieving Cells
If you need to retrieve a specific cell object there are two ways to do so:

**From Indices:**

```python
   # row and col must be integers
   row = 10
   col = 245

   # Get the cell with indices (row, col) in the backing array
   cell = fire.get_cell_from_indices(row, col)

```

If looking at a visualization, row 0 is the row along the bottom of the visualization, and column 0 is the column along the left side of the visualization.

**From Coordinates:**

```python
   # x_m and y_m are floats in meters
   x_m = 1240.0
   y_m = 245.7

   # Get the cell that contains the point (x_m, y_m) within it
   cell = fire.get_cell_from_xy(x_m, y_m)

```

If looking at a visualization, x = 0 is along the left edge of the visualization, y = 0 is along the bottom edge of the visualization.

## Setting State
At any point, you can set the state of a cell to one of the three available [states](fire_modelling.md#state) (`FUEL`, `FIRE` and `BURNT`).

**By passing in the cell object explicitly:**

```python
   from embrs.utilities.fire_util import CellStates

   state = CellStates.BURNT

   # Set cell's state to BURNT
   fire.set_state_at_cell(cell, state) # cell is an instance of 'Cell' class

```

**By passing in the x,y coordinates:**

```python
   from embrs.utilities.fire_util import CellStates

   # x_m and y_m are floats in meters
   x_m = 1205.4
   y_m = 24.6

   state = CellStates.FUEL

   # Set cell which contains (x,y)'s state to FUEL
   fire.set_state_at_xy(x_m, y_m, state)

```

**By passing in the indices:**

```python
   from embrs.utilities.fire_util import CellStates

   # row and col must be integers
   row = 120
   col = 17

   state = CellStates.BURNT

   # Set cell at indices (row, col)'s state to BURNT
   fire.set_state_at_indices(row, col, state)

```

!!! note
    While you can set a cell's state to FIRE using the above functions, it is recommended that you use the ignition functions below to do so.

## Starting Fires
To ignite a cell, use the `set_ignition_at_*` methods. These set the cell to the FIRE state and register it with the simulation's ignition tracking. Each can be called in three ways:

**By passing in the cell object explicitly:**

```python
   # Set ignition at cell
   fire.set_ignition_at_cell(cell) # cell is an instance of 'Cell' class

```

**By passing in the x,y coordinates:**

```python
   # x_m and y_m are floats in meters
   x_m = 1254.4
   y_m = 356.2

   # Set ignition at cell containing point (x,y)
   fire.set_ignition_at_xy(x_m, y_m)

```

**By passing in the indices:**

```python
   # row and col must be integers
   row = 40
   col = 250

   # Set ignition at cell whose indices are (row, col)
   fire.set_ignition_at_indices(row, col)

```

## Suppression Actions

### Fire Retardant

Long-term fire retardant can be applied to cells to reduce their rate of spread. The `effectiveness` parameter (0.0–1.0) controls how much the retardant slows fire spread, and `duration_hr` sets how long the effect lasts.

**By passing in the cell object explicitly:**

```python
   # Apply retardant with 80% effectiveness for 2 hours
   fire.add_retardant_at_cell(cell, duration_hr=2.0, effectiveness=0.8)

```

**By passing in the x,y coordinates:**

```python
   x_m = 1254.4
   y_m = 356.2

   fire.add_retardant_at_xy(x_m, y_m, duration_hr=2.0, effectiveness=0.8)

```

**By passing in the indices:**

```python
   row = 125
   col = 35

   fire.add_retardant_at_indices(row, col, duration_hr=2.0, effectiveness=0.8)

```

### Water Drops

Water drops can be modeled in two ways: as equivalent rainfall or as a direct moisture increase. Increasing fuel moisture slows fire spread; if moisture reaches the fuel model's dead moisture of extinction, the cell will not ignite.

#### As Equivalent Rainfall

Models the water drop as rainfall with a given depth in centimeters. The moisture effect is computed using the fuel moisture model.

```python
   # Apply water as 2 cm of equivalent rainfall at a cell
   fire.water_drop_at_cell_as_rain(cell, water_depth_cm=2.0)

   # By coordinates
   fire.water_drop_at_xy_as_rain(x_m, y_m, water_depth_cm=2.0)

   # By indices
   fire.water_drop_at_indices_as_rain(row, col, water_depth_cm=2.0)

```

#### As Direct Moisture Increase

Directly increases the cell's fuel moisture content by the specified fraction.

```python
   # Increase moisture content by 20%
   fire.water_drop_at_cell_as_moisture_bump(cell, moisture_inc=0.2)

   # By coordinates
   fire.water_drop_at_xy_as_moisture_bump(x_m, y_m, moisture_inc=0.2)

   # By indices
   fire.water_drop_at_indices_as_moisture_bump(row, col, moisture_inc=0.2)

```

### Fireline Construction

Fire breaks can be constructed along a Shapely `LineString` geometry. Firelines can be built instantly or progressively at a specified construction rate.

```python
   from shapely.geometry import LineString

   # Define a fireline path
   line = LineString([(100, 200), (300, 400), (500, 200)])

   # Build a 5-meter-wide fireline instantly
   fireline_id = fire.construct_fireline(line, width_m=5.0)

   # Build a fireline progressively at 0.5 m/s
   fireline_id = fire.construct_fireline(line, width_m=5.0, construction_rate=0.5)

   # Stop construction of an in-progress fireline
   fire.stop_fireline_construction(fireline_id)

```

## Querying Fire State

### Frontier

The frontier consists of FUEL cells adjacent to burning cells that could potentially ignite. Three methods are available depending on what data you need:

```python
   # Get set of cell IDs at the frontier
   frontier_ids = fire.get_frontier()

   # Get list of Cell objects at the frontier
   frontier_cells = fire.get_frontier_cells()

   # Get list of (x, y) positions at the frontier
   frontier_positions = fire.get_frontier_positions()

```

### Average Fire Position

To find the average (x,y) position of all burning cells:

```python
   x_avg, y_avg = fire.get_avg_fire_coord()

```

### Cells at Geometry

To retrieve all cells that intersect with a Shapely geometry (Point, LineString, or Polygon):

```python
   from shapely.geometry import Polygon

   area = Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])
   cells = fire.get_cells_at_geometry(area)

```

## Useful Properties

The interface provides read-only access to key properties of the simulation.

### Grid and Dimensions

```python
   # 2D numpy array of all Cell objects
   arr = fire.cell_grid

   # Dictionary mapping cell IDs to Cell objects
   cell_dict = fire.cell_dict

   # Grid dimensions as (rows, cols)
   rows, cols = fire.shape

   # Map size in meters as (width, height)
   width_m, height_m = fire.size

   # Max x and y coordinates in meters
   max_x = fire.x_lim
   max_y = fire.y_lim

   # Cell size in meters (distance between parallel sides of hexagon)
   cs = fire.cell_size

```

### Time

```python
   # Current simulation time
   time_s = fire.curr_time_s   # seconds
   time_m = fire.curr_time_m   # minutes
   time_h = fire.curr_time_h   # hours

   # Time step per iteration in seconds
   dt = fire.time_step

   # Number of iterations completed
   n = fire.iters

   # Total simulation duration in seconds
   dur = fire.sim_duration

   # Whether the simulation has finished
   done = fire.finished

```

### Fire State

```python
   # List of Cell objects currently on fire
   burning = fire.burning_cells

   # Initial ignition geometries
   ign = fire.initial_ignition

```

### Fire Breaks and Roads

```python
   # List of (LineString, width, id) tuples for each fire break
   breaks = fire.fire_breaks

   # List of Cell objects along fire breaks
   break_cells = fire.fire_break_cells

   # Road data: list of (road_coords, road_type, road_width) tuples
   roads = fire.roads

```

## Agents

Agents can be registered with the simulation for logging and visualization. See [Custom Control Classes — Agents](./user_code.md#agents) for details.

```python
   fire.add_agent(agent)  # agent must be an instance of AgentBase

```

!!! note
    For the complete API reference including all parameters, return types, and exceptions, see [Base Classes](./base_class_documentation.md).
