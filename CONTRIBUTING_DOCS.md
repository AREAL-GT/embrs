# CONTRIBUTING_DOCS.md

Documentation contract for the EMBRS codebase.

## Docstring Style

Use **Google-style** docstrings. All docstrings use triple double-quotes (`"""`).

```python
def example(param1: float, param2: int) -> bool:
    """One-line summary in imperative mood.

    Extended description if needed. Explain behavior, not implementation.

    Args:
        param1 (float): Description with units if applicable.
        param2 (int): Description.

    Returns:
        bool: Description of return value.

    Raises:
        ValueError: When this exception is raised.
    """
```

## Required Sections

### Modules

```python
"""One-line summary of module purpose.

Extended description explaining what the module provides and its role
in the system.

Classes:
    - ClassName: Brief description.

.. autoclass:: ClassName
    :members:
"""
```

### Classes

```python
class Example:
    """One-line summary.

    Extended description of the class purpose and usage.

    Attributes:
        attr1 (type): Description with units.
        attr2 (type): Description.
    """
```

- List only **public** attributes in the class docstring
- Private attributes (`_attr`) documented inline or omitted

### Functions/Methods

Required sections (when applicable):
- **Args**: Always include type in parentheses
- **Returns**: Include type, omit for `None`
- **Raises**: Only document explicitly raised exceptions

Optional sections:
- **Side Effects**: For methods that modify state beyond return value
- **Notes**: For important caveats or implementation details
- **Behavior**: For complex conditional logic

If a type hint is missing or in accurate you may change it in the function definition.

### Properties

```python
@property
def cell_size(self) -> float:
    """Size of the cell in meters.

    Measured as the side length of the hexagon.
    """
```

## Canonical Vocabulary

Use these terms consistently:

| Term | Definition | NOT |
|------|------------|-----|
| **cell** | Hexagonal simulation unit | node, tile, hex |
| **grid** | 2D array of cells | mesh, lattice |
| **cell_grid** | The numpy array backing the simulation | grid_array |
| **cell_dict** | Dictionary mapping cell ID to Cell object | cell_map |
| **arrival time** | Time when fire first reaches a cell | ignition time (for initial), burn time |
| **ignition** | Initial fire start (user-specified) | arrival (for spread) |
| **burning** | Cell actively on fire (state=FIRE) | aflame, lit |
| **burnt** | Cell fully consumed (state=BURNT) | burned out |
| **frontier** | Set of burning cells adjacent to fuel | fire front, perimeter |
| **ensemble** | Collection of predictions with varied parameters | batch, set |
| **prediction** | Forward simulation from current state | forecast |
| **rate of spread (ROS)** | Fire spread velocity | spread rate |
| **steady-state ROS** | Equilibrium spread rate (`r_ss`) | final ROS |
| **fireline intensity** | Energy release rate per unit length | fire intensity |
| **fuel model** | Anderson 13 or Scott-Burgan 40 classification | fuel type (ambiguous) |
| **fuel content** | Remaining fuel fraction (0.0-1.0) | fuel load (different meaning) |
| **moisture content** | Fuel moisture fraction | humidity |
| **aspect** | Upslope direction (0°=North) | facing direction |
| **canopy** | Forest overstory vegetation | crown (use for crown fire) |

## Units and Coordinates

### Always Specify Units

Append unit suffix to variable names or document in docstring:

| Quantity | Suffix | Example |
|----------|--------|---------|
| Distance | `_m` | `cell_size_m`, `elevation_m` |
| Area | `_m2` | `cell_area_m2` |
| Time | `_s`, `_hr` | `duration_s`, `time_horizon_hr` |
| Velocity | `_mps` | `wind_speed_mps` |
| Angle | `_deg`, `_rad` | `slope_deg`, `aspect_rad` |
| Temperature | `_c`, `_f` | `temp_c` |

If unit is in docstring, format as: `"Elevation of the cell (meters)."` or `"Elevation of the cell in meters."`

### Coordinate Conventions

- **Grid indices**: `(row, col)` - row increases bottom-to-top visually, col increases left-to-right
- **Spatial coordinates**: `(x, y)` in meters - x increases left-to-right, y increases bottom-to-top
- **Angles**: Degrees, 0° = Vector facing North, increasing clockwise (compass convention) for aspect/wind direction
- **Hexagon orientation**: Point-up (vertex at top and bottom)

### Position Calculations

```
x_pos = col * cell_size * sqrt(3)           # even rows
x_pos = (col + 0.5) * cell_size * sqrt(3)   # odd rows
y_pos = row * cell_size * 1.5
```

## TODO:verify Rule

Use `TODO:verify` when documentation cannot be confirmed from code inspection alone:

```python
def calc_spread(self):
    """Calculate fire spread rate.

    Args:
        wind_factor (float): Wind adjustment multiplier. TODO:verify range [0,1] or unbounded

    Returns:
        float: Spread rate in m/s. TODO:verify if this accounts for slope
    """
```

**When to use:**
- Physical units are unclear from variable name and context
- Parameter valid ranges are not enforced in code
- Return value semantics are ambiguous
- Inherited behavior from external models (Rothermel, etc.)

**When NOT to use:**
- Information is clearly stated in code (assertions, docstrings, comments)
- Standard library or well-documented external API
- You can trace the value through the codebase

## Placeholder Docstrings

Incomplete docstrings use `_summary_` and `_description_`:

```python
def incomplete_method(self, param):
    """_summary_

    Args:
        param (type): _description_
    """
```

These indicate documentation debt. When updating, replace with actual content or `TODO:verify`.


## Unused legacy documentation

Remove any existing documentation which is no longer reflected in the code