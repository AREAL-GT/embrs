# Fire Simulator

The `embrs.fire_simulator` package contains the core runtime components of EMBRS. `FireSim` drives the main simulation loop — advancing time steps, propagating fire, and coordinating weather, logging, and visualization. `Cell` represents a single hexagonal unit in the grid and stores all per-cell fire behavior state. `RealTimeVisualizer` provides a live rendering interface that runs alongside a simulation.

Most users interact with `FireSim` indirectly through a [`ControlClass`](user_code.md) subclass, using the methods inherited from [`BaseFireSim`](interface_reference.md). The classes below are primarily useful for understanding simulation internals or building custom tooling.

## FireSim

The main simulation class. Extends [`BaseFireSim`](interface_reference.md) with the Rothermel fire spread loop, crown fire checks, logging, and visualization hooks. User control code receives a `FireSim` instance as the `fire` argument in `process_state()`.

::: embrs.fire_simulator.fire

## Cell

Each cell tracks its terrain (elevation, slope, aspect), fuel model, fuel moisture, fire state, and spread geometry. Cells are typically accessed through `FireSim` lookup methods like `get_cell_from_xy()` or `get_cell_from_indices()` rather than constructed directly.

::: embrs.fire_simulator.cell

## RealTimeVisualizer

Renders fire spread, agent positions, and weather data in real-time during a simulation. Created internally by the simulation launcher — most users do not need to instantiate this directly.

::: embrs.fire_simulator.visualizer
