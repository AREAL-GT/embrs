# Rothermel Spread Model

This module implements Rothermel's (1972) surface fire spread equations, computing rate of spread (ROS), fireline intensity, and fire ellipse geometry for each cell in the simulation grid. It is the core fire behavior engine called by `BaseFireSim` at every time step — users do not call these functions directly but can inspect the cell-level outputs they produce (e.g., `cell.r_ss`, `cell.I_ss`).

::: embrs.models.rothermel
