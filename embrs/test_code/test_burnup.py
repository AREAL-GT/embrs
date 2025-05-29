from embrs.fire_simulator.cell import Cell
from embrs.models.fuel_models import Anderson13
from embrs.models.burnup import Burnup

import numpy as np
# Inputs
fuel_model = 8

duff_loading = 1.0
dfm = 0.1

fmois = 0.20
f_i = 50
t_i = 60
u = 0
depth = 0.3
tamb = 27


r_0 = 1.8
dr = 0.4
dt = 15



loading_class_strs = ["1hr", "10hr", "100hr", "Live_W", "Live_H"]


# Create cell
cell = Cell(0, 0, 0, 30)
fuel = Anderson13(fuel_model)
cell._set_cell_data(fuel, 0, 0, 0, 0, 0, duff_loading, fmois, fmois, fmois)

cell.wdry =  np.array([1, 1, 1, 1, 1])
cell.fmois = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
cell.sigma = np.array([1480, 394, 105, 30, 30])

# Compute burn history
cell.burn_history = []


burn_mgr = Burnup(cell)
burn_mgr.set_fire_data(3000, f_i, t_i, u, depth, tamb, r_0, dr, dt, duff_loading, dfm)
burn_mgr.fi_min = 0.1

burn_mgr.arrays()
now = 1
d_time = burn_mgr.ti
burn_mgr.duff_burn()

burn_mgr.fi = burn_mgr.fistart

if not (burn_mgr.start(d_time, now)):
    print("No ignition")
    exit()

burn_mgr.fi = burn_mgr.fire_intensity()


if d_time > burn_mgr.tdf:
    burn_mgr.dfi = 0

while now <= burn_mgr.ntimes:
    burn_mgr.step(burn_mgr.dt, burn_mgr.tis, burn_mgr.dfi)
    now += 1

    d_time += burn_mgr.dt
    if d_time > burn_mgr.tdf:
        burn_mgr.dfi = 0

    fi = burn_mgr.fire_intensity()
    
    if fi <= burn_mgr.fi_min:
        break

# if burn_mgr.start_loop():
#     cont = True
#     i = 0
#     while cont:
#         cont = burn_mgr.burn_loop()

remaining_fracs = burn_mgr.get_updated_fuel_loading()

for i in range(3):
    print(f"Class: {loading_class_strs[i]}")
    print(f"Remaining: {remaining_fracs[i] * cell.fuel.w_0[i]}")

print(f"burn time: {d_time}")























