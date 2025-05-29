
from embrs.models.fuel_models import Anderson13
from embrs.models.rothermel import calc_I_r, calc_moisture_damping
import matplotlib.pyplot as plt
import numpy as np


fuel_1 = Anderson13(1)
fuel_8 = Anderson13(8)
fuel_1.dead_mx = 0.3

fuels = [fuel_1, fuel_8]
fms = np.linspace(0.025, 0.30, 500)

live_moist_damping = 1


plt.figure(figsize=(10, 6))
for fuel in fuels:
    I_r_vals = []
    for fm in fms:
        dead_moist_damping = calc_moisture_damping(fm, 0.3)
        I_r = calc_I_r(fuel, dead_moist_damping, live_moist_damping)
        I_r_vals.append(I_r)

    plt.plot(fms, I_r_vals, label=f"Fuel Model {fuel.model_num}")

plt.xlabel("Dead fuel moisture (fraction)")
plt.ylabel("Reaction Intensity (Btu/ft^2/min)")
plt.grid(True)
plt.legend()
plt.show()




















