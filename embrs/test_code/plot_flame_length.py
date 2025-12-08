from embrs.models.fuel_models import Anderson13, ScottBurgan40
from embrs.utilities.data_classes import CellData
from embrs.fire_simulator.cell import Cell
from embrs.utilities.unit_conversions import mph_to_ft_min
from embrs.models.rothermel import *
import matplotlib.pyplot as plt

import numpy as np
wind_speeds_mph = np.linspace(0, 20, 20)

fuels = [Anderson13(4), ScottBurgan40(145), ScottBurgan40(147)]


for fuel in fuels:
    flame_lens = []
    rate_of_spread = []
    for wind_speed_mph in wind_speeds_mph:

        cell_data = CellData(fuel, 0, 0, 0, 0, 0, 0, 0, 0)
        cell = Cell(0, 0, 0, 30)
        cell._set_cell_data(cell_data)
        wind_speed_ft_min = mph_to_ft_min(wind_speed_mph)
        cell.curr_wind = (wind_speed_ft_min, 0)

        # According to RMRS-GTR-371 p. 47
        cell.fmois = np.array([0.06, 0.07, 0.08, 0.90, 0.90])

        R_h, R_0, I_r, _ = calc_r_h(cell)

        e = calc_eccentricity(fuel, R_h, R_0)

        R_h, I_h = calc_r_and_i_along_dir(cell, 0.0, R_h, I_r, 0.0, e)

        flame_len_ft = calc_flame_len(I_h)

        flame_lens.append(flame_len_ft)
        rate_of_spread.append(R_h)

    plt.figure(figsize=(10, 5))

    # Plot Flame Length vs Wind Speed
    plt.subplot(1, 2, 1)
    plt.plot(wind_speeds_mph, flame_lens)
    plt.xlabel('Wind Speed (mph)')
    plt.ylabel('Flame Length (ft)')
    plt.title('Flame Length vs Wind Speed')

    # Plot Rate of Spread vs Wind Speed
    plt.subplot(1, 2, 2)
    plt.plot(wind_speeds_mph, rate_of_spread)
    plt.xlabel('Wind Speed (mph)')
    plt.ylabel('Rate of Spread (ft/min)')
    plt.title('Rate of Spread vs Wind Speed')
    plt.legend([fuel.name])

    plt.tight_layout()


# plt.ylim(0, 50)
plt.xlim(0, 20)
# plt.xticks(np.arange(0, 21, 2))
# plt.yticks(np.arange(0, 51, 5))
plt.grid()
plt.show()