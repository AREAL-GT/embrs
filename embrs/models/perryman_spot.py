

from embrs.fire_simulator.cell import Cell
from embrs.utilities.unit_conversions import *

import numpy as np

# --- Define Model Constants (from Table 2 and text) ---
num_firebrands = 50
g = 9.8  # Acceleration due to gravity (m s-2) [4]
rN = 1.1 # Ambient gas density (kg m-3) [4]
cpg = 1121 # Specific heat of gas (kJ kg-1 K-1) [4] - Note: Source uses kJ, need J for energy calculations
TN = 300 # Ambient temperature (K) [4]

class PerrymanSpotting:

    def __init__(self, spot_delay_s, limits):
        self.embers = []
        self.spot_delay_s = spot_delay_s

        self.x_lim, self.y_lim = limits

    def loft(self, cell: Cell):
        cell.lofted = True

        if cell.curr_wind[0] == 0:
            return

        # Compute distances parallel and perpendicular to wind dir
        par_distances = self.compute_parallel_distances(cell)
        perp_distances = self.compute_perp_distances(cell)

        # Generate actual coordinate where each brand will land
        
        # Get the downwind distances
        wind_dir = np.deg2rad(cell.curr_wind[1])
        down_wind_x_vec = par_distances * np.sin(wind_dir)
        down_wind_y_vec = par_distances * np.cos(wind_dir)

        # Get the distances perpendicular to wind direction
        perp_dir = wind_dir + (np.pi / 2)
        perp_x_vec = perp_distances * np.sin(perp_dir)
        perp_y_vec = perp_distances * np.cos(perp_dir)

        # Add the vectors
        tot_vec_x = down_wind_x_vec + perp_x_vec
        tot_vec_y = down_wind_y_vec + perp_y_vec

        # Compute absolute coordinates relative to the cell position
        x_coords = cell.x_pos + tot_vec_x
        y_coords = cell.y_pos + tot_vec_y

        # Ensure coordinates are within the limits
        x_coords = np.clip(x_coords, 0, self.x_lim)
        y_coords = np.clip(y_coords, 0, self.y_lim)

        # Calculate the distance from the cell center to each ember
        d = np.sqrt((cell.x_pos - x_coords)**2 + (cell.y_pos - y_coords)**2)

        for i in range(len(d)):
            ember = {
                'x': x_coords[i],
                'y': y_coords[i],
                'd': d[i]
            }

            self.embers.append(ember)

    def compute_parallel_distances(self, cell: Cell):
        # Convert 20 ft wind speed to wind at canopy height
        # Albini & Baughman 1979 Res. Pap. INT-221
        wind_spd_ft_s = m_to_ft(cell.curr_wind[0])
        canopy_ht = m_to_ft(cell.canopy_height)
        u_h_ft_s = wind_spd_ft_s / (np.log((20 + 0.36 * canopy_ht)/(0.1313 * canopy_ht)))
        u_h = ft_to_m(u_h_ft_s)
        
        # Get the fireline intensity in kW/m
        I_f_btu_ft_min = np.max(cell.I_ss)
        I_f_kW_m = BTU_ft_min_to_kW_m(I_f_btu_ft_min)

        # Compute Froude number to compute the lognormal distribution
        denom = np.sqrt(g * (I_f_kW_m / (rN * cpg * TN * np.sqrt(g)))**(2/3.0))

        if denom == 0:
            Fr = float('inf')
        else:
            Fr = u_h / denom

        # Set lognormal distribution parameters
        if Fr <= 1:
            sigma = 0.86 * (I_f_kW_m**(-0.21)*u_h**(0.44)) + 0.19
            mu =    1.47 * (I_f_kW_m**(0.54)*u_h**(-0.55)) + 1.14

        else:
            sigma = 4.95 * (I_f_kW_m**(-0.01)*u_h**(-0.02)) - 3.48
            mu =    1.32 * (I_f_kW_m**(0.26)*u_h**(0.11)) - 0.02

        # Sample distances for firebrands
        parallel_distances = np.random.lognormal(mean=mu, sigma=sigma, size=num_firebrands)
        
        return parallel_distances

    def compute_perp_distances(self, cell: Cell):
        # Horizontal standard dev. is half of cell width
        flat_to_flat = cell.cell_size * (np.sqrt(3)/2)
        sv = flat_to_flat / 2

        perp_distances = np.random.normal(loc=0, scale=sv, size=num_firebrands)

        return perp_distances
