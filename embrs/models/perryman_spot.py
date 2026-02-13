"""Statistical firebrand spotting model based on Perryman et al.

Generate firebrand landing locations using lognormal and normal
distributions parameterized by fireline intensity and wind speed. The
parallel (downwind) distribution depends on the Froude number regime;
the perpendicular distribution is scaled by cell width.

Classes:
    - PerrymanSpotting: Sample firebrand landing positions from a burning cell.

References:
    Perryman, H. A., et al. (2013). A mathematical model of spot fire
    transport. International Journal of Wildland Fire, 22, 350-358.
"""

from embrs.fire_simulator.cell import Cell
from embrs.utilities.unit_conversions import *

import numpy as np

# Model constants (Perryman et al. 2013, Table 2)
num_firebrands = 50
g = 9.8       # Acceleration due to gravity (m/s²)
rN = 1.1      # Ambient gas density (kg/m³)
cpg = 1121    # Specific heat of gas (kJ/(kg·K))
TN = 300      # Ambient temperature (K)

class PerrymanSpotting:
    """Statistical firebrand landing model.

    Sample firebrand landing coordinates from probability distributions
    parameterized by wind speed and fireline intensity. Firebrands that
    land within 50 meters of the source cell are filtered out.

    Attributes:
        embers (list[dict]): Accumulated firebrand landing positions,
            each with keys 'x', 'y', 'd' (distance in meters).
        spot_delay_s (float): Delay before spotted fires can ignite
            (seconds).
    """

    def __init__(self, spot_delay_s: float, limits: tuple):
        """Initialize the Perryman spotting model.

        Args:
            spot_delay_s (float): Spot fire ignition delay (seconds).
            limits (tuple): ``(x_lim, y_lim)`` simulation domain bounds
                (meters).
        """
        self.embers = []
        self.spot_delay_s = spot_delay_s

        self.x_lim, self.y_lim = limits

    def loft(self, cell: Cell):
        """Sample firebrand landing locations from a burning cell.

        Compute downwind (parallel) and crosswind (perpendicular) landing
        distances, combine them into absolute coordinates, and store
        firebrands that land beyond 50 meters from the source.

        Args:
            cell (Cell): Burning cell to generate firebrands from.

        Side Effects:
            Sets ``cell.lofted = True``. Appends ember dicts to
            ``self.embers``.
        """
        cell.lofted = True

        wind_speed, wind_dir_deg = cell.curr_wind()

        if wind_speed == 0:
            return

        # Compute distances parallel and perpendicular to wind dir
        par_distances = self.compute_parallel_distances(cell)
        perp_distances = self.compute_perp_distances(cell)

        # Generate actual coordinate where each brand will land
        
        # Get the downwind distances
        wind_dir = np.deg2rad(wind_dir_deg)
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

        # Filter out distances below minimum spotting distance
        filtered_indices = np.where(d > 50)[0]
        filtered_d = d[filtered_indices]
        x_coords = x_coords[filtered_indices]
        y_coords = y_coords[filtered_indices]

        for i in range(len(filtered_d)):
            ember = {
                'x': x_coords[i],
                'y': y_coords[i],
                'd': filtered_d[i]
            }

            self.embers.append(ember)

    def compute_parallel_distances(self, cell: Cell) -> np.ndarray:
        """Sample downwind (parallel) firebrand distances.

        Compute the canopy-height wind speed, fireline intensity, and
        Froude number, then sample distances from a lognormal distribution
        whose parameters depend on the Froude number regime.

        Args:
            cell (Cell): Burning cell providing wind, canopy, and intensity.

        Returns:
            np.ndarray: Downwind distances for each firebrand (meters),
                shape ``(num_firebrands,)``.
        """
        wind_spd_ft_s = m_to_ft(cell.curr_wind()[0])
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

    def compute_perp_distances(self, cell: Cell) -> np.ndarray:
        """Sample crosswind (perpendicular) firebrand distances.

        Use a normal distribution centered at zero with standard deviation
        equal to half the cell's flat-to-flat width.

        Args:
            cell (Cell): Cell providing ``cell_size`` for scale computation.

        Returns:
            np.ndarray: Crosswind distances for each firebrand (meters),
                shape ``(num_firebrands,)``.
        """
        # Horizontal standard dev. is half of cell width
        flat_to_flat = cell.cell_size * (np.sqrt(3)/2)
        sv = flat_to_flat / 2

        perp_distances = np.random.normal(loc=0, scale=sv, size=num_firebrands)

        return perp_distances
