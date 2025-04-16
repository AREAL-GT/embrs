from embrs.utilities.rothermel import *
from embrs.utilities.fuel_models import Anderson13
from embrs.utilities.unit_conversions import *
from embrs.fire_simulator.cell import Cell

def calc_R10(cell: Cell) -> float:
    # Calculate R_10 value for active crown fire ROS as described in Rothermel (1991)
    # Computes the rate of spread using the surface fire rate of spread in Fuel Model 10 with 
    # wind speed reduced by a factor of 0.4

    wind_speed_m_s, wind_dir_deg = cell.curr_wind

    wind_speed_ft_min = m_s_to_ft_min(wind_speed_m_s)

    fuel = Anderson13(10)
    slope_angle = np.deg2rad(cell.slope_deg)

    slope_dir_deg = cell.aspect

    if slope_angle == 0:
        rel_wind_dir_deg = 0
    elif wind_speed_m_s == 0:
        rel_wind_dir_deg = slope_dir_deg
    else:
        rel_wind_dir_deg = wind_dir_deg - slope_dir_deg
        if rel_wind_dir_deg < 0:
            rel_wind_dir_deg += 360

    omega = np.deg2rad(rel_wind_dir_deg)


    # Ensure fuel moisture dimensions are right for Rothermel calcs
    fmois = cell.fmois
    if len(fuel.rel_indices) != len(fmois):
        # If there is only one fuel moisture value make it that for all 3 classes
        if len(fmois) == 1:
            fmois = np.append(fmois, np.array([fmois[0], fmois[0]]))

        # If there are two, make the third the same as the second
        elif len(fmois) == 2:
            fmois = np.append(fmois, fmois[1]) 

    R_0, _ = calc_r_0(fuel, fmois)

    wind_speed_ft_min *= 0.4 # Adjustment factor for wind speed

    # TODO: everyhing below this is copied from the original calc_R_h function
    # TODO: need to refactor this to avoid code duplication    
    phi_w = calc_wind_factor(fuel, wind_speed_ft_min)
    phi_s = calc_slope_factor(fuel, slope_angle)

    t = 60
    d_w = R_0 * phi_w * t
    d_s = R_0 * phi_s * t

    x = d_s + d_w * np.cos(omega)
    y = d_w * np.sin(omega)

    D_h = np.sqrt(x**2 + y**2)

    R_h = R_0 + (D_h / t)

    return R_h