from embrs.utilities.fuel_models import Fuel
from embrs.utilities.unit_conversions import *
from embrs.fire_simulator.cell import Cell
import numpy as np
from typing import Tuple

# TODO: fill in docstrings

def calc_propagation_in_cell(cell: Cell, R_h_in:float = None) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        cell (Cell): _description_
        R_h_in (float, optional): _description_. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    
    R_h, R_0, I_r, alpha = calc_r_h(cell)
    spread_directions = np.deg2rad(cell.directions)
    if R_h < R_0 or R_0 == 0:
        I_list = [0] * len(spread_directions)
        r_list = [0] * len(spread_directions)

        return np.array(r_list), np.array(I_list)

    cell.reaction_intensity = I_r

    if R_h_in is not None:
        R_h = R_h_in

    t_r = 384 / cell.fuel.sav_ratio # Residence time
    H_a = I_r * t_r
    I_h = H_a * R_h

    e = calc_eccentricity(cell.fuel, R_h, R_0)

    r_list, I_list = calc_vals_for_all_directions(cell, R_h, I_r, alpha, e)
    
    return r_list, I_list, ft_min_to_m_s(R_h), I_h

def calc_vals_for_all_directions(cell, R_h, I_r, alpha, e):
    spread_directions = np.deg2rad(cell.directions)

    r_list = []
    I_list = []
    for decomp_dir in spread_directions:
        # rate of spread along gamma in ft/min, fireline intensity along gamma in Btu/ft/min
        r_gamma, I_gamma = calc_r_and_i_along_dir(cell, decomp_dir, R_h, I_r, alpha, e)

        r_gamma = ft_min_to_m_s(r_gamma) # convert to m/s

        r_list.append(r_gamma)
        I_list.append(I_gamma)

    
    return np.array(r_list), np.array(I_list)


def calc_r_h(cell, R_0: float = None, I_r: float = None) -> Tuple[float, float, float, float]:
    wind_speed_m_s, wind_dir_deg = cell.curr_wind


    wind_speed_ft_min = m_s_to_ft_min(wind_speed_m_s)

    wind_speed_ft_min *= cell.wind_adj_factor

    slope_angle_deg = cell.slope_deg
    slope_dir_deg = cell.aspect

    if slope_angle_deg == 0:
        rel_wind_dir_deg = 0
        cell.aspect = wind_dir_deg

    elif wind_speed_m_s == 0:
        rel_wind_dir_deg = 0
        wind_dir_deg = cell.aspect

    else:
        rel_wind_dir_deg = wind_dir_deg - slope_dir_deg
        if rel_wind_dir_deg < 0:
            rel_wind_dir_deg += 360

    rel_wind_dir = np.deg2rad(rel_wind_dir_deg)
    slope_angle = np.deg2rad(slope_angle_deg)


    fuel = cell.fuel
    m_f = cell.fmois

    if R_0 is None or I_r is None:
        R_0, I_r = calc_r_0(fuel, m_f)

    if R_0 == 0:
        # No spread in this cell
        return 0, 0, 0, 0
    
    phi_w = calc_wind_factor(fuel, wind_speed_ft_min)
    phi_s = calc_slope_factor(fuel, slope_angle)
    
    vec_speed, alpha = calc_wind_slope_vec(R_0, phi_w, phi_s, rel_wind_dir)
    
    R_h = R_0 + vec_speed

    return R_h, R_0, I_r, alpha

def calc_wind_slope_vec(R_0: float, phi_w: float, phi_s: float, angle: float) -> Tuple[float, float]:
    """_summary_

    Args:
        R_0 (float): _description_
        phi_w (float): _description_
        phi_s (float): _description_

    Returns:
        Tuple[float, float]: _description_
    """
    d_w = R_0 * phi_w
    d_s = R_0 * phi_s

    x = d_s + d_w * np.cos(angle)
    y = d_w * np.sin(angle)
    vec_mag = np.sqrt(x**2 + y**2)

    vec_dir = np.arcsin(y/vec_mag)

    return vec_mag, vec_dir



def calc_r_0(fuel: Fuel, m_f: np.ndarray) -> Tuple[float, float]:
    """_summary_

    Args:
        fuel (Fuel): _description_
        m_f (float): _description_

    Returns:
        Tuple[float, float]: _description_
    """

    m_f = get_working_m_f(fuel, m_f)

    # Calculate moisture damping constants
    dead_mf, live_mf = get_characteristic_moistures(fuel, m_f)
    live_mx = calc_live_mx(fuel, dead_mf)
    live_moisture_damping = calc_moisture_damping(live_mf, live_mx)
    dead_moisture_damping = calc_moisture_damping(dead_mf, fuel.dead_mx)

    flux_ratio = calc_flux_ratio(fuel)
    I_r = calc_I_r(fuel, dead_moisture_damping, live_moisture_damping)
    heat_sink = calc_heat_sink(fuel, m_f)

    R_0 = (I_r * flux_ratio)/heat_sink

    return R_0, I_r

def get_working_m_f(fuel: Fuel, m_f: np.ndarray):
    indices = fuel.rel_indices

    m_f_temp = []

    j = 0
    for i in range(5):
        if i in indices:
            m_f_temp.append(m_f[j])
            j += 1
        else:
            m_f_temp.append(0)

    return np.array(m_f_temp)


def get_characteristic_moistures(fuel: Fuel, m_f: np.ndarray):

    dead_mf = np.dot(fuel.f_ij[0:3], m_f[0:3])

    live_mf = np.dot(fuel.f_ij[3:5], m_f[3:5])

    return dead_mf, live_mf

def calc_live_mx(fuel: Fuel, m_f: float):

    W = fuel.W

    if W == np.inf:
        return fuel.dead_mx

    num = 0

    for i in range(3):
        num += m_f * fuel.w_0[i] * np.exp(-138/fuel.s[i])

    den = 0
    for i in range(3):
        den += fuel.w_0[i] * np.exp(-138/fuel.s[i])

    mf_dead = num/den

    mx = 2.9 * fuel.W * (1 - mf_dead / fuel.dead_mx) - 0.226

    return max(mx, fuel.dead_mx)


def calc_I_r(fuel: Fuel, dead_moist_damping: float, live_moist_damping: float) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_
        m_f (float): _description_

    Returns:
        float: _description_
    """

    mineral_damping = calc_mineral_damping()

    A = 133 * fuel.sav_ratio ** (-0.7913)

    max_reaction_vel = (fuel.sav_ratio ** 1.5) / (495 + 0.0594 * fuel.sav_ratio ** 1.5)
    opt_reaction_vel = max_reaction_vel * (fuel.rel_packing_ratio ** A) * np.exp(A*(1-fuel.rel_packing_ratio))

    dead_calc = fuel.w_n_dead * fuel.heat_content * dead_moist_damping * mineral_damping
    live_calc = fuel.w_n_live * fuel.heat_content * live_moist_damping * mineral_damping

    I_r = opt_reaction_vel * (dead_calc + live_calc)

    return I_r

def calc_flux_ratio(fuel: Fuel) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_

    Returns:
        float: _description_
    """

    rho_b = fuel.rho_b
    rho_p = fuel.rho_p
    sav_ratio = fuel.sav_ratio

    packing_ratio = rho_b / rho_p
    flux_ratio = (192 + 0.2595*sav_ratio)**(-1) * np.exp((0.792 + 0.681*sav_ratio**0.5)*(packing_ratio + 0.1))

    return flux_ratio

def calc_heat_sink(fuel: Fuel, m_f: np.ndarray) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_
        m_f (float): _description_

    Returns:
        float: _description_
    """

    rho_b = fuel.rho_b

    Q_ig = 250 + 1116 * m_f


    # Compute the heat sink term as per the equation
    heat_sink = 0
    for i in range(2): # loop through live and dead
        if i == 0:
            start = 0
            end = 3
        
        else:
            start = 3
            end = 5

        inner_sum = 0
        for j in range(start, end):
            if fuel.s[j] != 0:
                inner_sum += fuel.f_ij[j] * np.exp(-138/fuel.s[j])*Q_ig[j]

        heat_sink += fuel.f_i[i] * inner_sum

    heat_sink *= rho_b

    return heat_sink

def calc_r_and_i_along_dir(cell: Cell, decomp_dir: float, R_h: float, I_r: float, alpha: float, e: float) -> Tuple[float, float]:
    """_summary_

    Args:
        cell (Cell): _description_
        decomp_dir (float): _description_
        R_h (float): _description_
        I_r (float): _description_
        alpha (float): _description_
        e (float): _description_

    Returns:
        Tuple[float, float]: _description_
    """

    fuel = cell.fuel
    slope_dir = np.deg2rad(cell.aspect)

    gamma = abs((alpha + slope_dir) - decomp_dir) % (2*np.pi)
    gamma = np.min([gamma, 2*np.pi - gamma])

    R_gamma = R_h * ((1 - e)/(1 - e * np.cos(gamma)))

    t_r = 384 / fuel.sav_ratio # Residence time
    H_a = I_r * t_r
    I_gamma = H_a * R_gamma

    return R_gamma, I_gamma

def calc_E_B_C(fuel:Fuel) -> Tuple[float, float, float]:
    """_summary_

    Args:
        fuel (Fuel): _description_

    Returns:
        Tuple[float, float, float]: _description_
    """

    sav_ratio = fuel.sav_ratio

    E = 0.715 * np.exp(-3.59e-4 * sav_ratio)
    B = 0.02526 * sav_ratio ** 0.54
    C = 7.47 * np.exp(-0.133 * sav_ratio**0.55)

    return E, B, C

def calc_wind_factor(fuel:Fuel , wind_speed: float) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_
        wind_speed (float): _description_

    Returns:
        float: _description_
    """

    E, B, C = calc_E_B_C(fuel)
    phi_w = C * (wind_speed ** B) * fuel.rel_packing_ratio ** (-E)

    return phi_w

def calc_slope_factor(fuel: Fuel, phi: float) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_
        phi (float): _description_

    Returns:
        float: _description_
    """

    packing_ratio = fuel.rho_b / 32

    phi_s = 5.275 * (packing_ratio ** (-0.3)) * (np.tan(phi)) ** 2

    return phi_s


def calc_moisture_damping(m_f: float, m_x: float) -> float:
    """_summary_

    Args:
        m_f (float): _description_
        m_x (float): _description_

    Returns:
        float: _description_
    """
    r_m = m_f / m_x

    moist_damping = 1 - 2.59 * r_m + 5.11 * (r_m)**2 - 3.52 * (r_m)**3

    return max(0, moist_damping)

def calc_mineral_damping(s_e:float = 0.010) -> float:
    """_summary_

    Args:
        s_e (float, optional): _description_. Defaults to 0.010.

    Returns:
        float: _description_
    """

    mineral_damping = 0.174 * s_e ** (-0.19)

    return mineral_damping


def calc_effective_wind_factor(R_h: float, R_0: float) -> float:
    """_summary_

    Args:
        R_h (float): _description_
        R_0 (float): _description_

    Returns:
        float: _description_
    """

    phi_e = (R_h / R_0) - 1

    return phi_e

def calc_effective_wind_speed(fuel: Fuel, R_h: float, R_0: float) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_
        R_h (float): _description_
        R_0 (float): _description_

    Returns:
        float: _description_
    """
    E, B, C = calc_E_B_C(fuel)
    phi_e = calc_effective_wind_factor(R_h, R_0)


    u_e = ((phi_e * (fuel.rel_packing_ratio**E))/C) ** (-B)

    return u_e

def calc_eccentricity(fuel: Fuel, R_h: float, R_0: float):
    """_summary_

    Args:
        fuel (Fuel): _description_
        R_h (float): _description_
        R_0 (float): _description_

    Returns:
        _type_: _description_
    """

    u_e = calc_effective_wind_speed(fuel, R_h, R_0)

    u_e_mph = u_e * 0.0113636

    z = 1 + 0.25 * u_e_mph
    e = ((z**2 - 1)**0.5)/z

    return e
