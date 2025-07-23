from embrs.models.fuel_models import Fuel
from embrs.utilities.fire_util import CrownStatus
from embrs.utilities.unit_conversions import *
from embrs.fire_simulator.cell import Cell
import numpy as np
from typing import Tuple

# TODO: fill in docstrings

def surface_fire(cell: Cell):
    """_summary_

    Args:
        cell (Cell): _description_
        R_h_in (float, optional): _description_. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    R_h, R_0, I_r, alpha = calc_r_h(cell)
    cell.alpha = alpha
    spread_directions = np.deg2rad(cell.directions)
    
    if R_h < R_0 or R_0 == 0:
        I_list = np.zeros_like(spread_directions)
        r_list = np.zeros_like(spread_directions)

        return np.array(r_list), np.array(I_list)

    cell.reaction_intensity = I_r

    e = calc_eccentricity(cell.fuel, R_h, R_0)
    cell.e = e

    r_list, I_list = calc_vals_for_all_directions(cell, R_h, I_r, alpha, e)

    cell.r_ss = r_list
    cell.I_ss = I_list
    cell.r_h_ss = np.max(r_list)

def accelerate(cell: Cell, time_step: float):
    # Mask where acceleration is needed
    mask_accel = cell.r_t < cell.r_ss

    # Mask for nonzero r_t (partial acceleration history)
    mask_nonzero = mask_accel & (cell.r_t != 0)

    # Mask for zero r_t (accelerate from rest)
    mask_zero = mask_accel & (cell.r_t == 0)

    # --- Handle nonzero r_t ---
    if np.any(mask_nonzero):
        r_t = cell.r_t[mask_nonzero]
        r_ss = cell.r_ss[mask_nonzero]
        a_a = cell.a_a

        ratio = np.clip(r_t / (r_ss + 1e-7), 0.0, 1.0 - 1e-7)
        T_t = np.log(1 - ratio) / (-a_a)

        D_t = r_ss * (T_t + np.exp(-a_a * T_t) / a_a - (1 / a_a))
        D_t1 = r_ss * (
            time_step + T_t + np.exp(-a_a * (time_step + T_t)) / a_a - (1 / a_a)
        )
        avg_ros = (D_t1 - D_t) / time_step
        r_t1 = r_ss * (1 - np.exp(-a_a * (time_step + T_t)))

        # Apply updates
        cell.r_t[mask_nonzero] = r_t1
        cell.avg_ros[mask_nonzero] = avg_ros
        cell.I_t[mask_nonzero] = (r_t1 / (r_ss + 1e-7)) * cell.I_ss[mask_nonzero]

    # --- Handle zero r_t ---
    if np.any(mask_zero):
        r_ss = cell.r_ss[mask_zero]
        a_a = cell.a_a if np.isscalar(cell.a_a) else cell.a_a[mask_zero]

        r_t1 = r_ss * (1 - np.exp(-a_a * time_step))
        D_t = r_ss * (
            time_step + np.exp(-a_a * time_step) / a_a - (1 / a_a)
        )
        avg_ros = D_t / time_step

        # Apply updates
        cell.r_t[mask_zero] = r_t1
        cell.avg_ros[mask_zero] = avg_ros
        cell.I_t[mask_zero] = (r_t1 / (r_ss + 1e-7)) * cell.I_ss[mask_zero]

    # --- Handle steady state ---
    mask_steady = ~mask_accel
    if np.any(mask_steady):
        cell.r_t[mask_steady] = cell.r_ss[mask_steady]
        cell.avg_ros[mask_steady] = cell.r_ss[mask_steady]


def calc_vals_for_all_directions(cell, R_h, I_r, alpha, e):
    spread_directions = np.deg2rad(cell.directions)

    gamma = np.abs(((alpha + np.deg2rad(cell.aspect)) - spread_directions) % (2*np.pi))
    gamma = np.minimum(gamma, 2*np.pi - gamma)

    R_gamma = R_h * ((1 - e)/(1 - e * np.cos(gamma)))

    t_r = 384 / cell.fuel.sav_ratio
    H_a = I_r * t_r
    I_gamma = H_a * R_gamma

    r_list = ft_min_to_m_s(R_gamma)
    return r_list, I_gamma

def calc_r_h(cell, R_0: float = None, I_r: float = None) -> Tuple[float, float, float, float]:
    wind_speed_m_s, wind_dir_deg = cell.curr_wind()
    
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

    # Enforce maximum wind speed
    U_max = 0.9 * I_r
    wind_speed_ft_min = np.min([U_max, wind_speed_ft_min])

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

    if vec_mag == 0:
        vec_dir = 0

    else:
        vec_dir = np.arctan2(y, x)

    return vec_mag, vec_dir

def calc_r_0(fuel: Fuel, m_f: np.ndarray) -> Tuple[float, float]:
    """_summary_

    Args:
        fuel (Fuel): _description_
        m_f (float): _description_

    Returns:
        Tuple[float, float]: _description_
    """

    # Calculate moisture damping constants
    dead_mf, live_mf = get_characteristic_moistures(fuel, m_f)
    live_mx = calc_live_mx(fuel, dead_mf)
    live_moisture_damping = calc_moisture_damping(live_mf, live_mx)
    dead_moisture_damping = calc_moisture_damping(dead_mf, fuel.dead_mx)

    I_r = calc_I_r(fuel, dead_moisture_damping, live_moisture_damping)
    heat_sink = calc_heat_sink(fuel, m_f)

    R_0 = (I_r * fuel.flux_ratio)/heat_sink

    return R_0, I_r # ft/min, BTU/ft^2-min


def get_characteristic_moistures(fuel: Fuel, m_f: np.ndarray):

    dead_mf = np.dot(fuel.f_dead_arr, m_f[0:4])
    live_mf = np.dot(fuel.f_live_arr, m_f[4:])

    return dead_mf, live_mf

def calc_live_mx(fuel: Fuel, m_f: float):

    W = fuel.W

    if W == np.inf:
        return fuel.dead_mx

    num = 0
    den = 0
    for i in range(4):
        if fuel.s[i] != 0:
            num += m_f * fuel.load[i] * np.exp(-138/fuel.s[i])
            den += fuel.load[i] * np.exp(-138/fuel.s[i])

    mf_dead = num/den

    mx = 2.9 * W * (1 - mf_dead / fuel.dead_mx) - 0.226

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

    dead_calc = fuel.w_n_dead * fuel.heat_content * dead_moist_damping * mineral_damping
    live_calc = fuel.w_n_live * fuel.heat_content * live_moist_damping * mineral_damping

    I_r = fuel.gamma * (dead_calc + live_calc)

    return I_r

def calc_heat_sink(fuel: Fuel, m_f: np.ndarray) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_
        m_f (float): _description_

    Returns:
        float: _description_
    """

    Q_ig = 250 + 1116 * m_f


    # Compute the heat sink term as per the equation
    heat_sink = 0
    
    dead_sum = 0
    for j in range(4):
        if fuel.s[j] != 0:
            dead_sum += fuel.f_dead_arr[j] * np.exp(-138/fuel.s[j]) * Q_ig[j]

    heat_sink += fuel.f_i[0] * dead_sum

    live_sum = 0
    for j in range(2):
        if fuel.s[4+j] != 0:
            live_sum += fuel.f_live_arr[j] * np.exp(-138/fuel.s[4+j]) * Q_ig[4+j]
        
    heat_sink += fuel.f_i[1] * live_sum
    heat_sink *= fuel.rho_b

    return heat_sink


def calc_wind_factor(fuel:Fuel , wind_speed: float) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_
        wind_speed (float): _description_

    Returns:
        float: _description_
    """
    phi_w = fuel.C * (wind_speed ** fuel.B) * fuel.rat ** (-fuel.E)

    return phi_w

def calc_slope_factor(fuel: Fuel, phi: float) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_
        phi (float): _description_

    Returns:
        float: _description_
    """
    packing_ratio = fuel.rho_b / fuel.rho_p

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


    if R_h <= R_0:
        phi_e = 0

    else: 
        phi_e = calc_effective_wind_factor(R_h, R_0)

    u_e = ((phi_e * (fuel.rat**fuel.E))/fuel.C) ** (1/fuel.B)

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
    u_e_ms = ft_min_to_m_s(u_e)
    z = 0.936 * np.exp(0.2566 * u_e_ms) + 0.461 * np.exp(-0.1548 * u_e_ms) - 0.397
    z = np.min([z, 8.0])
    e = ((z**2 - 1)**0.5)/z

    return e

def calc_flame_len(cell: Cell):
    # Fireline intensity in Btu/ft/min
    fli = np.max(cell.I_ss)
    fli /= 60 # convert to Btu/ft/s

    if cell._crown_status == CrownStatus.NONE:
        # Surface fire
        # Brown and Davis 1973 pg. 175
        flame_len_ft = 0.45 * fli ** (0.46)

    else:
        flame_len_ft = (0.2 * (fli ** (2/3))) # in feet

    return flame_len_ft