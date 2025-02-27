from embrs.utilities.fuel_models import Fuel
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
    
    wind_speed_m_s, wind_dir_deg = cell.curr_wind

    wind_speed_ft_min = 196.85 * wind_speed_m_s * cell.wind_adj_factor

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
    spread_directions = np.deg2rad(cell.directions)

    R_h, R_0, I_r, alpha = calc_r_h(cell, wind_speed_ft_min, slope_angle_deg, rel_wind_dir)

    if R_h_in is not None:
        R_h = R_h_in

    e = calc_eccentricity(cell.fuel_type, R_h, R_0)

    r_list = []
    I_list = []
    for decomp_dir in spread_directions:
        # rate of spread along gamma in ft/min, fireline intensity along gamma in Btu/ft/min
        r_gamma, I_gamma = calc_r_and_i_along_dir(cell, decomp_dir, R_h, I_r, alpha, e)

        r_gamma /= 196.85 # convert to m/s
        I_gamma *= 0.05767 # convert to kW/m # TODO: double check this conversion

        r_list.append(r_gamma)
        I_list.append(I_gamma)

    return np.array(r_list), np.array(I_list)

def calc_r_0(fuel: Fuel, m_f: float) -> Tuple[float, float]:
    """_summary_

    Args:
        fuel (Fuel): _description_
        m_f (float): _description_

    Returns:
        Tuple[float, float]: _description_
    """

    flux_ratio = calc_flux_ratio(fuel)
    I_r = calc_I_r(fuel, m_f)
    heat_sink = calc_heat_sink(fuel, m_f)

    R_0 = (I_r * flux_ratio)/heat_sink

    return R_0, I_r

def calc_I_r(fuel: Fuel, m_f: float) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_
        m_f (float): _description_

    Returns:
        float: _description_
    """

    moist_damping = calc_moisture_damping(m_f, fuel.m_x)
    mineral_damping = calc_mineral_damping()

    A = 133 * fuel.sav_ratio ** (-0.7913)

    max_reaction_vel = (fuel.sav_ratio ** 1.5) * (495 + 0.0594 * fuel.sav_ratio ** 1.5) ** (-1)
    opt_reaction_vel = max_reaction_vel * (fuel.rel_packing_ratio ** A) * np.exp(A*(1-fuel.rel_packing_ratio))

    I_r = opt_reaction_vel * fuel.net_fuel_load * fuel.heat_content * moist_damping * mineral_damping

    return I_r

def calc_flux_ratio(fuel: Fuel) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_

    Returns:
        float: _description_
    """

    rho_b = fuel.rho_b
    sav_ratio = fuel.sav_ratio

    packing_ratio = rho_b / 32    
    flux_ratio = (192 + 0.2595*sav_ratio)**(-1) * np.exp((0.792 + 0.681*sav_ratio**0.5)*(packing_ratio + 0.1))

    return flux_ratio

def calc_heat_sink(fuel: Fuel, m_f: float) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_
        m_f (float): _description_

    Returns:
        float: _description_
    """

    rho_b = fuel.rho_b
    sav_ratio = fuel.sav_ratio

    epsilon = np.exp(-138/sav_ratio)
    Q_ig = 250 + 1116 * m_f

    heat_sink = rho_b * epsilon * Q_ig

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

    fuel = cell.fuel_type
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

    return moist_damping

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


    u_e = (((phi_e * fuel.rel_packing_ratio**E)/C) ** (1/B))

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

def calc_r_h(cell: Cell, wind_speed: float, 
             slope_angle: float, omega: float,
             R_0: float = None, I_r: float = None)-> Tuple[float, float, float, float]:
    
    """_summary_

    Args:
        cell (Cell): _description_
        wind_speed (float): _description_
        slope_angle (float): _description_
        omega (float): _description_
        R_0 (float, optional): _description_. Defaults to None.
        I_r (float, optional): _description_. Defaults to None.

    Returns:
        Tuple[float, float, float, float]: _description_
    """
    fuel = cell.fuel_type
    m_f = cell.m_f
    
    if R_0 is None or I_r is None:
        R_0, I_r = calc_r_0(fuel, m_f)

    phi_w = calc_wind_factor(fuel, wind_speed)
    phi_s = calc_slope_factor(fuel, slope_angle)

    t = 60

    d_w = R_0 * phi_w * t
    d_s = R_0 * phi_s * t

    x = d_s + d_w * np.cos(omega)
    y = d_w * np.sin(omega)

    D_h = np.sqrt(x**2 + y**2)

    R_h = R_0 + (D_h / t)

    alpha = np.arcsin(y/D_h)

    return R_h, R_0, I_r, alpha

