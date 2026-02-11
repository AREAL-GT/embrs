"""Rothermel fire spread model implementation.

Compute surface fire rate of spread (ROS), fireline intensity, and related
quantities using Rothermel's (1972) equations. All internal calculations use
imperial units (ft/min, BTU/ft²·min); public outputs are converted to metric
(m/s) unless noted otherwise.

Functions:
    - surface_fire: Compute steady-state ROS and fireline intensity for a cell.
    - accelerate: Apply fire acceleration from rest toward steady-state ROS.
    - calc_r_h: Compute head-fire ROS combining wind and slope effects.
    - calc_r_0: Compute no-wind, no-slope base ROS and reaction intensity.
    - calc_eccentricity: Compute fire ellipse eccentricity from effective wind speed.
    - calc_flame_len: Estimate flame length from fireline intensity.

References:
    Rothermel, R. C. (1972). A mathematical model for predicting fire spread
    in wildland fuels. USDA Forest Service Research Paper INT-115.
"""

from embrs.models.fuel_models import Fuel
from embrs.utilities.fire_util import CrownStatus
from embrs.utilities.unit_conversions import *
from embrs.fire_simulator.cell import Cell
import numpy as np
from typing import Tuple

def surface_fire(cell: Cell):
    """Compute steady-state surface fire ROS and fireline intensity for a cell.

    Calculate the head-fire rate of spread (R_h), then resolve spread rates and
    fireline intensities along all 12 spread directions using fire ellipse
    geometry. Results are stored directly on the cell object.

    Args:
        cell (Cell): Cell to evaluate. Must have fuel, moisture, wind, slope,
            and direction attributes populated.

    Side Effects:
        Sets ``cell.r_ss`` (m/s), ``cell.I_ss`` (BTU/ft/min),
        ``cell.r_h_ss`` (m/s), ``cell.reaction_intensity`` (BTU/ft²/min),
        ``cell.alpha`` (radians), and ``cell.e`` (eccentricity) on the cell.
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

    # r in m/s, I in btu/ft/min
    cell.r_ss = r_list
    cell.I_ss = I_list
    cell.r_h_ss = np.max(r_list)

def accelerate(cell: Cell, time_step: float):
    """Apply fire acceleration toward steady-state ROS.

    Update the transient rate of spread (``cell.r_t``) and average ROS
    (``cell.avg_ros``) for each spread direction using the exponential
    acceleration model (McAlpine 1989). Directions already at or above
    steady-state are clamped.

    Args:
        cell (Cell): Burning cell with ``r_ss``, ``r_t``, ``a_a`` set.
        time_step (float): Simulation time step in seconds.

    Side Effects:
        Updates ``cell.r_t``, ``cell.avg_ros``, and ``cell.I_t`` in-place.
    """
    # Mask where acceleration is needed
    mask_accel = cell.r_t < cell.r_ss

    # Mask for nonzero r_t (partial acceleration history)
    mask_nonzero = mask_accel & (cell.r_t != 0)

    # Mask for zero r_t (accelerate from rest)
    mask_zero = mask_accel & (cell.r_t == 0)

    # --- Handle nonzero r_t ---
    if mask_nonzero.any():
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
    if mask_zero.any():
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
    if mask_steady.any():
        cell.r_t[mask_steady] = cell.r_ss[mask_steady]
        cell.avg_ros[mask_steady] = cell.r_ss[mask_steady]

def calc_vals_for_all_directions(cell: Cell, R_h: float, I_r: float, alpha: float,
                                 e: float, I_h: float = None):
    """Compute ROS and fireline intensity along all spread directions.

    Use the fire ellipse (eccentricity ``e``) and the combined wind/slope
    heading ``alpha`` to resolve the head-fire ROS into each of the cell's
    spread directions.

    Args:
        cell (Cell): Cell providing directions and fuel properties.
        R_h (float): Head-fire rate of spread (ft/min for surface, m/min
            for crown fire).
        I_r (float): Reaction intensity (BTU/ft²/min). Ignored when
            ``I_h`` is provided.
        alpha (float): Combined wind/slope heading in radians, relative to
            the cell's aspect (upslope direction).
        e (float): Fire ellipse eccentricity in [0, 1).
        I_h (float, optional): Head-fire fireline intensity (BTU/ft/min).
            When provided, directional intensities are scaled from this
            value instead of being computed from ``I_r``.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(r_list, I_list)`` where
            ``r_list`` is ROS in m/s per direction and ``I_list`` is
            fireline intensity in BTU/ft/min per direction.
    """
    spread_directions = np.deg2rad(cell.directions)

    gamma = np.abs(((alpha + np.deg2rad(cell.aspect)) - spread_directions) % (2*np.pi))
    gamma = np.minimum(gamma, 2*np.pi - gamma)

    R_gamma = R_h * ((1 - e)/(1 - e * np.cos(gamma)))

    if I_h is None:
        t_r = 384 / cell.fuel.sav_ratio
        H_a = I_r * t_r
        I_gamma = H_a * R_gamma # BTU/ft/min

    else:
        I_gamma = I_h * (R_gamma / R_h) # BTU/ft/min

    r_list = ft_min_to_m_s(R_gamma)
    return r_list, I_gamma

def calc_r_h(cell: Cell, R_0: float = None, I_r: float = None) -> Tuple[float, float, float, float]:
    """Compute head-fire rate of spread combining wind and slope effects.

    Resolve the wind and slope vectors to determine the maximum spread
    direction (``alpha``) and head-fire ROS (``R_h``). Wind speed is capped
    at 0.9 × reaction intensity per Rothermel's wind limit.

    Args:
        cell (Cell): Cell with wind, slope, fuel, and moisture data.
        R_0 (float, optional): Pre-computed no-wind, no-slope ROS (ft/min).
            Computed internally if None.
        I_r (float, optional): Pre-computed reaction intensity
            (BTU/ft²/min). Computed internally if None.

    Returns:
        Tuple[float, float, float, float]: ``(R_h, R_0, I_r, alpha)`` where
            ``R_h`` is head-fire ROS (ft/min), ``R_0`` is base ROS (ft/min),
            ``I_r`` is reaction intensity (BTU/ft²/min), and ``alpha`` is the
            combined wind/slope heading (radians).

    Side Effects:
        May update ``cell.aspect`` when slope is zero (set to wind direction).
    """
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
    """Compute the combined wind and slope vector magnitude and direction.

    Resolve wind and slope spread factors into a single resultant vector
    using Rothermel's vector addition method.

    Args:
        R_0 (float): No-wind, no-slope ROS (ft/min).
        phi_w (float): Wind factor (dimensionless).
        phi_s (float): Slope factor (dimensionless).
        angle (float): Angle between wind and upslope directions (radians).

    Returns:
        Tuple[float, float]: ``(vec_mag, vec_dir)`` where ``vec_mag`` is the
            combined wind/slope spread increment (ft/min) and ``vec_dir`` is
            the direction of the resultant vector (radians).
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
    """Compute no-wind, no-slope base rate of spread and reaction intensity.

    Evaluate the Rothermel (1972) equations for base ROS using fuel
    properties and moisture content. This is the fundamental spread rate
    before wind and slope adjustments.

    Args:
        fuel (Fuel): Fuel model with precomputed constants.
        m_f (np.ndarray): Fuel moisture content array of shape (6,) with
            entries [1h, 10h, 100h, dead herb, live herb, live woody] as
            fractions (g water / g fuel).

    Returns:
        Tuple[float, float]: ``(R_0, I_r)`` where ``R_0`` is base ROS
            (ft/min) and ``I_r`` is reaction intensity (BTU/ft²/min).
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


def get_characteristic_moistures(fuel: Fuel, m_f: np.ndarray) -> Tuple[float, float]:
    """Compute weighted characteristic dead and live fuel moisture contents.

    Use fuel weighting factors (``f_dead_arr``, ``f_live_arr``) to collapse
    the per-class moisture array into single dead and live values.

    Args:
        fuel (Fuel): Fuel model providing weighting arrays.
        m_f (np.ndarray): Moisture content array of shape (6,) as fractions.

    Returns:
        Tuple[float, float]: ``(dead_mf, live_mf)`` weighted characteristic
            moisture contents for dead and live fuel categories.
    """
    dead_mf = np.dot(fuel.f_dead_arr, m_f[0:4])
    live_mf = np.dot(fuel.f_live_arr, m_f[4:])

    return dead_mf, live_mf

def calc_live_mx(fuel: Fuel, m_f: float) -> float:
    """Compute live fuel moisture of extinction.

    Determine the threshold moisture content above which live fuels will
    not sustain combustion, based on the ratio of dead-to-live fuel loading
    (``fuel.W``) and the dead characteristic moisture.

    Args:
        fuel (Fuel): Fuel model with loading ratio ``W`` and ``dead_mx``.
        m_f (float): Weighted characteristic dead fuel moisture (fraction).

    Returns:
        float: Live fuel moisture of extinction (fraction). Clamped to be
            at least ``fuel.dead_mx``.
    """
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
    """Compute reaction intensity from fuel properties and moisture damping.

    Reaction intensity is the rate of heat release per unit area of the
    flaming front (Rothermel 1972, Eq. 27).

    Args:
        fuel (Fuel): Fuel model with net fuel loadings, heat content, and
            optimum reaction velocity (``gamma``).
        dead_moist_damping (float): Dead fuel moisture damping coefficient
            in [0, 1].
        live_moist_damping (float): Live fuel moisture damping coefficient
            in [0, 1].

    Returns:
        float: Reaction intensity (BTU/ft²/min).
    """
    mineral_damping = calc_mineral_damping()

    dead_calc = fuel.w_n_dead * fuel.heat_content * dead_moist_damping * mineral_damping
    live_calc = fuel.w_n_live * fuel.heat_content * live_moist_damping * mineral_damping

    I_r = fuel.gamma * (dead_calc + live_calc)

    return I_r

def calc_heat_sink(fuel: Fuel, m_f: np.ndarray) -> float:
    """Compute heat sink term for the Rothermel spread equation.

    The heat sink represents the energy required to raise the fuel ahead
    of the fire front to ignition temperature, weighted by fuel class
    properties and moisture contents.

    Args:
        fuel (Fuel): Fuel model with bulk density, weighting factors, and
            surface-area-to-volume ratios.
        m_f (np.ndarray): Fuel moisture content array of shape (6,) as
            fractions.

    Returns:
        float: Heat sink (BTU/ft³).
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


def calc_wind_factor(fuel: Fuel, wind_speed: float) -> float:
    """Compute the wind factor (phi_w) for the Rothermel spread equation.

    Args:
        fuel (Fuel): Fuel model with precomputed wind coefficients
            ``B``, ``C``, ``E``, and packing ratio ``rat``.
        wind_speed (float): Midflame wind speed (ft/min).

    Returns:
        float: Dimensionless wind factor (phi_w).
    """
    phi_w = fuel.C * (wind_speed ** fuel.B) * fuel.rat ** (-fuel.E)

    return phi_w

def calc_slope_factor(fuel: Fuel, phi: float) -> float:
    """Compute the slope factor (phi_s) for the Rothermel spread equation.

    Args:
        fuel (Fuel): Fuel model with bulk density ``rho_b`` and particle
            density ``rho_p``.
        phi (float): Slope angle (radians).

    Returns:
        float: Dimensionless slope factor (phi_s).
    """
    packing_ratio = fuel.rho_b / fuel.rho_p

    phi_s = 5.275 * (packing_ratio ** (-0.3)) * (np.tan(phi)) ** 2

    return phi_s


def calc_moisture_damping(m_f: float, m_x: float) -> float:
    """Compute moisture damping coefficient for dead or live fuel.

    Evaluates a cubic polynomial in the moisture ratio ``m_f / m_x``
    (Rothermel 1972, Eq. 29). Returns 0 when moisture of extinction is
    zero or when the polynomial evaluates to a negative value.

    Args:
        m_f (float): Characteristic fuel moisture content (fraction).
        m_x (float): Moisture of extinction (fraction).

    Returns:
        float: Moisture damping coefficient in [0, 1].
    """
    if m_x == 0:
        return 0

    r_m = m_f / m_x

    # Horner's form: fewer multiplications than expanded polynomial
    moist_damping = 1 + r_m * (-2.59 + r_m * (5.11 - 3.52 * r_m))

    return max(0, moist_damping)

def calc_mineral_damping(s_e: float = 0.010) -> float:
    """Compute mineral damping coefficient.

    Args:
        s_e (float): Effective mineral content (fraction). Defaults to
            0.010 (standard value for wildland fuels).

    Returns:
        float: Mineral damping coefficient (dimensionless).
    """

    mineral_damping = 0.174 * s_e ** (-0.19)

    return mineral_damping


def calc_effective_wind_factor(R_h: float, R_0: float) -> float:
    """Compute the effective wind factor from head-fire and base ROS.

    The effective wind factor (phi_e) represents the combined influence of
    wind and slope as if it were a single wind-only factor.

    Args:
        R_h (float): Head-fire rate of spread (ft/min).
        R_0 (float): No-wind, no-slope base ROS (ft/min).

    Returns:
        float: Effective wind factor (dimensionless).
    """
    phi_e = (R_h / R_0) - 1

    return phi_e

def calc_effective_wind_speed(fuel: Fuel, R_h: float, R_0: float) -> float:
    """Compute the effective wind speed from the effective wind factor.

    Invert the wind factor equation to recover the equivalent wind speed
    that produces the same effect as the combined wind and slope.

    Args:
        fuel (Fuel): Fuel model with wind coefficients ``B``, ``C``, ``E``,
            and packing ratio ``rat``.
        R_h (float): Head-fire rate of spread (ft/min).
        R_0 (float): No-wind, no-slope base ROS (ft/min).

    Returns:
        float: Effective wind speed (ft/min). Returns 0 when ``R_h <= R_0``.
    """


    if R_h <= R_0:
        phi_e = 0

    else: 
        phi_e = calc_effective_wind_factor(R_h, R_0)

    u_e = ((phi_e * (fuel.rat**fuel.E))/fuel.C) ** (1/fuel.B)

    return u_e

def calc_eccentricity(fuel: Fuel, R_h: float, R_0: float) -> float:
    """Compute fire ellipse eccentricity from effective wind speed.

    Convert the effective wind speed to m/s, then compute the length-to-
    breadth ratio ``z`` and derive eccentricity. Capped at ``z = 8.0``
    following Anderson (1983).

    Args:
        fuel (Fuel): Fuel model for effective wind speed calculation.
        R_h (float): Head-fire rate of spread (ft/min).
        R_0 (float): No-wind, no-slope base ROS (ft/min).

    Returns:
        float: Fire ellipse eccentricity in [0, 1).
    """
    u_e = calc_effective_wind_speed(fuel, R_h, R_0)
    u_e_ms = ft_min_to_m_s(u_e)
    z = 0.936 * np.exp(0.2566 * u_e_ms) + 0.461 * np.exp(-0.1548 * u_e_ms) - 0.397
    z = np.min([z, 8.0])
    e = ((z**2 - 1)**0.5)/z

    return e

def calc_flame_len(cell: Cell) -> float:
    """Estimate flame length from maximum fireline intensity.

    For surface fires, uses Brown and Davis (1973) correlation. For crown
    fires, uses Thomas (1963) correlation.

    Args:
        cell (Cell): Cell with ``I_ss`` (BTU/ft/min) and ``_crown_status``.

    Returns:
        float: Flame length in feet.
    """
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