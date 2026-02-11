"""Crown fire initiation and spread model.

Evaluate whether a surface fire transitions into a crown fire and, if so,
compute crown fire rate of spread and fireline intensity. Implements the
Van Wagner (1977) initiation criteria and the Rothermel (1991) crown fire
spread equations, with crown fraction burned (CFB) from Alexander (1988).

Functions:
    - crown_fire: Evaluate crown fire initiation and update cell state.
    - calc_R10: Compute the Fuel Model 10 reference ROS for crown fire.
    - calc_crown_vector: Resolve crown fire spread direction and ROS.
    - calc_crown_propagation: Compute directional crown fire ROS and intensity.
    - crown_loading_burned: Compute crown fuel loading consumed.
    - crown_intensity: Compute crown fireline intensity.

References:
    Van Wagner, C. E. (1977). Conditions for the start and spread of crown
    fire. Canadian Journal of Forest Research, 7(1), 23-34.

    Rothermel, R. C. (1991). Predicting behavior and size of crown fires in
    the Northern Rocky Mountains. USDA Forest Service Research Paper INT-438.

    Scott, J. H. & Reinhardt, E. D. (2001). Assessing Crown Fire Potential.
    USDA Forest Service Research Paper RMRS-RP-29.

    Alexander, M. E. (1985). Estimating the length-to-breadth ratio of
    elliptical forest fire patterns. Pages 287-304 in Proceedings of the
    Eighth Conference on Fire and Forest Meteorology. SAF Publication 85-04.
"""

from embrs.models.rothermel import *
from embrs.models.fuel_models import Anderson13
from embrs.utilities.unit_conversions import *
from embrs.fire_simulator.cell import Cell

from embrs.utilities.fire_util import CrownStatus

# Module-level cache for Fuel Model 10 (Anderson 13 classification)
# Avoids creating a new instance on every crown_fire() call
_FUEL_MODEL_10_CACHE = None


def _get_fuel_model_10():
    """Return cached Fuel Model 10 instance, creating it on first call."""
    global _FUEL_MODEL_10_CACHE
    if _FUEL_MODEL_10_CACHE is None:
        _FUEL_MODEL_10_CACHE = Anderson13(10)
    return _FUEL_MODEL_10_CACHE


def crown_fire(cell: Cell, fmc: float):
    """Evaluate crown fire initiation and update cell spread parameters.

    Check whether the surface fireline intensity exceeds the Van Wagner
    (1977) crown fire initiation threshold. If so, determine whether the
    crown fire is passive or active, compute crown fire ROS, and update
    the cell's spread and intensity arrays.

    Args:
        cell (Cell): Burning cell with surface fire ROS (``r_ss``) and
            fireline intensity (``I_ss``) already computed. Must have
            canopy attributes (``canopy_base_height``, ``canopy_bulk_density``,
            ``canopy_height``).
        fmc (float): Foliar moisture content (percent).

    Side Effects:
        Updates ``cell._crown_status``, ``cell.cfb``, ``cell.a_a``,
        ``cell.r_ss``, ``cell.I_ss``, ``cell.r_h_ss``, and ``cell.e``.
        Sets crown status to NONE, PASSIVE, or ACTIVE.
    """
    # Return if crown fire not possible
    if not cell.has_canopy:
        return
    
    # Calculate crown fire intensity threshold
    I_o = (0.01 * cell.canopy_base_height * (460 + 25.9 * fmc))**(3/2) # kW/m

    # Get the max rate of spread and fireline intensity within the cell
    R_m_s = np.max(cell.r_ss)
    R = R_m_s * 60 # R in m/min
    I_btu_ft_min = np.max(cell.I_ss)
    I_t = BTU_ft_min_to_kW_m(I_btu_ft_min) # I in kw/m

    # Check if fireline intensity is high enough to initiate crown fire
    if I_t >= I_o:
        # Surface fire will initiate a crown fire
        # Check if crown should be passive or active

        # Threshold for active crown spread rate (Alexander 1988)
        rac = 3.0 / cell.canopy_bulk_density # m/min

        # Critical surface fire spread rate
        R_0 = I_o * (R/I_t) # m/min

        # Surface fuel consumed 
        sfc = I_t / (300 * R) # kg/m^2

        # CFB scaling exponent
        a_c = -np.log(0.1) / (0.9 * (rac - R_0))

        # Crown fraction burned, proportion of the trees involved in crowning phase
        cfb = 1 - np.exp(-a_c * (R - R_0))

        # Set the crown fraction burned in the cell
        cell.cfb = cfb
        
        set_accel_constant(cell, cfb)
        
        # Forward surface fire spread rate for fuel model 10 using 0.4 wind reduction factor
        R_10 = calc_R10(cell)
        
        # Calculate maximum crown fire spread rate and the wind slope vector
        R_cmax, crown_dir, vec_mag = calc_crown_vector(cell, R_10)
        R_cmax = ft_min_to_m_s(R_cmax) * 60 # R_10 in m/min

        if cell._crown_status != CrownStatus.NONE:
            # Check if surface fire intensity too low in already burning crown fires
            t_r = 384 / cell.fuel.sav_ratio
            H_a = cell.reaction_intensity * t_r
            if (R_cmax * H_a) < I_o:
                cell._crown_status = CrownStatus.NONE
                return

        r_actual = R + cfb * (R_cmax - R) # m/min
        
        if r_actual >= rac:
            # Active crown fire
            # Actual active crown fire spread rate
            cell._crown_status = CrownStatus.ACTIVE

        else:
            # Passive crown fire
            # Passive crown fire rate set to surface rate of spread
            r_actual = cell.r_h_ss * 60 # m/min
            cell._crown_status = CrownStatus.PASSIVE
        
        # Set rate of spread based on crown fire equations
        cell.r_ss, cell.I_ss = calc_crown_propagation(cell, r_actual, crown_dir, vec_mag, sfc, cfb)
        cell.r_h_ss = np.max(cell.r_ss)

    else:
        cell._crown_status = CrownStatus.NONE

def set_accel_constant(cell: Cell, cfb: float):
    """Set the fire acceleration constant based on crown fraction burned.

    Compute a crown-fire-adjusted acceleration constant and store it on
    the cell. The formula reduces acceleration as CFB increases.

    Args:
        cell (Cell): Cell to update.
        cfb (float): Crown fraction burned in [0, 1].

    Side Effects:
        Sets ``cell.a_a`` (1/s).
    """
    a = (0.3 - 18.8 * (cfb ** 2.5) * np.exp(-8 * cfb)) / 60

    cell.a_a = a

def calc_R10(cell: Cell) -> float:
    """Compute no-wind no-slope ROS for Fuel Model 10 (Rothermel 1991).

    Calculate the base ROS using Anderson 13 Fuel Model 10 properties
    and the cell's current fuel moisture. This value is used as a
    reference for the crown fire spread rate calculation.

    Args:
        cell (Cell): Cell providing fuel moisture (``fmois``).

    Returns:
        float: Base ROS for Fuel Model 10 (ft/min).
    """
    fuel = _get_fuel_model_10()

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

    return R_0

def get_wind_slope_vector(cell: Cell, phi_w: float, phi_s: float,
                          slope_speed: float) -> Tuple[float, float, float]:
    """Compute the combined wind and slope vector for crown fire spread.

    Resolve wind and slope influences into a resultant speed, magnitude,
    and direction using the law of cosines. Used by ``calc_crown_vector``.

    Args:
        cell (Cell): Cell providing current wind and aspect.
        phi_w (float): Wind factor (dimensionless).
        phi_s (float): Slope factor (dimensionless).
        slope_speed (float): Equivalent slope wind speed (ft/min).

    Returns:
        Tuple[float, float, float]: ``(vec_speed, vec_mag, vec_dir)`` where
            ``vec_speed`` is the combined factor magnitude (dimensionless),
            ``vec_mag`` is the resultant speed (ft/min), and ``vec_dir``
            is the resultant direction (degrees, compass convention).
    """
    wind_speed, wind_dir = cell.curr_wind()

    angle = np.abs(wind_dir - cell.aspect) # degrees

    wind_speed_ft_min = m_s_to_ft_min(wind_speed) # ft/min
    wind_speed = 0.5 * wind_speed_ft_min

    if angle != 180:
        vec_speed = np.sqrt(phi_w**2 + phi_s**2 + 2 * phi_w * phi_s * np.cos(np.deg2rad(angle)))
        vec_mag = np.sqrt(wind_speed ** 2 + slope_speed ** 2 + (2 * wind_speed * slope_speed * np.cos(np.deg2rad(angle))))
    
    else:
        vec_speed = np.abs(phi_s - phi_w)
        vec_mag = np.abs(slope_speed - wind_speed)

    if phi_s >= phi_w:
        aside = phi_w
        bside = phi_s
        cside = vec_speed
    else:
        aside = phi_s
        bside = phi_w
        cside = vec_speed
 
    if bside != 0 and cside != 0:
        vangle = (aside**2 - bside**2 - cside**2) / (-2 * bside * cside)

        if vangle > 1:
            vangle = 1
        else:
            if vangle < 0:
                vangle = np.pi
            else:
                vangle = np.arccos(vangle)
                vangle = np.abs(vangle)

    else:
        vangle = 0

    vangle = np.rad2deg(vangle)

    if angle < 90:
        if angle > 0:
            if phi_w >= phi_s:
                vec_dir = wind_dir - vangle
            else:
                vec_dir = cell.aspect + vangle

        else:
            if phi_w >= phi_s:
                vec_dir = wind_dir + vangle
            else:
                vec_dir = cell.aspect - vangle
    else:
        if angle > 0:
            if phi_w >= phi_s:
                vec_dir = wind_dir + vangle
            else:
                vec_dir = cell.aspect - vangle
        else:
            if phi_w >= phi_s:
                vec_dir = wind_dir - vangle
            else:
                vec_dir = cell.aspect + vangle
    if vec_dir < 0:
        vec_dir += 360

    if vec_dir > 360:
        vec_dir -= 360

    return vec_speed, vec_mag, vec_dir # ft/min, degrees

def calc_slope_speed(cell: Cell, phi_s: float) -> float:
    """Compute equivalent wind speed from slope factor.

    Invert the wind factor equation to find the wind speed that would
    produce the same spread effect as the slope factor.

    Args:
        cell (Cell): Cell providing fuel model with wind coefficients.
        phi_s (float): Slope factor (dimensionless).

    Returns:
        float: Equivalent slope wind speed (ft/min).
    """
    fuel = cell.fuel

    part1 = fuel.C * fuel.sav_ratio ** -fuel.E
    slope_speed = (phi_s/part1) ** (1/fuel.B)
    
    return slope_speed # ft/min

def calc_crown_vector(cell: Cell, R10: float) -> Tuple[float, float, float]:
    """Compute crown fire maximum ROS and spread direction.

    Combine the Fuel Model 10 base ROS (``R10``) with wind and slope
    effects, then scale by the 3.34 crown fire multiplier
    (Rothermel 1991). Wind speed is reduced by 0.4 for the crown fire
    calculation.

    Args:
        cell (Cell): Cell providing wind, slope, and fuel data.
        R10 (float): No-wind no-slope Fuel Model 10 ROS (ft/min).

    Returns:
        Tuple[float, float, float]: ``(R_cmax, crown_dir, vec_mag)`` where
            ``R_cmax`` is maximum crown fire ROS (ft/min), ``crown_dir`` is
            the spread heading (radians), and ``vec_mag`` is the
            wind/slope vector magnitude (ft/min).
    """
    wind_speed, wind_dir = cell.curr_wind()

    wind_ft_min = m_s_to_ft_min(wind_speed) # ft/min
    phi_w = calc_wind_factor(cell.fuel, wind_ft_min * 0.4) # Reduce wind speed by 0.4 to get R_10 (Rothermel 1991)

    slope_rad = np.deg2rad(cell.slope_deg)
    phi_s = calc_slope_factor(cell.fuel, slope_rad)

    slope_speed = calc_slope_speed(cell, phi_s)

    if cell.slope_deg > 0:
        vec_speed, vec_mag, vec_dir = get_wind_slope_vector(cell, phi_w, phi_s, slope_speed)
        vec_ros = R10 * (1 + vec_speed)

    else:
        vec_dir = wind_dir
        vec_mag = wind_ft_min * 0.5
        vec_ros = R10 * (1 + phi_w)

    vec_ros *= 3.34 # R10 * 3.34 to get crown fire spread rate

    return vec_ros, np.deg2rad(vec_dir), vec_mag # ft/min, radians, ft/min

def calc_crown_eccentricity(wind_slope_vec_mag: float) -> float:
    """Compute crown fire ellipse eccentricity from wind/slope vector magnitude.

    Similar to the surface fire eccentricity but uses different exponential
    coefficients and converts input from ft/min to mph.

    Based on Alexander, M. E. (1985). Estimating the length-to-breadth ratio
    of elliptical forest fire patterns. Pages 287-304 in Proceedings of the
    Eighth Conference on Fire and Forest Meteorology. SAF Publication 85-04.

    Args:
        wind_slope_vec_mag (float): Combined wind/slope vector magnitude
            (ft/min).

    Returns:
        float: Crown fire ellipse eccentricity in [0, 1).
    """
    wind_slope_vec_mag = ft_min_to_mph(wind_slope_vec_mag) # convert to mph

    z = 0.936 * np.exp(0.1147 * wind_slope_vec_mag) + 0.461 * np.exp(-0.0692 * wind_slope_vec_mag) - 0.397

    e = ((z**2 - 1)**0.5)/z

    return e

def calc_crown_propagation(cell: Cell, r_actual: float, alpha: float,
                           vec_mag: float, sfc: float,
                           cfb: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute directional crown fire ROS and fireline intensity.

    Calculate crown fireline intensity (Scott & Reinhardt 2001, Eq. 22),
    crown fire eccentricity, and resolve ROS and intensity along all
    spread directions.

    Args:
        cell (Cell): Cell providing canopy and spread direction data.
        r_actual (float): Actual crown fire ROS (m/min).
        alpha (float): Crown fire spread heading (radians).
        vec_mag (float): Wind/slope vector magnitude (ft/min).
        sfc (float): Surface fuel consumed (kg/m²).
        cfb (float): Crown fraction burned in [0, 1].

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(r_list, I_list)`` where
            ``r_list`` is ROS in m/s per direction and ``I_list`` is
            fireline intensity in BTU/ft/min per direction.
    """
    # Calculate Fireline intensity (Based on Equation 22 of Scott Reinhardt Crown Fire [RMRS-RP-29])
    clb = crown_loading_burned(cell, cfb)
    I_h = crown_intensity(r_actual, sfc, clb)

    # Calculate Eccentricity
    e = calc_crown_eccentricity(vec_mag)
    cell.e = e

    # calculate ros and I along each direction based on e and alpha
    r_list, I_list = calc_vals_for_all_directions(cell, r_actual, 0, alpha, e, I_h=I_h)

    # return values in m/s and BTU/ft/min
    return r_list, I_list

def crown_loading_burned(cell: Cell, cfb: float) -> float:
    """Compute crown fuel loading consumed by the crown fire.

    Based on Van Wagner (1990): crown load burned = CFB * CBD * (CH - CBH).

    Args:
        cell (Cell): Cell with ``canopy_bulk_density`` (kg/m³),
            ``canopy_height`` (meters), and ``canopy_base_height`` (meters).
        cfb (float): Crown fraction burned in [0, 1].

    Returns:
        float: Crown loading burned (kg/m²).
    """
    cbd = cell.canopy_bulk_density
    ch = cell.canopy_height
    cbh = cell.canopy_base_height

    # Compute crown loading burned (kg/m2)
    crown_load_burned = cfb * cbd * np.abs(ch - cbh) # Van Wagner 1990

    return crown_load_burned

def crown_intensity(R: float, sfc: float, clb: float) -> float:
    """Compute crown fireline intensity (Rothermel 1991, pp. 10-11).

    Args:
        R (float): Crown fire ROS (m/min). Converted to ft/s internally.
        sfc (float): Surface fuel consumed (kg/m²).
        clb (float): Crown loading burned (kg/m²).

    Returns:
        float: Crown fireline intensity (BTU/ft/min).
    """
    # Convert R to ft/s
    R /= (0.3048 * 60.0) # m/min to ft/s

    I_h = np.abs(R * (sfc + clb) * 1586.01) # btu/ft/s

    I_h *= 60 # convert to btu/ft/min

    return I_h