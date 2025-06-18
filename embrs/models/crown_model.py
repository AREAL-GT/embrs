from embrs.models.rothermel import *
from embrs.models.fuel_models import Anderson13
from embrs.utilities.unit_conversions import *
from embrs.fire_simulator.cell import Cell

from embrs.utilities.fire_util import CrownStatus


def crown_fire(cell: Cell, fmc: float):
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

            # Actual active crown fire spread rate
            r_actual = R + cfb * (R_cmax - R) # m/min

            if r_actual >= rac:
                # Active crown fire
                cell._crown_status = CrownStatus.ACTIVE

            else:
                # Passive crown fire
                cell._crown_status = CrownStatus.PASSIVE
            
            # Set rate of spread based on crown fire equations
            cell.r_ss, cell.I_ss = calc_crown_propagation(cell, r_actual, crown_dir, vec_mag, sfc, cfb)

        else:
            cell._crown_status = CrownStatus.NONE

def set_accel_constant(cell, cfb):
    # Set the acceleration constant for the cell
    a = 0.115 - 18.8 * (cfb ** 2.5) * np.exp(-8 * cfb)

    cell.a_a = a

def calc_R10(cell: Cell) -> float:
    # Calculate R_10 value for active crown fire ROS as described in Rothermel (1991)
    # Computes the rate of spread using the surface fire rate of spread in Fuel Model 10 with 
    # wind speed reduced by a factor of 0.4
    fuel = Anderson13(10)

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

def get_wind_slope_vector(cell, phi_w, phi_s, slope_speed):
    angle = np.abs(cell.curr_wind[1] - cell.aspect) # degrees

    wind_speed_ft_min = m_s_to_ft_min(cell.curr_wind[0]) # ft/min
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
                vec_dir = cell.curr_wind[1] - vangle
            else:
                vec_dir = cell.aspect + vangle

        else:
            if phi_w >= phi_s:
                vec_dir = cell.curr_wind[1] + vangle
            else:
                vec_dir = cell.aspect - vangle
    else:
        if angle > 0:
            if phi_w >= phi_s:
                vec_dir = cell.curr_wind[1] + vangle
            else:
                vec_dir = cell.aspect - vangle
        else:
            if phi_w >= phi_s:
                vec_dir = cell.curr_wind[1] - vangle
            else:
                vec_dir = cell.aspect + vangle
    if vec_dir < 0:
        vec_dir += 360

    if vec_dir > 360:
        vec_dir -= 360

    return vec_speed, vec_mag, vec_dir # ft/min, degrees

def calc_slope_speed(cell, phi_s):

    e, b, c = calc_E_B_C(cell.fuel)

    part1 = c * cell.fuel.sav_ratio ** -e

    slope_speed = (phi_s/part1) ** (1/b)
    
    return slope_speed # ft/min

def calc_crown_vector(cell, R10):

    wind_ft_min = m_s_to_ft_min(cell.curr_wind[0]) # ft/min
    phi_w = calc_wind_factor(cell.fuel, wind_ft_min * 0.4) # Reduce wind speed by 0.4 to get R_10 (Rothermel 1991)

    slope_rad = np.deg2rad(cell.slope_deg)
    phi_s = calc_slope_factor(cell.fuel, slope_rad)

    slope_speed = calc_slope_speed(cell, phi_s)

    if cell.slope_deg > 0:
        vec_speed, vec_mag, vec_dir = get_wind_slope_vector(cell, phi_w, phi_s, slope_speed)
        vec_ros = R10 * (1 + vec_speed)

    else:
        vec_dir = cell.curr_wind[1]
        vec_mag = cell.curr_wind[0] * 0.5
        vec_ros = R10 * (1 + phi_w)

    vec_ros *= 3.34 # R10 * 3.34 to get crown fire spread rate

    return vec_ros, vec_dir, vec_mag # ft/min, degrees, ft/min

def calc_crown_eccentricity(wind_slope_vec_mag: float):
    # wind_slope_vec_mag should be in ft/min
    wind_slope_vec_mag = ft_min_to_mph(wind_slope_vec_mag) # convert to mph

    z = 0.936 * np.exp(0.1147 * wind_slope_vec_mag) + 0.461 * np.exp(-0.0692 * wind_slope_vec_mag) - 0.397

    e = ((z**2 - 1)**0.5)/z

    return e

def calc_crown_propagation(cell, r_actual, alpha, vec_mag, sfc, cfb):
    # Calculate Fireline intensity (Based on Equation 22 of Scott Reinhardt Crown Fire [RMRS-RP-29])
    clb = crown_loading_burned(cell, cfb)
    I_h = crown_intensity(r_actual, sfc, clb)

    # Calculate Crown Eccentricity
    e = calc_crown_eccentricity(vec_mag)

    # calculate ros and I along each direction based on e and alpha
    r_list, I_list = calc_vals_for_all_directions(cell, r_actual, I_h, alpha, e)

    # return values in m/s and BTU/ft/min
    return r_list, I_list

def crown_loading_burned(cell, cfb):
    cbd = cell.canopy_bulk_density
    ch = cell.canopy_height
    cbh = cell.canopy_base_height

    # Compute crown loading burned (kg/m2)
    crown_load_burned = cfb * cbd * np.abs(ch - cbh) # Van Wagner 1990

    return crown_load_burned

def crown_intensity(R, sfc, clb):
    # Calculate crown fireline intensity and flame length
    # From Rothermel 1991 pp 10, 11

    # Convert R to ft/s
    R /= 60 # m/min to m/s
    R = m_s_to_ft_min(R)/60 # m/s to ft/s

    I_h = np.abs(R * (sfc + clb) * 1586.01) # btu/ft/s

    I_h *= 60 # convert to btu/ft/min

    return I_h