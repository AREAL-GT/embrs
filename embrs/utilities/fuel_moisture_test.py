import numpy as np

mtoft = 3.280839895

## Canadian Hourly Fine Fuel Moisture Code (FFMC)

measurement_elev = 25 # TODO: this will be fed from response.Elevation() (m asl)

m_0 = 10 # initial fine fuel moisture (%) # TODO: this should be calculated using condition over the last few days of weather
H = 10 # relative humidity (%)
T = 25 # temperature (C)
T_d = 20 # dew point (C)
W = 20 # wind speed (km/h)






def apply_elevation_correction(local_elev: float, T: float, T_d: float):
    elev_diff = local_elev - measurement_elev # feet

    temp_slope = mtoft * (5/9) * (-3.5/1000) # convert F/ft to C/m
    dew_slope = mtoft * (5/9) * (-1.1/1000) # convert F/ft to C/m

    delta_T = temp_slope * elev_diff # Change in temp in degrees C
    T = T + delta_T


    delta_T_d = dew_slope * elev_diff
    T_d = T_d + delta_T_d


    return T, T_d


def calc_RH(T: float, T_d: float):

    num = np.exp((17.625 * T_d)/(243.04 + T_d))
    den = np.exp((17.625 * T)/(243.04 + T))

    RH = 100 * (num/den)

    return RH

def get_next_hour_moisture(m_0: float, H: float, T: float, W: float):
    # Using Canadian Hourly Fine Fuel Moisture Code (FFMC)
    E_d = 0.942 * (H ** 0.679) + 11 * np.exp((H - 100)/10) + 0.18*(21.1 - T)*(1 - np.exp(-0.115*H)) # EMC for drying (%)
    E_w = 0.618 * (H ** 0.753) + 10 * np.exp((H - 100)/10) + 0.18*(21.1 - T)*(1 - np.exp(-0.115*H)) # EMC for wetting (%)
    
    if m_0 > E_d:
        k_a = 0.424 * (1 - (H/100)**1.7) + 0.0694*(W**0.5)*(1 - (H/100)**8)
        k_d = 0.0579 * k_a * np.exp(0.0365 * T)

        m = E_d + (m_0 - E_d) * np.exp(-2.303 * k_d)

    elif m_0 < E_d:
        k_b = 0.424 * (1 - ((100 - H)/100)**1.7) + 0.0694 * (W**0.5) * (1 - (H/100)**8) # Log drying rate for hourly computation, log to base 10
        k_w = 0.0579 * k_b * np.exp(0.0365 * T) # Log wetting rate for hourly computation, log to base 10

        m = E_w - (E_w - m_0) * np.exp(-2.303 * k_w) # Fuel moisture (%)

    if m_0 == E_d or m_0 == E_w or E_d > m_0 > E_w :
        m = m_0


    return m

def get_tree_shading(psi, A_u):
    J = optical attenutation coefficient
    A = solar altitude angle
    D = crown diameter
    L = crown length (dist from base of crown to top)
    h = crown height
    C = crown closure




    if crown_type == 'confiers':

        if np.tan(A) >= 2*L/D:
            A_h = np.pi*D**2/4
        else:
            G = np.arccos((2 * (L/D) * (1/np.tan(A)))**-1)

            A_h = (np.pi - G)*D**2/4 + D*L*(1/np.tan(A))*np.sin(G)

        X = np.pi * D**2 * L / (12*A_h*np.sin(A))

    elif crown_type == 'deciduous':
        G_prime = np.arctan(-(L/D)*(1/np.tan(A)))
        r_prime = np.sin(G_prime - A)
        l = r_prime * np.sin(G_prime - A)
        A_h = (np.pi * D * l)/(2*np.sin(A))
        X = np.pi * D**2 * L / (6*A_h*np.sin(A))


    A_b = 0.0093 * h**2 * (1/np.tan(A))
    A_h_prime = A_b + A_h * (1 - np.exp(-J*X))
    A_s = A_h_prime * (np.cos(psi) * np.sin(A)) / np.sin(A + psi)


    N_1 = A_s / A_u

    n = -4 * A_u * np.log(1 - C)/(np.pi * D**2)

    tree_shading = np.exp(-n*N_1)




def calc_inputs():

    T_a, T_d = apply_elevation_correction(local_elev, T, T_d)

    H_a = calc_RH(T, T_d)

    I_raw = get_irradiance()

    I = I_raw * tree_shading

    T_f = I / (0.015 * U_h + 0.026) + T # TODO: Fahrenheit input output needs to be convereted to C
    H_f = H_a * np.exp(-0.033*(T_f - T_a) # TODO: These expect Fahrenheit inputs

