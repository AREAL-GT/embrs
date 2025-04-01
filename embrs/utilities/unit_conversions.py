"""This module contains functions for unit conversions"""

def F_to_C(f_f: float) -> float:
    """Converts from Fahrenheit to Celsius

    Args:
        f_f (float): Fahrenheit

    Returns:
        _type_: float
    """
    g = 5 / 9
    h = 32
    c = g * (f_f - h)

    return c

def m_to_ft(f_m: float) -> float:
    """Converts from meters to feet

    Args:
        f_m (float): meters

    Returns:
        _type_: float
    """
    g = 3.28084
    f = f_m * g

    return f

def ft_to_m(f_ft: float) -> float:
    """Converts from feet to meters

    Args:
        f_ft (float): feet
    Returns:
        _type_: float
    """
    g = 1 / m_to_ft(1)
    f = f_ft * g

    return f

def ft_min_to_m_s(f_ft_min: float) -> float:
    """Converts from ft/min to m/s

    Args:
        f_ft_min (float): ft/min

    Returns:
        _type_: float
    """
    g = 0.00508
    f = f_ft_min * g

    return f

def m_s_to_ft_min(m_s: float) -> float:
    """_summary_

    Args:
        m_s (float): _description_

    Returns:
        float: _description_
    """
    g = 1 / ft_min_to_m_s(1)
    f = m_s * g
    return f

def Lbsft2_to_KiSq(f_libsft2: float) -> float:
    """Converts from lbs/ft^2 to kW/m^2

    Args:
        f_libsft2 (float): lbs/ft^2

    Returns:
        _type_: float
    """
    g = 4.88243
    f = f_libsft2 * g

    return f

def KiSq_to_Lbsft2(f_kisq: float) -> float:
    """_summary_

    Args:
        f_kisq (float): _description_

    Returns:
        float: _description_
    """
    g = 1 / Lbsft2_to_KiSq(1)
    f = f_kisq * g

    return f

def TPA_to_KiSq(f_tpa: float) -> float:
    """_summary_

    Args:
        f_tpa (float): _description_

    Returns:
        _type_: _description_
    """
    g = 4.46
    f = f_tpa / g
    return f

def KiSq_to_TPA(f_kisq: float) -> float:
    """_summary_

    Args:
        f_kisq (float): _description_

    Returns:
        float: _description_
    """
    f = 1 / TPA_to_KiSq(1)
    g = f_kisq * f
    return g

def BTU_ft2_min_to_kW_m2(f_btu_ft2_min: float) -> float:
    """_summary_

    Args:
        f_btu_ft2_min (float): _description_

    Returns:
        _type_: _description_
    """
    g = 0.189276
    f = f_btu_ft2_min * g
    return f

def BTU_ft_min_to_kW_m(f_btu_ft_min: float) -> float:
    """_summary_

    Args:
        f_btu_ft_min (float): _description_

    Returns:
        _type_: _description_
    """
    g = 0.05767
    f = f_btu_ft_min * g
    return f
