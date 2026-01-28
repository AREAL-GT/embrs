"""Unit conversion functions for fire modeling calculations.

This module provides conversion functions between imperial and metric units
commonly used in fire behavior modeling (Rothermel equations use imperial units
internally).

Functions are organized by conversion type: temperature, length, speed,
fuel loading, heat flux, and heat content.
"""


def F_to_C(f_f: float) -> float:
    """Convert temperature from Fahrenheit to Celsius.

    Args:
        f_f (float): Temperature in degrees Fahrenheit.

    Returns:
        float: Temperature in degrees Celsius.
    """
    g = 5 / 9
    h = 32
    c = g * (f_f - h)

    return c


def m_to_ft(f_m: float) -> float:
    """Convert length from meters to feet.

    Args:
        f_m (float): Length in meters.

    Returns:
        float: Length in feet.
    """
    g = 3.28084
    f = f_m * g

    return f


def ft_to_m(f_ft: float) -> float:
    """Convert length from feet to meters.

    Args:
        f_ft (float): Length in feet.

    Returns:
        float: Length in meters.
    """
    g = 1 / m_to_ft(1)
    f = f_ft * g

    return f


def ft_min_to_m_s(f_ft_min: float) -> float:
    """Convert speed from feet per minute to meters per second.

    Args:
        f_ft_min (float): Speed in ft/min.

    Returns:
        float: Speed in m/s.
    """
    g = 0.00508
    f = f_ft_min * g

    return f


def m_s_to_ft_min(m_s: float) -> float:
    """Convert speed from meters per second to feet per minute.

    Args:
        m_s (float): Speed in m/s.

    Returns:
        float: Speed in ft/min.
    """
    g = 1 / ft_min_to_m_s(1)
    f = m_s * g
    return f


def ft_min_to_mph(f_ft_min: float) -> float:
    """Convert speed from feet per minute to miles per hour.

    Args:
        f_ft_min (float): Speed in ft/min.

    Returns:
        float: Speed in mph.
    """
    g = 1 / mph_to_ft_min(1)
    f = f_ft_min * g

    return f


def mph_to_ft_min(f_mph: float) -> float:
    """Convert speed from miles per hour to feet per minute.

    Args:
        f_mph (float): Speed in mph.

    Returns:
        float: Speed in ft/min.
    """
    g = 88
    f = f_mph * g

    return f


def Lbsft2_to_KiSq(f_libsft2: float) -> float:
    """Convert fuel loading from lb/ft^2 to kg/m^2.

    Args:
        f_libsft2 (float): Fuel loading in lb/ft^2.

    Returns:
        float: Fuel loading in kg/m^2.
    """
    g = 4.88243
    f = f_libsft2 * g

    return f


def KiSq_to_Lbsft2(f_kisq: float) -> float:
    """Convert fuel loading from kg/m^2 to lb/ft^2.

    Args:
        f_kisq (float): Fuel loading in kg/m^2.

    Returns:
        float: Fuel loading in lb/ft^2.
    """
    g = 1 / Lbsft2_to_KiSq(1)
    f = f_kisq * g

    return f


def TPA_to_KiSq(f_tpa: float) -> float:
    """Convert fuel loading from tons per acre to kg/m^2.

    Args:
        f_tpa (float): Fuel loading in tons per acre.

    Returns:
        float: Fuel loading in kg/m^2.
    """
    g = 4.46
    f = f_tpa / g
    return f


def TPA_to_Lbsft2(f_tpa: float) -> float:
    """Convert fuel loading from tons per acre to lb/ft^2.

    Args:
        f_tpa (float): Fuel loading in tons per acre.

    Returns:
        float: Fuel loading in lb/ft^2.
    """
    g = 0.04591
    f = f_tpa * g

    return f


def Lbsft2_to_TPA(f_lbsft2: float) -> float:
    """Convert fuel loading from lb/ft^2 to tons per acre.

    Args:
        f_lbsft2 (float): Fuel loading in lb/ft^2.

    Returns:
        float: Fuel loading in tons per acre.
    """
    g = 1 / TPA_to_Lbsft2(1)
    f = f_lbsft2 * g

    return f


def KiSq_to_TPA(f_kisq: float) -> float:
    """Convert fuel loading from kg/m^2 to tons per acre.

    Args:
        f_kisq (float): Fuel loading in kg/m^2.

    Returns:
        float: Fuel loading in tons per acre.
    """
    f = 1 / TPA_to_KiSq(1)
    g = f_kisq * f
    return g


def BTU_ft2_min_to_kW_m2(f_btu_ft2_min: float) -> float:
    """Convert heat flux from BTU/(ft^2*min) to kW/m^2.

    Args:
        f_btu_ft2_min (float): Heat flux in BTU/(ft^2*min).

    Returns:
        float: Heat flux in kW/m^2.
    """
    g = 0.189276
    f = f_btu_ft2_min * g
    return f


def BTU_ft_min_to_kW_m(f_btu_ft_min: float) -> float:
    """Convert fireline intensity from BTU/(ft*min) to kW/m.

    Args:
        f_btu_ft_min (float): Fireline intensity in BTU/(ft*min).

    Returns:
        float: Fireline intensity in kW/m.
    """
    g = 0.05767
    f = f_btu_ft_min * g
    return f


def cal_g_to_BTU_lb(f_cal_g: float) -> float:
    """Convert heat content from cal/g to BTU/lb.

    Args:
        f_cal_g (float): Heat content in cal/g.

    Returns:
        float: Heat content in BTU/lb.
    """
    g = 1.8
    f = f_cal_g * g
    return f


def BTU_lb_to_cal_g(f_btu_lb: float) -> float:
    """Convert heat content from BTU/lb to cal/g.

    Args:
        f_btu_lb (float): Heat content in BTU/lb.

    Returns:
        float: Heat content in cal/g.
    """
    g = 0.555
    f = f_btu_lb * g
    return f
