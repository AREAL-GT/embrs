"""Unit conversion functions for fire modeling calculations.

This module provides conversion functions between imperial and metric units
commonly used in fire behavior modeling (Rothermel equations use imperial units
internally).

Functions are organized by conversion type: temperature, length, speed,
fuel loading, heat flux, and heat content.
"""

# ============================================================================
# Pre-computed conversion constants
# These avoid function calls during conversion calculations
# ============================================================================

# Length conversions
_M_TO_FT = 3.28084
_FT_TO_M = 1.0 / _M_TO_FT  # 0.3048

# Speed conversions
_FT_MIN_TO_M_S = 0.00508
_M_S_TO_FT_MIN = 1.0 / _FT_MIN_TO_M_S  # 196.8503937...
_MPH_TO_FT_MIN = 88.0
_FT_MIN_TO_MPH = 1.0 / _MPH_TO_FT_MIN  # 0.01136363...

# Fuel loading conversions
_LBSFT2_TO_KISQ = 4.88243
_KISQ_TO_LBSFT2 = 1.0 / _LBSFT2_TO_KISQ  # 0.204816...
_TPA_TO_LBSFT2 = 0.04591
_LBSFT2_TO_TPA = 1.0 / _TPA_TO_LBSFT2  # 21.781...
_TPA_TO_KISQ_DIVISOR = 4.46
_KISQ_TO_TPA = _TPA_TO_KISQ_DIVISOR  # 4.46

# Heat flux conversions
_BTU_FT2_MIN_TO_KW_M2 = 0.189276
_BTU_FT_MIN_TO_KW_M = 0.05767

# Heat content conversions
_CAL_G_TO_BTU_LB = 1.8
_BTU_LB_TO_CAL_G = 0.555


# ============================================================================
# Temperature conversions
# ============================================================================

def F_to_C(f_f: float) -> float:
    """Convert temperature from Fahrenheit to Celsius.

    Args:
        f_f (float): Temperature in degrees Fahrenheit.

    Returns:
        float: Temperature in degrees Celsius.
    """
    return (5.0 / 9.0) * (f_f - 32.0)


# ============================================================================
# Length conversions
# ============================================================================

def m_to_ft(f_m: float) -> float:
    """Convert length from meters to feet.

    Args:
        f_m (float): Length in meters.

    Returns:
        float: Length in feet.
    """
    return f_m * _M_TO_FT


def ft_to_m(f_ft: float) -> float:
    """Convert length from feet to meters.

    Args:
        f_ft (float): Length in feet.

    Returns:
        float: Length in meters.
    """
    return f_ft * _FT_TO_M


# ============================================================================
# Speed conversions
# ============================================================================

def ft_min_to_m_s(f_ft_min: float) -> float:
    """Convert speed from feet per minute to meters per second.

    Args:
        f_ft_min (float): Speed in ft/min.

    Returns:
        float: Speed in m/s.
    """
    return f_ft_min * _FT_MIN_TO_M_S


def m_s_to_ft_min(m_s: float) -> float:
    """Convert speed from meters per second to feet per minute.

    Args:
        m_s (float): Speed in m/s.

    Returns:
        float: Speed in ft/min.
    """
    return m_s * _M_S_TO_FT_MIN


def ft_min_to_mph(f_ft_min: float) -> float:
    """Convert speed from feet per minute to miles per hour.

    Args:
        f_ft_min (float): Speed in ft/min.

    Returns:
        float: Speed in mph.
    """
    return f_ft_min * _FT_MIN_TO_MPH


def mph_to_ft_min(f_mph: float) -> float:
    """Convert speed from miles per hour to feet per minute.

    Args:
        f_mph (float): Speed in mph.

    Returns:
        float: Speed in ft/min.
    """
    return f_mph * _MPH_TO_FT_MIN


# ============================================================================
# Fuel loading conversions
# ============================================================================

def Lbsft2_to_KiSq(f_libsft2: float) -> float:
    """Convert fuel loading from lb/ft^2 to kg/m^2.

    Args:
        f_libsft2 (float): Fuel loading in lb/ft^2.

    Returns:
        float: Fuel loading in kg/m^2.
    """
    return f_libsft2 * _LBSFT2_TO_KISQ


def KiSq_to_Lbsft2(f_kisq: float) -> float:
    """Convert fuel loading from kg/m^2 to lb/ft^2.

    Args:
        f_kisq (float): Fuel loading in kg/m^2.

    Returns:
        float: Fuel loading in lb/ft^2.
    """
    return f_kisq * _KISQ_TO_LBSFT2


def TPA_to_KiSq(f_tpa: float) -> float:
    """Convert fuel loading from tons per acre to kg/m^2.

    Args:
        f_tpa (float): Fuel loading in tons per acre.

    Returns:
        float: Fuel loading in kg/m^2.
    """
    return f_tpa / _TPA_TO_KISQ_DIVISOR


def TPA_to_Lbsft2(f_tpa: float) -> float:
    """Convert fuel loading from tons per acre to lb/ft^2.

    Args:
        f_tpa (float): Fuel loading in tons per acre.

    Returns:
        float: Fuel loading in lb/ft^2.
    """
    return f_tpa * _TPA_TO_LBSFT2


def Lbsft2_to_TPA(f_lbsft2: float) -> float:
    """Convert fuel loading from lb/ft^2 to tons per acre.

    Args:
        f_lbsft2 (float): Fuel loading in lb/ft^2.

    Returns:
        float: Fuel loading in tons per acre.
    """
    return f_lbsft2 * _LBSFT2_TO_TPA


def KiSq_to_TPA(f_kisq: float) -> float:
    """Convert fuel loading from kg/m^2 to tons per acre.

    Args:
        f_kisq (float): Fuel loading in kg/m^2.

    Returns:
        float: Fuel loading in tons per acre.
    """
    return f_kisq * _KISQ_TO_TPA


# ============================================================================
# Heat flux conversions
# ============================================================================

def BTU_ft2_min_to_kW_m2(f_btu_ft2_min: float) -> float:
    """Convert heat flux from BTU/(ft^2*min) to kW/m^2.

    Args:
        f_btu_ft2_min (float): Heat flux in BTU/(ft^2*min).

    Returns:
        float: Heat flux in kW/m^2.
    """
    return f_btu_ft2_min * _BTU_FT2_MIN_TO_KW_M2


def BTU_ft_min_to_kW_m(f_btu_ft_min: float) -> float:
    """Convert fireline intensity from BTU/(ft*min) to kW/m.

    Args:
        f_btu_ft_min (float): Fireline intensity in BTU/(ft*min).

    Returns:
        float: Fireline intensity in kW/m.
    """
    return f_btu_ft_min * _BTU_FT_MIN_TO_KW_M


# ============================================================================
# Heat content conversions
# ============================================================================

def cal_g_to_BTU_lb(f_cal_g: float) -> float:
    """Convert heat content from cal/g to BTU/lb.

    Args:
        f_cal_g (float): Heat content in cal/g.

    Returns:
        float: Heat content in BTU/lb.
    """
    return f_cal_g * _CAL_G_TO_BTU_LB


def BTU_lb_to_cal_g(f_btu_lb: float) -> float:
    """Convert heat content from BTU/lb to cal/g.

    Args:
        f_btu_lb (float): Heat content in BTU/lb.

    Returns:
        float: Heat content in cal/g.
    """
    return f_btu_lb * _BTU_LB_TO_CAL_G
