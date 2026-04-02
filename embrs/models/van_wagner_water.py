"""Van Wagner & Taylor (2022) energy-balance water suppression model.

Implements a **binary-threshold** suppression model based on
"Theoretical Amounts of Water to Put Out Forest Fires" (Van Wagner &
Taylor 2022, Information Report BC-X-458).

Water drops accumulate cooling energy (``water_applied_kJ``) that decays
exponentially over time. Each simulation tick, the energy required for
extinguishment is computed from the current fire state using Van Wagner's
Eq. 7b (flame quenching) and Eq. 10b (fuel cooling), with an
intensity-dependent efficiency multiplier. When the accumulated energy
meets or exceeds the extinguishment threshold the cell's dead fuel
moisture is pushed to the extinction value, zeroing the rate of spread
via Rothermel's moisture damping. Fire burns completely unperturbed
until that threshold is crossed -- there is no intermediate moisture
injection or partial suppression from water.

Key functions:
    - ``heat_absorbed_per_kg_water`` (Eq. 1a/1b)
    - ``water_depth_quench_flame_mm`` (Eq. 7b)
    - ``water_depth_cool_fuel_mm`` (Eq. 10b)
    - ``heat_to_extinguish_kJ`` -- full pipeline
    - ``flame_depth_m`` (Eq. 2a-2b, Thomas 1963)
    - ``burning_zone_area_m2`` -- active flame strip area
    - ``efficiency_for_intensity`` -- piecewise-linear lookup table
    - ``volume_L_to_energy_kJ`` -- water volume to cooling energy
    - ``compute_suppression_ratio`` -- ratio of applied to needed energy

All functions are pure math with no simulation dependencies.

Reference:
    Van Wagner, C.E. & Taylor, S.W. (2022). Theoretical Amounts of Water
    to Put Out Forest Fires. The Forestry Chronicle.
"""

import numpy as np

# Physical constants from Van Wagner Table 1, p.1
_KCAL_TO_KJ = 4.184
_C_S = 0.46       # heat capacity of steam (kcal/kg·°C)
_C_F = 0.264      # heat capacity of flame gas (kcal/kg·°C)
_C_1 = 0.35       # heat capacity of hot fuel (kcal/kg·°C)


def heat_absorbed_per_kg_water(T_a: float = 20.0, T_e: float = 100.0,
                                c_s: float = _C_S) -> float:
    """Heat absorbed per kg of water applied to fire.

    Eq. 1a (general): H_w = (100 - T_a) + 540 + (T_e - 100)*c_s  [kcal/kg]
    Eq. 1b (T_e=100): H_w = (100 - T_a) + 540  [kcal/kg]

    Args:
        T_a: Ambient air temperature in °C. Default 20.
        T_e: Exit temperature of steam in °C. Default 100 (no superheating).
        c_s: Heat capacity of steam in kcal/(kg·°C). Default 0.46.

    Returns:
        Heat absorbed in kJ/kg.
    """
    H_w_kcal = (100.0 - T_a) + 540.0 + (T_e - 100.0) * c_s
    return H_w_kcal * _KCAL_TO_KJ


def water_depth_quench_flame_mm(I_kW_m: float) -> float:
    """Water depth to quench flame (Mechanism I).

    Eq. 7b: D_w = 3.84e-4 × I^(2/3)  [mm]
    where I is in kcal/(s·m).

    Args:
        I_kW_m: Fireline intensity in kW/m.

    Returns:
        Water depth in mm on the burning zone.
    """
    I_kcal = I_kW_m / _KCAL_TO_KJ
    return 3.84e-4 * I_kcal ** (2.0 / 3.0)


def water_depth_cool_fuel_mm(W_1_kg_m2: float) -> float:
    """Water depth to cool hot fuel behind the flame front (Mechanism II).

    Eq. 10b: D_w = 0.226 × W_1  [mm]
    where W_1 is dead fuel loading in kg/m².

    Args:
        W_1_kg_m2: Dead fuel loading in kg/m².

    Returns:
        Water depth in mm on the burning zone.
    """
    return 0.226 * W_1_kg_m2


def water_depth_combined_mm(I_kW_m: float, W_1_kg_m2: float) -> float:
    """Combined water depth for Mechanisms I + II.

    Sum of flame quenching (Eq. 7b) and fuel cooling (Eq. 10b),
    as recommended in Discussion §4, p.4.

    Args:
        I_kW_m: Fireline intensity in kW/m.
        W_1_kg_m2: Dead fuel loading in kg/m².

    Returns:
        Combined water depth in mm.
    """
    return water_depth_quench_flame_mm(I_kW_m) + water_depth_cool_fuel_mm(W_1_kg_m2)


def flame_depth_m(I_kcal_s_m: float) -> float:
    """Flame depth from Thomas' correlation (Van Wagner Eq. 2a-2b).

    L = 0.0792 × I^(2/3)  [m]  (Eq. 2b, Thomas 1963)
    D = L / 2              [m]  (Eq. 2a)

    This uses Van Wagner's specific parameterization of Thomas (1963),
    which is coupled to the water depth equations (Eq. 7b, 10b).
    Do NOT substitute other flame length correlations (e.g. Brown &
    Davis 1973 from rothermel.calc_flame_len) — the suppression
    energy balance depends on internal consistency with Eq. 2b.

    Args:
        I_kcal_s_m: Fireline intensity in kcal/(s·m).
            Convert from BTU/(ft·min) via BTU_ft_min_to_kcal_s_m().

    Returns:
        Flame depth in meters. Returns 0.0 if I_kcal_s_m <= 0.
    """
    if I_kcal_s_m <= 0.0:
        return 0.0
    L = 0.0792 * I_kcal_s_m ** (2.0 / 3.0)
    return L / 2.0


def burning_zone_area_m2(I_kcal_s_m: float, front_length_m: float) -> float:
    """Area of the active burning zone (flame depth x fire front length).

    The burning zone is the strip of actively flaming fuel behind the
    fire front, with depth D from Thomas' correlation (Eq. 2a-2b).

    Args:
        I_kcal_s_m: Fireline intensity in kcal/(s·m).
        front_length_m: Length of fire front in meters.

    Returns:
        Burning zone area in m². Returns 0.0 if either input is <= 0.
    """
    if I_kcal_s_m <= 0.0 or front_length_m <= 0.0:
        return 0.0
    return flame_depth_m(I_kcal_s_m) * front_length_m


# Default intensity-efficiency breakpoints (kW/m, efficiency_multiplier).
# Sources: Van Wagner Table 4 (<350), Plucinski 2019 (350-2000),
# Andrews 2018 RMRS-GTR-371 (2000-3500), NWCG (>3500).
_DEFAULT_EFFICIENCY_TABLE = [
    (0.0,    2.0),
    (350.0,  2.5),
    (2000.0, 5.0),
    (3500.0, 8.0),
]

def efficiency_for_intensity(I_kW_m: float,
                             table: list = None) -> float:
    """Piecewise-linear efficiency multiplier based on fireline intensity.

    Args:
        I_kW_m: Fireline intensity in kW/m. This function uses kW/m
            because the suppression literature (Andrews 2018, Plucinski
            2019) reports thresholds in kW/m.
        table: List of (intensity_kW_m, efficiency) breakpoints, sorted
            by intensity ascending. If None, uses default table.

    Returns:
        Interpolated efficiency multiplier.
    """
    if table is None:
        table = _DEFAULT_EFFICIENCY_TABLE

    if I_kW_m <= table[0][0]:
        return table[0][1]
    if I_kW_m >= table[-1][0]:
        return table[-1][1]

    for i in range(len(table) - 1):
        I_lo, e_lo = table[i]
        I_hi, e_hi = table[i + 1]
        if I_lo <= I_kW_m <= I_hi:
            t = (I_kW_m - I_lo) / (I_hi - I_lo)
            return e_lo + t * (e_hi - e_lo)

    return table[-1][1]


def heat_to_extinguish_kJ(I_kW_m: float, W_1_kg_m2: float,
                           fire_area_m2: float, efficiency: float = 2.5,
                           T_a: float = 20.0) -> float:
    """Total heat energy that must be removed to extinguish a fire.

    Pipeline:
        D_mm = water_depth_combined_mm(I, W) × efficiency  (Table 4)
        mass_kg = (D_mm / 1000) × fire_area_m2 × 1000  (water density)
        energy_kJ = mass_kg × heat_absorbed_per_kg_water(T_a)

    Args:
        I_kW_m: Fireline intensity in kW/m.
        W_1_kg_m2: Dead fuel loading in kg/m².
        fire_area_m2: Fire area in m². Returns 0.0 if <= 0.
        efficiency: Application efficiency multiplier (Table 4, p.5).
            Real applications need 2–4× theoretical water. Default 2.5.
        T_a: Ambient air temperature in °C. Default 20.

    Returns:
        Energy in kJ required for extinguishment.
    """
    if fire_area_m2 <= 0.0:
        return 0.0

    D_mm = water_depth_combined_mm(I_kW_m, W_1_kg_m2) * efficiency
    # D_mm of water over fire_area_m2: volume_m3 = (D_mm/1000) * fire_area_m2
    # mass_kg = volume_m3 * 1000 (water density kg/m³)
    mass_kg = (D_mm / 1000.0) * fire_area_m2 * 1000.0
    return mass_kg * heat_absorbed_per_kg_water(T_a)


def volume_L_to_energy_kJ(volume_L: float, T_a: float = 20.0) -> float:
    """Convert water volume to cooling energy.

    1 L of water = 1 kg. Energy = mass × H_w (Eq. 1b).

    Args:
        volume_L: Water volume in liters.
        T_a: Ambient air temperature in °C. Default 20.

    Returns:
        Cooling energy in kJ.
    """
    return volume_L * heat_absorbed_per_kg_water(T_a)


def compute_suppression_ratio(water_applied_kJ: float,
                               heat_to_extinguish: float) -> float:
    """Ratio of applied cooling energy to energy needed for extinguishment.

    Args:
        water_applied_kJ: Cumulative cooling energy from water drops (kJ).
        heat_to_extinguish: Energy needed to extinguish (kJ).

    Returns:
        Ratio clamped to [0.0, 1.0]. Returns 1.0 if heat_to_extinguish <= 0
        (trivially suppressible).
    """
    if heat_to_extinguish <= 0.0:
        return 1.0
    return min(water_applied_kJ / heat_to_extinguish, 1.0)