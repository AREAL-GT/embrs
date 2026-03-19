"""Van Wagner & Taylor (2022) energy-balance water suppression model.

Implements "Theoretical Amounts of Water to Put Out Forest Fires" equations
for computing the energy required to extinguish a fire and the corresponding
moisture injection from applied water.

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


def compute_moisture_injection(current_fmois: np.ndarray, dead_mx: float,
                                suppression_ratio: float) -> np.ndarray:
    """Compute moisture injection toward extinction for dead fuel classes.

    Dead fuel classes (indices 0–3) are pushed toward dead_mx proportionally
    to suppression_ratio. Live fuel classes (indices 4–5) are unchanged.

    Args:
        current_fmois: Current fuel moisture array, shape (6,).
        dead_mx: Dead fuel moisture of extinction (fraction).
        suppression_ratio: Ratio in [0, 1] from compute_suppression_ratio().

    Returns:
        New moisture array (copy; input is not modified).
    """
    new_fmois = current_fmois.copy()
    # Dead fuel classes: indices 0, 1, 2, 3
    for i in range(min(4, len(new_fmois))):
        new_fmois[i] = current_fmois[i] + suppression_ratio * (dead_mx - current_fmois[i])
    return new_fmois
