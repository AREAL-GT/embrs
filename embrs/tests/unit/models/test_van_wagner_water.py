"""Tests for Van Wagner & Taylor (2022) energy-balance water suppression model.

All expected values are hand-calculated from the paper equations.
"""

import pytest
import numpy as np

from embrs.models.van_wagner_water import (
    heat_absorbed_per_kg_water,
    water_depth_quench_flame_mm,
    water_depth_cool_fuel_mm,
    water_depth_combined_mm,
    heat_to_extinguish_kJ,
    volume_L_to_energy_kJ,
    compute_suppression_ratio,
    compute_moisture_injection,
)

_KCAL_TO_KJ = 4.184


class TestHeatAbsorbed:
    """Tests for heat_absorbed_per_kg_water (Eq. 1a/1b)."""

    def test_default_620_kcal(self):
        """Eq. 1b default: T_a=20, T_e=100 -> H_w = 620 kcal/kg = 2594.08 kJ/kg."""
        result = heat_absorbed_per_kg_water()
        expected = 620.0 * _KCAL_TO_KJ  # 2594.08
        assert result == pytest.approx(expected, rel=1e-6)

    def test_cold_ambient(self):
        """T_a=0 -> H_w = (100-0) + 540 = 640 kcal/kg."""
        result = heat_absorbed_per_kg_water(T_a=0.0, T_e=100.0)
        expected = 640.0 * _KCAL_TO_KJ
        assert result == pytest.approx(expected, rel=1e-6)

    def test_superheated_steam(self):
        """Eq. 1a with T_e=200: H_w = (100-20) + 540 + (200-100)*0.46 = 666 kcal/kg."""
        result = heat_absorbed_per_kg_water(T_a=20.0, T_e=200.0)
        expected = 666.0 * _KCAL_TO_KJ
        assert result == pytest.approx(expected, rel=1e-6)

    def test_warm_ambient(self):
        """T_a=30 -> H_w = (100-30) + 540 = 610 kcal/kg."""
        result = heat_absorbed_per_kg_water(T_a=30.0, T_e=100.0)
        expected = 610.0 * _KCAL_TO_KJ
        assert result == pytest.approx(expected, rel=1e-6)


class TestWaterDepthQuenchFlame:
    """Tests for water_depth_quench_flame_mm (Eq. 7b).

    Table 3 values (in kcal/s-m): I=100 -> 0.008, I=1000 -> 0.038, I=10000 -> 0.18 mm.
    """

    def test_I_100_kcal(self):
        """I=100 kcal/s-m -> D = 3.84e-4 * 100^(2/3) = 0.00826 mm."""
        I_kW_m = 100.0 * _KCAL_TO_KJ  # Convert kcal/s-m to kW/m
        result = water_depth_quench_flame_mm(I_kW_m)
        expected = 3.84e-4 * 100.0 ** (2.0 / 3.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_I_1000_kcal(self):
        """I=1000 kcal/s-m -> D = 3.84e-4 * 1000^(2/3) ≈ 0.0384 mm."""
        I_kW_m = 1000.0 * _KCAL_TO_KJ
        result = water_depth_quench_flame_mm(I_kW_m)
        expected = 3.84e-4 * 1000.0 ** (2.0 / 3.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_I_10000_kcal(self):
        """I=10000 kcal/s-m -> D = 3.84e-4 * 10000^(2/3) ≈ 0.178 mm."""
        I_kW_m = 10000.0 * _KCAL_TO_KJ
        result = water_depth_quench_flame_mm(I_kW_m)
        expected = 3.84e-4 * 10000.0 ** (2.0 / 3.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_zero_intensity(self):
        """Zero intensity -> zero water depth."""
        assert water_depth_quench_flame_mm(0.0) == 0.0


class TestWaterDepthCoolFuel:
    """Tests for water_depth_cool_fuel_mm (Eq. 10b).

    Table 3: W_1=1.25 -> 0.28, W_1=2.5 -> 0.56, W_1=5.0 -> 1.13 mm.
    """

    def test_W1_1_25(self):
        """W_1=1.25 kg/m² -> D = 0.226 * 1.25 = 0.2825 mm."""
        result = water_depth_cool_fuel_mm(1.25)
        assert result == pytest.approx(0.226 * 1.25, rel=1e-6)

    def test_W1_2_5(self):
        """W_1=2.5 kg/m² -> D = 0.226 * 2.5 = 0.565 mm."""
        result = water_depth_cool_fuel_mm(2.5)
        assert result == pytest.approx(0.226 * 2.5, rel=1e-6)

    def test_W1_5_0(self):
        """W_1=5.0 kg/m² -> D = 0.226 * 5.0 = 1.13 mm."""
        result = water_depth_cool_fuel_mm(5.0)
        assert result == pytest.approx(0.226 * 5.0, rel=1e-6)

    def test_zero_loading(self):
        """Zero fuel loading -> zero water depth."""
        assert water_depth_cool_fuel_mm(0.0) == 0.0


class TestWaterDepthCombined:
    """Tests for water_depth_combined_mm (Mechanism I + II)."""

    def test_sum_of_mechanisms(self):
        """Combined should equal sum of quench + cool."""
        I_kW_m = 1000.0 * _KCAL_TO_KJ
        W_1 = 2.5

        combined = water_depth_combined_mm(I_kW_m, W_1)
        separate = water_depth_quench_flame_mm(I_kW_m) + water_depth_cool_fuel_mm(W_1)

        assert combined == pytest.approx(separate, rel=1e-10)

    def test_zero_both(self):
        """Zero intensity and loading -> zero combined."""
        assert water_depth_combined_mm(0.0, 0.0) == 0.0


class TestHeatToExtinguish:
    """Tests for heat_to_extinguish_kJ pipeline."""

    def test_known_values(self):
        """Pipeline test with known inputs."""
        I_kW_m = 500.0 * _KCAL_TO_KJ
        W_1 = 2.0
        area = 10.0
        eff = 2.5
        T_a = 20.0

        D_mm = water_depth_combined_mm(I_kW_m, W_1) * eff
        mass_kg = (D_mm / 1000.0) * area * 1000.0
        expected = mass_kg * heat_absorbed_per_kg_water(T_a)

        result = heat_to_extinguish_kJ(I_kW_m, W_1, area, eff, T_a)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_zero_area_returns_zero(self):
        """Zero fire area -> zero energy needed."""
        result = heat_to_extinguish_kJ(1000.0, 2.0, 0.0)
        assert result == 0.0

    def test_negative_area_returns_zero(self):
        """Negative fire area -> zero energy needed."""
        result = heat_to_extinguish_kJ(1000.0, 2.0, -5.0)
        assert result == 0.0

    def test_higher_intensity_needs_more_energy(self):
        """Higher intensity fire should need more energy to extinguish."""
        low = heat_to_extinguish_kJ(100.0, 2.0, 10.0)
        high = heat_to_extinguish_kJ(1000.0, 2.0, 10.0)
        assert high > low

    def test_larger_area_needs_more_energy(self):
        """Larger fire area should need more energy."""
        small = heat_to_extinguish_kJ(500.0, 2.0, 5.0)
        large = heat_to_extinguish_kJ(500.0, 2.0, 50.0)
        assert large > small


class TestVolumeLToEnergy:
    """Tests for volume_L_to_energy_kJ."""

    def test_1L_at_20C(self):
        """1 L at 20°C = 1 kg * 620 kcal/kg * 4.184 = 2594.08 kJ."""
        result = volume_L_to_energy_kJ(1.0, T_a=20.0)
        expected = 620.0 * _KCAL_TO_KJ
        assert result == pytest.approx(expected, rel=1e-6)

    def test_zero_volume(self):
        """Zero volume -> zero energy."""
        assert volume_L_to_energy_kJ(0.0) == 0.0

    def test_scales_linearly(self):
        """Energy should scale linearly with volume."""
        e1 = volume_L_to_energy_kJ(1.0)
        e5 = volume_L_to_energy_kJ(5.0)
        assert e5 == pytest.approx(5.0 * e1, rel=1e-10)


class TestSuppressionRatio:
    """Tests for compute_suppression_ratio."""

    def test_half_suppressed(self):
        """Half the needed energy -> ratio 0.5."""
        assert compute_suppression_ratio(500.0, 1000.0) == pytest.approx(0.5)

    def test_fully_suppressed(self):
        """Exactly enough energy -> ratio 1.0."""
        assert compute_suppression_ratio(1000.0, 1000.0) == pytest.approx(1.0)

    def test_over_suppressed_clamped(self):
        """More than needed -> clamped to 1.0."""
        assert compute_suppression_ratio(2000.0, 1000.0) == pytest.approx(1.0)

    def test_zero_heat_returns_one(self):
        """Zero heat needed (trivially suppressible) -> 1.0."""
        assert compute_suppression_ratio(100.0, 0.0) == pytest.approx(1.0)

    def test_zero_water_returns_zero(self):
        """No water applied -> 0.0."""
        assert compute_suppression_ratio(0.0, 1000.0) == pytest.approx(0.0)

    def test_negative_heat_returns_one(self):
        """Negative heat needed -> 1.0."""
        assert compute_suppression_ratio(50.0, -10.0) == pytest.approx(1.0)


class TestMoistureInjection:
    """Tests for compute_moisture_injection."""

    def test_zero_ratio_no_change(self):
        """Zero suppression ratio -> moisture unchanged."""
        fmois = np.array([0.06, 0.07, 0.08, 0.06, 0.30, 0.80])
        result = compute_moisture_injection(fmois, 0.25, 0.0)
        np.testing.assert_array_almost_equal(result, fmois)

    def test_full_ratio_reaches_dead_mx(self):
        """Ratio 1.0 -> dead fuel classes reach dead_mx."""
        fmois = np.array([0.06, 0.07, 0.08, 0.06, 0.30, 0.80])
        dead_mx = 0.25
        result = compute_moisture_injection(fmois, dead_mx, 1.0)

        # Dead fuel classes (0-3) should be at dead_mx
        for i in range(4):
            assert result[i] == pytest.approx(dead_mx)

        # Live fuel classes (4-5) unchanged
        assert result[4] == pytest.approx(0.30)
        assert result[5] == pytest.approx(0.80)

    def test_half_ratio_halfway(self):
        """Ratio 0.5 -> dead fuel classes halfway to dead_mx."""
        fmois = np.array([0.06, 0.07, 0.08, 0.06, 0.30, 0.80])
        dead_mx = 0.25
        result = compute_moisture_injection(fmois, dead_mx, 0.5)

        for i in range(4):
            expected = fmois[i] + 0.5 * (dead_mx - fmois[i])
            assert result[i] == pytest.approx(expected)

    def test_does_not_modify_input(self):
        """Input array should not be modified."""
        fmois = np.array([0.06, 0.07, 0.08, 0.06, 0.30, 0.80])
        original = fmois.copy()
        compute_moisture_injection(fmois, 0.25, 1.0)
        np.testing.assert_array_equal(fmois, original)

    def test_live_fuel_unchanged(self):
        """Live fuel indices (4, 5) should always remain unchanged."""
        fmois = np.array([0.10, 0.10, 0.10, 0.10, 0.50, 0.90])
        result = compute_moisture_injection(fmois, 0.30, 0.8)

        assert result[4] == pytest.approx(0.50)
        assert result[5] == pytest.approx(0.90)
