"""Tests for IRPG Fuel Moisture Model."""
import pytest
import numpy as np

from embrs.models.irpg_moisture_model import (
    IRPGMoistureModel,
    FuelMoisturePriors,
    RFM_TABLE,
    DELTA_TABLES,
    MONTH_TO_TABLE,
)


class TestFuelMoisturePriors:
    """Tests for FuelMoisturePriors validation."""

    def test_default_priors_valid(self):
        """Default priors should be valid."""
        priors = FuelMoisturePriors()
        model = IRPGMoistureModel(priors)
        assert model.priors.p_shaded == 0.5

    def test_invalid_p_shaded_negative(self):
        """p_shaded < 0 should raise ValueError."""
        priors = FuelMoisturePriors(p_shaded=-0.1)
        with pytest.raises(ValueError, match="p_shaded must be in"):
            IRPGMoistureModel(priors)

    def test_invalid_p_shaded_greater_than_one(self):
        """p_shaded > 1 should raise ValueError."""
        priors = FuelMoisturePriors(p_shaded=1.5)
        with pytest.raises(ValueError, match="p_shaded must be in"):
            IRPGMoistureModel(priors)

    def test_invalid_aspect_probs_length(self):
        """aspect_probs with wrong length should raise ValueError."""
        priors = FuelMoisturePriors(aspect_probs=[0.5, 0.5])
        with pytest.raises(ValueError, match="aspect_probs must have 4 elements"):
            IRPGMoistureModel(priors)

    def test_invalid_aspect_probs_sum(self):
        """aspect_probs not summing to 1 should raise ValueError."""
        priors = FuelMoisturePriors(aspect_probs=[0.1, 0.1, 0.1, 0.1])
        with pytest.raises(ValueError, match="aspect_probs must sum to 1.0"):
            IRPGMoistureModel(priors)

    def test_invalid_slope_probs_length(self):
        """slope_probs with wrong length should raise ValueError."""
        priors = FuelMoisturePriors(slope_probs=[1.0])
        with pytest.raises(ValueError, match="slope_probs must have 2 elements"):
            IRPGMoistureModel(priors)

    def test_invalid_slope_probs_sum(self):
        """slope_probs not summing to 1 should raise ValueError."""
        priors = FuelMoisturePriors(slope_probs=[0.3, 0.3])
        with pytest.raises(ValueError, match="slope_probs must sum to 1.0"):
            IRPGMoistureModel(priors)

    def test_invalid_elev_probs_length(self):
        """elev_probs with wrong length should raise ValueError."""
        priors = FuelMoisturePriors(elev_probs=[0.5, 0.5])
        with pytest.raises(ValueError, match="elev_probs must have 3 elements"):
            IRPGMoistureModel(priors)

    def test_invalid_elev_probs_sum(self):
        """elev_probs not summing to 1 should raise ValueError."""
        priors = FuelMoisturePriors(elev_probs=[0.1, 0.1, 0.1])
        with pytest.raises(ValueError, match="elev_probs must sum to 1.0"):
            IRPGMoistureModel(priors)


class TestRFMBinning:
    """Tests for Reference Fuel Moisture (RFM) lookup."""

    @pytest.fixture
    def model(self):
        return IRPGMoistureModel(FuelMoisturePriors())

    def test_rfm_first_bin(self, model):
        """RH=0, T=10 should map to first cell."""
        assert model.rfm(T=10, RH=0) == float(RFM_TABLE[0][0])
        assert model.rfm(T=10, RH=4) == float(RFM_TABLE[0][0])

    def test_rfm_second_rh_bin(self, model):
        """RH=5 should map to second RH bin."""
        assert model.rfm(T=10, RH=5) == float(RFM_TABLE[1][0])
        assert model.rfm(T=10, RH=9) == float(RFM_TABLE[1][0])

    def test_rfm_last_rh_bin(self, model):
        """RH=100 should map to last RH bin."""
        assert model.rfm(T=10, RH=100) == float(RFM_TABLE[20][0])

    def test_rfm_second_t_bin(self, model):
        """T=30 should map to second T bin."""
        assert model.rfm(T=30, RH=0) == float(RFM_TABLE[0][1])
        assert model.rfm(T=49, RH=0) == float(RFM_TABLE[0][1])

    def test_rfm_last_t_bin(self, model):
        """T=110+ should map to last T bin (109+)."""
        assert model.rfm(T=110, RH=0) == float(RFM_TABLE[0][5])
        assert model.rfm(T=200, RH=0) == float(RFM_TABLE[0][5])

    def test_rfm_boundary_rh_100(self, model):
        """RH=100 at various temperatures."""
        # From Table A, RH=100 column
        assert model.rfm(T=20, RH=100) == 14.0   # T 10-29
        assert model.rfm(T=40, RH=100) == 13.0   # T 30-49
        assert model.rfm(T=60, RH=100) == 13.0   # T 50-69
        assert model.rfm(T=80, RH=100) == 13.0   # T 70-89
        assert model.rfm(T=100, RH=100) == 13.0  # T 90-109
        assert model.rfm(T=120, RH=100) == 12.0  # T 109+
        assert model.rfm(T=75, RH=43) == 6.0
        assert model.rfm(T=69, RH=74) == 9.0   # T 50-69, RH 70-74


    def test_rfm_clamps_low_rh(self, model):
        """Negative RH should be clamped to 0."""
        assert model.rfm(T=20, RH=-10) == model.rfm(T=20, RH=0)

    def test_rfm_clamps_high_rh(self, model):
        """RH > 100 should be clamped to 100."""
        assert model.rfm(T=20, RH=150) == model.rfm(T=20, RH=100)

    def test_rfm_clamps_low_t(self, model):
        """T < 10 should be clamped to first bin."""
        assert model.rfm(T=0, RH=50) == model.rfm(T=10, RH=50)
        assert model.rfm(T=-20, RH=50) == model.rfm(T=10, RH=50)


class TestMonthMapping:
    """Tests for month to table ID mapping."""

    def test_table_b_months(self):
        """May, June, July should map to Table B."""
        assert MONTH_TO_TABLE[5] == "B"
        assert MONTH_TO_TABLE[6] == "B"
        assert MONTH_TO_TABLE[7] == "B"

    def test_table_c_months(self):
        """Feb, Mar, Apr, Aug, Sep, Oct should map to Table C."""
        assert MONTH_TO_TABLE[2] == "C"
        assert MONTH_TO_TABLE[3] == "C"
        assert MONTH_TO_TABLE[4] == "C"
        assert MONTH_TO_TABLE[8] == "C"
        assert MONTH_TO_TABLE[9] == "C"
        assert MONTH_TO_TABLE[10] == "C"

    def test_table_d_months(self):
        """Nov, Dec, Jan should map to Table D."""
        assert MONTH_TO_TABLE[11] == "D"
        assert MONTH_TO_TABLE[12] == "D"
        assert MONTH_TO_TABLE[1] == "D"


class TestTimeBinMapping:
    """Tests for time hour to time bin mapping."""

    @pytest.fixture
    def model(self):
        return IRPGMoistureModel(FuelMoisturePriors())

    def test_time_before_0800(self, model):
        """Hours before 8 AM should use 1800> bin."""
        for hour in [0, 1, 5, 7]:
            assert model._time_hr_to_time_bin(hour) == "1800>"

    def test_time_0800_bin(self, model):
        """Hours 8-9 should use 0800> bin."""
        assert model._time_hr_to_time_bin(8) == "0800>"
        assert model._time_hr_to_time_bin(9) == "0800>"

    def test_time_1000_bin(self, model):
        """Hours 10-11 should use 1000> bin."""
        assert model._time_hr_to_time_bin(10) == "1000>"
        assert model._time_hr_to_time_bin(11) == "1000>"

    def test_time_1200_bin(self, model):
        """Hours 12-13 should use 1200> bin."""
        assert model._time_hr_to_time_bin(12) == "1200>"
        assert model._time_hr_to_time_bin(13) == "1200>"

    def test_time_1400_bin(self, model):
        """Hours 14-15 should use 1400> bin."""
        assert model._time_hr_to_time_bin(14) == "1400>"
        assert model._time_hr_to_time_bin(15) == "1400>"

    def test_time_1600_bin(self, model):
        """Hours 16-17 should use 1600> bin."""
        assert model._time_hr_to_time_bin(16) == "1600>"
        assert model._time_hr_to_time_bin(17) == "1600>"

    def test_time_1800_bin(self, model):
        """Hours 18+ should use 1800> bin."""
        for hour in [18, 19, 20, 23]:
            assert model._time_hr_to_time_bin(hour) == "1800>"


class TestDeltaLookup:
    """Tests for delta correction table lookups."""

    @pytest.fixture
    def model(self):
        return IRPGMoistureModel(FuelMoisturePriors())

    def test_exposed_lookup_table_b(self, model):
        """Test exposed lookup for Table B (May-July)."""
        # Table B, N 0-30%, 1200>, L (col index 7)
        delta = model._delta_lookup_exposed("B", "N", "0_30", "1200>", "L")
        assert delta == 0

        # Table B, W 31%+, 0800>, A (col index 2)
        delta = model._delta_lookup_exposed("B", "W", "31_plus", "0800>", "A")
        assert delta == 6

    def test_exposed_lookup_table_d(self, model):
        """Test exposed lookup for Table D (Nov-Jan)."""
        # Table D, N 31%+, any time should be 4, 5, or 6
        delta = model._delta_lookup_exposed("D", "N", "31_plus", "1200>", "B")
        assert delta == 4

    def test_shaded_lookup_table_b(self, model):
        """Test shaded lookup for Table B."""
        # Table B, N, 1200>, L
        delta = model._delta_lookup_shaded("B", "N", "1200>", "L")
        assert delta == 3

    def test_shaded_lookup_table_d(self, model):
        """Test shaded lookup for Table D (all values 4, 5, 6)."""
        # Table D shaded is all 4, 5, 6 for B, L, A elevations
        delta = model._delta_lookup_shaded("D", "N", "1200>", "B")
        assert delta == 4
        delta = model._delta_lookup_shaded("D", "N", "1200>", "L")
        assert delta == 5
        delta = model._delta_lookup_shaded("D", "N", "1200>", "A")
        assert delta == 6


class TestSampleDelta:
    """Tests for stochastic delta sampling."""

    @pytest.fixture
    def model(self):
        return IRPGMoistureModel(FuelMoisturePriors())

    def test_sample_delta_returns_valid_range(self, model):
        """sample_delta should return values in plausible range."""
        rng = np.random.default_rng(123)
        for _ in range(100):
            delta = model.sample_delta(month=6, local_time_hr=13, rng=rng)
            assert isinstance(delta, float)
            assert 0 <= delta <= 6

    def test_sample_delta_deterministic_with_seed(self, model):
        """Same seed should produce same results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        results1 = [model.sample_delta(6, 13, rng1) for _ in range(10)]
        results2 = [model.sample_delta(6, 13, rng2) for _ in range(10)]

        assert results1 == results2

    def test_sample_delta_uses_correct_table(self, model):
        """Delta values should match expected tables for different months."""
        rng = np.random.default_rng(999)

        # Summer (Table B) - typically lower corrections at midday
        summer_deltas = [model.sample_delta(6, 12, rng) for _ in range(50)]

        # Winter (Table D) - typically higher corrections
        rng = np.random.default_rng(999)
        winter_deltas = [model.sample_delta(12, 12, rng) for _ in range(50)]

        # Winter should have higher average correction
        assert np.mean(winter_deltas) >= np.mean(summer_deltas)


class TestSampleFDFM:
    """Tests for full FDFM sampling."""

    @pytest.fixture
    def model(self):
        return IRPGMoistureModel(FuelMoisturePriors())

    def test_sample_fdfm_equals_rfm_plus_delta(self, model):
        """FDFM should equal RFM + delta."""
        T, RH, month, hour = 75.0, 35.0, 7, 14

        # Get RFM (deterministic)
        rfm = model.rfm(T, RH)

        # Sample delta with specific seed
        rng1 = np.random.default_rng(123)
        delta = model.sample_delta(month, hour, rng1)

        # Sample FDFM with same seed
        rng2 = np.random.default_rng(123)
        fdfm = model.sample_fdfm(T, RH, month, hour, rng2)

        assert fdfm == rfm + delta

    def test_sample_fdfm_returns_plausible_values(self, model):
        """FDFM should be in reasonable range (1-20%)."""
        rng = np.random.default_rng(42)

        for _ in range(100):
            fdfm = model.sample_fdfm(
                T=np.random.uniform(50, 100),
                RH=np.random.uniform(10, 90),
                month=np.random.randint(1, 13),
                local_time_hr=np.random.randint(0, 24),
                rng=rng
            )
            assert 1 <= fdfm <= 20

    def test_sample_fdfm_deterministic(self, model):
        """Same seed should produce same FDFM."""
        params = (75.0, 35.0, 7, 14)

        rng1 = np.random.default_rng(42)
        fdfm1 = model.sample_fdfm(*params, rng=rng1)

        rng2 = np.random.default_rng(42)
        fdfm2 = model.sample_fdfm(*params, rng=rng2)

        assert fdfm1 == fdfm2


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def model(self):
        return IRPGMoistureModel(FuelMoisturePriors())

    def test_extreme_temperature(self, model):
        """Model should handle extreme temperatures."""
        rng = np.random.default_rng(1)

        # Very cold
        fdfm = model.sample_fdfm(T=-40, RH=50, month=1, local_time_hr=12, rng=rng)
        assert fdfm > 0

        # Very hot
        fdfm = model.sample_fdfm(T=150, RH=50, month=7, local_time_hr=12, rng=rng)
        assert fdfm > 0

    def test_all_shaded(self):
        """Model with p_shaded=1 should only use shaded tables."""
        priors = FuelMoisturePriors(p_shaded=1.0)
        model = IRPGMoistureModel(priors)
        rng = np.random.default_rng(42)

        # Table D shaded values are all 4, 5, or 6
        for _ in range(20):
            delta = model.sample_delta(month=12, local_time_hr=12, rng=rng)
            assert delta in [4.0, 5.0, 6.0]

    def test_all_exposed(self):
        """Model with p_shaded=0 should only use exposed tables."""
        priors = FuelMoisturePriors(p_shaded=0.0)
        model = IRPGMoistureModel(priors)
        rng = np.random.default_rng(42)

        # Table B exposed at midday can have 0 values
        deltas = [model.sample_delta(month=6, local_time_hr=12, rng=rng) for _ in range(50)]
        assert 0.0 in deltas  # Exposed tables have 0 values
