"""Tests for embrs.fire_danger.nfdrs_index.

Verification strategy (per plan §4.1, M0-12):
1. Boundary tests for ``cure`` and the slope-class table.
2. Structural invariants (D1 branch active, NaN propagation, failure paths).
3. Physical-direction tests (dry vs. wet, low vs. high wind, low vs. high
   slope, KBDI drought effect).
4. **Regression anchors** for all five fuel models under one fixed input
   set, cross-checked against ``iCalcIndexes`` control flow by inspection.
   PSW-82 has no published worked numeric example, so the anchors are the
   primary line of defence against an accidental refactor of the port.
"""
from __future__ import annotations

import math

import pytest

from embrs.fire_danger.nfdrs_fuel_models import NFDRS_FUEL_MODELS
from embrs.fire_danger.nfdrs_index import (
    _SLPFCT,
    calc_indexes,
    cure,
)


# ---------------------------------------------------------------------------
# cure()
# ---------------------------------------------------------------------------


def test_cure_below_greenup_is_fully_cured():
    assert cure(0.0) == 1.0
    assert cure(0.49) == 1.0


def test_cure_at_greenup_is_one():
    # Just AT greenup (0.5): linear formula gives (-2)*(0.5)+2 = 1.0
    assert cure(0.5) == pytest.approx(1.0)


def test_cure_at_max_gsi_is_zero():
    assert cure(1.0) == pytest.approx(0.0, abs=1e-12)


def test_cure_linear_interpolation_above_greenup():
    # Halfway between greenup (0.5) and max (1.0): gsi=0.75 -> fct=0.5
    assert cure(0.75) == pytest.approx(0.5, rel=1e-9)


def test_cure_nan_is_dormant():
    assert cure(float("nan")) == 1.0


def test_cure_clamped_to_unit_interval():
    # gsi > 1.0 would otherwise give a negative fct; test the clamp.
    assert cure(1.5) == 0.0
    assert cure(-0.1, greenup_threshold=0.5) == 1.0


# ---------------------------------------------------------------------------
# Slope-class table — PSW-82 D3 values.
# ---------------------------------------------------------------------------


def test_slope_class_table_matches_psw82():
    assert _SLPFCT == (0.267, 0.533, 1.068, 2.134, 4.273)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_invalid_slope_class_raises():
    with pytest.raises(ValueError, match="slope_class"):
        calc_indexes(NFDRS_FUEL_MODELS["Y"], 6, 8, 12, 15, 70, 100, 0.5, 300, 10, 0)
    with pytest.raises(ValueError, match="slope_class"):
        calc_indexes(NFDRS_FUEL_MODELS["Y"], 6, 8, 12, 15, 70, 100, 0.5, 300, 10, 6)


def test_negative_wind_raises():
    with pytest.raises(ValueError, match="wind_mph"):
        calc_indexes(NFDRS_FUEL_MODELS["Y"], 6, 8, 12, 15, 70, 100, 0.5, 300, -1, 2)


def test_d1_branch_active_for_all_five_models():
    """D1: SGWood=1500 > 1200 and SGHerb=2000 > 1200 -> WLIVEN = WTOTLN."""
    for fm in NFDRS_FUEL_MODELS.values():
        assert fm.SGWood > 1200
        assert fm.SGHerb > 1200


# ---------------------------------------------------------------------------
# Regression anchors — cross-checked against iCalcIndexes by inspection.
# Inputs: moderate-dry summer day, slope class 2, moderate KBDI = 300.
# ---------------------------------------------------------------------------


_ANCHOR_INPUTS = dict(
    mc1=6.0, mc10=8.0, mc100=12.0, mc1000=15.0,
    mcherb=70.0, mcwood=100.0,
    gsi=0.5, kbdi=300.0,
    wind_mph=10.0, slope_class=2,
)

# Expected values from the current port. If the port is refactored these
# anchors should be re-derived against the C++ flow by inspection before
# being updated.
_ANCHORS = {
    "V": (86.3856, 6.1759, 54.0820),
    "W": (27.6388, 8.7425, 37.5681),
    "X": (74.1612, 76.5106, 160.4574),
    "Y": (4.5001, 41.1995, 33.2583),
    "Z": (17.1217, 93.8182, 89.7945),
}


@pytest.mark.parametrize("model", list(_ANCHORS))
def test_anchor_values(model):
    sc, erc, bi = _ANCHORS[model]
    r = calc_indexes(NFDRS_FUEL_MODELS[model], **_ANCHOR_INPUTS)
    assert r.sc == pytest.approx(sc, rel=1e-4)
    assert r.erc == pytest.approx(erc, rel=1e-4)
    assert r.bi == pytest.approx(bi, rel=1e-4)


def test_anchor_physics_grass_vs_timber():
    """V (grass): high SC, low ERC. Y (timber): low SC, high ERC."""
    v = calc_indexes(NFDRS_FUEL_MODELS["V"], **_ANCHOR_INPUTS)
    y = calc_indexes(NFDRS_FUEL_MODELS["Y"], **_ANCHOR_INPUTS)
    assert v.sc > y.sc * 5      # grass spreads much faster
    assert y.erc > v.erc * 5    # timber holds much more energy


# ---------------------------------------------------------------------------
# Physical-direction tests
# ---------------------------------------------------------------------------


def _calc_Y(**overrides):
    inputs = dict(_ANCHOR_INPUTS)
    inputs.update(overrides)
    return calc_indexes(NFDRS_FUEL_MODELS["Y"], **inputs)


def test_drier_dead_fuel_raises_bi():
    dry = _calc_Y(mc1=3.0, mc10=4.0, mc100=6.0, mc1000=8.0)
    moist = _calc_Y(mc1=15.0, mc10=18.0, mc100=22.0, mc1000=25.0)
    assert dry.bi > moist.bi
    assert dry.sc > moist.sc
    assert dry.erc > moist.erc


def test_higher_wind_raises_sc_and_bi():
    calm = _calc_Y(wind_mph=0.0)
    windy = _calc_Y(wind_mph=20.0)
    assert windy.sc > calm.sc
    assert windy.bi > calm.bi
    # ERC does not depend on wind.
    assert windy.erc == pytest.approx(calm.erc, rel=1e-12)


def test_steeper_slope_raises_sc_and_bi():
    flat = _calc_Y(slope_class=1)
    steep = _calc_Y(slope_class=5)
    assert steep.sc > flat.sc
    assert steep.bi > flat.bi
    # ERC does not depend on slope.
    assert steep.erc == pytest.approx(flat.erc, rel=1e-12)


def test_higher_kbdi_raises_erc_and_bi_for_drought_fuels():
    """Model Y (LDrought=5.0) responds to KBDI."""
    base = _calc_Y(kbdi=100.0)        # at threshold -> no transfer
    drought = _calc_Y(kbdi=600.0)     # well above threshold
    assert drought.erc > base.erc
    assert drought.bi > base.bi


def test_kbdi_has_no_effect_for_zero_drought_load_model():
    """Model V has LDrought=0 -> KBDI shouldn't change BI."""
    base = calc_indexes(NFDRS_FUEL_MODELS["V"], kbdi=100.0,
                        **{k: v for k, v in _ANCHOR_INPUTS.items() if k != "kbdi"})
    drought = calc_indexes(NFDRS_FUEL_MODELS["V"], kbdi=600.0,
                           **{k: v for k, v in _ANCHOR_INPUTS.items() if k != "kbdi"})
    assert drought.bi == pytest.approx(base.bi, rel=1e-9)


def test_nan_gsi_treated_as_dormant_via_cure():
    """NaN GSI -> cure returns 1.0 -> all herbaceous transferred to dead."""
    r_nan = _calc_Y(gsi=float("nan"))
    r_dormant = _calc_Y(gsi=0.0)
    assert r_nan.sc == pytest.approx(r_dormant.sc, rel=1e-12)
    assert r_nan.erc == pytest.approx(r_dormant.erc, rel=1e-12)
    assert r_nan.bi == pytest.approx(r_dormant.bi, rel=1e-12)
