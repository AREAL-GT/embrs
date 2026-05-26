"""Tests for embrs.fire_danger.nfdrs_fuel_models — spot-check the §6.3 table."""
from __future__ import annotations

import pytest

from embrs.fire_danger.nfdrs_fuel_models import (
    CTA,
    KBDI_THRESHOLD_DEFAULT,
    NFDRS_FUEL_MODELS,
)


def test_all_five_models_present():
    assert set(NFDRS_FUEL_MODELS) == {"V", "W", "X", "Y", "Z"}


def test_shared_sav_values():
    """All five share SG1/10/100/1000/Wood/Herb and HD — per scope §6.3."""
    for fm in NFDRS_FUEL_MODELS.values():
        assert fm.SG1 == 2000
        assert fm.SG10 == 109
        assert fm.SG100 == 30
        assert fm.SG1000 == 8
        assert fm.SGWood == 1500
        assert fm.SGHerb == 2000
        assert fm.HD == 8000


@pytest.mark.parametrize("model,L1,L10,L100,L1000,LWood,LHerb,Depth,MXD,SCM,LDrought,WNDFC", [
    ("V", 0.1, 0.0,  0.0,  0.0,   0.0, 1.0,  1.0,  15, 108, 0.0, 0.6),
    ("W", 0.5, 0.5,  0.0,  0.0,   1.0, 0.6,  1.5,  15,  62, 1.0, 0.4),
    ("X", 4.5, 2.45, 0.0,  0.0,   7.0, 1.55, 4.4,  25, 104, 2.5, 0.4),
    ("Y", 2.5, 2.2,  3.6, 10.16,  0.0, 0.0,  0.6,  25,   5, 5.0, 0.2),
    ("Z", 4.5, 4.25, 4.0,  4.0,   0.0, 0.0,  1.0,  25,  19, 7.0, 0.4),
])
def test_per_model_values(model, L1, L10, L100, L1000, LWood, LHerb, Depth,
                          MXD, SCM, LDrought, WNDFC):
    fm = NFDRS_FUEL_MODELS[model]
    assert fm.L1 == pytest.approx(L1)
    assert fm.L10 == pytest.approx(L10)
    assert fm.L100 == pytest.approx(L100)
    assert fm.L1000 == pytest.approx(L1000)
    assert fm.LWood == pytest.approx(LWood)
    assert fm.LHerb == pytest.approx(LHerb)
    assert fm.Depth == pytest.approx(Depth)
    assert fm.MXD == pytest.approx(MXD)
    assert fm.SCM == pytest.approx(SCM)
    assert fm.LDrought == pytest.approx(LDrought)
    assert fm.WNDFC == pytest.approx(WNDFC)


def test_constants():
    assert CTA == pytest.approx(0.0459137)
    assert KBDI_THRESHOLD_DEFAULT == 100.0
