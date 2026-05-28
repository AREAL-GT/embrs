"""Component 5a — NFDRS V/W/X/Y/Z fuel model parameter table.

Transcribed from ``NFDRS4py/lib/NFDRS4/src/nfdrs4.cpp:CreateFuelModels()``
(lines 604-721). See plan §6.3.

All five models share these surface-area-to-volume ratios and heat of
combustion (verified via M0-9, scope §6.2):

- ``SG1 = 2000``, ``SG10 = 109``, ``SG100 = 30``, ``SG1000 = 8``
- ``SGWood = 1500``, ``SGHerb = 2000``
- ``HD = 8000`` BTU/lb

Loadings ``L*`` are tons/acre, ``Depth`` ft, ``MXD`` %, ``WNDFC``
dimensionless, ``SCM`` the spread-component normalizer (used only by IC —
unused for BI), ``LDrought`` the drought-transfer load.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NFDRSFuelModel:
    """Static parameters for one NFDRS fuel model."""

    model: str           # 'V'..'Z'
    description: str

    # Loadings (tons/acre)
    L1: float
    L10: float
    L100: float
    L1000: float
    LWood: float
    LHerb: float
    LDrought: float

    # Geometry / extinction / wind / IC normalizer
    Depth: float         # ft
    MXD: float           # % (dead moisture of extinction)
    SCM: float           # unused for BI (IC only)
    WNDFC: float         # wind reduction factor (dimensionless)

    # SAV ratios (1/ft) — all five share these defaults
    SG1: float = 2000.0
    SG10: float = 109.0
    SG100: float = 30.0
    SG1000: float = 8.0
    SGWood: float = 1500.0
    SGHerb: float = 2000.0

    # Heat of combustion (BTU/lb) — single value used for both dead and live
    # (scope §6.2 delta D2).
    HD: float = 8000.0


# Constants from nfdrs4.cpp:48 / nfdrs4.h.
CTA: float = 0.0459137   # tons/acre -> lb/ft^2
KBDI_THRESHOLD_DEFAULT: float = 100.0


# fmt: off
NFDRS_FUEL_MODELS: dict[str, NFDRSFuelModel] = {
    "V": NFDRSFuelModel(
        model="V", description="Grass",
        L1=0.1, L10=0.0, L100=0.0, L1000=0.0,
        LWood=0.0, LHerb=1.0, LDrought=0.0,
        Depth=1.0, MXD=15.0, SCM=108.0, WNDFC=0.6,
    ),
    "W": NFDRSFuelModel(
        model="W", description="Grass-Shrub",
        L1=0.5, L10=0.5, L100=0.0, L1000=0.0,
        LWood=1.0, LHerb=0.6, LDrought=1.0,
        Depth=1.5, MXD=15.0, SCM=62.0, WNDFC=0.4,
    ),
    "X": NFDRSFuelModel(
        model="X", description="Brush",
        L1=4.5, L10=2.45, L100=0.0, L1000=0.0,
        LWood=7.0, LHerb=1.55, LDrought=2.5,
        Depth=4.4, MXD=25.0, SCM=104.0, WNDFC=0.4,
    ),
    "Y": NFDRSFuelModel(
        model="Y", description="Timber",
        L1=2.5, L10=2.2, L100=3.6, L1000=10.16,
        LWood=0.0, LHerb=0.0, LDrought=5.0,
        Depth=0.6, MXD=25.0, SCM=5.0, WNDFC=0.2,
    ),
    "Z": NFDRSFuelModel(
        model="Z", description="Slash/Blowdown",
        L1=4.5, L10=4.25, L100=4.0, L1000=4.0,
        LWood=0.0, LHerb=0.0, LDrought=7.0,
        Depth=1.0, MXD=25.0, SCM=19.0, WNDFC=0.4,
    ),
}
# fmt: on
