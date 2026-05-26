"""Component 5b — NFDRS Spread / Energy Release / Burning Index port.

Faithful port of ``NFDRS4py/.../nfdrs4.cpp::iCalcIndexes`` (lines 762-1037)
and ``Cure`` (1140-1151), with the deltas documented in scope §6.2:

- **D1**: ``WLIVEN = WTOTLN`` when ``SGWood > 1200 && SGHerb > 1200`` (true
  for all V/W/X/Y/Z).
- **D2**: single ``HD`` for dead and live.
- **D3**: 5-class slope table only; the continuous-slope branch is omitted
  (slope class is constrained 1-5).
- **D4**: floats throughout — no rounding (would quantize the trajectory).
- **D5**: GSI-based :func:`cure` instead of PSW-82 ``FCTCUR``.
- **D6**: drought block in :mod:`kbdi` (called before curing).
- **D7**: ``SAHERB`` uses ``WHERBP``; underflow guards on ``HNHERB`` / ``HNWOOD``.
- **D8**: Ignition Component skipped — not needed for BI.

The port returns :class:`IndexResult` with ``SC`` / ``ERC`` / ``BI`` as
floats. ``SADEAD <= 0`` (no dead-fuel surface area at all) returns NaN for
all three (data-driven failure — OQ-14 path).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from embrs.fire_danger.kbdi import apply_drought_load_transfer
from embrs.fire_danger.nfdrs_fuel_models import (
    CTA,
    KBDI_THRESHOLD_DEFAULT,
    NFDRSFuelModel,
)


# Shared constants from iCalcIndexes:765-767.
_STD = 0.0555
_STL = 0.0555
_RHOD = 32.0
_RHOL = 32.0
_ETASD = 0.4173969
_ETASL = 0.4173969


# 5-class slope factor table from iCalcIndexes:956-973.
_SLPFCT: tuple[float, ...] = (0.267, 0.533, 1.068, 2.134, 4.273)


@dataclass(frozen=True)
class IndexResult:
    """Output of :func:`calc_indexes`."""

    sc: float
    erc: float
    bi: float


def cure(
    gsi: float,
    greenup_threshold: float = 0.5,
    gsi_max: float = 1.0,
) -> float:
    """Return the herbaceous curing fraction ``fct_cur`` in ``[0, 1]``.

    Port of ``NFDRS4::Cure`` (nfdrs4.cpp:1140-1151). Below ``greenup_threshold``
    the herbaceous is fully cured (``1.0``); above, it interpolates down to
    ``0.0`` at ``gsi = gsi_max``. NaN GSI is treated as dormant (``1.0``) —
    matches the seed behaviour when the GSI tracker has < 2 days buffered.
    """
    if not (gsi == gsi):  # NaN
        return 1.0
    if gsi < greenup_threshold:
        return 1.0
    fct = (-1.0 / (1.0 - greenup_threshold)) * (gsi / gsi_max) + 1.0 / (1.0 - greenup_threshold)
    if fct < 0.0:
        return 0.0
    if fct > 1.0:
        return 1.0
    return fct


def calc_indexes(
    fuel: NFDRSFuelModel,
    mc1: float,
    mc10: float,
    mc100: float,
    mc1000: float,
    mcherb: float,
    mcwood: float,
    gsi: float,
    kbdi: float,
    wind_mph: float,
    slope_class: int,
    kbdi_threshold: float = KBDI_THRESHOLD_DEFAULT,
) -> IndexResult:
    """Compute SC / ERC / BI for one fuel model at one timestep.

    All moisture-content inputs are in **percent**; ``wind_mph`` is mph at
    NFDRS 20-ft reference height; ``slope_class`` is 1..5 (PSW-82 table).

    Returns:
        :class:`IndexResult` with SC, ERC, BI as floats (no rounding — D4).

    Raises:
        ValueError: For caller bugs — slope class out of [1, 5], negative
            wind, non-positive fuel-bed depth.
    """
    if not (1 <= slope_class <= 5):
        raise ValueError(f"slope_class must be in [1, 5], got {slope_class!r}")
    if wind_mph < 0:
        raise ValueError(f"wind_mph must be non-negative, got {wind_mph!r}")
    if fuel.Depth <= 0:
        raise ValueError(f"fuel.Depth must be positive, got {fuel.Depth!r}")

    # -----------------------------------------------------------------
    # 1. Drought load transfer (iCalcIndexes:806-823, in kbdi.py)
    # -----------------------------------------------------------------
    adj = apply_drought_load_transfer(fuel, kbdi, kbdi_threshold)
    W1, W10, W100, W1000 = adj.W1, adj.W10, adj.W100, adj.W1000
    WHERB, WWOOD = adj.W_herb, adj.W_wood
    DEPTH = adj.depth

    # -----------------------------------------------------------------
    # 2. Cure / load transfer (iCalcIndexes:827-829 + Cure:1148-1149)
    # -----------------------------------------------------------------
    fct_cur = cure(gsi)
    W1P = W1 + WHERB * fct_cur
    WHERBP = WHERB * (1.0 - fct_cur)

    # -----------------------------------------------------------------
    # 3. Preliminary calculations (iCalcIndexes:831-842)
    # -----------------------------------------------------------------
    WTOTD = W1P + W10 + W100 + W1000
    WTOTL = WHERBP + WWOOD
    WTOT = WTOTD + WTOTL
    W1N = W1P * (1.0 - _STD)
    W10N = W10 * (1.0 - _STD)
    W100N = W100 * (1.0 - _STD)
    WHERBN = WHERBP * (1.0 - _STL)
    WWOODN = WWOOD * (1.0 - _STL)
    WTOTLN = WTOTL * (1.0 - _STL)
    RHOBED = (WTOT - W1000) / DEPTH
    RHOBAR = ((WTOTL * _RHOL) + (WTOTD * _RHOD)) / WTOT if WTOT > 0 else 0.0
    BETBAR = RHOBED / RHOBAR if RHOBAR > 0 else 0.0

    # -----------------------------------------------------------------
    # 4. Live moisture of extinction (iCalcIndexes:844-871)
    # -----------------------------------------------------------------
    if WTOTLN > 0:
        HN1 = W1N * math.exp(-138.0 / fuel.SG1)
        HN10 = W10N * math.exp(-138.0 / fuel.SG10)
        HN100 = W100N * math.exp(-138.0 / fuel.SG100)
        HNHERB = 0.0 if (-500.0 / fuel.SGHerb) < -180.218 \
                 else WHERBN * math.exp(-500.0 / fuel.SGHerb)
        HNWOOD = 0.0 if (-500.0 / fuel.SGWood) < -180.218 \
                 else WWOODN * math.exp(-500.0 / fuel.SGWood)
        if (HNHERB + HNWOOD) == 0.0:
            WRAT = 0.0
        else:
            WRAT = (HN1 + HN10 + HN100) / (HNHERB + HNWOOD)
        denom = HN1 + HN10 + HN100
        MCLFE = ((mc1 * HN1) + (mc10 * HN10) + (mc100 * HN100)) / denom if denom > 0 else 0.0
        MXL = (2.9 * WRAT * (1.0 - MCLFE / fuel.MXD) - 0.226) * 100.0
    else:
        MXL = 0.0
    if MXL < fuel.MXD:
        MXL = fuel.MXD

    # -----------------------------------------------------------------
    # 5. Surface areas + weighting factors (iCalcIndexes:872-902)
    # -----------------------------------------------------------------
    SA1 = (W1P / _RHOD) * fuel.SG1
    SA10 = (W10 / _RHOD) * fuel.SG10
    SA100 = (W100 / _RHOD) * fuel.SG100
    SAHERB = (WHERBP / _RHOL) * fuel.SGHerb
    SAWOOD = (WWOOD / _RHOL) * fuel.SGWood
    SADEAD = SA1 + SA10 + SA100
    SALIVE = SAHERB + SAWOOD

    if SADEAD <= 0:
        return IndexResult(sc=float("nan"), erc=float("nan"), bi=float("nan"))

    F1 = SA1 / SADEAD
    F10 = SA10 / SADEAD
    F100 = SA100 / SADEAD
    if WTOTL <= 0:
        FHERB = 0.0
        FWOOD = 0.0
    else:
        FHERB = SAHERB / SALIVE
        FWOOD = SAWOOD / SALIVE
    total_sa = SADEAD + SALIVE
    FDEAD = SADEAD / total_sa
    FLIVE = SALIVE / total_sa
    WDEADN = F1 * W1N + F10 * W10N + F100 * W100N

    # D1: all V/W/X/Y/Z hit the first branch (SGWood=1500, SGHerb=2000).
    if fuel.SGWood > 1200 and fuel.SGHerb > 1200:
        WLIVEN = WTOTLN
    else:
        WLIVEN = FWOOD * WWOODN + FHERB * WHERBN

    # -----------------------------------------------------------------
    # 6. Reaction velocity + SC (iCalcIndexes:904-988)
    # -----------------------------------------------------------------
    SGBRD = F1 * fuel.SG1 + F10 * fuel.SG10 + F100 * fuel.SG100
    SGBRL = FHERB * fuel.SGHerb + FWOOD * fuel.SGWood
    SGBRT = FDEAD * SGBRD + FLIVE * SGBRL

    BETOP = 3.348 * math.pow(SGBRT, -0.8189)
    GMAMX = math.pow(SGBRT, 1.5) / (495.0 + 0.0594 * math.pow(SGBRT, 1.5))
    AD = 133.0 * math.pow(SGBRT, -0.7913)
    GMAOP = GMAMX * math.pow(BETBAR / BETOP, AD) * math.exp(AD * (1.0 - BETBAR / BETOP))

    ZETA = math.exp((0.792 + 0.681 * math.pow(SGBRT, 0.5)) * (BETBAR + 0.1))
    ZETA = ZETA / (192.0 + 0.2595 * SGBRT)

    WTMCD = F1 * mc1 + F10 * mc10 + F100 * mc100
    WTMCL = FHERB * mcherb + FWOOD * mcwood
    DEDRT = WTMCD / fuel.MXD
    LIVRT = WTMCL / MXL if MXL > 0 else 0.0
    ETAMD = 1.0 - 2.59 * DEDRT + 5.11 * DEDRT * DEDRT - 3.52 * DEDRT ** 3
    ETAML = 1.0 - 2.59 * LIVRT + 5.11 * LIVRT * LIVRT - 3.52 * LIVRT ** 3
    ETAMD = max(0.0, min(1.0, ETAMD))
    ETAML = max(0.0, min(1.0, ETAML))

    B = 0.02526 * math.pow(SGBRT, 0.54)
    C = 7.47 * math.exp(-0.133 * math.pow(SGBRT, 0.55))
    E = 0.715 * math.exp(-3.59e-4 * SGBRT)
    UFACT = C * math.pow(BETBAR / BETOP, -E)

    # IR uses HD for both dead and live (D2).
    IR = GMAOP * (WDEADN * fuel.HD * _ETASD * ETAMD
                  + WLIVEN * fuel.HD * _ETASL * ETAML)

    if 88.0 * wind_mph * fuel.WNDFC > 0.9 * IR:
        PHIWND = UFACT * math.pow(0.9 * IR, B)
    else:
        PHIWND = UFACT * math.pow(wind_mph * 88.0 * fuel.WNDFC, B)

    slpfct = _SLPFCT[slope_class - 1]
    PHISLP = slpfct * math.pow(BETBAR, -0.3)

    XF1 = F1 * math.exp(-138.0 / fuel.SG1) * (250.0 + 11.16 * mc1)
    XF10 = F10 * math.exp(-138.0 / fuel.SG10) * (250.0 + 11.16 * mc10)
    XF100 = F100 * math.exp(-138.0 / fuel.SG100) * (250.0 + 11.16 * mc100)
    XFHERB = FHERB * math.exp(-138.0 / fuel.SGHerb) * (250.0 + 11.16 * mcherb)
    XFWOOD = FWOOD * math.exp(-138.0 / fuel.SGWood) * (250.0 + 11.16 * mcwood)
    HTSINK = RHOBED * (FDEAD * (XF1 + XF10 + XF100) + FLIVE * (XFHERB + XFWOOD))

    SC = IR * ZETA * (1.0 + PHISLP + PHIWND) / HTSINK if HTSINK > 0 else 0.0

    # -----------------------------------------------------------------
    # 7. ERC (iCalcIndexes:990-1032)
    # -----------------------------------------------------------------
    F1E = W1P / WTOTD
    F10E = W10 / WTOTD
    F100E = W100 / WTOTD
    F1000E = W1000 / WTOTD
    if WTOTL <= 0:
        FHERBE = 0.0
        FWOODE = 0.0
    else:
        FHERBE = WHERBP / WTOTL
        FWOODE = WWOOD / WTOTL
    FDEADE = WTOTD / WTOT
    FLIVEE = WTOTL / WTOT
    WDEDNE = WTOTD * (1.0 - _STD)
    WLIVNE = WTOTL * (1.0 - _STL)
    SGBRDE = (F1E * fuel.SG1 + F10E * fuel.SG10
              + F100E * fuel.SG100 + F1000E * fuel.SG1000)
    SGBRLE = FHERBE * fuel.SGHerb + FWOODE * fuel.SGWood
    SGBRTE = FDEADE * SGBRDE + FLIVEE * SGBRLE
    BETOPE = 3.348 * math.pow(SGBRTE, -0.8189)
    GMAMXE = math.pow(SGBRTE, 1.5) / (495.0 + 0.0594 * math.pow(SGBRTE, 1.5))
    ADE = 133.0 * math.pow(SGBRTE, -0.7913)
    GMAOPE = GMAMXE * math.pow(BETBAR / BETOPE, ADE) * math.exp(ADE * (1.0 - BETBAR / BETOPE))

    WTMCDE = (F1E * mc1 + F10E * mc10
              + F100E * mc100 + F1000E * mc1000)
    WTMCLE = FHERBE * mcherb + FWOODE * mcwood
    DEDRTE = WTMCDE / fuel.MXD
    LIVRTE = WTMCLE / MXL if MXL > 0 else 0.0
    # ERC polynomial coefficients DIFFER from SC's (1 - 2x + 1.5x^2 - 0.5x^3)
    ETAMDE = 1.0 - 2.0 * DEDRTE + 1.5 * DEDRTE * DEDRTE - 0.5 * DEDRTE ** 3
    ETAMLE = 1.0 - 2.0 * LIVRTE + 1.5 * LIVRTE * LIVRTE - 0.5 * LIVRTE ** 3
    ETAMDE = max(0.0, min(1.0, ETAMDE))
    ETAMLE = max(0.0, min(1.0, ETAMLE))

    IRE = FDEADE * WDEDNE * fuel.HD * _ETASD * ETAMDE
    IRE = GMAOPE * (IRE + FLIVEE * WLIVNE * fuel.HD * _ETASL * ETAMLE)

    # TAU uses the surface-area-weighted SGBRT (not SGBRTE) — see scope §6.2.
    TAU = 384.0 / SGBRT
    ERC = 0.04 * IRE * TAU

    # -----------------------------------------------------------------
    # 8. BI (iCalcIndexes:1037) — transcribed exactly as in the C++.
    # -----------------------------------------------------------------
    BI = 0.301 * math.pow(SC * ERC, 0.46) * 10.0

    return IndexResult(sc=SC, erc=ERC, bi=BI)
