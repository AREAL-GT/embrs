"""Component 6 — SB40 / Anderson-13 → NFDRS V/W/X/Y/Z crosswalk.

Approved tables per scope §9. Non-burnable codes ``{91, 92, 93, 98, 99}``
(urban, snow/ice, agriculture, water, barren) are excluded entirely from the
area-weighting denominator (OQ-9 RESOLVED).

The TU family (timber-understory, 161-165) maps to **Y (Timber)** per
OQ-5 (RESOLVED) — chosen to retain the timber 1000-hr fuel load so ERC keeps
its drought signal.
"""
from __future__ import annotations

from typing import Optional


NON_BURNABLE: frozenset[int] = frozenset({91, 92, 93, 98, 99})


# SB40 codes -> NFDRS char (scope §9 table).
SB40_TO_NFDRS: dict[int, str] = {
    # GR (grass) 101-109 -> V
    **{c: "V" for c in range(101, 110)},
    # GS (grass-shrub) 121-124 -> W
    **{c: "W" for c in range(121, 125)},
    # SH (shrub) 141-149 -> X
    **{c: "X" for c in range(141, 150)},
    # TU (timber-understory) 161-165 -> Y (OQ-5)
    **{c: "Y" for c in range(161, 166)},
    # TL (timber-litter) 181-189 -> Y
    **{c: "Y" for c in range(181, 190)},
    # SB (slash-blowdown) 201-204 -> Z
    **{c: "Z" for c in range(201, 205)},
}


# Anderson-13 codes -> NFDRS char (approximate fallback, scope §9).
ANDERSON13_TO_NFDRS: dict[int, str] = {
    1: "V", 2: "V", 3: "V",
    4: "X", 5: "X", 6: "X", 7: "X",
    8: "Y", 9: "Y", 10: "Y",
    11: "Z", 12: "Z", 13: "Z",
}


def crosswalk_code(code: int, fbfm_type: str) -> Optional[str]:
    """Map a single LANDFIRE fuel code to an NFDRS model character.

    Returns:
        ``'V'`` ... ``'Z'`` for burnable codes; ``None`` for non-burnable
        codes, NoData (``-9999``), and any code not in the active table.
    """
    if code in NON_BURNABLE or code == -9999:
        return None
    table = SB40_TO_NFDRS if fbfm_type == "ScottBurgan" else ANDERSON13_TO_NFDRS
    return table.get(int(code))
