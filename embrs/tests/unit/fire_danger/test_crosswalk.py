"""Tests for embrs.fire_danger.crosswalk."""
from __future__ import annotations

import pytest

from embrs.fire_danger.crosswalk import (
    ANDERSON13_TO_NFDRS,
    NON_BURNABLE,
    SB40_TO_NFDRS,
    crosswalk_code,
)


def test_sb40_families_map_as_documented():
    # GR -> V
    assert all(SB40_TO_NFDRS[c] == "V" for c in range(101, 110))
    # GS -> W
    assert all(SB40_TO_NFDRS[c] == "W" for c in range(121, 125))
    # SH -> X
    assert all(SB40_TO_NFDRS[c] == "X" for c in range(141, 150))
    # TU -> Y (OQ-5 RESOLVED)
    assert all(SB40_TO_NFDRS[c] == "Y" for c in range(161, 166))
    # TL -> Y
    assert all(SB40_TO_NFDRS[c] == "Y" for c in range(181, 190))
    # SB -> Z
    assert all(SB40_TO_NFDRS[c] == "Z" for c in range(201, 205))


def test_anderson13_families_map_as_documented():
    for c in (1, 2, 3):
        assert ANDERSON13_TO_NFDRS[c] == "V"
    for c in (4, 5, 6, 7):
        assert ANDERSON13_TO_NFDRS[c] == "X"
    for c in (8, 9, 10):
        assert ANDERSON13_TO_NFDRS[c] == "Y"
    for c in (11, 12, 13):
        assert ANDERSON13_TO_NFDRS[c] == "Z"


def test_non_burnable_returns_none():
    for c in (91, 92, 93, 98, 99):
        assert crosswalk_code(c, "ScottBurgan") is None
        assert crosswalk_code(c, "Anderson") is None


def test_nodata_returns_none():
    assert crosswalk_code(-9999, "ScottBurgan") is None


def test_unmapped_code_returns_none():
    assert crosswalk_code(500, "ScottBurgan") is None    # not in any table
    assert crosswalk_code(99, "Anderson") is None        # non-burnable


def test_crosswalk_dispatches_on_fbfm_type():
    # Code 5 means different things in SB40 (not in table) vs A13 (X)
    assert crosswalk_code(5, "Anderson") == "X"
    assert crosswalk_code(5, "ScottBurgan") is None
