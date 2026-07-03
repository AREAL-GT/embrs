"""Regression tests for wind adjustment factor lifecycle on fuel swaps.

Cells overwritten from urban to a burnable fuel (road modelling /
``_overwrite_urban_fuel``) previously kept the non-burnable WAF of 1.0,
so those cells burned with unadjusted 20-ft wind as their midflame wind
(~2.5x effective wind on grass). ``_set_fuel_type`` must recompute the WAF
for the new fuel.
"""

import numpy as np
import pytest

from embrs.fire_simulator.cell import Cell
from embrs.models.fuel_models import ScottBurgan40
from embrs.utilities.data_classes import CellData


def _make_cell(fuel_num: int) -> Cell:
    fuel = ScottBurgan40(fuel_num)
    cd = CellData(
        fuel_type=fuel, elevation=0.0, aspect=0.0, slope_deg=0.0,
        canopy_cover=0.0, canopy_height=0.0, canopy_base_height=0.0,
        canopy_bulk_density=0.0, init_dead_mf=[0.08] * 3,
        live_h_mf=0.3, live_w_mf=0.6,
    )
    cell = Cell(0, 15.0, 15.0, 30)
    cell._set_cell_data(cd)
    return cell


class TestWafOnFuelSwap:

    def test_urban_to_grass_recomputes_waf(self):
        """The ex-urban corridor case: urban (WAF 1.0) -> grass."""
        cell = _make_cell(91)          # urban, non-burnable
        assert cell.wind_adj_factor == 1

        cell._set_fuel_type(ScottBurgan40(105))   # overwrite to grass

        expected = _make_cell(105).wind_adj_factor  # Albini unsheltered value
        assert cell.wind_adj_factor == pytest.approx(expected)
        assert cell.wind_adj_factor < 0.5   # grass WAF is far from 1.0

    def test_grass_to_urban_sets_waf_one(self):
        """Fireline urban conversion keeps the non-burnable WAF of 1."""
        cell = _make_cell(105)
        assert cell.wind_adj_factor < 0.5

        cell._set_fuel_type(ScottBurgan40(91))

        assert cell.wind_adj_factor == 1

    def test_grass_to_grass_tracks_fuel_depth(self):
        """Swapping between burnable fuels recomputes from the new depth."""
        cell = _make_cell(105)
        waf_105 = cell.wind_adj_factor

        cell._set_fuel_type(ScottBurgan40(101))   # short grass, shallower bed

        assert cell.wind_adj_factor != pytest.approx(waf_105)
        assert cell.wind_adj_factor == pytest.approx(
            _make_cell(101).wind_adj_factor)
