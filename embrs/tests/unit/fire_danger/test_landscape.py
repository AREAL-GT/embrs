"""Tests for embrs.fire_danger.landscape."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from embrs.fire_danger.landscape import (
    NODATA_SENTINEL,
    Landscape,
    _slope_pct_to_class,
    compute_fuel_composition,
    derive_slope_class,
    load_landscape,
    resolve_geo,
)


# ---------------------------------------------------------------------------
# Helpers: build a synthetic 4-band LCP in EPSG:5070 30m on disk.
# ---------------------------------------------------------------------------


def _write_synthetic_lcp(
    tmp_path,
    fuel_grid: np.ndarray,
    slope_grid: np.ndarray | None = None,
    n_bands: int = 4,
) -> str:
    if slope_grid is None:
        slope_grid = np.full_like(fuel_grid, 5, dtype=np.int16)
    h, w = fuel_grid.shape
    bands = []
    for b in range(1, n_bands + 1):
        if b == 2:
            bands.append(slope_grid.astype(np.int16))
        elif b == 4:
            bands.append(fuel_grid.astype(np.int16))
        else:
            bands.append(np.full((h, w), 0, dtype=np.int16))
    transform = from_origin(west=-1_000_000, north=2_000_000, xsize=30.0, ysize=30.0)
    path = str(tmp_path / "synthetic.tif")
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=n_bands, dtype="int16", crs="EPSG:5070", transform=transform,
    ) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr, i)
    return path


# ---------------------------------------------------------------------------
# load_landscape
# ---------------------------------------------------------------------------


def test_load_landscape_basic(tmp_path):
    fuel = np.full((10, 10), 101, dtype=np.int16)
    slope = np.full((10, 10), 5, dtype=np.int16)
    path = _write_synthetic_lcp(tmp_path, fuel, slope)
    L = load_landscape(path)
    assert L.fuel.shape == (10, 10)
    assert L.slope.shape == (10, 10)
    assert L.pixel_area_m2 == pytest.approx(900.0)
    assert L.fbfm_type == "ScottBurgan"


def test_load_landscape_anderson_autodetect(tmp_path):
    fuel = np.full((10, 10), 8, dtype=np.int16)  # all Anderson-13 codes
    path = _write_synthetic_lcp(tmp_path, fuel)
    assert load_landscape(path).fbfm_type == "Anderson"


def test_load_landscape_handles_internal_nodata(tmp_path):
    fuel = np.full((10, 10), 101, dtype=np.int16)
    fuel[0, :] = NODATA_SENTINEL  # edge NoData
    path = _write_synthetic_lcp(tmp_path, fuel)
    # Should NOT raise — the BI tool excludes NoData by pixel, not crop.
    L = load_landscape(path)
    assert (L.fuel == NODATA_SENTINEL).sum() == 10


def test_load_landscape_too_few_bands(tmp_path):
    fuel = np.full((4, 4), 101, dtype=np.int16)
    path = _write_synthetic_lcp(tmp_path, fuel, n_bands=2)
    with pytest.raises(ValueError, match="≥ 4 bands"):
        load_landscape(path)


def test_load_landscape_all_nodata_fuel(tmp_path):
    fuel = np.full((4, 4), NODATA_SENTINEL, dtype=np.int16)
    path = _write_synthetic_lcp(tmp_path, fuel)
    with pytest.raises(ValueError, match="entirely NoData"):
        load_landscape(path)


# ---------------------------------------------------------------------------
# compute_fuel_composition
# ---------------------------------------------------------------------------


def _landscape_from_fuel(fuel: np.ndarray) -> Landscape:
    return Landscape(
        fuel=fuel, slope=np.full_like(fuel, 5, dtype=np.int16),
        crs=rasterio.crs.CRS.from_epsg(5070),
        bounds=rasterio.coords.BoundingBox(0, 0, 100, 100),
        pixel_area_m2=900.0,
        fbfm_type="ScottBurgan",
    )


def test_fuel_composition_pure_grass():
    L = _landscape_from_fuel(np.full((10, 10), 101, dtype=np.int16))
    comp = compute_fuel_composition(L, min_area_frac=0.0)
    assert comp.fractions == {"V": pytest.approx(1.0)}
    assert comp.n_burnable_pixels == 100
    assert comp.n_total_pixels == 100


def test_fuel_composition_excludes_nodata_and_nonburnable():
    fuel = np.full((10, 10), 101, dtype=np.int16)
    fuel[0:2, :] = NODATA_SENTINEL  # 20 NoData
    fuel[2:4, :] = 98               # 20 water (non-burnable)
    L = _landscape_from_fuel(fuel)
    comp = compute_fuel_composition(L, min_area_frac=0.0)
    assert comp.n_total_pixels == 100
    assert comp.n_burnable_pixels == 60   # only grass survives
    assert comp.fractions == {"V": pytest.approx(1.0)}


def test_fuel_composition_renormalizes_after_min_area():
    """A grid with 90% V and 10% W (boundary case at min_area_frac=0.05
    keeps both; at 0.15 drops W)."""
    fuel = np.full((100,), 101, dtype=np.int16)  # 100 V
    fuel[:10] = 121  # 10 W (GS1)
    L = _landscape_from_fuel(fuel.reshape(10, 10))
    # Both retained
    comp = compute_fuel_composition(L, min_area_frac=0.05)
    assert set(comp.fractions) == {"V", "W"}
    assert sum(comp.fractions.values()) == pytest.approx(1.0)
    # W dropped
    comp = compute_fuel_composition(L, min_area_frac=0.15)
    assert set(comp.fractions) == {"V"}
    assert comp.fractions["V"] == pytest.approx(1.0)


def test_fuel_composition_no_burnable_raises():
    fuel = np.full((4, 4), 98, dtype=np.int16)  # all water
    L = _landscape_from_fuel(fuel)
    with pytest.raises(ValueError, match="No burnable pixels"):
        compute_fuel_composition(L)


def test_fuel_composition_all_below_threshold_raises():
    """Five NFDRS families each at 20% — all dropped when threshold > 0.2."""
    # 20 px each of V (101), W (121), X (141), Y (181), Z (201)
    codes = [101, 121, 141, 181, 201]
    fuel = np.array(sum([[c] * 20 for c in codes], []), dtype=np.int16).reshape(10, 10)
    L = _landscape_from_fuel(fuel)
    # All five make up exactly 20%; threshold 0.5 drops them all.
    with pytest.raises(ValueError, match="below min_area_frac"):
        compute_fuel_composition(L, min_area_frac=0.5)


# ---------------------------------------------------------------------------
# derive_slope_class
# ---------------------------------------------------------------------------


def test_slope_pct_to_class_breakpoints():
    assert _slope_pct_to_class(0) == 1
    assert _slope_pct_to_class(25) == 1
    assert _slope_pct_to_class(26) == 2
    assert _slope_pct_to_class(40) == 2
    assert _slope_pct_to_class(41) == 3
    assert _slope_pct_to_class(55) == 3
    assert _slope_pct_to_class(56) == 4
    assert _slope_pct_to_class(75) == 4
    assert _slope_pct_to_class(76) == 5
    assert _slope_pct_to_class(200) == 5


def test_derive_slope_class_uniform_10deg():
    # 10 degrees = tan(10°)*100 ≈ 17.6% -> class 1
    fuel = np.full((10, 10), 101, dtype=np.int16)
    slope = np.full((10, 10), 10, dtype=np.int16)
    L = Landscape(fuel=fuel, slope=slope,
                  crs=rasterio.crs.CRS.from_epsg(5070),
                  bounds=rasterio.coords.BoundingBox(0, 0, 100, 100),
                  pixel_area_m2=900.0, fbfm_type="ScottBurgan")
    assert derive_slope_class(L) == 1


def test_derive_slope_class_uniform_20deg():
    # 20° = tan(20°)*100 ≈ 36.4% -> class 2
    fuel = np.full((10, 10), 101, dtype=np.int16)
    slope = np.full((10, 10), 20, dtype=np.int16)
    L = Landscape(fuel=fuel, slope=slope,
                  crs=rasterio.crs.CRS.from_epsg(5070),
                  bounds=rasterio.coords.BoundingBox(0, 0, 100, 100),
                  pixel_area_m2=900.0, fbfm_type="ScottBurgan")
    assert derive_slope_class(L) == 2


def test_derive_slope_class_uniform_45deg():
    # 45° = 100% -> class 5
    fuel = np.full((10, 10), 101, dtype=np.int16)
    slope = np.full((10, 10), 45, dtype=np.int16)
    L = Landscape(fuel=fuel, slope=slope,
                  crs=rasterio.crs.CRS.from_epsg(5070),
                  bounds=rasterio.coords.BoundingBox(0, 0, 100, 100),
                  pixel_area_m2=900.0, fbfm_type="ScottBurgan")
    assert derive_slope_class(L) == 5


# ---------------------------------------------------------------------------
# resolve_geo
# ---------------------------------------------------------------------------


def test_resolve_geo_on_real_sample():
    path = "/Users/rjdp3/Documents/Research/embrs_map/scenario_4/cropped_lcp.tif"
    if not os.path.exists(path):
        pytest.skip("sample LANDFIRE tile not present")
    L = load_landscape(path)
    geo = resolve_geo(L)
    # CONUS-ish lat/lon
    assert -130 < geo.center_lon < -65
    assert 25 < geo.center_lat < 50
    assert geo.timezone is not None and "/" in geo.timezone
