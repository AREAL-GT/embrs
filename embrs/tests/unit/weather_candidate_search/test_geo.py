"""Tests for landscape geo extraction."""
from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from embrs.weather_candidate_search.geo import load_landscape_geo, NODATA_SENTINEL


def _write_synthetic_lcp(
    tmp_path,
    h=10,
    w=10,
    fuel_code=181,
    elevation_m=1200,
    n_bands=4,
):
    """Synthetic LANDFIRE-style raster in EPSG:5070 (CONUS Albers)."""
    bands = []
    elev = np.full((h, w), elevation_m, dtype=np.int16)
    slope = np.full((h, w), 10, dtype=np.int16)
    fuel = np.full((h, w), fuel_code, dtype=np.int16)
    for b in range(1, n_bands + 1):
        if b == 1:
            bands.append(elev)
        elif b == 2:
            bands.append(slope)
        elif b == 4:
            bands.append(fuel)
        else:
            bands.append(np.zeros((h, w), dtype=np.int16))
    # 30 m pixels, EPSG:5070
    transform = from_origin(west=-1_000_000, north=2_000_000, xsize=30.0, ysize=30.0)
    path = str(tmp_path / "lcp.tif")
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=n_bands, dtype="int16",
        crs="EPSG:5070", transform=transform,
    ) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr, i)
    return path


def test_load_landscape_geo_returns_plausible_values(tmp_path):
    path = _write_synthetic_lcp(tmp_path, elevation_m=1500, fuel_code=181)
    lgeo = load_landscape_geo(path)
    assert lgeo.geo.center_lat is not None
    assert lgeo.geo.center_lon is not None
    assert lgeo.geo.timezone is not None and lgeo.geo.timezone.startswith("America/")
    # 1500 m → ~4921 ft
    assert 4900 < lgeo.elevation_ft < 4950


def test_load_landscape_geo_burnable_only_excludes_non_burnable(tmp_path):
    # Half the raster is non-burnable code 91 with a very different elevation.
    h = 10
    w = 10
    elev = np.full((h, w), 1500, dtype=np.int16)
    elev[:, w // 2 :] = 100   # non-burnable half is much lower
    slope = np.full((h, w), 10, dtype=np.int16)
    fuel = np.full((h, w), 181, dtype=np.int16)
    fuel[:, w // 2 :] = 91   # urban — non-burnable
    bands = [elev, slope, np.zeros((h, w), dtype=np.int16), fuel]
    transform = from_origin(west=-1_000_000, north=2_000_000, xsize=30.0, ysize=30.0)
    path = str(tmp_path / "mixed.tif")
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=4, dtype="int16",
        crs="EPSG:5070", transform=transform,
    ) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr, i)
    lgeo = load_landscape_geo(path, burnable_only=True)
    # Burnable-only mean → 1500 m → ~4921 ft (not the mixed 800 m mean)
    assert 4900 < lgeo.elevation_ft < 4950


def test_load_landscape_geo_handles_nodata(tmp_path):
    h = 10
    w = 10
    elev = np.full((h, w), 1500, dtype=np.int16)
    elev[0, 0] = NODATA_SENTINEL
    slope = np.full((h, w), 10, dtype=np.int16)
    fuel = np.full((h, w), 181, dtype=np.int16)
    bands = [elev, slope, np.zeros((h, w), dtype=np.int16), fuel]
    transform = from_origin(west=-1_000_000, north=2_000_000, xsize=30.0, ysize=30.0)
    path = str(tmp_path / "nodata.tif")
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=4, dtype="int16",
        crs="EPSG:5070", transform=transform,
    ) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr, i)
    lgeo = load_landscape_geo(path)
    # NoData pixel excluded; mean is still ~1500 m
    assert 4900 < lgeo.elevation_ft < 4950
