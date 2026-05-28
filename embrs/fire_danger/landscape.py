"""Component 6 — read a LANDFIRE raster and derive fuel composition,
slope class, and geo info.

The BI tool does **no clipping of its own** (per the revised design — the
user passes a ``.tif`` already at whatever extent they want). NoData
(``-9999``) is handled by **pixel exclusion**, not crop rejection (plan
M0-7).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pyproj
import rasterio
import rasterio.coords

from embrs.fire_danger.config import FuelComposition
from embrs.fire_danger.crosswalk import (
    ANDERSON13_TO_NFDRS,
    NON_BURNABLE,
    SB40_TO_NFDRS,
    crosswalk_code,
)
from embrs.utilities.data_classes import GeoInfo


NODATA_SENTINEL: int = -9999


@dataclass
class Landscape:
    """A LANDFIRE raster after read.

    ``fuel`` is band 4 (1-indexed). ``slope`` is band 2, in **degrees**
    (LANDFIRE LCP convention — M0-8, user-confirmed). NoData remains the
    raw ``-9999`` sentinel; downstream consumers filter it explicitly.
    """

    fuel: np.ndarray
    slope: np.ndarray
    crs: "pyproj.CRS"
    bounds: rasterio.coords.BoundingBox
    pixel_area_m2: float
    fbfm_type: str          # 'ScottBurgan' | 'Anderson'


def load_landscape(raster_path: str) -> Landscape:
    """Open a LANDFIRE ``.tif`` / ``.lcp`` and return the bands we need.

    Reads band 4 (fuel) and band 2 (slope). Pixel area is computed from the
    raster's affine transform so non-EPSG:5070 inputs work without
    hard-coding 900 m². FBFM type is autodetected via the rule
    ``np.any(fuel >= 101)`` (matches ``embrs.map_generator``).

    Raises:
        ValueError: If the raster has < 4 bands or contains only NoData in
            the fuel band.
    """
    with rasterio.open(raster_path) as src:
        if src.count < 4:
            raise ValueError(
                f"{raster_path}: expected ≥ 4 bands (LANDFIRE LCP order), "
                f"got {src.count}"
            )
        fuel = src.read(4)
        slope = src.read(2)
        crs = src.crs
        bounds = src.bounds
        # Geographic CRSes give |a*e| in degrees^2 — meaningless. The BI
        # tool is built for projected metre rasters (EPSG:5070 in CONUS);
        # warn but still proceed for compatibility.
        pixel_area_m2 = abs(src.transform.a * src.transform.e)

    valid_fuel = fuel[fuel != NODATA_SENTINEL]
    if valid_fuel.size == 0:
        raise ValueError(f"{raster_path}: fuel band is entirely NoData")

    fbfm_type = "ScottBurgan" if bool(np.any(valid_fuel >= 101)) else "Anderson"

    return Landscape(
        fuel=fuel, slope=slope,
        crs=crs, bounds=bounds,
        pixel_area_m2=pixel_area_m2, fbfm_type=fbfm_type,
    )


def compute_fuel_composition(
    landscape: Landscape,
    min_area_frac: float = 0.05,
) -> FuelComposition:
    """Crosswalk pixels and compute area fractions over burnable area.

    NoData (``-9999``) and non-burnable codes ``{91, 92, 93, 98, 99}`` are
    dropped before computing the denominator (OQ-9). Models below
    ``min_area_frac`` are dropped after crosswalk and the remaining
    fractions are renormalised to sum to 1.0 (OQ-8).

    Raises:
        ValueError: If no burnable pixels survive the filter.
    """
    flat = landscape.fuel.ravel()
    burnable_mask = (flat != NODATA_SENTINEL) & ~np.isin(flat, list(NON_BURNABLE))
    burnable_codes = flat[burnable_mask]
    n_burnable = int(burnable_codes.size)
    if n_burnable == 0:
        raise ValueError(
            "No burnable pixels in landscape (all NoData or non-burnable codes)."
        )

    table = SB40_TO_NFDRS if landscape.fbfm_type == "ScottBurgan" else ANDERSON13_TO_NFDRS
    nfdrs_counts: Counter[str] = Counter()
    for code in burnable_codes.tolist():
        nfdrs = table.get(int(code))
        if nfdrs is None:
            # Burnable but unmapped — skip (shouldn't happen for in-table codes).
            continue
        nfdrs_counts[nfdrs] += 1

    total_mapped = sum(nfdrs_counts.values())
    if total_mapped == 0:
        raise ValueError(
            f"No burnable pixels could be crosswalked from "
            f"fbfm_type={landscape.fbfm_type!r}."
        )

    # Initial fractions over the mapped-burnable total.
    fractions = {m: c / total_mapped for m, c in nfdrs_counts.items()}

    # Apply min-area threshold after crosswalk, then renormalize.
    fractions = {m: f for m, f in fractions.items() if f >= min_area_frac}
    if not fractions:
        raise ValueError(
            f"All NFDRS fractions fell below min_area_frac={min_area_frac}; "
            f"no fuel models retained."
        )
    s = sum(fractions.values())
    fractions = {m: f / s for m, f in fractions.items()}

    # Defer slope_class — caller computes via derive_slope_class.
    return FuelComposition(
        fractions=fractions,
        slope_class=-1,            # filled by orchestrator
        fbfm_type=landscape.fbfm_type,
        n_burnable_pixels=n_burnable,
        n_total_pixels=int(flat.size),
        pixel_area_m2=landscape.pixel_area_m2,
    )


def _slope_pct_to_class(slope_pct: float) -> int:
    """Map slope percent to the NFDRS class 1-5 (PSW-82 breakpoints)."""
    if slope_pct <= 25:
        return 1
    if slope_pct <= 40:
        return 2
    if slope_pct <= 55:
        return 3
    if slope_pct <= 75:
        return 4
    return 5


def derive_slope_class(landscape: Landscape) -> int:
    """Area-weighted mean slope (degrees) → percent → NFDRS class 1-5.

    LANDFIRE LCP band 2 is in **degrees** (closes OQ-16). The mean slope is
    converted to percent via ``tan(d_rad) * 100`` and mapped to the PSW-82
    breakpoints: 1 (0-25%), 2 (26-40%), 3 (41-55%), 4 (56-75%), 5 (>75%).
    """
    slope = landscape.slope
    fuel = landscape.fuel
    mask = (slope != NODATA_SENTINEL) & (fuel != NODATA_SENTINEL) & \
           ~np.isin(fuel, list(NON_BURNABLE))
    valid = slope[mask]
    if valid.size == 0:
        # No burnable cells with valid slope — fall back to flat (class 1).
        return 1
    mean_deg = float(np.mean(valid.astype(float)))
    mean_pct = math.tan(math.radians(mean_deg)) * 100.0
    return _slope_pct_to_class(mean_pct)


def resolve_geo(landscape: Landscape) -> GeoInfo:
    """Build a :class:`GeoInfo` with centroid lat/lon and IANA timezone.

    Reuses :meth:`GeoInfo.calc_center_coords` to reproject the raster's
    bounds centroid to EPSG:4326 and :meth:`GeoInfo.calc_time_zone` for the
    timezone (closes OQ-4).
    """
    geo = GeoInfo(bounds=landscape.bounds)
    geo.calc_center_coords(landscape.crs)
    geo.calc_time_zone()
    return geo
