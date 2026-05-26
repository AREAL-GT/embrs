"""Centroid + elevation + IANA timezone extraction from a LANDFIRE ``.tif``.

Reuses :class:`embrs.utilities.data_classes.GeoInfo` for centroid /
timezone resolution (plan §4.3 — same machinery the BI pipeline uses).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyproj
import rasterio
import rasterio.coords

from embrs.fire_danger.crosswalk import NON_BURNABLE
from embrs.utilities.data_classes import GeoInfo
from embrs.utilities.unit_conversions import m_to_ft

logger = logging.getLogger(__name__)

NODATA_SENTINEL: int = -9999


@dataclass
class LandscapeGeo:
    """Result of :func:`load_landscape_geo`.

    ``elevation_ft`` is the mean elevation over burnable pixels, in feet.
    ``NaN`` if the raster does not have a usable elevation band — callers
    can fall back to the Open-Meteo ``Elevation()`` value.
    """

    geo: GeoInfo
    crs: "pyproj.CRS"
    bounds: rasterio.coords.BoundingBox
    elevation_ft: float


def _mean_elevation_ft(
    elevation_band: np.ndarray, fuel_band: np.ndarray, burnable_only: bool
) -> float:
    """Mean elevation over the (optionally burnable-only) mask, in feet.

    LANDFIRE LCP band 1 is elevation in **metres** for FBFM40 CONUS tiles.
    """
    valid = elevation_band != NODATA_SENTINEL
    if burnable_only:
        valid = (
            valid
            & (fuel_band != NODATA_SENTINEL)
            & ~np.isin(fuel_band, list(NON_BURNABLE))
        )
    if not valid.any():
        # Fall back to whole-raster non-NoData mean.
        valid = elevation_band != NODATA_SENTINEL
        if not valid.any():
            return float("nan")
        logger.warning(
            "No burnable pixels with valid elevation; falling back to whole-raster mean."
        )
    mean_m = float(np.mean(elevation_band[valid].astype(float)))
    return float(m_to_ft(mean_m))


def load_landscape_geo(
    landscape_tif: str, burnable_only: bool = True
) -> LandscapeGeo:
    """Open a LANDFIRE raster and return centroid/tz/elevation.

    Args:
        landscape_tif: Path to a LANDFIRE ``.tif`` / ``.lcp``.
        burnable_only: If True, average elevation only over burnable pixels
            (excluding non-burnable codes ``{91, 92, 93, 98, 99}`` and
            ``-9999`` NoData). Falls back to whole-raster mean if the
            burnable mask is empty.

    Returns:
        :class:`LandscapeGeo` with centroid coords, tz, and elevation.

    Raises:
        ValueError: If the raster has < 1 band or no readable elevation.
    """
    with rasterio.open(landscape_tif) as src:
        crs = src.crs
        bounds = src.bounds
        band_count = src.count
        elevation_band: Optional[np.ndarray] = src.read(1) if band_count >= 1 else None
        fuel_band: Optional[np.ndarray] = src.read(4) if band_count >= 4 else None

    if elevation_band is None:
        raise ValueError(
            f"{landscape_tif}: raster has no readable bands (count={band_count})"
        )

    if fuel_band is None and burnable_only:
        logger.warning(
            "%s: no fuel band (need band 4); computing elevation over whole raster.",
            landscape_tif,
        )
        burnable_only = False

    elevation_ft = _mean_elevation_ft(
        elevation_band,
        fuel_band if fuel_band is not None else np.full_like(elevation_band, 0),
        burnable_only=burnable_only,
    )

    geo = GeoInfo(bounds=bounds)
    geo.calc_center_coords(crs)
    geo.calc_time_zone()

    return LandscapeGeo(
        geo=geo,
        crs=crs,
        bounds=bounds,
        elevation_ft=elevation_ft,
    )
