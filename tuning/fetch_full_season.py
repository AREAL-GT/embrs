"""Fetch a region's real-season hourly weather and write a ``full_season.wxs``.

This is the minimal "backdrop source" step for the scenario_weather workflow.
It reproduces steps 1-5 of the weather_candidate_search pipeline (resolve the
landscape centroid -> Open-Meteo history pull -> wind correction -> write .wxs)
and **skips the BI candidate search**, which the controlled-scenario approach no
longer uses. Real file + __main__ guard for spawn safety.

Usage:
  python -m tuning.fetch_full_season \
      --landscape-tif /path/to/cropped_lcp.tif \
      --year 2022 --season-start-month 5 --season-end-month 10 \
      --out embrs/weather_candidate_search/search_outputs/region_c_clearwater/full_season.wxs
"""
import argparse
import calendar
import datetime as dt
import os

from embrs.weather_candidate_search.config import WindConversionConfig
from embrs.weather_candidate_search.geo import load_landscape_geo
from embrs.weather_candidate_search.openmeteo_client import (
    OpenMeteoPullSpec,
    fetch_history,
)
from embrs.weather_candidate_search.wxs_writer import (
    WxsWriteSpec,
    correct_wind_speed_10m_to_20ft,
    write_wxs,
)

_MPS_TO_MPH = 2.23693629


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landscape-tif", required=True,
                    help="region LANDFIRE .tif / cropped_lcp.tif (for centroid/elev/tz)")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--season-start-month", type=int, required=True)
    ap.add_argument("--season-end-month", type=int, required=True)
    ap.add_argument("--out", required=True, help="output full_season.wxs path")
    ap.add_argument("--cache-dir", default="./.openmeteo_cache/")
    ap.add_argument("--surface-roughness-m", type=float, default=0.06,
                    help="10m->20ft log-profile roughness (immaterial for "
                         "scenario_weather, which overwrites wind)")
    a = ap.parse_args()

    geo = load_landscape_geo(a.landscape_tif)
    elev_ft = geo.elevation_ft if geo.elevation_ft == geo.elevation_ft else 0.0  # NaN guard
    print(f"Centroid: lat={geo.geo.center_lat:.4f} lon={geo.geo.center_lon:.4f} "
          f"tz={geo.geo.timezone} elev={elev_ft:.0f}ft")

    start = dt.date(a.year, a.season_start_month, 1)
    end = dt.date(a.year, a.season_end_month,
                  calendar.monthrange(a.year, a.season_end_month)[1])
    print(f"Pull span: {start} .. {end}")

    om = fetch_history(
        OpenMeteoPullSpec(lat=float(geo.geo.center_lat), lon=float(geo.geo.center_lon),
                          start_date=start, end_date=end, timezone="auto"),
        cache_dir=a.cache_dir,
    )
    print(f"Open-Meteo: {len(om.df)} hours (source={om.source}, NaN={om.nan_hour_count})")

    if elev_ft == 0.0 and om.elevation_m == om.elevation_m:
        from embrs.utilities.unit_conversions import m_to_ft
        elev_ft = float(m_to_ft(om.elevation_m))
        print(f"Using Open-Meteo elevation: {elev_ft:.0f}ft")

    df = om.df.copy()
    df["wind_mph"] = correct_wind_speed_10m_to_20ft(
        df["wind_mps"].to_numpy(dtype=float), a.surface_roughness_m) * _MPS_TO_MPH

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    write_wxs(
        WxsWriteSpec(df=df, elevation_ft=int(round(elev_ft)),
                     wind_correction=WindConversionConfig(enabled=False,
                                                          surface_roughness_m=a.surface_roughness_m),
                     wind_mph_precomputed="wind_mph"),
        a.out,
    )
    print(f"Wrote {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
