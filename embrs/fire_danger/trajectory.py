"""Component 7 — pipeline orchestration + area weighting."""
from __future__ import annotations

from datetime import datetime, time
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from embrs.fire_danger.config import (
    Config,
    DEFAULT_AVG_ANN_PRECIP_IN,
    FuelComposition,
    TrajectoryResult,
)
from embrs.fire_danger.dead_moisture import compute_dead_moisture
from embrs.fire_danger.kbdi import (
    compute_kbdi_series,
    resolve_avg_ann_precip_in,
)
from embrs.fire_danger.landscape import (
    compute_fuel_composition,
    derive_slope_class,
    load_landscape,
    resolve_geo,
)
from embrs.fire_danger.live_moisture import compute_live_moisture
from embrs.fire_danger.nfdrs_fuel_models import NFDRS_FUEL_MODELS
from embrs.fire_danger.nfdrs_index import calc_indexes
from embrs.fire_danger.solar import synthesize_solar
from embrs.fire_danger.weather_loader import load_wxs


# Forward-fill seeds for the pre-first-daily-value hours (per plan §2.8).
_GSI_SEED = float("nan")          # cure() treats NaN as dormant -> fct=1
_MCHERB_SEED = 30.0               # dormant herbaceous (live_moisture._gsi_to_live_moisture_pct(-1))
_MCWOOD_SEED = 60.0
_KBDI_SEED = 100.0


def _maybe_localize_scenario_start(scenario_start: datetime, tz_name: str) -> pd.Timestamp:
    """Ensure ``scenario_start`` is comparable against a tz-aware index."""
    ts = pd.Timestamp(scenario_start)
    if ts.tz is None:
        ts = ts.tz_localize(pytz.timezone(tz_name))
    return ts


def _attach_phase_and_snow(
    weather_df: pd.DataFrame, scenario_start: datetime, snow_mode: str, tz_name: str
) -> None:
    """Add ``phase`` and ``snow`` columns to ``weather_df`` in place."""
    ts = _maybe_localize_scenario_start(scenario_start, tz_name)
    weather_df["phase"] = np.where(
        weather_df.index < ts, "conditioning", "scenario"
    )
    if snow_mode == "none":
        weather_df["snow"] = False
    elif snow_mode == "temp-derived":
        # Simple heuristic: snow when temp_C below 0 AND precip > 0.
        weather_df["snow"] = (weather_df["temp_C"] <= 0.0) & (weather_df["precip_cm_hr"] > 0.0)
    else:
        raise ValueError(f"snow_mode must be 'none' or 'temp-derived', got {snow_mode!r}")


def _forward_fill_daily_to_hourly(
    daily_df: pd.DataFrame, hourly_index: pd.DatetimeIndex, seeds: dict[str, float]
) -> pd.DataFrame:
    """Forward-fill a daily series onto an hourly index.

    For each hour in ``hourly_index``, use the most recent daily value with
    ``daily_date <= hour``. Hours before any daily value use ``seeds``.
    """
    if daily_df.empty:
        return pd.DataFrame({c: np.full(len(hourly_index), v) for c, v in seeds.items()},
                            index=hourly_index)
    # asof requires sorted indices on both sides.
    daily_sorted = daily_df.sort_index()
    hourly_sorted = hourly_index.sort_values()
    # Per-column asof merge.
    out = {}
    daily_ts = daily_sorted.index
    if hourly_sorted.tz is not None and daily_ts.tz is None:
        daily_sorted = daily_sorted.copy()
        daily_sorted.index = daily_ts.tz_localize(hourly_sorted.tz)
    elif hourly_sorted.tz is None and daily_ts.tz is not None:
        daily_sorted = daily_sorted.copy()
        daily_sorted.index = daily_ts.tz_convert(None)
    for col, seed in seeds.items():
        merged = pd.merge_asof(
            pd.DataFrame(index=hourly_sorted),
            daily_sorted[[col]].reset_index().rename(
                columns={daily_sorted.index.name or "index": "_t"}
            ),
            left_index=True, right_on="_t", direction="backward",
        )
        merged.index = hourly_sorted
        out[col] = merged[col].fillna(seed).to_numpy()
    return pd.DataFrame(out, index=hourly_sorted).reindex(hourly_index)


def compute_bi_trajectory(cfg: Config) -> TrajectoryResult:
    """End-to-end BI trajectory pipeline.

    See plan §2.8 for the step-by-step ordering.
    """
    # 1. Landscape (independent of weather; M3 parallelizable)
    landscape = load_landscape(cfg.landscape_path)
    composition = compute_fuel_composition(landscape, cfg.min_area_frac)
    geo = resolve_geo(landscape)
    if cfg.lat_override is not None:
        geo.center_lat = float(cfg.lat_override)
    slope_class = cfg.slope_class if cfg.slope_class is not None \
                  else derive_slope_class(landscape)
    composition.slope_class = slope_class

    # 2. Weather load (tz-naive)
    weather = load_wxs(cfg.wxs_path)

    # Validate scenario_start is inside the weather span.
    naive_start = weather.raw_start
    naive_end = weather.raw_end
    if cfg.scenario_start < naive_start or cfg.scenario_start > naive_end:
        raise ValueError(
            f"scenario_start ({cfg.scenario_start}) is outside the .wxs span "
            f"[{naive_start}, {naive_end}]."
        )

    # 3. Solar synthesis — also localizes the index to geo.timezone in place.
    synthesize_solar(weather, geo, cfg.cloud_scale)

    # 4. Phase + snow columns
    _attach_phase_and_snow(weather.df, cfg.scenario_start, cfg.snow_mode, geo.timezone)

    # 5/6/7. Moisture + KBDI series
    dead = compute_dead_moisture(weather).df            # hourly MC1/10/100/1000
    live = compute_live_moisture(weather, geo).df        # daily GSI/MCHERB/MCWOOD
    avg_precip, precip_source = resolve_avg_ann_precip_in(
        explicit=cfg.avg_ann_precip_in,
        lat=geo.center_lat, lon=geo.center_lon,
    )
    kbdi = compute_kbdi_series(weather, avg_precip, cfg.reg_obs_hr).df  # daily KBDI

    # Forward-fill daily series onto the hourly index.
    live_hourly = _forward_fill_daily_to_hourly(
        live, weather.df.index,
        seeds={"GSI": _GSI_SEED, "MCHERB": _MCHERB_SEED, "MCWOOD": _MCWOOD_SEED},
    )
    kbdi_hourly = _forward_fill_daily_to_hourly(
        kbdi, weather.df.index, seeds={"KBDI": _KBDI_SEED}
    )

    # 8. Per-fuel-model index loop
    hourly_index = weather.df.index
    mc1 = dead["MC1"].to_numpy()
    mc10 = dead["MC10"].to_numpy()
    mc100 = dead["MC100"].to_numpy()
    mc1000 = dead["MC1000"].to_numpy()
    mcherb = live_hourly["MCHERB"].to_numpy()
    mcwood = live_hourly["MCWOOD"].to_numpy()
    gsi = live_hourly["GSI"].to_numpy()
    kbdi_arr = kbdi_hourly["KBDI"].to_numpy()
    wind_mph = weather.df["wind_mph"].to_numpy()

    per_model: dict[str, dict[str, np.ndarray]] = {}
    n = len(hourly_index)
    for model in composition.fractions:
        fuel = NFDRS_FUEL_MODELS[model]
        sc = np.empty(n)
        erc = np.empty(n)
        bi = np.empty(n)
        for i in range(n):
            r = calc_indexes(
                fuel, float(mc1[i]), float(mc10[i]), float(mc100[i]),
                float(mc1000[i]), float(mcherb[i]), float(mcwood[i]),
                float(gsi[i]), float(kbdi_arr[i]),
                float(wind_mph[i]), slope_class,
            )
            sc[i] = r.sc
            erc[i] = r.erc
            bi[i] = r.bi
        per_model[model] = {"SC": sc, "ERC": erc, "BI": bi}

    # 9. Area-weighted BI with NaN-renormalization (OQ-14)
    weights = np.array(
        [composition.fractions[m] for m in per_model], dtype=float
    )
    bi_matrix = np.stack([per_model[m]["BI"] for m in per_model], axis=1)  # (n, k)
    valid = np.isfinite(bi_matrix)
    weights_mat = np.broadcast_to(weights, bi_matrix.shape) * valid
    sum_weights = weights_mat.sum(axis=1)
    safe_bi = np.where(valid, bi_matrix, 0.0)
    weighted = (safe_bi * weights_mat).sum(axis=1)
    bi_weighted = np.where(sum_weights > 0, weighted / sum_weights, np.nan)

    # 10. Assemble output DataFrame
    out_cols: dict[str, np.ndarray] = {}
    out_cols["phase"] = weather.df["phase"].to_numpy()
    for model, blk in per_model.items():
        out_cols[f"BI_{model}"] = blk["BI"]
        out_cols[f"SC_{model}"] = blk["SC"]
        out_cols[f"ERC_{model}"] = blk["ERC"]
    out_cols["BI_area_weighted"] = bi_weighted
    # Debug columns
    out_cols["MC1"] = mc1
    out_cols["MC10"] = mc10
    out_cols["MC100"] = mc100
    out_cols["MC1000"] = mc1000
    out_cols["MCHERB"] = mcherb
    out_cols["MCWOOD"] = mcwood
    out_cols["GSI"] = gsi
    out_cols["KBDI"] = kbdi_arr
    out_cols["wind_mph"] = wind_mph

    df_out = pd.DataFrame(out_cols, index=hourly_index)

    # peak BI = 97th pct of scenario rows (OQ-15)
    scenario_mask = df_out["phase"] == "scenario"
    scenario_bi = df_out.loc[scenario_mask, "BI_area_weighted"].dropna()
    if scenario_bi.empty:
        peak_bi = float("nan")
    else:
        peak_bi = float(np.percentile(scenario_bi.to_numpy(), 97))

    metadata = {
        "geo_center_lat": geo.center_lat,
        "geo_center_lon": geo.center_lon,
        "geo_timezone": geo.timezone,
        "avg_ann_precip_in": avg_precip,
        "avg_ann_precip_source": precip_source,
        "slope_class_used": slope_class,
        "fbfm_type": composition.fbfm_type,
        "n_burnable_pixels": composition.n_burnable_pixels,
        "n_total_pixels": composition.n_total_pixels,
        "pixel_area_m2": composition.pixel_area_m2,
        "ref_elev_m": weather.ref_elev_m,
        "n_hours": len(df_out),
    }

    return TrajectoryResult(
        df=df_out,
        peak_bi=peak_bi,
        fuel_composition=composition,
        config=cfg,
        metadata=metadata,
    )
