"""Diagnostic plotters for scenario weather (spec §4.7).

Two matplotlib plotters for eyeballing inputs and outputs (diagnostics, not part
of the class metric):

- :func:`plot_temprh` — a candidate **backdrop window** (search-tool output)
  before any wind is added: hourly temperature (°F) and RH (%) on twin axes,
  with daily-max-temp / daily-min-RH (and daily-peak VPD) marked.
- :func:`plot_wxs` — a multi-panel view of a generated/edited ``.wxs``
  (temp+RH; wind speed with the deterministic mean and the backburn threshold;
  wind direction with the prevailing bearing; precip/cloud), shading the
  backburn-feasible windows.

Both reuse the timezone fix from
``embrs.weather_candidate_search.artifacts.plot_candidate``: pass the data's
local timezone to the date locator/formatter so an afternoon peak renders in
local time, not UTC (a bug already hit and fixed there).
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from embrs.fire_danger.weather_loader import load_wxs
from embrs.scenario_weather.config import WindModelConfig
from embrs.scenario_weather.period_search import vapor_pressure_deficit_kpa
from embrs.scenario_weather.wind_model import diurnal_mean_ms

_MPS_TO_MPH: float = 2.23693629


def _apply_local_time_axis(ax, index: pd.DatetimeIndex) -> None:
    """Render ``ax``'s date axis in the data's own local timezone (spec §4.7).

    matplotlib otherwise labels tz-aware timestamps in UTC, shifting an
    afternoon peak ~5-6 h so it appears near midnight. Passing the index tz to
    the locator/formatter keeps the axis in local time. For tz-naive data this
    is a no-op (``tz=None``).
    """
    axis_tz = getattr(index, "tz", None)
    locator = mdates.AutoDateLocator(tz=axis_tz)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=axis_tz))


def _localize(df: pd.DataFrame, local_tz: Optional[str]) -> pd.DataFrame:
    """Attach ``local_tz`` to a tz-naive frame so axes label in local time."""
    if local_tz and df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize(local_tz, nonexistent="shift_forward", ambiguous="NaT")
    return df


def plot_temprh(
    source,
    out_path: str,
    *,
    local_tz: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Plot a backdrop window's hourly temperature and RH (spec §4.7).

    Args:
        source: A ``.wxs`` path, or a dataframe with ``temp_F``/``rh_pct``/
            ``temp_C`` columns and a ``DatetimeIndex`` (e.g. from
            :func:`period_search.slice_window_df`).
        out_path: Output PNG path.
        local_tz: Optional IANA zone to render the x-axis in local time.
        title: Optional plot title.

    Returns:
        ``out_path``.
    """
    if isinstance(source, str):
        df = load_wxs(source).df
    else:
        df = source
    df = _localize(df, local_tz)

    temp_F = df["temp_F"].to_numpy()
    rh_pct = df["rh_pct"].to_numpy()
    vpd = vapor_pressure_deficit_kpa(df["temp_C"].to_numpy(), rh_pct)

    # Daily extremes for the markers. Keep idxmax results as lists of (possibly
    # tz-aware) Timestamps — Series.values would coerce tz-aware to UTC-naive
    # datetime64 and break the .loc lookup / plot positions.
    daily = pd.DataFrame({"temp_F": temp_F, "rh_pct": rh_pct, "vpd": vpd}, index=df.index)
    tmax_times = list(daily["temp_F"].groupby(daily.index.normalize()).idxmax())
    rhmin_times = list(daily["rh_pct"].groupby(daily.index.normalize()).idxmin())
    vpd_peak_times = list(daily["vpd"].groupby(daily.index.normalize()).idxmax())

    fig, ax_t = plt.subplots(figsize=(12, 4.5))
    ax_t.plot(df.index, temp_F, color="firebrick", linewidth=1.2, label="Temp (°F)")
    ax_t.scatter(tmax_times, daily.loc[tmax_times, "temp_F"],
                 color="firebrick", s=22, zorder=5, label="daily max temp")
    ax_t.set_ylabel("Temp (°F)", color="firebrick")
    ax_t.tick_params(axis="y", labelcolor="firebrick")

    ax_rh = ax_t.twinx()
    ax_rh.plot(df.index, rh_pct, color="steelblue", linewidth=1.0, label="RH (%)")
    ax_rh.scatter(rhmin_times, daily.loc[rhmin_times, "rh_pct"],
                  color="steelblue", s=22, zorder=5, label="daily min RH")
    ax_rh.set_ylabel("RH (%)", color="steelblue")
    ax_rh.tick_params(axis="y", labelcolor="steelblue")
    ax_rh.set_ylim(0, 100)

    # Mark daily peak-VPD (hot/dry afternoon) hours with faint vertical lines.
    for ts in vpd_peak_times:
        ax_t.axvline(ts, color="goldenrod", alpha=0.25, linewidth=0.8)

    ax_t.set_xlabel("Local time")
    ax_t.set_title(title or "Backdrop window — temperature & RH")
    ax_t.grid(True, alpha=0.3)
    _apply_local_time_axis(ax_t, df.index)
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _shade_mask(ax, index: pd.DatetimeIndex, mask: np.ndarray, color: str) -> None:
    """Shade contiguous True runs of an hourly boolean mask."""
    n = len(index)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        ax.axvspan(index[i], index[j] + pd.Timedelta(hours=1), alpha=0.18, color=color)
        i = j + 1


def plot_wxs(
    wxs_path: str,
    out_path: str,
    *,
    wind_cfg: Optional[WindModelConfig] = None,
    peak_scale: Optional[float] = None,
    backburn_threshold_m_s: float = 10.0,
    suitable_mask: Optional[np.ndarray] = None,
    local_tz: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Multi-panel view of a generated/edited ``.wxs`` (spec §4.7).

    Panels: (1) temp + RH; (2) wind speed — realized series plus the
    deterministic diurnal mean (if ``wind_cfg`` given) and the backburn wind
    threshold; (3) wind direction with the prevailing bearing; (4) precip +
    cloud. Backburn-feasible windows are shaded (from ``suitable_mask`` if
    provided, else the pure wind-under-threshold lulls).

    Args:
        wxs_path: The ``.wxs`` to plot.
        out_path: Output PNG path.
        wind_cfg: If given, overlays the deterministic diurnal mean and marks
            the prevailing bearing.
        peak_scale: Peak override for the overlaid mean (the tuning knob).
        backburn_threshold_m_s: Reference wind threshold drawn on the speed
            panel and used for lull shading.
        suitable_mask: Optional hourly boolean mask of backburn-suitable hours
            (e.g. from :func:`backburn_check.validate`'s logic) to shade instead
            of the pure wind-under-threshold mask.
        local_tz: Optional IANA zone to render the x-axis in local time.
        title: Optional overall title.

    Returns:
        ``out_path``.
    """
    df = _localize(load_wxs(wxs_path).df, local_tz)
    index = df.index
    wind_mph = df["wind_mph"].to_numpy()
    wind_dir = df["wind_dir_deg"].to_numpy()
    thresh_mph = backburn_threshold_m_s * _MPS_TO_MPH
    shade = suitable_mask if suitable_mask is not None else (wind_mph <= thresh_mph)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 11), sharex=True)
    ax_th, ax_w, ax_d, ax_pc = axes

    # Panel 1: temp + RH.
    ax_th.plot(index, df["temp_F"].to_numpy(), color="firebrick", label="Temp (°F)")
    ax_th.set_ylabel("Temp (°F)", color="firebrick")
    ax_th.tick_params(axis="y", labelcolor="firebrick")
    ax_rh = ax_th.twinx()
    ax_rh.plot(index, df["rh_pct"].to_numpy(), color="steelblue", linewidth=1.0, label="RH (%)")
    ax_rh.set_ylabel("RH (%)", color="steelblue")
    ax_rh.tick_params(axis="y", labelcolor="steelblue")
    ax_rh.set_ylim(0, 100)
    _shade_mask(ax_th, index, shade, "green")
    ax_th.set_title(title or f"Scenario .wxs — {os.path.basename(wxs_path)}")
    ax_th.grid(True, alpha=0.3)

    # Panel 2: wind speed (realized + deterministic mean + threshold).
    ax_w.plot(index, wind_mph, color="darkgreen", linewidth=1.0, label="WindSpd realized (mph, 20 ft)")
    if wind_cfg is not None:
        hod = np.asarray(index.hour + index.minute / 60.0, dtype=float)
        mean_mph = diurnal_mean_ms(hod, wind_cfg, peak_scale=peak_scale) * _MPS_TO_MPH
        ax_w.plot(index, mean_mph, color="black", linewidth=1.2, linestyle="--",
                  label="diurnal mean")
    ax_w.axhline(thresh_mph, color="red", linestyle=":", linewidth=1.0,
                 label=f"backburn threshold ({thresh_mph:.0f} mph)")
    ax_w.set_ylabel("Wind speed (mph)")
    _shade_mask(ax_w, index, shade, "green")
    ax_w.legend(loc="upper left", fontsize=8)
    ax_w.grid(True, alpha=0.3)

    # Panel 3: wind direction.
    ax_d.scatter(index, wind_dir, s=6, color="purple", alpha=0.5, label="WindDir (°)")
    if wind_cfg is not None:
        ax_d.axhline(wind_cfg.prevailing_dir_deg % 360, color="purple", alpha=0.4,
                     linewidth=1.0, label=f"prevailing {wind_cfg.prevailing_dir_deg:.0f}°")
    ax_d.set_ylabel("Wind dir (°)")
    ax_d.set_ylim(0, 360)
    ax_d.legend(loc="upper left", fontsize=8)
    ax_d.grid(True, alpha=0.3)

    # Panel 4: precip + cloud.
    ax_pc.plot(index, df["precip_in_hr"].to_numpy(), color="navy", label="Precip (in/hr)")
    ax_pc.set_ylabel("Precip (in/hr)", color="navy")
    ax_pc.tick_params(axis="y", labelcolor="navy")
    ax_cc = ax_pc.twinx()
    ax_cc.plot(index, df["cloud_cover"].to_numpy(), color="gray", linewidth=1.0, label="Cloud (%)")
    ax_cc.set_ylabel("Cloud (%)", color="gray")
    ax_cc.tick_params(axis="y", labelcolor="gray")
    ax_cc.set_ylim(0, 100)
    ax_pc.set_xlabel("Local time")
    ax_pc.grid(True, alpha=0.3)

    _apply_local_time_axis(ax_pc, index)
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
