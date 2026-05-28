"""Component 8 — CSV writer + matplotlib plot for the trajectory result."""
from __future__ import annotations

import os
from typing import Iterable

import matplotlib

# Use a non-interactive backend so plotting works headless (CI, batch tuning).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from embrs.fire_danger.config import TrajectoryResult


def write_csv(result: TrajectoryResult, path: str) -> None:
    """Write the trajectory DataFrame to CSV.

    Timestamp is written as ISO-8601 local (the index is tz-aware after
    solar synthesis localized it).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df = result.df.copy()
    df.index.name = df.index.name or "timestamp"
    df.to_csv(path, index=True)


_REG_OBS_HOUR: int = 13   # NFDRS regular observation hour (1 PM local)


def _plot_hourly(ax, result: TrajectoryResult) -> None:
    """Top panel — hourly area-weighted BI with per-model thin lines."""
    df = result.df

    for col in df.columns:
        if col.startswith("BI_") and col != "BI_area_weighted":
            ax.plot(df.index, df[col].to_numpy(), linewidth=0.8, alpha=0.6,
                    label=col)

    ax.plot(df.index, df["BI_area_weighted"].to_numpy(),
            linewidth=2.2, color="black", label="BI_area_weighted")

    conditioning = df.index[df["phase"] == "conditioning"]
    if len(conditioning) > 0:
        ax.axvspan(conditioning.min(), conditioning.max(),
                   alpha=0.10, color="gray", label="conditioning")

    if result.peak_bi == result.peak_bi:  # not NaN
        ax.axhline(result.peak_bi, color="red", linestyle="--", linewidth=1.0,
                   label=f"peak BI (97th pct) = {result.peak_bi:.1f}")

    ax.set_ylabel("Burning Index (hourly)")
    ax.set_title(
        "Hourly trajectory — "
        + ", ".join(f"{m}={f:.0%}"
                    for m, f in result.fuel_composition.fractions.items())
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def _plot_daily_reg_obs(ax, result: TrajectoryResult) -> None:
    """Bottom panel — one BI value per day at the NFDRS regular obs hour.

    Sampled at 13:00 local (1 PM). Mirrors how NFDRS BI is operationally
    reported: a single afternoon value per day, stripping the diurnal
    saw-tooth driven by the 1-hr fuel moisture cycle. Days where no
    13:00 row is present in the trajectory are dropped.
    """
    df = result.df
    daily_df = df[df.index.hour == _REG_OBS_HOUR]
    if daily_df.empty:
        ax.text(0.5, 0.5, "(no rows at 13:00 — skipping daily-1pm plot)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    for col in daily_df.columns:
        if col.startswith("BI_") and col != "BI_area_weighted":
            ax.plot(daily_df.index, daily_df[col].to_numpy(),
                    linewidth=1.0, alpha=0.6, marker="o", markersize=3,
                    label=col)

    ax.plot(daily_df.index, daily_df["BI_area_weighted"].to_numpy(),
            linewidth=2.2, color="black", marker="o", markersize=4,
            label="BI_area_weighted (1 PM)")

    conditioning = daily_df.index[daily_df["phase"] == "conditioning"]
    if len(conditioning) > 0:
        ax.axvspan(conditioning.min(), conditioning.max(),
                   alpha=0.10, color="gray")

    ax.set_xlabel("Date (local)")
    ax.set_ylabel("Burning Index (1 PM)")
    ax.set_title("Daily 1 PM BI — one sample per day at the NFDRS RegObsHr")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_trajectory(result: TrajectoryResult, path: str) -> None:
    """Two-panel plot: hourly BI (top) and daily 1 PM BI (bottom).

    - **Top**: ``BI_area_weighted`` plus per-model thin lines at hourly
      resolution. Conditioning period shaded gray. Horizontal marker at
      ``peak_bi`` (97th percentile of scenario hours).
    - **Bottom**: one BI value per day sampled at 13:00 local — the NFDRS
      regular observation hour. Strips the diurnal MC1-driven saw-tooth
      and matches how operational fire-danger reports describe a period.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    _plot_hourly(ax_top, result)
    _plot_daily_reg_obs(ax_bot, result)

    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
