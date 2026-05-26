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


_NOON_HOUR: int = 12


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


def _plot_daily_noon(ax, result: TrajectoryResult) -> None:
    """Bottom panel — one BI value per day, sampled at 12:00 local.

    Mirrors how NFDRS BI is normally read: a single afternoon value per day
    rather than the hourly saw-tooth driven by the diurnal MC1 cycle.
    Days where no 12:00 row is present in the trajectory are dropped.
    """
    df = result.df
    noon_df = df[df.index.hour == _NOON_HOUR]
    if noon_df.empty:
        ax.text(0.5, 0.5, "(no rows at 12:00 — skipping noon plot)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    for col in noon_df.columns:
        if col.startswith("BI_") and col != "BI_area_weighted":
            ax.plot(noon_df.index, noon_df[col].to_numpy(),
                    linewidth=1.0, alpha=0.6, marker="o", markersize=3,
                    label=col)

    ax.plot(noon_df.index, noon_df["BI_area_weighted"].to_numpy(),
            linewidth=2.2, color="black", marker="o", markersize=4,
            label="BI_area_weighted (noon)")

    conditioning = noon_df.index[noon_df["phase"] == "conditioning"]
    if len(conditioning) > 0:
        ax.axvspan(conditioning.min(), conditioning.max(),
                   alpha=0.10, color="gray")

    ax.set_xlabel("Date (local)")
    ax.set_ylabel("Burning Index (noon)")
    ax.set_title("Daily noon BI — one sample per day at 12:00 local")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_trajectory(result: TrajectoryResult, path: str) -> None:
    """Two-panel plot: hourly BI (top) and daily noon BI (bottom).

    - **Top**: ``BI_area_weighted`` plus per-model thin lines at hourly
      resolution. Conditioning period shaded gray. Horizontal marker at
      ``peak_bi`` (97th percentile of scenario hours).
    - **Bottom**: one BI value per day sampled at 12:00 local — easier to
      read for tuning since it strips the diurnal MC1-driven saw-tooth.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    _plot_hourly(ax_top, result)
    _plot_daily_noon(ax_bot, result)

    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
