"""Benchmark script for measuring how EMBRS scales with cell_size and time_step.

Runs a fixed fire scenario (spot_test map, 48h duration) across a grid of
(cell_size, time_step) combinations and records wall-clock times, iteration
counts, and burning-cell statistics.

Produces a JSON results file suitable for generating tables and plots for
the computational performance discussion in the manuscript.

Usage:
    python -m embrs.tests.benchmarks.benchmark_scaling
    python -m embrs.tests.benchmarks.benchmark_scaling --output results.json
    python -m embrs.tests.benchmarks.benchmark_scaling --cell-sizes 30 50 75 --time-steps 5 15
    python -m embrs.tests.benchmarks.benchmark_scaling --map-folder /path/to/map --weather-file /path/to/weather.wxs
"""

import argparse
import json
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ── Default configuration ────────────────────────────────────────────────────

DEFAULT_MAP_FOLDER = "/Users/rjdp3/Documents/Research/embrs_map/spot_test"
DEFAULT_WEATHER_FILE = "/Users/rjdp3/Documents/Research/embrs_weather/two_week_backburn_balanced.wxs"

# Simulation window: 48 hours starting July 1 08:00
DEFAULT_START = "2025-07-01T08:00:00"
DEFAULT_END = "2025-07-03T08:00:00"

# Conditioning period starts at the beginning of the weather file
DEFAULT_CONDITIONING_START = "2025-07-01T00:00:00"

DEFAULT_CELL_SIZES = [20, 30, 50, 75, 100]
DEFAULT_TIME_STEPS = [5, 10, 15, 30]

# Sample per-iteration timing every N iterations (keeps overhead low)
ITER_SAMPLE_INTERVAL = 50

SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_sim_params(
    map_folder: str,
    weather_file: str,
    cell_size: int,
    time_step: int,
    start_dt: str,
    end_dt: str,
    conditioning_start: str,
):
    """Build a SimParams object programmatically (no .cfg file needed)."""
    from embrs.utilities.data_classes import SimParams, WeatherParams

    # Load pre-built map params
    pkl_path = os.path.join(map_folder, "map_params.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"map_params.pkl not found in {map_folder}")

    with open(pkl_path, "rb") as f:
        map_params = pickle.load(f)

    start = datetime.fromisoformat(start_dt)
    end = datetime.fromisoformat(end_dt)
    cond = datetime.fromisoformat(conditioning_start)
    duration_s = (end - start).total_seconds()

    weather_params = WeatherParams(
        input_type="File",
        file=weather_file,
        mesh_resolution=250,
        conditioning_start=cond,
        start_datetime=start,
        end_datetime=end,
    )

    sim_params = SimParams(
        map_params=map_params,
        log_folder=None,
        weather_input=weather_params,
        t_step_s=time_step,
        cell_size=cell_size,
        init_mf=[0.06, 0.07, 0.08],
        model_spotting=False,
        duration_s=duration_s,
        visualize=False,
        num_runs=1,
        write_logs=False,
    )

    return sim_params


def compute_grid_info(map_folder: str, cell_size: int) -> dict:
    """Compute grid dimensions for a given cell size without initializing sim."""
    pkl_path = os.path.join(map_folder, "map_params.pkl")
    with open(pkl_path, "rb") as f:
        map_params = pickle.load(f)

    rows, cols = map_params.shape(cell_size)
    return {
        "rows": rows,
        "cols": cols,
        "total_cells": rows * cols,
        "map_width_m": map_params.lcp_data.width_m,
        "map_height_m": map_params.lcp_data.height_m,
    }


def run_single_benchmark(
    map_folder: str,
    weather_file: str,
    cell_size: int,
    time_step: int,
    start_dt: str,
    end_dt: str,
    conditioning_start: str,
) -> dict:
    """Run one simulation and collect performance metrics."""
    from embrs.fire_simulator.fire import FireSim

    # Reproducibility
    np.random.seed(SEED)
    random.seed(SEED)

    grid_info = compute_grid_info(map_folder, cell_size)
    total_cells = grid_info["total_cells"]

    start = datetime.fromisoformat(start_dt)
    end = datetime.fromisoformat(end_dt)
    duration_s = (end - start).total_seconds()
    expected_iters = int(duration_s / time_step)

    print(f"\n{'='*60}")
    print(f"  cell_size={cell_size}m  time_step={time_step}s")
    print(f"  grid: {grid_info['rows']}x{grid_info['cols']} = {total_cells:,} cells")
    print(f"  expected iterations: {expected_iters:,}")
    print(f"{'='*60}")

    # ── Initialization timing ────────────────────────────────────────────
    sim_params = build_sim_params(
        map_folder, weather_file, cell_size, time_step,
        start_dt, end_dt, conditioning_start,
    )

    t0 = time.perf_counter()
    sim = FireSim(sim_params)
    init_time = time.perf_counter() - t0

    print(f"  Init time: {init_time:.2f}s")

    # ── Simulation loop with instrumentation ─────────────────────────────
    iter_count = 0
    burning_history = []          # (iteration, n_burning) — sampled
    iter_time_samples = []        # (iteration, n_burning, ms) — sampled
    peak_burning = 0
    total_burning_cell_iters = 0  # sum of n_burning across all iterations

    sim_start = time.perf_counter()

    while not sim.finished:
        n_burning = len(sim.burning_cells)
        total_burning_cell_iters += n_burning

        if n_burning > peak_burning:
            peak_burning = n_burning

        # Sample per-iteration timing periodically
        should_sample = (iter_count % ITER_SAMPLE_INTERVAL == 0)

        if should_sample:
            t_iter_start = time.perf_counter()

        sim.iterate()

        if should_sample:
            iter_ms = (time.perf_counter() - t_iter_start) * 1000
            burning_history.append((iter_count, n_burning))
            iter_time_samples.append((iter_count, n_burning, round(iter_ms, 4)))

        iter_count += 1

        # Progress update every 10% of expected iterations
        if expected_iters > 0 and iter_count % max(1, expected_iters // 10) == 0:
            pct = iter_count / expected_iters * 100
            elapsed = time.perf_counter() - sim_start
            print(f"  {pct:5.1f}% — iter {iter_count:,}, "
                  f"burning: {n_burning:,}, elapsed: {elapsed:.1f}s")

    sim_time = time.perf_counter() - sim_start
    total_time = init_time + sim_time

    print(f"  Sim time:   {sim_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Iterations: {iter_count:,}")
    print(f"  Peak burning cells: {peak_burning:,}")

    # Compute mean iteration time from samples
    if iter_time_samples:
        sample_times = [s[2] for s in iter_time_samples]
        mean_iter_ms = sum(sample_times) / len(sample_times)
    else:
        mean_iter_ms = (sim_time * 1000) / max(iter_count, 1)

    return {
        "cell_size_m": cell_size,
        "time_step_s": time_step,
        "grid_rows": grid_info["rows"],
        "grid_cols": grid_info["cols"],
        "total_cells": total_cells,
        "duration_s": duration_s,
        "total_iterations": iter_count,
        "init_time_s": round(init_time, 3),
        "sim_time_s": round(sim_time, 3),
        "total_time_s": round(total_time, 3),
        "peak_burning_cells": peak_burning,
        "total_burning_cell_iters": total_burning_cell_iters,
        "mean_iter_ms": round(mean_iter_ms, 4),
        "iter_time_samples": iter_time_samples,
        "burning_history": burning_history,
    }


# ── Summary printing ─────────────────────────────────────────────────────────

def print_summary_table(results: List[dict]):
    """Print a formatted summary table of all runs."""
    print("\n" + "=" * 110)
    print("SCALING BENCHMARK SUMMARY")
    print("=" * 110)
    header = (
        f"{'cell(m)':>8} {'dt(s)':>6} {'grid':>12} {'cells':>8} "
        f"{'iters':>9} {'init(s)':>8} {'sim(s)':>9} {'total(s)':>9} "
        f"{'peak_burn':>10} {'ms/iter':>9}"
    )
    print(header)
    print("-" * 110)

    for r in results:
        grid_str = f"{r['grid_rows']}x{r['grid_cols']}"
        print(
            f"{r['cell_size_m']:>8} {r['time_step_s']:>6} {grid_str:>12} "
            f"{r['total_cells']:>8,} {r['total_iterations']:>9,} "
            f"{r['init_time_s']:>8.1f} {r['sim_time_s']:>9.1f} "
            f"{r['total_time_s']:>9.1f} {r['peak_burning_cells']:>10,} "
            f"{r['mean_iter_ms']:>9.3f}"
        )

    print("=" * 110)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark EMBRS scaling with cell_size and time_step"
    )
    parser.add_argument(
        "--map-folder", type=str, default=DEFAULT_MAP_FOLDER,
        help="Path to map data folder containing map_params.pkl"
    )
    parser.add_argument(
        "--weather-file", type=str, default=DEFAULT_WEATHER_FILE,
        help="Path to weather file (.wxs or .json)"
    )
    parser.add_argument(
        "--cell-sizes", type=int, nargs="+", default=DEFAULT_CELL_SIZES,
        help="Cell sizes to test (meters)"
    )
    parser.add_argument(
        "--time-steps", type=int, nargs="+", default=DEFAULT_TIME_STEPS,
        help="Time steps to test (seconds)"
    )
    parser.add_argument(
        "--start", type=str, default=DEFAULT_START,
        help="Simulation start datetime (ISO format)"
    )
    parser.add_argument(
        "--end", type=str, default=DEFAULT_END,
        help="Simulation end datetime (ISO format)"
    )
    parser.add_argument(
        "--conditioning-start", type=str, default=DEFAULT_CONDITIONING_START,
        help="Fuel moisture conditioning start datetime (ISO format)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON file path (default: benchmark_scaling_results.json in cwd)"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(os.path.join(args.map_folder, "map_params.pkl")):
        print(f"Error: map_params.pkl not found in {args.map_folder}")
        return 1
    if not os.path.exists(args.weather_file):
        print(f"Error: weather file not found: {args.weather_file}")
        return 1

    output_path = args.output or "benchmark_scaling_results.json"

    # Print grid info for all cell sizes
    print("\nGrid dimensions for each cell size:")
    for cs in args.cell_sizes:
        info = compute_grid_info(args.map_folder, cs)
        print(f"  cell_size={cs:>3}m -> {info['rows']}x{info['cols']} = {info['total_cells']:>6,} cells")

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    duration_h = (end - start).total_seconds() / 3600
    print(f"\nDuration: {duration_h:.0f} hours")
    print(f"Time steps: {args.time_steps}")
    total_runs = len(args.cell_sizes) * len(args.time_steps)
    print(f"Total runs: {total_runs}")

    # Run benchmarks (largest cell_size first → fastest runs first, for quick sanity check)
    results = []
    run_num = 0

    for cell_size in sorted(args.cell_sizes, reverse=True):
        for time_step in sorted(args.time_steps, reverse=True):
            run_num += 1
            print(f"\n>>> Run {run_num}/{total_runs}")

            result = run_single_benchmark(
                map_folder=args.map_folder,
                weather_file=args.weather_file,
                cell_size=cell_size,
                time_step=time_step,
                start_dt=args.start,
                end_dt=args.end,
                conditioning_start=args.conditioning_start,
            )
            results.append(result)

            # Write intermediate results after each run
            output = {
                "metadata": {
                    "map_folder": args.map_folder,
                    "weather_file": args.weather_file,
                    "start_datetime": args.start,
                    "end_datetime": args.end,
                    "duration_h": duration_h,
                    "seed": SEED,
                    "timestamp": datetime.now().isoformat(),
                },
                "results": results,
            }
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

    # Sort results for display: by cell_size ascending, then time_step ascending
    results.sort(key=lambda r: (r["cell_size_m"], r["time_step_s"]))
    print_summary_table(results)

    # Final write with sorted results
    output["results"] = results
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
