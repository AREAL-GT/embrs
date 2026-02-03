"""Benchmark script for measuring fire simulation iteration performance.

This script measures the time taken for fire simulation iterations and provides
statistics useful for tracking performance improvements.

Usage:
    python -m embrs.tests.benchmarks.benchmark_iteration --config path/to/config.cfg
    python -m embrs.tests.benchmarks.benchmark_iteration --iterations 200
"""

import argparse
import statistics
import time
from pathlib import Path


def benchmark_iteration(config_path: str, num_iterations: int = 100, warmup: int = 10):
    """Benchmark fire simulation iteration performance.

    Args:
        config_path: Path to simulation configuration file.
        num_iterations: Number of iterations to measure (after warmup).
        warmup: Number of warmup iterations before measurement.

    Returns:
        dict: Statistics including mean, median, stdev, min, max times.
    """
    from embrs.main import load_sim_params
    from embrs.fire_simulator.fire import FireSim

    print(f"Loading configuration from: {config_path}")
    sim_params = load_sim_params(config_path)

    print("Initializing simulation...")
    sim = FireSim(sim_params)

    # Warmup iterations (allows JIT compilation, cache warming, etc.)
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        if sim.finished:
            print("Warning: Simulation finished during warmup")
            break
        sim.iterate()

    # Measured iterations
    print(f"Running {num_iterations} measured iterations...")
    times = []
    for i in range(num_iterations):
        if sim.finished:
            print(f"Simulation finished after {i} measured iterations")
            break

        start = time.perf_counter()
        sim.iterate()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    if len(times) < 2:
        print("Error: Not enough iterations completed for statistics")
        return None

    results = {
        'num_iterations': len(times),
        'mean_ms': statistics.mean(times) * 1000,
        'median_ms': statistics.median(times) * 1000,
        'stdev_ms': statistics.stdev(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'total_s': sum(times),
    }

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted table."""
    if results is None:
        print("No results to display")
        return

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Iterations measured: {results['num_iterations']}")
    print(f"Total time:          {results['total_s']:.3f} s")
    print("-" * 50)
    print(f"Mean:                {results['mean_ms']:.3f} ms/iteration")
    print(f"Median:              {results['median_ms']:.3f} ms/iteration")
    print(f"Std Dev:             {results['stdev_ms']:.3f} ms")
    print(f"Min:                 {results['min_ms']:.3f} ms")
    print(f"Max:                 {results['max_ms']:.3f} ms")
    print("=" * 50)


def find_default_config() -> str:
    """Find a default configuration file for benchmarking."""
    search_paths = [
        Path("embrs/configs"),
        Path("configs"),
        Path("."),
    ]

    for path in search_paths:
        if path.exists():
            configs = list(path.glob("*.cfg"))
            if configs:
                return str(configs[0])

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fire simulation iteration performance"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to simulation configuration file"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of iterations to measure (default: 100)"
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )

    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = find_default_config()
        if config_path is None:
            print("Error: No configuration file specified and no default found")
            print("Usage: python -m embrs.tests.benchmarks.benchmark_iteration --config path/to/config.cfg")
            return 1

    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1

    results = benchmark_iteration(
        config_path,
        num_iterations=args.iterations,
        warmup=args.warmup
    )

    print_results(results)
    return 0


if __name__ == "__main__":
    exit(main())
