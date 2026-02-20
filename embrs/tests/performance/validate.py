"""Performance validation harness for EMBRS fire simulation.

Runs test cases against baseline outputs, validates correctness (fire arrival times
and cell-level fire state), and measures wall-clock speedup.

Usage:
    # Generate baseline (first run on unmodified code):
    python -m embrs.tests.performance.validate --generate-baseline

    # Validate against baseline:
    python -m embrs.tests.performance.validate

    # Run a single test case:
    python -m embrs.tests.performance.validate --case tc01_small_flat_no_spot

    # Increase tolerance:
    python -m embrs.tests.performance.validate --atol 1e-4
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PERF_DIR = Path(__file__).parent
CONFIGS_DIR = PERF_DIR / "configs"
BASELINE_DIR = PERF_DIR / "baseline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_test_cases() -> List[str]:
    """Return sorted list of test case names from config files."""
    configs = sorted(CONFIGS_DIR.glob("tc*.cfg"))
    return [c.stem for c in configs]


def run_simulation(cfg_path: str, seed: int = 42) -> Tuple[dict, float]:
    """Run a fire simulation and capture outputs.

    Returns:
        Tuple of (outputs_dict, wall_clock_seconds).
        outputs_dict keys:
            - cell_states: dict mapping cell_id -> final CellStates int
            - arrival_times: dict mapping cell_id -> arrival_time (float, -999 if not burned)
            - num_iters: int, total iterations run
            - num_burning_cells_history: list of int, burning cell count per iteration
            - final_time_s: float, final simulation time
    """
    from embrs.main import load_sim_params
    from embrs.fire_simulator.fire import FireSim

    np.random.seed(seed)

    sim_params = load_sim_params(cfg_path)
    sim_params.visualize = False
    sim_params.write_logs = False

    t0 = time.perf_counter()

    sim = FireSim(sim_params)

    init_time = time.perf_counter() - t0

    # Run simulation
    iter_count = 0
    burning_history = []

    sim_start = time.perf_counter()
    while not sim.finished:
        sim.iterate()
        iter_count += 1
        burning_history.append(len(sim.burning_cells))

    sim_time = time.perf_counter() - sim_start
    total_time = time.perf_counter() - t0

    # Capture cell states and arrival times
    cell_states = {}
    arrival_times = {}

    for cell_id, cell in sim.cell_dict.items():
        cell_states[cell_id] = int(cell.state)
        arrival_times[cell_id] = float(cell._arrival_time)

    outputs = {
        "cell_states": cell_states,
        "arrival_times": arrival_times,
        "num_iters": iter_count,
        "num_burning_cells_history": burning_history,
        "final_time_s": float(sim.curr_time_s),
        "init_time_s": init_time,
        "sim_time_s": sim_time,
        "total_time_s": total_time,
    }

    return outputs, total_time


def save_baseline(case_name: str, outputs: dict):
    """Save baseline outputs for a test case."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"{case_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(outputs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved baseline to {path}")


def load_baseline(case_name: str) -> Optional[dict]:
    """Load baseline outputs for a test case."""
    path = BASELINE_DIR / f"{case_name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def validate_outputs(
    baseline: dict,
    current: dict,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Tuple[bool, List[str]]:
    """Compare current outputs against baseline.

    Returns:
        Tuple of (passed: bool, messages: list of str).
    """
    messages = []
    passed = True

    # 1. Check cell-level fire state (exact match)
    base_states = baseline["cell_states"]
    curr_states = current["cell_states"]

    if set(base_states.keys()) != set(curr_states.keys()):
        messages.append(f"FAIL: Cell ID sets differ. "
                       f"Baseline has {len(base_states)} cells, current has {len(curr_states)}")
        passed = False
    else:
        state_mismatches = []
        for cell_id in base_states:
            if base_states[cell_id] != curr_states[cell_id]:
                state_mismatches.append(cell_id)

        if state_mismatches:
            n = len(state_mismatches)
            total = len(base_states)
            messages.append(f"FAIL: Cell state mismatch in {n}/{total} cells")
            # Show first few
            for cid in state_mismatches[:5]:
                messages.append(f"  Cell {cid}: baseline={base_states[cid]}, current={curr_states[cid]}")
            if n > 5:
                messages.append(f"  ... and {n - 5} more")
            passed = False
        else:
            messages.append(f"PASS: All {len(base_states)} cell states match exactly")

    # 2. Check fire arrival times (within tolerance)
    base_arrivals = baseline["arrival_times"]
    curr_arrivals = current["arrival_times"]

    arrival_diffs = []
    for cell_id in base_arrivals:
        if cell_id not in curr_arrivals:
            continue
        b_val = base_arrivals[cell_id]
        c_val = curr_arrivals[cell_id]

        # Both unburned
        if b_val == -999 and c_val == -999:
            continue

        # One burned, other didn't
        if (b_val == -999) != (c_val == -999):
            arrival_diffs.append((cell_id, b_val, c_val, float("inf")))
            continue

        # Both burned - check tolerance
        diff = abs(b_val - c_val)
        rel_diff = diff / (abs(b_val) + 1e-10) if b_val != 0 else diff
        if diff > atol and rel_diff > rtol:
            arrival_diffs.append((cell_id, b_val, c_val, diff))

    if arrival_diffs:
        n = len(arrival_diffs)
        total_burned = sum(1 for v in base_arrivals.values() if v != -999)
        messages.append(f"FAIL: Arrival time mismatch in {n} cells "
                       f"(of {total_burned} burned, atol={atol}, rtol={rtol})")
        for cid, bv, cv, d in arrival_diffs[:5]:
            messages.append(f"  Cell {cid}: baseline={bv:.6f}, current={cv:.6f}, diff={d:.6f}")
        if n > 5:
            messages.append(f"  ... and {n - 5} more")
        passed = False
    else:
        total_burned = sum(1 for v in base_arrivals.values() if v != -999)
        messages.append(f"PASS: All arrival times match within tolerance "
                       f"({total_burned} burned cells)")

    # 3. Check iteration count
    if baseline["num_iters"] != current["num_iters"]:
        messages.append(f"WARN: Iteration count differs: "
                       f"baseline={baseline['num_iters']}, current={current['num_iters']}")

    # 4. Report timing
    base_time = baseline["total_time_s"]
    curr_time = current["total_time_s"]
    speedup = base_time / curr_time if curr_time > 0 else float("inf")
    messages.append(f"Timing: baseline={base_time:.2f}s, current={curr_time:.2f}s, "
                   f"speedup={speedup:.2f}x")

    base_sim = baseline.get("sim_time_s", base_time)
    curr_sim = current.get("sim_time_s", curr_time)
    sim_speedup = base_sim / curr_sim if curr_sim > 0 else float("inf")
    messages.append(f"  Sim loop: baseline={base_sim:.2f}s, current={curr_sim:.2f}s, "
                   f"speedup={sim_speedup:.2f}x")

    return passed, messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EMBRS Performance Validation Harness"
    )
    parser.add_argument(
        "--generate-baseline",
        action="store_true",
        help="Generate baseline outputs (run on unmodified code first)"
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Run only the specified test case (e.g., tc01_small_flat_no_spot)"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for arrival time comparison (default: 1e-6)"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for arrival time comparison (default: 1e-6)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Discover test cases
    all_cases = discover_test_cases()
    if args.case:
        if args.case not in all_cases:
            print(f"Error: Test case '{args.case}' not found. Available: {all_cases}")
            return 1
        cases = [args.case]
    else:
        cases = all_cases

    print(f"{'=' * 70}")
    print(f"EMBRS Performance Validation Harness")
    print(f"{'=' * 70}")
    print(f"Mode: {'GENERATE BASELINE' if args.generate_baseline else 'VALIDATE'}")
    print(f"Test cases: {len(cases)}")
    if not args.generate_baseline:
        print(f"Tolerances: atol={args.atol}, rtol={args.rtol}")
    print(f"Random seed: {args.seed}")
    print(f"{'=' * 70}\n")

    results = {}
    overall_pass = True

    for case_name in cases:
        cfg_path = str(CONFIGS_DIR / f"{case_name}.cfg")
        print(f"\n{'─' * 60}")
        print(f"Test Case: {case_name}")
        print(f"Config: {cfg_path}")
        print(f"{'─' * 60}")

        if not os.path.exists(cfg_path):
            print(f"  SKIP: Config file not found")
            continue

        try:
            outputs, wall_time = run_simulation(cfg_path, seed=args.seed)
            print(f"  Completed in {wall_time:.2f}s "
                  f"({outputs['num_iters']} iters, "
                  f"{sum(1 for v in outputs['arrival_times'].values() if v != -999)} cells burned)")

        except Exception as e:
            print(f"  ERROR: Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            overall_pass = False
            results[case_name] = {"status": "ERROR", "error": str(e)}
            continue

        if args.generate_baseline:
            save_baseline(case_name, outputs)
            results[case_name] = {
                "status": "BASELINE_SAVED",
                "total_time_s": outputs["total_time_s"],
                "sim_time_s": outputs.get("sim_time_s", outputs["total_time_s"]),
                "num_iters": outputs["num_iters"],
            }
        else:
            baseline = load_baseline(case_name)
            if baseline is None:
                print(f"  SKIP: No baseline found. Run with --generate-baseline first.")
                results[case_name] = {"status": "NO_BASELINE"}
                continue

            passed, messages = validate_outputs(
                baseline, outputs,
                atol=args.atol, rtol=args.rtol
            )

            for msg in messages:
                print(f"  {msg}")

            status = "PASS" if passed else "FAIL"
            print(f"\n  Result: {status}")

            if not passed:
                overall_pass = False

            results[case_name] = {
                "status": status,
                "baseline_time_s": baseline["total_time_s"],
                "current_time_s": outputs["total_time_s"],
                "speedup": baseline["total_time_s"] / outputs["total_time_s"]
                    if outputs["total_time_s"] > 0 else float("inf"),
                "baseline_sim_time_s": baseline.get("sim_time_s", baseline["total_time_s"]),
                "current_sim_time_s": outputs.get("sim_time_s", outputs["total_time_s"]),
            }

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    if args.generate_baseline:
        print(f"\n{'Case':<40} {'Time (s)':>10} {'Iters':>8}")
        print(f"{'-' * 60}")
        for case_name, r in results.items():
            if r["status"] == "BASELINE_SAVED":
                print(f"{case_name:<40} {r['total_time_s']:>10.2f} {r['num_iters']:>8}")
        print(f"\nBaseline generation complete.")

    else:
        print(f"\n{'Case':<35} {'Status':>8} {'Base(s)':>8} {'Curr(s)':>8} {'Speedup':>8}")
        print(f"{'-' * 70}")

        speedups = []
        for case_name, r in results.items():
            if r["status"] in ("PASS", "FAIL"):
                status_str = r["status"]
                print(f"{case_name:<35} {status_str:>8} "
                      f"{r['baseline_time_s']:>8.2f} {r['current_time_s']:>8.2f} "
                      f"{r['speedup']:>7.2f}x")
                speedups.append(r["speedup"])
            else:
                print(f"{case_name:<35} {r['status']:>8}")

        if speedups:
            geo_mean = np.exp(np.mean(np.log(speedups)))
            print(f"\n{'Geometric mean speedup:':<45} {geo_mean:.2f}x")

        print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")

    # Save results JSON
    results_path = PERF_DIR / "last_results.json"
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: (vv if not isinstance(vv, float) or np.isfinite(vv) else str(vv))
                          for kk, vv in v.items()}

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
