"""
Profiling script for EMBRS + Backburn simulation.

Captures two layers of data:
  1. cProfile full call graph (saved as .prof for snakeviz/gprof2dot)
  2. Targeted wall-clock instrumentation on key functions (saved as JSON)

Usage:
    python overnight_profile.py

    # Option A: with py-spy (best, but needs sudo on macOS):
    sudo py-spy record --subprocesses --format speedscope \
        --output pyspy_profile.json -- python overnight_profile.py

    # Option B: standalone (no sudo needed):
    python overnight_profile.py

Analysis after the run:
    # Interactive flame graph from cProfile data:
    pip install snakeviz
    snakeviz profile_output.prof

    # Or generate a dot graph:
    pip install gprof2dot
    gprof2dot -f pstats profile_output.prof | dot -Tpng -o profile_graph.png

    # Read the timing breakdown:
    cat timing_breakdown.json | python -m json.tool

    # If you used py-spy, open in browser:
    # Go to https://www.speedscope.app and load pyspy_profile.json
"""

import cProfile
import pstats
import time
import sys
import os
import json
import functools
import atexit
from collections import defaultdict

# ============================================================================
# Layer 2: Targeted wall-clock instrumentation
# ============================================================================

_timing_data = defaultdict(lambda: {"calls": 0, "total_s": 0.0, "max_s": 0.0, "min_s": float('inf'), "history": []})
_MAX_HISTORY = 200  # Keep per-call times for the first N calls per function

def _instrument(cls, method_name, label=None):
    """Monkey-patch a method with wall-clock timing."""
    if not hasattr(cls, method_name):
        return
    orig = getattr(cls, method_name)
    tag = label or f"{cls.__name__}.{method_name}"

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        d = _timing_data[tag]
        d["calls"] += 1
        d["total_s"] += elapsed
        d["max_s"] = max(d["max_s"], elapsed)
        d["min_s"] = min(d["min_s"], elapsed)
        if d["calls"] <= _MAX_HISTORY:
            d["history"].append(round(elapsed, 4))
        return result

    setattr(cls, method_name, wrapper)


def _save_timing_report():
    """Save timing data to JSON on exit."""
    out = {}
    total = sum(v["total_s"] for v in _timing_data.values())
    for name, d in sorted(_timing_data.items(), key=lambda x: -x[1]["total_s"]):
        avg = d["total_s"] / d["calls"] if d["calls"] else 0
        pct = (d["total_s"] / total * 100) if total > 0 else 0
        out[name] = {
            "calls": d["calls"],
            "total_s": round(d["total_s"], 3),
            "avg_s": round(avg, 3),
            "max_s": round(d["max_s"], 3),
            "min_s": round(d["min_s"], 3) if d["min_s"] != float('inf') else None,
            "pct_of_measured": round(pct, 1),
            "first_N_call_times": d["history"],
        }

    report = {
        "total_measured_time_s": round(total, 2),
        "functions": out,
    }

    path = os.path.join(os.path.dirname(__file__), "timing_breakdown_3.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTiming breakdown saved to {path}")


# ============================================================================
# Patch key functions BEFORE they get imported by the simulation
# ============================================================================

sys.path.insert(0, "/Users/rjdp3/Documents/Research/embrs")
sys.path.insert(0, "/Users/rjdp3/Documents/Research/ra-cbba-core")

# --- embrs core ---
from embrs.fire_simulator.fire import FireSim
_instrument(FireSim, "iterate")

from embrs.tools.fire_predictor import FirePredictor
_instrument(FirePredictor, "run_ensemble")
_instrument(FirePredictor, "run")
_instrument(FirePredictor, "generate_forecast_pool")
_instrument(FirePredictor, "prepare_for_serialization")

try:
    from embrs.tools.forecast_pool import ForecastPoolManager
    _instrument(ForecastPoolManager, "generate_forecasts")
except Exception:
    pass

# --- ra-cbba-core backburn ---
try:
    from applications.backburn.backburn import Backburn
    _instrument(Backburn, "__init__")
    _instrument(Backburn, "process_state")
except Exception as e:
    print(f"Warning: could not patch Backburn: {e}")

try:
    from applications.backburn.fire_prediction_manager import FirePredictionManager
    _instrument(FirePredictionManager, "run_prediction")
    _instrument(FirePredictionManager, "rollout_task")
except Exception:
    pass

try:
    from applications.backburn.task_generator import TaskGenerator
    _instrument(TaskGenerator, "generate_tasks")
    _instrument(TaskGenerator, "rollout_proposed_tasks")
    _instrument(TaskGenerator, "find_time_windows_for_seg")
    _instrument(TaskGenerator, "find_time_windows_for_seg_probabilistic")
    _instrument(TaskGenerator, "evaluate_task_prediction")
    _instrument(TaskGenerator, "time_suitable")
    _instrument(TaskGenerator, "backburn_wind_check")
except Exception:
    pass

try:
    from applications.backburn.surveillance import Surveillance
    _instrument(Surveillance, "update")
    _instrument(Surveillance, "update_and_create_tasks")
    _instrument(Surveillance, "sample_state_estimates")
    _instrument(Surveillance, "set_initial_state")
except Exception:
    pass

try:
    from applications.backburn.task_dispatcher import TaskDispatcher
    _instrument(TaskDispatcher, "dispatch_tasks")
    _instrument(TaskDispatcher, "tick_cbba")
except Exception:
    pass

try:
    from applications.backburn.action_commit_handler import ActionCommitHandler
    _instrument(ActionCommitHandler, "commit_pending_actions")
except Exception:
    pass

try:
    from applications.backburn.containment_geometry import ContainmentGeometryManager
    _instrument(ContainmentGeometryManager, "get_relevant_segments")
    _instrument(ContainmentGeometryManager, "get_backburn_cells")
except Exception:
    pass

# --- ra-cbba-core CBBA ---
try:
    from cbba.cbba import CBBAEngine
    _instrument(CBBAEngine, "tick")
    _instrument(CBBAEngine, "build_bundle")
except Exception:
    pass

try:
    from cbba.agents.resource_aware_cost_model import ResourceAwareCostModel
    _instrument(ResourceAwareCostModel, "get_marginal_scores")
    _instrument(ResourceAwareCostModel, "reactive_score")
    _instrument(ResourceAwareCostModel, "calc_path_score")
    _instrument(ResourceAwareCostModel, "find_best_recharge_for_path")
except Exception:
    pass

try:
    from cbba.scheduler import CBBAScheduler
    _instrument(CBBAScheduler, "tick")
except Exception:
    pass

# Guard against forked worker processes inheriting atexit/signal handlers.
# multiprocessing.parent_process() returns None only in the original main process.
import multiprocessing

def _is_main_process():
    return multiprocessing.parent_process() is None

def _save_if_main():
    if _is_main_process():
        _save_timing_report()

atexit.register(_save_if_main)

import signal
def _sigint_handler(signum, frame):
    if _is_main_process():
        print("\n\nInterrupted — saving timing data before exit...")
        _save_timing_report()
    sys.exit(0)
signal.signal(signal.SIGINT, _sigint_handler)

# ============================================================================
# Run simulation under cProfile
# ============================================================================

from embrs.main import load_sim_params, sim_loop

def run():
    cfg_path = "/Users/rjdp3/Documents/Research/embrs/embrs/configs/scenario_1.cfg"
    sim_params = load_sim_params(cfg_path)
    sim_loop(sim_params)


if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)

    wall_start = time.perf_counter()

    profiler = cProfile.Profile()
    profiler.enable()

    run()

    profiler.disable()

    wall_total = time.perf_counter() - wall_start

    # Save binary profile (for snakeviz / gprof2dot)
    prof_path = os.path.join(out_dir, "profile_output_3.prof")
    profiler.dump_stats(prof_path)
    print(f"\ncProfile data saved to {prof_path}")

    # Save text summary
    txt_path = os.path.join(out_dir, "profile_summary_3.txt")
    with open(txt_path, "w") as f:
        f.write(f"Total wall-clock time: {wall_total:.1f}s ({wall_total/60:.1f} min)\n\n")

        f.write("=" * 100 + "\n")
        f.write("TOP 100 BY CUMULATIVE TIME\n")
        f.write("=" * 100 + "\n")
        s = pstats.Stats(profiler, stream=f)
        s.strip_dirs().sort_stats("cumulative").print_stats(100)

        f.write("\n" + "=" * 100 + "\n")
        f.write("TOP 100 BY SELF TIME\n")
        f.write("=" * 100 + "\n")
        s2 = pstats.Stats(profiler, stream=f)
        s2.strip_dirs().sort_stats("tottime").print_stats(100)

        f.write("\n" + "=" * 100 + "\n")
        f.write("BACKBURN / CBBA FUNCTIONS\n")
        f.write("=" * 100 + "\n")
        s3 = pstats.Stats(profiler, stream=f)
        s3.strip_dirs().sort_stats("cumulative").print_stats(
            "backburn|cbba|cost_model|surveillance|task_gen|dispatcher|containment", 200
        )

        f.write("\n" + "=" * 100 + "\n")
        f.write("EMBRS FUNCTIONS\n")
        f.write("=" * 100 + "\n")
        s4 = pstats.Stats(profiler, stream=f)
        s4.strip_dirs().sort_stats("cumulative").print_stats(
            "fire|predictor|rothermel|crown|grid|weather|cell|forecast", 200
        )

        f.write("\n" + "=" * 100 + "\n")
        f.write("CALLERS (top 60 by cumulative)\n")
        f.write("=" * 100 + "\n")
        s5 = pstats.Stats(profiler, stream=f)
        s5.strip_dirs().sort_stats("cumulative").print_callers(60)

    print(f"cProfile text summary saved to {txt_path}")
    print(f"\nTotal wall-clock time: {wall_total:.1f}s ({wall_total/60:.1f} min)")
    print(f"\nTo visualize:  snakeviz {prof_path}")
