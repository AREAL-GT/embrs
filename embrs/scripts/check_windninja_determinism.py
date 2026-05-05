"""Pre-flight check: is WindNinja deterministic given byte-identical inputs?

The seed-determinism plan assumes WindNinja, when given identical weather
and terrain inputs, produces identical output bytes. This script verifies
that assumption by running ``run_windninja`` twice with the same inputs
and comparing SHA-256 hashes of the resulting forecast arrays.

Exit codes:
    0  PASS — outputs are byte-identical, WindNinja is in-contract.
    1  FAIL — outputs differ; the script prints diff statistics.
    2  COULD-NOT-RUN — environment problem (missing CLI, bad cfg, etc.).
       Treat the determinism question as unresolved and flag for follow-up.

Usage:
    python -m embrs.scripts.check_windninja_determinism --config <path.cfg>

If --config is omitted, defaults to ``embrs/config_files/example.cfg``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_COULD_NOT_RUN = 2


def _hash_array(arr) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _diff_stats(a, b) -> str:
    import numpy as np

    if a.shape != b.shape:
        return f"shape mismatch: {a.shape} vs {b.shape}"
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    n_diff = int((diff > 0).sum())
    return (
        f"  shape={a.shape}\n"
        f"  cells_differing={n_diff}/{a.size}\n"
        f"  max_abs_diff={float(diff.max()):.6e}\n"
        f"  mean_abs_diff={float(diff.mean()):.6e}\n"
        f"  diff_at_p99={float(np.quantile(diff, 0.99)):.6e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to .cfg file (default: embrs/config_files/example.cfg)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not clean up the temp dirs after running (for debugging).",
    )
    args = parser.parse_args()

    cfg_path = args.config
    if cfg_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        cfg_path = repo_root / "embrs" / "config_files" / "example.cfg"
    cfg_path = Path(cfg_path)

    if not cfg_path.exists():
        print(f"[COULD-NOT-RUN] config not found: {cfg_path}", file=sys.stderr)
        return EXIT_COULD_NOT_RUN

    try:
        from embrs.main import load_sim_params
        from embrs.models.wind_forecast import run_windninja
        from embrs.models.weather import WeatherStream
    except Exception as e:
        print(f"[COULD-NOT-RUN] import error: {e}", file=sys.stderr)
        traceback.print_exc()
        return EXIT_COULD_NOT_RUN

    try:
        sim_params = load_sim_params(str(cfg_path))
    except Exception as e:
        print(f"[COULD-NOT-RUN] could not load sim params: {e}", file=sys.stderr)
        return EXIT_COULD_NOT_RUN

    if sim_params.weather_input is None or sim_params.map_params is None:
        print("[COULD-NOT-RUN] cfg lacks weather_input or map_params", file=sys.stderr)
        return EXIT_COULD_NOT_RUN

    try:
        weather = WeatherStream(sim_params)
    except Exception as e:
        print(f"[COULD-NOT-RUN] could not build WeatherStream: {e}", file=sys.stderr)
        traceback.print_exc()
        return EXIT_COULD_NOT_RUN

    tmp_a = tempfile.mkdtemp(prefix="wn_det_a_")
    tmp_b = tempfile.mkdtemp(prefix="wn_det_b_")
    try:
        print(f"Running WindNinja pass A in {tmp_a} ...")
        try:
            arr_a = run_windninja(weather, sim_params.map_params, custom_temp_dir=tmp_a, num_workers=1)
        except FileNotFoundError as e:
            print(f"[COULD-NOT-RUN] WindNinja CLI not found: {e}", file=sys.stderr)
            return EXIT_COULD_NOT_RUN
        except Exception as e:
            print(f"[COULD-NOT-RUN] pass A failed: {e}", file=sys.stderr)
            traceback.print_exc()
            return EXIT_COULD_NOT_RUN

        print(f"Running WindNinja pass B in {tmp_b} ...")
        try:
            arr_b = run_windninja(weather, sim_params.map_params, custom_temp_dir=tmp_b, num_workers=1)
        except Exception as e:
            print(f"[COULD-NOT-RUN] pass B failed: {e}", file=sys.stderr)
            traceback.print_exc()
            return EXIT_COULD_NOT_RUN

        h_a = _hash_array(arr_a)
        h_b = _hash_array(arr_b)
        print(f"  pass A sha256 = {h_a}")
        print(f"  pass B sha256 = {h_b}")

        if h_a == h_b:
            print("[PASS] WindNinja produced byte-identical output across two runs.")
            return EXIT_PASS

        print("[FAIL] WindNinja output differs across runs:", file=sys.stderr)
        print(_diff_stats(arr_a, arr_b), file=sys.stderr)
        return EXIT_FAIL
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp_a, ignore_errors=True)
            shutil.rmtree(tmp_b, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
