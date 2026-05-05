"""Seed-determinism regression tests for EMBRS (no firefighting controller).

Two integration tests, both ``@pytest.mark.slow`` because each constructs
a FireSim from a real scenario cfg (which invokes WindNinja). They skip
gracefully if the required external assets (WindNinja CLI, map data,
weather file) are missing on this machine.

Phase 2 status: tests now exercise the seeded paths owned by Phase 2
(``FireSim._breach_rng``, ``Embers._rng``). The grid hash is computed
via ``hash_fire_grid`` from the firefighting test helper module.

Phases 3+ widen the coverage: predictor RNG, firefighting subsystems,
deterministic IDs. The same-seed/different-seed assertions here remain
valid as later phases land — they just exercise more state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The hash helper lives in the firefighting application's test package; we
# import it directly via path so this test does not depend on installing
# ra-cbba-core. If the path is unavailable, skip.
_RA_CBBA_REPO = Path("/Users/rui/Documents/Research/Code/ra-cbba-core")
_HELPERS_PATH = _RA_CBBA_REPO / "applications" / "firefighting" / "tests"
if str(_HELPERS_PATH) not in sys.path and _HELPERS_PATH.exists():
    sys.path.insert(0, str(_HELPERS_PATH))

try:
    from _determinism_helpers import hash_fire_grid, hash_prediction_output  # type: ignore
except Exception:  # pragma: no cover — skip below
    hash_fire_grid = None  # type: ignore
    hash_prediction_output = None  # type: ignore


# Cfgs whose external assets (map dir, weather file) we expect to be present
# in the typical EMBRS dev setup. The first one whose deps resolve wins.
_CANDIDATE_CFGS = [
    "fireline_construction.cfg",
    "burnout_demo.cfg",
]


def _locate_runnable_cfg() -> Path | None:
    """Pick a cfg whose external assets exist on this machine.

    Returns the cfg path or None if no candidate is fully resolvable.
    """
    # tests/integration/test_determinism.py -> tests/ -> embrs/ (the package)
    embrs_pkg = Path(__file__).resolve().parents[2]
    cfg_dir = embrs_pkg / "config_files"
    for name in _CANDIDATE_CFGS:
        cfg_path = cfg_dir / name
        if not cfg_path.exists():
            continue
        text = cfg_path.read_text(encoding="utf-8", errors="replace")
        ok = True
        for line in text.splitlines():
            stripped = line.strip()
            for key in ("folder", "file"):
                if stripped.startswith(f"{key} ="):
                    val = stripped.split("=", 1)[1].strip()
                    p = Path(val)
                    if key == "folder":
                        ok &= (p / "map_params.pkl").exists()
                    else:
                        ok &= p.exists() or val == ""
        if ok:
            return cfg_path
    return None


def _build_fire(cfg_path: Path, seed: int):
    """Load a cfg, override the seed, build a FireSim. Returns the FireSim."""
    from embrs.main import load_sim_params
    from embrs.fire_simulator.fire import FireSim

    sp = load_sim_params(str(cfg_path))
    sp.seed = seed
    return FireSim(sp)


def _run_n_iters(fire, n: int):
    """Iterate the fire up to n times (or until finished). Returns the fire."""
    for _ in range(n):
        if fire.finished:
            break
        fire.iterate()
    return fire


# Iteration cap. Even a few iterations exercises the breach/embers paths and
# all the pre-computed cell state; we don't need to run the full sim.
_N_ITERS = 5


@pytest.mark.slow
def test_same_seed_same_outcome():
    """Same master seed -> byte-identical fire grid hash across two runs."""
    if hash_fire_grid is None:
        pytest.skip("ra-cbba-core determinism helpers not on this machine")
    cfg = _locate_runnable_cfg()
    if cfg is None:
        pytest.skip("no EMBRS cfg with locally-available map+weather assets")

    fire_a = _build_fire(cfg, seed=42)
    _run_n_iters(fire_a, _N_ITERS)
    h_a = hash_fire_grid(fire_a)

    fire_b = _build_fire(cfg, seed=42)
    _run_n_iters(fire_b, _N_ITERS)
    h_b = hash_fire_grid(fire_b)

    assert h_a == h_b, (
        f"hash_fire_grid differs between two runs with the same seed.\n"
        f"  run A: {h_a}\n"
        f"  run B: {h_b}"
    )


def _build_state_estimate(fire):
    """Build a tiny StateEstimate for predictor.run() — uses the fire's
    current burning + burnt cells. No-op if there's nothing burning."""
    from embrs.utilities.data_classes import StateEstimate
    from embrs.utilities.fire_util import UtilFuncs
    burning = UtilFuncs.get_cell_polygons(fire._burning_cells)
    burnt = UtilFuncs.get_cell_polygons(fire._burnt_cells)
    return StateEstimate(burnt_polys=burnt, burning_polys=burning, start_time_s=None)


def _run_predictor(fire, time_horizon_hr: float = 0.5):
    """Construct a FirePredictor, run it once, return the PredictionOutput."""
    from embrs.tools.fire_predictor import FirePredictor
    from embrs.utilities.data_classes import PredictorParams

    params = PredictorParams(
        time_horizon_hr=time_horizon_hr,
        time_step_s=30,
        cell_size_m=fire.cell_size,
        dead_mf=0.08,
        live_mf=0.30,
        model_spotting=False,
    )
    pred = FirePredictor(params, fire)
    return pred.run(fire_estimate=_build_state_estimate(fire), visualize=False)


@pytest.mark.slow
def test_predictor_same_seed_same_outcome():
    """Same master seed -> identical PredictionOutput from FirePredictor.run().

    Exercises the seeded paths added in Phase 3:
      - FirePredictor._rng_breach (firebreak Bernoulli)
      - FirePredictor._rng_spot   (spot ignition Bernoulli)
      - FirePredictor._rng_wind   (perturbed weather seeds)
      - PerrymanSpotting._rng     (spawn off the embers SeedSequence)
    Single-process (no run_ensemble); the worker-path is exercised by the
    full-stack firefighting test in Phase 4+.
    """
    if hash_prediction_output is None:
        pytest.skip("ra-cbba-core determinism helpers not on this machine")
    cfg = _locate_runnable_cfg()
    if cfg is None:
        pytest.skip("no EMBRS cfg with locally-available map+weather assets")

    fire_a = _build_fire(cfg, seed=42)
    _run_n_iters(fire_a, _N_ITERS)
    out_a = _run_predictor(fire_a)
    h_a = hash_prediction_output(out_a)

    fire_b = _build_fire(cfg, seed=42)
    _run_n_iters(fire_b, _N_ITERS)
    out_b = _run_predictor(fire_b)
    h_b = hash_prediction_output(out_b)

    assert h_a == h_b, (
        f"hash_prediction_output differs between two predictor runs with the same seed.\n"
        f"  run A: {h_a}\n"
        f"  run B: {h_b}"
    )


@pytest.mark.slow
def test_predictor_different_seed_different_outcome():
    """Different master seeds -> different per-stream RNG state in the predictor.

    Same sanity-check pattern as test_different_seed_changes_rng_state, but
    on the predictor's owned generators.
    """
    if hash_prediction_output is None:
        pytest.skip("ra-cbba-core determinism helpers not on this machine")
    cfg = _locate_runnable_cfg()
    if cfg is None:
        pytest.skip("no EMBRS cfg with locally-available map+weather assets")

    from embrs.tools.fire_predictor import FirePredictor
    from embrs.utilities.data_classes import PredictorParams

    fire_a = _build_fire(cfg, seed=42)
    fire_b = _build_fire(cfg, seed=42)
    fire_c = _build_fire(cfg, seed=99)

    params = PredictorParams(
        time_horizon_hr=0.5, time_step_s=30, cell_size_m=fire_a.cell_size,
        dead_mf=0.08, live_mf=0.30, model_spotting=False,
    )
    pred_a = FirePredictor(params, fire_a)
    pred_b = FirePredictor(params, fire_b)
    pred_c = FirePredictor(params, fire_c)

    for name in ("_rng_breach", "_rng_spot", "_rng_wind"):
        a = float(getattr(pred_a, name).random())
        b = float(getattr(pred_b, name).random())
        c = float(getattr(pred_c, name).random())
        assert a == b, f"same seed produced different {name} draws"
        assert a != c, f"different seeds produced identical {name} draws"


@pytest.mark.slow
def test_predictor_run_ensemble_n2_deterministic():
    """Same master seed -> identical EnsemblePredictionOutput from run_ensemble (N=2).

    Closes the Phase 3 gap: the ProcessPoolExecutor / spawn-context worker
    contract was correct by inspection but not exercised by a passing test.
    Forces the multi-process path with two ensemble members and asserts
    byte-equal hashes of each member's PredictionOutput across two runs.
    """
    if hash_prediction_output is None:
        pytest.skip("ra-cbba-core determinism helpers not on this machine")
    cfg = _locate_runnable_cfg()
    if cfg is None:
        pytest.skip("no EMBRS cfg with locally-available map+weather assets")

    from embrs.tools.fire_predictor import FirePredictor
    from embrs.utilities.data_classes import PredictorParams

    def _run(seed: int):
        fire = _build_fire(cfg, seed=seed)
        _run_n_iters(fire, _N_ITERS)
        params = PredictorParams(
            time_horizon_hr=0.25, time_step_s=30, cell_size_m=fire.cell_size,
            dead_mf=0.08, live_mf=0.30, model_spotting=False,
        )
        pred = FirePredictor(params, fire)
        # Two state estimates from the current fire — forces N=2 workers
        # and exercises the spawn-context ProcessPoolExecutor path.
        state_a = _build_state_estimate(fire)
        state_b = _build_state_estimate(fire)
        ensemble_out = pred.run_ensemble(
            state_estimates=[state_a, state_b],
            return_individual=True,
            num_workers=2,
        )
        # Hash each individual prediction in submission order. Phase 3
        # guaranteed predictions are stored at predictions[member_idx], so
        # the list order is deterministic regardless of worker completion.
        return [hash_prediction_output(p) for p in ensemble_out.individual_predictions]

    h_a = _run(seed=42)
    h_b = _run(seed=42)
    assert h_a == h_b, (
        f"run_ensemble N=2 produced different hashes across same-seed runs.\n"
        f"  run A: {h_a}\n  run B: {h_b}"
    )


@pytest.mark.slow
def test_different_seed_changes_rng_state():
    """Different master seeds -> different seeded RNG state.

    Sanity check that the seed actually plumbs through. We can't always rely
    on the breach/embers paths to *fire* during a short scenario run, so we
    inspect the seeded Generators directly: same seed -> same first draw,
    different seed -> different first draw.
    """
    if hash_fire_grid is None:
        pytest.skip("ra-cbba-core determinism helpers not on this machine")
    cfg = _locate_runnable_cfg()
    if cfg is None:
        pytest.skip("no EMBRS cfg with locally-available map+weather assets")

    fire_42a = _build_fire(cfg, seed=42)
    fire_42b = _build_fire(cfg, seed=42)
    fire_99 = _build_fire(cfg, seed=99)

    a = float(fire_42a._breach_rng.random())
    b = float(fire_42b._breach_rng.random())
    c = float(fire_99._breach_rng.random())
    assert a == b, "same seed produced different breach RNG draws"
    assert a != c, "different seeds produced identical breach RNG draws"

    g_42a = fire_42a.child_generator("embrs.embers")
    g_42b = fire_42b.child_generator("embrs.embers")
    g_99 = fire_99.child_generator("embrs.embers")
    assert float(g_42a.random()) == float(g_42b.random())
    assert float(g_42a.random()) != float(g_99.random())
