"""Write a temporary ``.cfg`` and build a ``FireSim`` in-process.

Lifted from the proven plumbing in ``tuning/flame_sweep/sweep_one.py`` (spec §3,
§8). The cfg text is the contract with :func:`embrs.main.load_sim_params`, which
parses ``use_gsi``/``live_*_mf``/``init_mf`` and the no-conditioning behaviour
for us — so we reuse it verbatim rather than constructing ``SimParams`` by hand.

Heavy EMBRS imports are deferred to call time so importing this module stays
cheap (and import-safe under multiprocessing/spawn, spec §9).
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from embrs.scenario_weather.config import RunConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from embrs.fire_simulator.fire import FireSim


def write_cfg(
    cfg_path: str,
    *,
    map_dir: str,
    wxs_path: str,
    start_datetime: datetime,
    end_datetime: datetime,
    run_cfg: RunConfig,
    log_folder: Optional[str] = None,
    write_logs: bool = False,
) -> str:
    """Write a ``.cfg`` for a classifier/scenario run and return its path.

    No conditioning period is used (``conditioning_start == start_datetime``);
    fuel state comes entirely from ``run_cfg`` (spec §3). The cfg is emitted to
    ``cfg_path``; its parent directory is created if needed.

    Args:
        cfg_path: Where to write the ``.cfg``.
        map_dir: Folder containing ``map_params.pkl`` (the real scenario map
            with its real ignition region).
        wxs_path: Path to the weather ``.wxs`` file.
        start_datetime: Scenario start (also the conditioning start).
        end_datetime: Scenario end.
        run_cfg: Fixed run settings (moisture, seed, spotting, grid).
        log_folder: Log directory; only used when ``write_logs`` is True.
        write_logs: Whether EMBRS should write parquet logs (off for the
            in-process classifier — we read ``cell.I_ss`` directly).

    Returns:
        ``cfg_path``.
    """
    if end_datetime <= start_datetime:
        raise ValueError(
            f"end_datetime ({end_datetime}) must be after start_datetime "
            f"({start_datetime})"
        )

    os.makedirs(os.path.dirname(os.path.abspath(cfg_path)) or ".", exist_ok=True)
    log_folder = log_folder or os.path.join(
        os.path.dirname(os.path.abspath(cfg_path)), "logs"
    )
    init_mf = ", ".join(f"{v}" for v in run_cfg.init_mf)
    iso = "%Y-%m-%dT%H:%M:%S"

    text = f"""[Simulation]
log_folder = {log_folder}
t_step_s = {run_cfg.t_step_s}
cell_size_m = {run_cfg.cell_size_m}
visualize = False
num_runs = 1
write_logs = {write_logs}
model_spotting = {run_cfg.model_spotting}
seed = {run_cfg.seed}
[Weather]
input_type = File
file = {wxs_path}
mesh_resolution = {run_cfg.mesh_resolution}
conditioning_start = {start_datetime.strftime(iso)}
start_datetime = {start_datetime.strftime(iso)}
end_datetime = {end_datetime.strftime(iso)}
solar_source = {run_cfg.solar_source}
use_gsi = False
live_herb_mf = {run_cfg.live_herb_mf}
live_woody_mf = {run_cfg.live_woody_mf}
init_mf = {init_mf}
[Map]
folder = {map_dir}
"""
    with open(cfg_path, "w") as fh:
        fh.write(text)
    return cfg_path


def build_fire(cfg_path: str) -> "FireSim":
    """Load a ``.cfg`` and construct a :class:`FireSim` ready to step.

    Args:
        cfg_path: Path to a ``.cfg`` written by :func:`write_cfg`.

    Returns:
        An initialised ``FireSim`` (not yet stepped).
    """
    from embrs.main import load_sim_params
    from embrs.fire_simulator.fire import FireSim

    sim_params = load_sim_params(cfg_path)
    return FireSim(sim_params)
