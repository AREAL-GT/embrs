#!/usr/bin/env python3
"""Assemble the thesis-evaluation config set from the weather-tuning outputs.

For each region x weather-class, the authoritative "final" candidate is the iter
whose peak-scale matches ``final_peak_scale_ms`` in that class's
``tuning_result.json``. This script copies the selected ``.cfg``/``.wxs`` into a
clean, PACE-portable tree under ``embrs/configs/thesis_eval/`` and emits, per
cell, a Mac variant (``<class>.cfg``) and a PACE variant (``<class>_pace.cfg``)
that differ only in their absolute path roots. The ``.wxs`` is committed once and
travels with the repo, so the PACE cfg's weather path resolves automatically.

Re-run this freely after re-tuning a cell (e.g. b_front_range/mild) — it is
idempotent and rewrites the destination tree from scratch.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path("/Users/rjdp3/Documents/Research/embrs")
SRC = REPO / "embrs/scenario_weather/outputs"
DST = REPO / "embrs/configs/thesis_eval"

REGIONS = ["a_flint_hills", "b_front_range_foothills", "c_clearwater"]
CLASSES = ["mild", "moderate", "extreme"]

# --- Mac (local) absolute roots ------------------------------------------------
MAC_LOG = "/Users/rjdp3/Documents/Research/embrs_logs/thesis_eval_logs"
MAC_MAP = "/Users/rjdp3/Documents/Research/embrs_map/thesis_eval_maps"
MAC_FC = "/Users/rjdp3/Documents/Research/ra-cbba-core/applications/firefighting/fire_coordinator.py"
MAC_WX_ROOT = str(DST)  # wxs lives beside the cfg

# --- PACE (Georgia Tech Phoenix) absolute roots --------------------------------
PACE_HOME = "/storage/home/hcoda1/0/rjdp3"
PACE_RUI = f"{PACE_HOME}/r-jrogers8-0/rui"
PACE_LOG = f"{PACE_HOME}/scratch/embrs_logs/thesis_eval"          # write-heavy -> scratch
PACE_MAP = f"{PACE_RUI}/embrs_data/maps/thesis_eval"             # NOT yet transferred; rsync here
PACE_FC = f"{PACE_RUI}/ra-cbba-core/applications/firefighting/fire_coordinator.py"
PACE_WX_ROOT = f"{PACE_RUI}/embrs/embrs/configs/thesis_eval"      # wxs travels with the embrs repo


def selected_iter(region: str, fire_class: str) -> tuple[Path, dict]:
    """Return (source .cfg path, tuning_result dict) for the selected candidate."""
    cell = SRC / region / fire_class
    result = json.loads((cell / "tuning_result.json").read_text())
    ps = result["final_peak_scale_ms"]
    matches = sorted(cell.glob(f"*_ps{ps:.3f}.cfg"))
    if len(matches) != 1:
        raise RuntimeError(
            f"{region}/{fire_class}: expected exactly one cfg matching "
            f"ps{ps:.3f}, found {[m.name for m in matches]}"
        )
    return matches[0], result


def rewrite_cfg(src_lines: list[str], region: str, fire_class: str, *, pace: bool) -> str:
    """Rewrite the path roots in a source cfg and wire in the FireCoordinator."""
    log_root = PACE_LOG if pace else MAC_LOG
    map_root = PACE_MAP if pace else MAC_MAP
    wx_root = PACE_WX_ROOT if pace else MAC_WX_ROOT
    fc_path = PACE_FC if pace else MAC_FC

    out: list[str] = []
    injected_controller = False
    for line in src_lines:
        stripped = line.strip()
        if stripped.startswith("log_folder"):
            out.append(f"log_folder = {log_root}/{region}/{fire_class}\n")
            continue
        if stripped.startswith("file") and "=" in stripped:
            out.append(f"file = {wx_root}/{region}/{fire_class}.wxs\n")
            continue
        if stripped.startswith("folder") and "=" in stripped:
            out.append(f"folder = {map_root}/{region}\n")
            continue
        # Insert the controller hookup at the end of [Simulation], just before
        # the [Weather] header. The harness strips this for baseline runs and
        # supplies the FireCoordinator .toml via the BACKBURN_CFG env var.
        if stripped.startswith("[Weather]") and not injected_controller:
            out.append(f"user_path = {fc_path}\n")
            out.append("user_class = FireCoordinator\n")
            injected_controller = True
        out.append(line)
    return "".join(out)


def main() -> None:
    # Clear any prior top-level cfgs (the old flat layout) and region dirs.
    for old in DST.glob("*.cfg"):
        old.unlink()
    for region in REGIONS:
        rdir = DST / region
        if rdir.exists():
            shutil.rmtree(rdir)

    provenance: list[dict] = []
    for region in REGIONS:
        (DST / region).mkdir(parents=True, exist_ok=True)
        for fire_class in CLASSES:
            src_cfg, result = selected_iter(region, fire_class)
            src_wxs = src_cfg.with_suffix(".wxs")
            src_lines = src_cfg.read_text().splitlines(keepends=True)

            shutil.copyfile(src_wxs, DST / region / f"{fire_class}.wxs")
            (DST / region / f"{fire_class}.cfg").write_text(
                rewrite_cfg(src_lines, region, fire_class, pace=False)
            )
            (DST / region / f"{fire_class}_pace.cfg").write_text(
                rewrite_cfg(src_lines, region, fire_class, pace=True)
            )

            provenance.append(
                dict(
                    region=region,
                    fire_class=fire_class,
                    source=src_cfg.name,
                    target_ft=result["target_ft"],
                    flame_ft=result["final_flame_ft"],
                    peak_scale_ms=result["final_peak_scale_ms"],
                    converged=result["converged"],
                )
            )

    write_readme(provenance)
    print(f"Assembled {len(provenance)} cells into {DST}")


def write_readme(rows: list[dict]) -> None:
    lines = [
        "# Thesis-evaluation configs\n",
        "\n",
        "Final per-region x weather-class EMBRS configs + tuned weather streams for the\n",
        "thesis firefighting evaluation. **Generated** by `assemble_thesis_eval.py` from\n",
        "`embrs/scenario_weather/outputs/<region>/<class>/` — re-run that script after\n",
        "re-tuning a cell; do not hand-edit selections here.\n",
        "\n",
        "## Layout\n",
        "```\n",
        "thesis_eval/<region>/\n",
        "    <class>.cfg        # Mac paths (local runs)\n",
        "    <class>_pace.cfg   # PACE /storage paths (cluster runs via the sweep harness)\n",
        "    <class>.wxs        # tuned weather stream (committed; travels to PACE via git)\n",
        "```\n",
        "`<region>` in {a_flint_hills, b_front_range_foothills, c_clearwater}; "
        "`<class>` in {mild, moderate, extreme}.\n",
        "\n",
        "## PACE notes\n",
        "- `.cfg` paths are absolute and NOT `~`-expanded. The `_pace.cfg` roots are:\n",
        f"  weather `{PACE_WX_ROOT}` (in-repo, auto via git), "
        f"map `{PACE_MAP}/<region>`, logs `{PACE_LOG}` (scratch).\n",
        "- **Maps are not yet on PACE.** `rsync` each map folder to "
        f"`{PACE_MAP}/<region>` keeping the folder name unchanged (MapParams.load rebases\n",
        "  the baked absolute paths onto the load location, but the logger still locates\n",
        "  map metadata as `<folder>/<foldername>.json`).\n",
        "- The FireCoordinator is wired via `user_path`/`user_class`; the sweep harness\n",
        "  supplies the strategy `.toml` (one per region) via the `BACKBURN_CFG` env var\n",
        "  and strips the controller for no-suppression baselines.\n",
        "- Strategy tomls live in "
        "`ra-cbba-core/applications/firefighting/configs/thesis_eval/<region>.toml`.\n",
        "\n",
        "## Provenance\n",
        "| Region | Class | Target (ft) | Flame (ft) | peak_scale (m/s) | Converged | Source |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for r in rows:
        conv = "yes" if r["converged"] else "**NO**"
        lines.append(
            f"| {r['region']} | {r['fire_class']} | {r['target_ft']:.1f} | "
            f"{r['flame_ft']:.2f} | {r['peak_scale_ms']:.3f} | {conv} | `{r['source']}` |\n"
        )
    (DST / "README.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
