"""Shift a map's initial ignition polygon(s) in place, portably.

Updates both ``map_params.pkl`` (the actual sim input) and the generator's
sidecar ``<folder>.json`` (visualization/metadata), and scrubs machine-specific
absolute paths so the folder loads identically on any machine via
``MapParams.load()``.

Coordinate convention (matches grid_manager.get_cell_from_xy):
    (0,0) is the lower-left corner, x increases right, y increases up.
    So "up"   -> y_off > 0,  "down"  -> y_off < 0
       "right"-> x_off > 0,  "left"  -> x_off < 0

Example: move 1000 m up and 500 m left:
    python shift_ignition.py <map_folder> -500 1000
"""

import os
import sys
import json
import pickle
import shutil
import argparse

from shapely.affinity import translate
from shapely.geometry import shape, mapping

# Imported so pickle can resolve the dataclasses when unpickling.
from embrs.utilities.data_classes import (  # noqa: F401
    MapParams, MapDrawerData, GeoInfo, LandscapeData,
)


def _centroids(geoms):
    return [tuple(round(c, 1) for c in g.centroid.coords[0]) for g in geoms]


def shift_map_ignition(folder, x_off, y_off, backup=True):
    folder = os.path.abspath(folder)
    pkl_path = os.path.join(folder, "map_params.pkl")
    json_path = os.path.join(folder, os.path.basename(folder) + ".json")

    # ---- 1. pickle: shift polygons, then scrub machine-specific paths -------
    mp = MapParams.load(folder)  # rebases folder / cropped_lcp_path on load
    before = _centroids(mp.scenario_data.initial_ign)
    mp.scenario_data.initial_ign = [
        translate(g, xoff=x_off, yoff=y_off) for g in mp.scenario_data.initial_ign
    ]
    after = _centroids(mp.scenario_data.initial_ign)

    # load() unconditionally overwrites these every load, so storing this
    # machine's abspath is pointless and leaks usernames. Store portable forms.
    mp.folder = os.path.basename(folder)
    mp.cropped_lcp_path = "cropped_lcp.tif"
    if mp.lcp_filepath:
        mp.lcp_filepath = os.path.basename(mp.lcp_filepath)

    if backup:
        shutil.copy2(pkl_path, pkl_path + ".bak")
    with open(pkl_path, "wb") as f:
        pickle.dump(mp, f)

    # ---- 2. sidecar JSON: shift ignition + relativize abs file paths --------
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)

        if data.get("initial_igntion"):  # note: generator's spelling
            data["initial_igntion"] = [
                mapping(translate(shape(g), xoff=x_off, yoff=y_off))
                for g in data["initial_igntion"]
            ]

        li = data.get("landscape_info") or {}
        for k, v in list(li.items()):
            if isinstance(v, str) and os.path.isabs(v):
                li[k] = os.path.basename(v)
        roads = data.get("roads")
        if isinstance(roads, dict) and isinstance(roads.get("file"), str):
            roads["file"] = os.path.basename(roads["file"])

        if backup:
            shutil.copy2(json_path, json_path + ".bak")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
    else:
        print(f"(no sidecar JSON at {json_path}; skipped)")

    print(f"folder : {folder}")
    print(f"shift  : x {x_off:+g} m, y {y_off:+g} m")
    for b, a in zip(before, after):
        print(f"  ignition centroid {b} -> {a}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="map folder containing map_params.pkl")
    ap.add_argument("x_off", type=float, help="meters: +right / -left")
    ap.add_argument("y_off", type=float, help="meters: +up / -down")
    ap.add_argument("--no-backup", action="store_true",
                    help="do not write .bak copies before overwriting")
    args = ap.parse_args()
    shift_map_ignition(args.folder, args.x_off, args.y_off, backup=not args.no_backup)
