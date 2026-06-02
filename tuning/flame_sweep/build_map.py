"""Clone a thesis map into a sweep map with a single POINT ignition.

The ignition is placed at the original map's ignition centroid (known-burnable
with clearance). Only scenario_data.initial_ign changes; fuel/terrain are
untouched. Usage:

    python build_map.py <src_map_dir> <dst_map_dir> [x y]
"""
import os, pickle, shutil, sys
from shapely.geometry import Point


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    mp = pickle.load(open(os.path.join(src, "map_params.pkl"), "rb"))
    if len(sys.argv) >= 5:
        x, y = float(sys.argv[3]), float(sys.argv[4])
    else:
        c = mp.scenario_data.initial_ign[0].centroid
        x, y = c.x, c.y
    mp.scenario_data.initial_ign = [Point(x, y)]

    os.makedirs(dst, exist_ok=True)
    # Copy the npy/json inputs so the cloned map_params (which may rebase to
    # its own dir) resolves them locally.
    for f in os.listdir(src):
        if f == "map_params.pkl":
            continue
        s = os.path.join(src, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, f))
    with open(os.path.join(dst, "map_params.pkl"), "wb") as fh:
        pickle.dump(mp, fh)
    print(f"sweep map written to {dst} with POINT ignition at ({x:.0f}, {y:.0f})")


if __name__ == "__main__":
    main()
