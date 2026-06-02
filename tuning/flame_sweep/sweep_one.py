"""Run ONE controlled fire-behavior point and print a JSON result.

Constant weather + pinned fuel moisture (use_gsi=False) + point ignition, so
head fireline intensity / flame length / spread become a clean function of the
swept driver (wind). Prints JSON: {wind_mph, rh, bi, flame_len_ft, head_I_kW_m,
ros_m_min, burned_cells, burned_area_m2}.

Usage: python sweep_one.py --wind 20 --rh 25 --temp 85 --hours 6 --map <dir> --workdir <dir>
"""
import argparse, json, os, tempfile, datetime as dt
import numpy as np


def write_constant_wxs(path, temp_F, rh, wind_mph, wind_dir, cond_start, end, step_h=1):
    rows = ["RAWS_UNITS: English", "RAWS_ELEVATION: 1337", "RAWS: 1",
            "Year  Mth  Day   Time    Temp     RH  HrlyPcp  WindSpd WindDir CloudCov"]
    t = cond_start
    while t <= end:
        rows.append(f"{t.year:4d}  {t.month:<3d}  {t.day:<3d}  {t.hour:02d}00    "
                    f"{temp_F:5.1f}    {int(rh):3d}    0.00      {wind_mph:.1f}     "
                    f"{int(wind_dir):3d}     {0:3d}")
        t += dt.timedelta(hours=step_h)
    open(path, "w").write("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wind", type=float, required=True)
    ap.add_argument("--rh", type=float, default=25)
    ap.add_argument("--temp", type=float, default=85)
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--wind-dir", type=float, default=180)   # from south -> head north
    ap.add_argument("--live-herb", type=float, default=30)   # cured grass
    ap.add_argument("--live-woody", type=float, default=60)
    ap.add_argument("--map", required=True)
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    tag = f"w{a.wind:.0f}_rh{a.rh:.0f}"
    wxs = os.path.join(a.workdir, f"{tag}.wxs")

    cond_start = dt.datetime(2022, 7, 1, 0)
    start = dt.datetime(2022, 7, 8, 0)
    end = start + dt.timedelta(hours=int(a.hours))
    write_constant_wxs(wxs, a.temp, a.rh, a.wind, a.wind_dir, cond_start,
                       end + dt.timedelta(hours=2))

    cfg = os.path.join(a.workdir, f"{tag}.cfg")
    open(cfg, "w").write(f"""[Simulation]
log_folder = {a.workdir}/logs
t_step_s = 30
cell_size_m = 30
visualize = False
num_runs = 1
write_logs = False
model_spotting = False
seed = 42
[Weather]
input_type = File
file = {wxs}
mesh_resolution = 250
conditioning_start = {cond_start.strftime('%Y-%m-%dT%H:%M:%S')}
start_datetime = {start.strftime('%Y-%m-%dT%H:%M:%S')}
end_datetime = {end.strftime('%Y-%m-%dT%H:%M:%S')}
solar_source = offline
use_gsi = False
live_herb_mf = {a.live_herb}
live_woody_mf = {a.live_woody}
init_mf = 0.06, 0.07, 0.08
[Map]
folder = {a.map}
""")

    from embrs.main import load_sim_params
    from embrs.fire_simulator.fire import FireSim
    from embrs.utilities.unit_conversions import BTU_ft_min_to_kW_m

    sp = load_sim_params(cfg)
    fire = FireSim(sp)

    head_I = []   # (t_s, head fireline intensity kW/m)
    while not fire.finished:
        fire.iterate()
        bc = fire.burning_cells
        if bc:
            iss_btu = max((float(np.max(c.I_ss)) for c in bc), default=0.0)
            head_I.append((fire.curr_time_s, BTU_ft_min_to_kW_m(iss_btu)))

    # steady head intensity = 75th pct over the latter half (after establishment)
    arr = np.array([v for _, v in head_I]) if head_I else np.array([0.0])
    half = arr[len(arr) // 2:] if len(arr) > 4 else arr
    head_kw = float(np.percentile(half, 75)) if half.size else 0.0
    flame_m = 0.0775 * head_kw ** 0.46 if head_kw > 0 else 0.0   # Byram
    flame_ft = flame_m * 3.28084

    burned = sum(1 for c in fire.cell_dict.values() if c.state in (0, 2))
    area = burned * (30 * 30)

    # BI for the same conditions (avg precip fixed to avoid network)
    from embrs.fire_danger import Config as FDC, compute_bi_trajectory
    TIF = ("/Users/rjdp3/Library/Mobile Documents/com~apple~CloudDocs/Documents/"
           "Research/Thesis/Final Evaluation/Phase 1/Regions/region_a/flint_hills/"
           "data/landfire/Landscape_LF2023_FBFM40_CONUS/Landscape_LF2023_FBFM40_CONUS.tif")
    bidf = compute_bi_trajectory(FDC(landscape_path=TIF, wxs_path=wxs,
                                     scenario_start=start, avg_ann_precip_in=30.0)).df
    sc = bidf[bidf["phase"] == "scenario"]["BI_area_weighted"].dropna()
    bi = float(sc.median()) if len(sc) else float("nan")

    print(json.dumps({"wind_mph": a.wind, "rh": a.rh, "bi": round(bi, 1),
                      "flame_len_ft": round(flame_ft, 1), "head_I_kW_m": round(head_kw, 0),
                      "burned_cells": burned, "burned_area_m2": area}))


if __name__ == "__main__":
    main()
