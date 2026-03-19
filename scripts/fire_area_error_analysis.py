"""Error analysis: Triangle Fan vs Trapezoidal Polar for fire_area_m2.

Compares relative error of both methods against the exact ellipse area
as a function of (1) eccentricity and (2) cell_size / fire_size ratio.
"""

import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Geometry (same as main visualization) ──────────────────────────────────

def get_directions(n_loc):
    """Return direction angles (degrees) for a given ignition location."""
    if n_loc == 0:
        return np.array([d % 360 for d in np.linspace(30, 360, 12)])
    elif n_loc % 2 == 0:
        sa = (30 * n_loc + 120) % 360
        ea = sa + 120
        raw = [sa, sa+19.107, sa+30, sa+46.102, sa+60,
               ea-46.102, ea-30, ea-19.107, ea]
    else:
        sa = (30 * n_loc + 90) % 360
        ea = sa + 180
        raw = [sa, sa+30, sa+40.893, sa+60, sa+73.898, sa+90,
               ea-73.898, ea-60, ea-40.893, ea-30, ea]
    return np.array([d % 360 for d in raw])


def get_distances(n_loc, cell_size):
    """Return boundary distances for each direction."""
    s = cell_size
    odd_d = {1: s/2, 2: np.sqrt(3)/2*s, 3: np.sqrt(7)/2*s,
             4: 3*s/2, 5: np.sqrt(13)/2*s, 6: np.sqrt(3)*s}
    even_d = {2: s, 3: np.sqrt(7)/2*s, 4: np.sqrt(3)*s,
              5: np.sqrt(13)/2*s, 6: 2*s}

    if n_loc == 0:
        start_ep, end_ep = 1, 12
    elif n_loc % 2 == 0:
        start_ep = (n_loc + 2) % 12 or 12
        end_ep = (start_ep + 8) % 12 or 12
    else:
        start_ep = (n_loc + 1) % 12 or 12
        end_ep = (12 + (n_loc - 1)) % 12 or 12

    if end_ep < start_ep:
        endpoints = np.concatenate([np.arange(start_ep, 13),
                                    np.arange(1, end_ep + 1)]).astype(int)
    else:
        endpoints = np.arange(start_ep, end_ep + 1, dtype=int)

    dists = []
    for ep in endpoints:
        diff = abs(int(ep) - n_loc)
        if diff > 6:
            diff = 12 - diff
        if n_loc == 0:
            dists.append(s if diff % 2 == 0 else np.sqrt(3)*s/2)
        elif n_loc % 2 == 0:
            dists.append(even_d[diff])
        else:
            dists.append(odd_d[diff])
    return np.array(dists)


def angular_range(n_loc):
    """Raw start/end angles (degrees) for the sweep."""
    if n_loc == 0:
        return 30, 390
    if n_loc % 2 == 0:
        sa = (30 * n_loc + 120) % 360
        return sa, sa + 120
    sa = (30 * n_loc + 90) % 360
    return sa, sa + 180


def get_frac(n_loc):
    if n_loc == 0:
        return 1.0
    return 0.5 if n_loc % 2 == 1 else 1/6


def ellipse_r(theta, a, b, heading):
    d = np.deg2rad(theta - heading)
    return (a * b) / np.sqrt((b * np.cos(d))**2 + (a * np.sin(d))**2)


# ── Area methods ───────────────────────────────────────────────────────────

def area_exact(a, b, n_loc, cell_size, heading):
    """High-resolution numerical integration as ground truth.

    Integrates (1/2)∫r²dθ with fine angular resolution, clamping r to
    the boundary distance.  Uses vectorized numpy for speed.
    """
    sa, ea = angular_range(n_loc)
    n_pts = 10_000
    th = np.linspace(sa, ea, n_pts + 1)

    # Vectorized ellipse radius
    d = np.deg2rad(th - heading)
    r = (a * b) / np.sqrt((b * np.cos(d))**2 + (a * np.sin(d))**2)

    # Clamp to interpolated boundary distances
    directions = get_directions(n_loc)
    distances = get_distances(n_loc, cell_size)

    dirs_sweep = directions.copy().astype(float)
    if n_loc == 0:
        dirs_sweep[dirs_sweep < sa] += 360
    sort_idx = np.argsort(dirs_sweep)

    boundary_r = np.interp(th, dirs_sweep[sort_idx], distances[sort_idx])
    r = np.minimum(r, boundary_r)

    # Trapezoidal integration
    dth = np.deg2rad(np.diff(th))
    return 0.5 * np.sum(dth * (r[:-1]**2 + r[1:]**2) / 2)


def area_triangle_fan(fire_spread, directions, n_loc):
    dr = np.deg2rad(directions)
    rx = fire_spread * np.sin(dr)
    ry = fire_spread * np.cos(dr)
    n = len(directions)
    area = 0.0
    for i in range(n - 1):
        area += rx[i]*ry[i+1] - rx[i+1]*ry[i]
    if n_loc == 0:
        area += rx[-1]*ry[0] - rx[0]*ry[-1]
    return abs(area) * 0.5


def area_trapezoidal(fire_spread, directions, n_loc):
    n = len(directions)
    area = 0.0
    for i in range(n - 1):
        dth = np.deg2rad((directions[i+1] - directions[i]) % 360)
        area += dth * (fire_spread[i]**2 + fire_spread[i+1]**2) / 2
    if n_loc == 0:
        dth = np.deg2rad((directions[0] - directions[-1] + 360) % 360)
        area += dth * (fire_spread[-1]**2 + fire_spread[0]**2) / 2
    return area * 0.5


# ── Analysis ───────────────────────────────────────────────────────────────

def run_eccentricity_sweep(n_loc, cell_size, heading, fire_radius):
    """Vary eccentricity while keeping fire area constant (a*b = fire_radius²)."""
    # Eccentricity of ellipse: e = sqrt(1 - (b/a)²)
    # We parameterize by length-to-breadth ratio LB = a/b
    # Keeping a*b = fire_radius² means a = fire_radius * sqrt(LB), b = fire_radius / sqrt(LB)
    lb_ratios = np.linspace(1.0, 8.0, 200)
    eccentricities = np.sqrt(1 - 1/lb_ratios**2)

    directions = get_directions(n_loc)
    distances = get_distances(n_loc, cell_size)

    err_tf = []
    err_tr = []

    for lb in lb_ratios:
        a = fire_radius * np.sqrt(lb)
        b = fire_radius / np.sqrt(lb)

        fs = np.array([min(ellipse_r(d, a, b, heading), distances[i])
                       for i, d in enumerate(directions)])

        exact = area_exact(a, b, n_loc, cell_size, heading)
        if exact < 1e-6:
            err_tf.append(0)
            err_tr.append(0)
            continue

        atf = area_triangle_fan(fs, directions, n_loc)
        atr = area_trapezoidal(fs, directions, n_loc)

        err_tf.append((atf - exact) / exact * 100)
        err_tr.append((atr - exact) / exact * 100)

    return eccentricities, np.array(err_tf), np.array(err_tr)


def run_cell_size_sweep(n_loc, heading, a_true, b_true):
    """Vary cell_size while keeping fire ellipse fixed."""
    # Minimum cell size where the fire fits without clipping:
    # max spread ≈ a_true, min boundary dist ≈ s/2 for edge ignition
    # So we go from cell_size = a_true (heavy clipping) to 5*a_true (no clipping)
    cell_sizes = np.linspace(a_true * 0.8, a_true * 5, 200)

    err_tf = []
    err_tr = []

    for cs in cell_sizes:
        directions = get_directions(n_loc)
        distances = get_distances(n_loc, cs)

        fs = np.array([min(ellipse_r(d, a_true, b_true, heading), distances[i])
                       for i, d in enumerate(directions)])

        exact = area_exact(a_true, b_true, n_loc, cs, heading)
        if exact < 1e-6:
            err_tf.append(0)
            err_tr.append(0)
            continue

        atf = area_triangle_fan(fs, directions, n_loc)
        atr = area_trapezoidal(fs, directions, n_loc)

        err_tf.append((atf - exact) / exact * 100)
        err_tr.append((atr - exact) / exact * 100)

    return cell_sizes / a_true, np.array(err_tf), np.array(err_tr)


# ── Plotting ───────────────────────────────────────────────────────────────

n_locs = [0, 1, 2]
loc_names = {0: "n_loc=0 (center, 12 dirs)",
             1: "n_loc=1 (edge, 11 dirs)",
             2: "n_loc=2 (corner, 9 dirs)"}
headings = {0: 60.0, 1: 210.0, 2: 240.0}
loc_colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# ── Top panel: error vs eccentricity ────────────────────────────────────
ax = axes[0]
cell_size = 60.0   # large enough to avoid clipping at moderate eccentricity
fire_radius = 15.0  # sqrt(a*b) held constant

for n_loc in n_locs:
    ecc, err_tf, err_tr = run_eccentricity_sweep(
        n_loc, cell_size, headings[n_loc], fire_radius)

    ax.plot(ecc, err_tf, "-",  color=loc_colors[n_loc], lw=2,
            label=f"TF  {loc_names[n_loc]}")
    ax.plot(ecc, err_tr, "--", color=loc_colors[n_loc], lw=2,
            label=f"TP  {loc_names[n_loc]}")

ax.axhline(0, color="black", lw=0.5, ls=":")
ax.set_xlabel("Eccentricity  e = √(1 − b²/a²)", fontsize=11)
ax.set_ylabel("Relative Error  (%)", fontsize=11)
ax.set_title("Error vs Eccentricity  (fire_radius = 15 m, cell_size = 60 m, no clipping)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8, ncol=2, loc="lower left")
ax.grid(True, alpha=0.3)

# ── Bottom panel: error vs cell_size/fire_size ratio ────────────────────
ax = axes[1]
a_true, b_true = 20.0, 8.0  # LB ≈ 2.5, e ≈ 0.92

for n_loc in n_locs:
    ratio, err_tf, err_tr = run_cell_size_sweep(
        n_loc, headings[n_loc], a_true, b_true)

    ax.plot(ratio, err_tf, "-",  color=loc_colors[n_loc], lw=2,
            label=f"TF  {loc_names[n_loc]}")
    ax.plot(ratio, err_tr, "--", color=loc_colors[n_loc], lw=2,
            label=f"TP  {loc_names[n_loc]}")

ax.axhline(0, color="black", lw=0.5, ls=":")
ax.set_xlabel("cell_size / a  (larger = less clipping)", fontsize=11)
ax.set_ylabel("Relative Error  (%)", fontsize=11)
ax.set_title("Error vs Cell Size  (a = 20 m, b = 8 m, e ≈ 0.92)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.grid(True, alpha=0.3)

fig.suptitle(
    "Triangle Fan (TF, solid) vs Trapezoidal Polar (TP, dashed)\n"
    "Relative error vs exact high-resolution integral",
    fontsize=13, fontweight="bold", y=1.03,
)

plt.tight_layout()
plt.savefig("scripts/fire_area_error_analysis.png", dpi=150,
            bbox_inches="tight")
print("Saved scripts/fire_area_error_analysis.png")

# ── Print key data points ──────────────────────────────────────────────
print("\n── Key data points ──")
print(f"{'':30} {'TF error':>10} {'TP error':>10}")

# Circle (e=0) with n_loc=0
directions = get_directions(0)
distances = get_distances(0, 60)
R = 15.0
fs = np.array([min(R, distances[i]) for i in range(len(directions))])
exact = area_exact(R, R, 0, 60, 60)
print(f"{'Circle, n_loc=0:':<30} "
      f"{(area_triangle_fan(fs, directions, 0) - exact)/exact*100:>9.2f}% "
      f"{(area_trapezoidal(fs, directions, 0) - exact)/exact*100:>9.2f}%")

# High eccentricity (e≈0.97) with n_loc=0
a, b = 30.0, 7.5  # LB=4, e=sqrt(1-1/16)≈0.97
fs = np.array([min(ellipse_r(d, a, b, 60), distances[i])
               for i, d in enumerate(directions)])
exact = area_exact(a, b, 0, 60, 60)
print(f"{'e≈0.97, n_loc=0:':<30} "
      f"{(area_triangle_fan(fs, directions, 0) - exact)/exact*100:>9.2f}% "
      f"{(area_trapezoidal(fs, directions, 0) - exact)/exact*100:>9.2f}%")

# Same but n_loc=2
directions2 = get_directions(2)
distances2 = get_distances(2, 60)
fs2 = np.array([min(ellipse_r(d, a, b, 240), distances2[i])
                for i, d in enumerate(directions2)])
exact2 = area_exact(a, b, 2, 60, 240)
print(f"{'e≈0.97, n_loc=2:':<30} "
      f"{(area_triangle_fan(fs2, directions2, 2) - exact2)/exact2*100:>9.2f}% "
      f"{(area_trapezoidal(fs2, directions2, 2) - exact2)/exact2*100:>9.2f}%")
