"""Standalone visualization of four fire_area_m2 computation methods.

Shows a 4×3 grid:
  Rows — area computation methods:
    1. Triangle fan (exact polygon area)
    2. Trapezoidal polar integration (½∫r²dθ)
    3. Perpendicular minor-axis lookup (π·a·b·frac)
    4. Least-squares ellipse fit (π·a_fit·b_fit·frac)
  Columns — ignition location types:
    n_loc=0  (center):        12 directions, full ellipse
    n_loc=1  (edge midpoint): 11 directions, half-ellipse
    n_loc=2  (corner vertex):  9 directions, 60° sector

Each panel shows the hexagonal cell, the discretized spread arrows, the
geometric shape whose area the method actually computes (filled), the true
continuous ellipse (dashed gray), and an info box with the resulting area.
"""

import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import least_squares

# ── Cell geometry ──────────────────────────────────────────────────────────
CELL_SIZE = 30.0
CELL_AREA = (3 * np.sqrt(3) / 2) * CELL_SIZE ** 2


def hex_vertices(s):
    """Point-up hexagon vertices: top then clockwise."""
    sq3 = np.sqrt(3)
    return np.array([
        [0, s], [sq3 / 2 * s, s / 2], [sq3 / 2 * s, -s / 2],
        [0, -s], [-sq3 / 2 * s, -s / 2], [-sq3 / 2 * s, s / 2],
    ])


def boundary_point_xy(idx, s):
    """(x, y) of boundary point 1-12.  Even=corner, odd=edge midpoint."""
    verts = hex_vertices(s)
    corner = {12: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5}
    if idx in corner:
        return verts[corner[idx]]
    mid = {1: (12, 2), 3: (2, 4), 5: (4, 6),
           7: (6, 8), 9: (8, 10), 11: (10, 12)}
    a, b = mid[idx]
    return (boundary_point_xy(a, s) + boundary_point_xy(b, s)) / 2


# ── Replicate EMBRS ignition geometry ──────────────────────────────────────

def get_ign_parameters(edge_loc, s):
    """Mirrors UtilFuncs.get_ign_parameters."""
    if edge_loc == 0:
        dirs_raw = np.linspace(30, 360, 12).tolist()
        s_ep, e_ep = 1, 12
    elif edge_loc % 2 == 0:
        sa = (30 * edge_loc + 120) % 360
        ea = sa + 120
        dirs_raw = [sa, sa + 19.107, sa + 30, sa + 46.102, sa + 60,
                    ea - 46.102, ea - 30, ea - 19.107, ea]
        s_ep = (edge_loc + 2) % 12 or 12
        e_ep = (s_ep + 8) % 12 or 12
    else:
        sa = (30 * edge_loc + 90) % 360
        ea = sa + 180
        dirs_raw = [sa, sa + 30, sa + 40.893, sa + 60, sa + 73.898, sa + 90,
                    ea - 73.898, ea - 60, ea - 40.893, ea - 30, ea]
        s_ep = (edge_loc + 1) % 12 or 12
        e_ep = (12 + (edge_loc - 1)) % 12 or 12

    directions = np.array([d % 360 for d in dirs_raw])

    if e_ep < s_ep:
        endpoints = np.concatenate([np.arange(s_ep, 13),
                                    np.arange(1, e_ep + 1)]).astype(int)
    else:
        endpoints = np.arange(s_ep, e_ep + 1, dtype=int)

    odd_d = {1: s / 2, 2: np.sqrt(3) / 2 * s, 3: np.sqrt(7) / 2 * s,
             4: 3 * s / 2, 5: np.sqrt(13) / 2 * s, 6: np.sqrt(3) * s}
    even_d = {2: s, 3: np.sqrt(7) / 2 * s, 4: np.sqrt(3) * s,
              5: np.sqrt(13) / 2 * s, 6: 2 * s}

    distances = []
    for ep in endpoints:
        diff = abs(int(ep) - edge_loc)
        if diff > 6:
            diff = 12 - diff
        if edge_loc == 0:
            distances.append(s if diff % 2 == 0 else np.sqrt(3) * s / 2)
        elif edge_loc % 2 == 0:
            distances.append(even_d[diff])
        else:
            distances.append(odd_d[diff])

    return directions, np.array(distances), endpoints


def compass_to_xy(bearing_deg):
    """Compass bearing → (x, y) unit vector.  0°→+y, 90°→+x."""
    r = np.deg2rad(bearing_deg)
    return np.sin(r), np.cos(r)


def ellipse_radius(theta, a, b, heading):
    """Polar radius of ellipse at compass bearing *theta* (degrees)."""
    d = np.deg2rad(theta - heading)
    return (a * b) / np.sqrt((b * np.cos(d)) ** 2 + (a * np.sin(d)) ** 2)


def get_frac(n_loc):
    if n_loc == 0:
        return 1.0
    return 0.5 if n_loc % 2 == 1 else 1 / 6


def angular_range_raw(n_loc):
    """Start/end angles (pre-mod) for the direction sweep."""
    if n_loc == 0:
        return 30, 390          # full circle
    if n_loc % 2 == 0:
        sa = (30 * n_loc + 120) % 360
        return sa, sa + 120
    sa = (30 * n_loc + 90) % 360
    return sa, sa + 180


def simulate_spread(directions, distances, a_true, b_true, heading):
    return np.array([
        min(ellipse_radius(d, a_true, b_true, heading), distances[i])
        for i, d in enumerate(directions)
    ])


def original_area(fire_spread, n_loc):
    """Original fire_area_m2 (max / median)."""
    a = float(np.max(fire_spread))
    b = float(np.median(fire_spread))
    return min(np.pi * a * b * get_frac(n_loc), CELL_AREA)


# ── Common drawing helpers ─────────────────────────────────────────────────

def draw_hex_and_arrows(ax, n_loc, directions, endpoints, fire_spread, ign):
    """Draw hex boundary, endpoint dots, spread arrows, ignition marker."""
    # Hex outline
    verts = hex_vertices(CELL_SIZE)
    ax.add_patch(plt.Polygon(verts, closed=True, fill=False,
                             edgecolor="black", lw=1.5, zorder=2))
    # Boundary endpoint dots
    for ep in endpoints:
        bx, by = boundary_point_xy(ep, CELL_SIZE)
        ax.plot(bx, by, "o", color="gray", ms=3, zorder=3)

    # Spread arrows
    dirs_rad = np.deg2rad(directions)
    for i in range(len(directions)):
        tip = ign + fire_spread[i] * np.array([np.sin(dirs_rad[i]),
                                                np.cos(dirs_rad[i])])
        ax.annotate("", xy=tip, xytext=ign,
                    arrowprops=dict(arrowstyle="-|>", color="steelblue",
                                   lw=1, alpha=0.6),
                    zorder=4)

    # Ignition point
    ax.plot(*ign, "k+", ms=8, mew=1.5, zorder=5)

    # Axes
    m = CELL_SIZE * 1.45
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)


def draw_true_ellipse(ax, n_loc, a_true, b_true, heading, ign):
    """Dashed gray reference ellipse."""
    sa, ea = angular_range_raw(n_loc)
    th = np.linspace(sa, ea, 400)
    r = np.array([ellipse_radius(t, a_true, b_true, heading) for t in th])
    ex = ign[0] + r * np.sin(np.deg2rad(th))
    ey = ign[1] + r * np.cos(np.deg2rad(th))
    ax.plot(ex, ey, "--", color="gray", lw=1, alpha=0.5, zorder=1)


def tips_xy(directions, fire_spread, ign):
    """(x, y) arrays of spread-tip positions."""
    dr = np.deg2rad(directions)
    return (ign[0] + fire_spread * np.sin(dr),
            ign[1] + fire_spread * np.cos(dr))


# ── Method 1: Triangle Fan (exact polygon area) ───────────────────────────

def method_triangle_fan(ax, n_loc, directions, fire_spread, ign):
    n = len(directions)
    tx, ty = tips_xy(directions, fire_spread, ign)
    rx, ry = tx - ign[0], ty - ign[1]

    # Shoelace from origin
    area = 0.0
    for i in range(n - 1):
        area += rx[i] * ry[i + 1] - rx[i + 1] * ry[i]
    if n_loc == 0:
        area += rx[-1] * ry[0] - rx[0] * ry[-1]
    area = abs(area) * 0.5

    # Draw polygon
    if n_loc == 0:
        px = np.append(tx, tx[0])
        py = np.append(ty, ty[0])
    else:
        px = np.concatenate([[ign[0]], tx, [ign[0]]])
        py = np.concatenate([[ign[1]], ty, [ign[1]]])
    ax.fill(px, py, color="royalblue", alpha=0.18, zorder=1)
    # Polygon edges (tip-to-tip only)
    ax.plot(np.append(tx, tx[0]) if n_loc == 0 else tx,
            np.append(ty, ty[0]) if n_loc == 0 else ty,
            "o-", color="royalblue", ms=3, lw=1.5, zorder=3)

    return min(area, CELL_AREA)


# ── Method 2: Trapezoidal Polar Integration ────────────────────────────────

def method_trapezoidal(ax, n_loc, directions, fire_spread, ign):
    n = len(directions)

    # Build index pairs (consecutive, plus closing pair for center)
    pairs = [(i, i + 1) for i in range(n - 1)]
    if n_loc == 0:
        pairs.append((n - 1, 0))

    area = 0.0
    bnd_x, bnd_y = [], []

    for i, j in pairs:
        r_i, r_j = fire_spread[i], fire_spread[j]
        dth = (directions[j] - directions[i]) % 360
        dth_rad = np.deg2rad(dth)
        area += dth_rad * (r_i ** 2 + r_j ** 2) / 2

        # Interpolate boundary: r² linear in θ
        n_seg = 25
        last = (1 if (i, j) == pairs[-1] else 0)
        for k in range(n_seg + last):
            t = k / n_seg
            th = directions[i] + t * dth
            r = np.sqrt(max(r_i ** 2 + t * (r_j ** 2 - r_i ** 2), 0))
            bnd_x.append(ign[0] + r * np.sin(np.deg2rad(th)))
            bnd_y.append(ign[1] + r * np.cos(np.deg2rad(th)))

    area *= 0.5
    bx, by = np.array(bnd_x), np.array(bnd_y)

    if n_loc == 0:
        fx, fy = bx, by
    else:
        fx = np.concatenate([[ign[0]], bx, [ign[0]]])
        fy = np.concatenate([[ign[1]], by, [ign[1]]])
    ax.fill(fx, fy, color="darkorange", alpha=0.18, zorder=1)
    ax.plot(bx, by, "-", color="darkorange", lw=1.5, zorder=3)
    # Show sample points
    tx, ty = tips_xy(directions, fire_spread, ign)
    ax.plot(tx, ty, "o", color="darkorange", ms=3, zorder=4)

    return min(area, CELL_AREA)


# ── Method 3: Perpendicular Minor-Axis Lookup ─────────────────────────────

def method_perp_minor(ax, n_loc, directions, fire_spread, ign):
    a = float(np.max(fire_spread))
    a_idx = int(np.argmax(fire_spread))
    heading = directions[a_idx]

    perp_target = (heading + 90) % 360
    diffs = np.minimum(np.abs(directions - perp_target),
                       360 - np.abs(directions - perp_target))
    b_idx = int(np.argmin(diffs))
    b = fire_spread[b_idx]
    perp_err = diffs[b_idx]

    frac = get_frac(n_loc)
    area = min(np.pi * a * b * frac, CELL_AREA)

    # Draw the π·a·b ellipse (within angular range)
    sa, ea = angular_range_raw(n_loc)
    th = np.linspace(sa, ea, 400)
    r = np.array([ellipse_radius(t, a, b, heading) for t in th])
    ex = ign[0] + r * np.sin(np.deg2rad(th))
    ey = ign[1] + r * np.cos(np.deg2rad(th))

    if n_loc == 0:
        fx, fy = ex, ey
    else:
        fx = np.concatenate([[ign[0]], ex, [ign[0]]])
        fy = np.concatenate([[ign[1]], ey, [ign[1]]])
    ax.fill(fx, fy, color="forestgreen", alpha=0.15, zorder=1)
    ax.plot(ex, ey, "-", color="forestgreen", lw=1.5, zorder=3)

    # a-axis arrow
    da = np.array(compass_to_xy(heading))
    ax.annotate("", xy=ign + da * a, xytext=ign,
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5), zorder=6)
    ax.annotate(f"a = {a:.1f}", ign + da * a, fontsize=8, color="red",
                fontweight="bold", xytext=(4, 4), textcoords="offset points",
                bbox=dict(fc="white", ec="red", alpha=0.9,
                          boxstyle="round,pad=0.2"), zorder=7)

    # b-axis arrow (toward actual perpendicular target, length = b)
    db = np.array(compass_to_xy(perp_target))
    ax.annotate("", xy=ign + db * b, xytext=ign,
                arrowprops=dict(arrowstyle="-|>", color="darkgreen", lw=2.5),
                zorder=6)
    blbl = f"b = {b:.1f}"
    if perp_err > 5:
        blbl += f"\n({perp_err:.0f}° off perp)"
    ax.annotate(blbl, ign + db * b, fontsize=8, color="darkgreen",
                fontweight="bold", xytext=(4, 4), textcoords="offset points",
                bbox=dict(fc="white", ec="darkgreen", alpha=0.9,
                          boxstyle="round,pad=0.2"), zorder=7)

    return area, perp_err


# ── Method 4: Least-Squares Ellipse Fit ────────────────────────────────────

def method_ellipse_fit(ax, n_loc, directions, fire_spread, ign):
    dirs_rad = np.deg2rad(directions)

    def residuals(p):
        a, b, phi = p
        d = dirs_rad - phi
        rm = (a * b) / np.sqrt((b * np.cos(d)) ** 2 + (a * np.sin(d)) ** 2)
        return fire_spread - rm

    a0 = float(np.max(fire_spread))
    b0 = float(np.min(fire_spread))
    phi0 = float(dirs_rad[np.argmax(fire_spread)])

    res = least_squares(residuals, [a0, b0, phi0],
                        bounds=([0.01, 0.01, -2 * np.pi],
                                [200, 200, 2 * np.pi]))
    a_fit, b_fit, phi_fit = res.x
    if b_fit > a_fit:
        a_fit, b_fit = b_fit, a_fit
        phi_fit += np.pi / 2
    heading_fit = np.rad2deg(phi_fit) % 360

    frac = get_frac(n_loc)
    area = min(np.pi * a_fit * b_fit * frac, CELL_AREA)

    # Draw fitted ellipse
    sa, ea = angular_range_raw(n_loc)
    th = np.linspace(sa, ea, 400)
    r = np.array([ellipse_radius(t, a_fit, b_fit, heading_fit) for t in th])
    ex = ign[0] + r * np.sin(np.deg2rad(th))
    ey = ign[1] + r * np.cos(np.deg2rad(th))

    if n_loc == 0:
        fx, fy = ex, ey
    else:
        fx = np.concatenate([[ign[0]], ex, [ign[0]]])
        fy = np.concatenate([[ign[1]], ey, [ign[1]]])
    ax.fill(fx, fy, color="mediumpurple", alpha=0.15, zorder=1)
    ax.plot(ex, ey, "-", color="mediumpurple", lw=1.5, zorder=3)

    # a-axis
    da = np.array(compass_to_xy(heading_fit))
    ax.annotate("", xy=ign + da * a_fit, xytext=ign,
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5), zorder=6)
    ax.annotate(f"a = {a_fit:.1f}", ign + da * a_fit, fontsize=8,
                color="red", fontweight="bold",
                xytext=(4, 4), textcoords="offset points",
                bbox=dict(fc="white", ec="red", alpha=0.9,
                          boxstyle="round,pad=0.2"), zorder=7)

    # b-axis
    db = np.array(compass_to_xy((heading_fit + 90) % 360))
    ax.annotate("", xy=ign + db * b_fit, xytext=ign,
                arrowprops=dict(arrowstyle="-|>", color="purple", lw=2.5),
                zorder=6)
    ax.annotate(f"b = {b_fit:.1f}", ign + db * b_fit, fontsize=8,
                color="purple", fontweight="bold",
                xytext=(4, 4), textcoords="offset points",
                bbox=dict(fc="white", ec="purple", alpha=0.9,
                          boxstyle="round,pad=0.2"), zorder=7)

    # Data points
    tx, ty = tips_xy(directions, fire_spread, ign)
    ax.plot(tx, ty, "o", color="purple", ms=3, zorder=5)

    return area, a_fit, b_fit, heading_fit


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

cases = [
    # (n_loc, a_true, b_true, heading_deg)
    (0, 22.0, 8.0, 60.0),
    (1, 18.0, 7.0, 210.0),
    (2, 14.0, 6.0, 240.0),
]

method_labels = [
    "1. Triangle Fan\n(Polygon Area)",
    "2. Trapezoidal Polar\n(½ ∫r² dθ)",
    "3. Perp. Minor-Axis Lookup\n(π · a · b · frac)",
    "4. Least-Squares Ellipse Fit\n(π · a_fit · b_fit · frac)",
]
method_colors = ["royalblue", "darkorange", "forestgreen", "mediumpurple"]

fig, axes = plt.subplots(4, 3, figsize=(19, 25))

loc_names = {0: "center", 1: "edge midpoint", 2: "corner vertex"}

print(f"cell_size = {CELL_SIZE} m,  cell_area = {CELL_AREA:.1f} m²\n")
print(f"{'Method':<38} {'n_loc=0':>12} {'n_loc=1':>12} {'n_loc=2':>12}")
print("-" * 76)

summary_rows = []

for row, (label, color) in enumerate(zip(method_labels, method_colors)):
    areas = []
    for col, (n_loc, a_t, b_t, hd) in enumerate(cases):
        ax = axes[row, col]

        directions, distances, endpoints = get_ign_parameters(n_loc, CELL_SIZE)
        ign = (np.array([0.0, 0.0]) if n_loc == 0
               else boundary_point_xy(n_loc, CELL_SIZE))
        fs = simulate_spread(directions, distances, a_t, b_t, hd)
        orig = original_area(fs, n_loc)
        frac = get_frac(n_loc)

        # Base drawing
        draw_hex_and_arrows(ax, n_loc, directions, endpoints, fs, ign)
        draw_true_ellipse(ax, n_loc, a_t, b_t, hd, ign)

        # Method-specific drawing
        if row == 0:
            area = method_triangle_fan(ax, n_loc, directions, fs, ign)
            detail = f"A = Σ triangles\n  = {area:.1f} m²"
        elif row == 1:
            area = method_trapezoidal(ax, n_loc, directions, fs, ign)
            detail = f"A = ½∫r²dθ\n  = {area:.1f} m²"
        elif row == 2:
            area, perr = method_perp_minor(ax, n_loc, directions, fs, ign)
            frac_s = f"{frac}" if frac in (1.0, 0.5) else "1/6"
            detail = (f"A = π·a·b·{frac_s}\n  = {area:.1f} m²"
                      + (f"\n⊥ err = {perr:.0f}°" if perr > 5 else ""))
        else:
            area, af, bf, hf = method_ellipse_fit(ax, n_loc, directions,
                                                   fs, ign)
            frac_s = f"{frac}" if frac in (1.0, 0.5) else "1/6"
            detail = f"A = π·{af:.1f}·{bf:.1f}·{frac_s}\n  = {area:.1f} m²"

        areas.append(area)

        # Info box
        info = f"{detail}\n\noriginal = {orig:.1f} m²"
        ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=8,
                va="bottom", fontfamily="monospace",
                bbox=dict(boxstyle="round", fc="lightyellow", ec="gray",
                          alpha=0.92))

        # Column header (top row only)
        if row == 0:
            n_d = len(directions)
            frac_s = f"{frac}" if frac in (1.0, 0.5) else "1/6"
            ax.set_title(f"n_loc = {n_loc}  ({loc_names[n_loc]})\n"
                         f"{n_d} dirs · frac = {frac_s}",
                         fontsize=10, fontweight="bold")

    # Row label
    axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold",
                            labelpad=15)

    summary_rows.append(areas)
    clean_label = label.replace("\n", " ")
    print(f"{clean_label:<38} {areas[0]:>10.1f}  {areas[1]:>10.1f}  "
          f"{areas[2]:>10.1f}")

# Print original row for reference
orig_areas = []
for n_loc, a_t, b_t, hd in cases:
    dirs, dists, _ = get_ign_parameters(n_loc, CELL_SIZE)
    fs = simulate_spread(dirs, dists, a_t, b_t, hd)
    orig_areas.append(original_area(fs, n_loc))
print(f"{'Original (π·max·median·frac)':<38} {orig_areas[0]:>10.1f}  "
      f"{orig_areas[1]:>10.1f}  {orig_areas[2]:>10.1f}")

# Shared legend
handles = [
    mpatches.Patch(color="royalblue", alpha=0.3, label="Triangle fan polygon"),
    mpatches.Patch(color="darkorange", alpha=0.3, label="Trapezoidal boundary"),
    mpatches.Patch(color="forestgreen", alpha=0.25, label="Perp-axis ellipse"),
    mpatches.Patch(color="mediumpurple", alpha=0.25, label="Fitted ellipse"),
    plt.Line2D([0], [0], color="gray", ls="--", lw=1, label="True ellipse"),
    plt.Line2D([0], [0], color="steelblue", marker=">", ls="-", lw=1,
               alpha=0.6, label="Spread directions"),
]
fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9,
           bbox_to_anchor=(0.5, 1.0))

fig.suptitle(
    "fire_area_m2 — Four Computation Methods Compared\n"
    f"cell_size = {CELL_SIZE} m,  cell_area = {CELL_AREA:.0f} m²",
    fontsize=14, fontweight="bold", y=1.04,
)

plt.tight_layout(h_pad=1.5)
plt.savefig("scripts/cell_fire_area_visualization.png", dpi=150,
            bbox_inches="tight")
plt.show()
