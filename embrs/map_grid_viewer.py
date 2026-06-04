"""Static EMBRS map viewer with a coordinate grid overlay.

Renders an EMBRS map (the kind produced by ``map_generator`` and stored as a
folder containing ``map_params.pkl``) exactly the way
:class:`~embrs.base_classes.base_visualizer.BaseVisualizer` draws the hexagonal
fuel grid, but adds a labeled coordinate grid so you can read off the world
(meters) coordinates of key locations. Distracting overlays that the real
visualizer draws -- the north-pointing compass, the fuel legend, and the
topography contour map -- are intentionally omitted.

Roads, fire breaks, and the initial ignition region are drawn (matching the
base visualizer) since those are useful spatial references; pass ``--bare`` to
hide them.

Usage::

    python -m embrs.map_grid_viewer /path/to/embrs_map/scenario_4
    python -m embrs.map_grid_viewer scenario_4 --cell-size 50 --grid-spacing 2000
    python -m embrs.map_grid_viewer scenario_4 --save scenario_4_grid.png

Interactions (when shown on screen):
    - The matplotlib toolbar continuously shows the cursor's (x, y) in meters.
    - Left-click anywhere to print that location's coordinates (and the fuel
      type under the cursor) to the terminal.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from shapely.geometry import LineString, Polygon, Point

from embrs.utilities.data_classes import MapParams
from embrs.models.fuel_models import FuelConstants as fc
from embrs.utilities.fire_util import RoadConstants as rc

# Color used for any fuel id that is missing from the fuel color table.
_UNKNOWN_FUEL_COLOR = "#dddddd"


def _resolve_map_folder(path: str) -> str:
    """Return an absolute folder path that contains ``map_params.pkl``.

    Accepts either a direct path to a map folder, or a bare map name that is
    looked up under the sibling ``embrs_map`` directory.
    """
    candidates = [path]

    # Allow passing a bare map name (e.g. "scenario_4") resolved against the
    # conventional embrs_map directory next to the repo.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_map_root = os.path.join(os.path.dirname(repo_root), "embrs_map")
    candidates.append(os.path.join(default_map_root, path))

    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "map_params.pkl")):
            return os.path.abspath(cand)

    raise FileNotFoundError(
        f"Could not find 'map_params.pkl' for '{path}'. Tried: "
        + ", ".join(os.path.abspath(c) for c in candidates)
    )


def _compute_cell_positions(num_rows: int, num_cols: int, cell_size: float):
    """Replicate ``GridManager.compute_all_cell_positions`` for a fresh grid."""
    rows, cols = np.meshgrid(
        np.arange(num_rows), np.arange(num_cols), indexing="ij"
    )
    hex_width = np.sqrt(3) * cell_size
    all_x = (cols + 0.5 * (rows % 2)) * hex_width
    all_y = rows * cell_size * 1.5
    return all_x, all_y


def _sample_fuel(all_x, all_y, fuel_map, data_res):
    """Sample fuel ids at each cell center.

    Mirrors ``BaseFireSim`` which flips the fuel raster vertically (row 0 ->
    y = 0) before indexing with ``floor(coord / resolution)``.
    """
    flipped = np.flipud(fuel_map)
    data_rows, data_cols = flipped.shape
    col_idx = np.clip(np.floor(all_x / data_res).astype(np.int32), 0, data_cols - 1)
    row_idx = np.clip(np.floor(all_y / data_res).astype(np.int32), 0, data_rows - 1)
    return flipped[row_idx, col_idx]


def _build_cell_collection(map_params: MapParams, cell_size: float):
    """Build a PatchCollection of fuel-colored hexagons for the whole map."""
    lcp = map_params.lcp_data

    num_rows = int(np.floor(lcp.height_m / (1.5 * cell_size))) + 1
    num_cols = int(np.floor(lcp.width_m / (np.sqrt(3) * cell_size))) + 1

    all_x, all_y = _compute_cell_positions(num_rows, num_cols, cell_size)
    fuels = _sample_fuel(all_x, all_y, lcp.fuel_map, lcp.resolution)

    polygons = []
    colors = []
    flat_x = all_x.ravel()
    flat_y = all_y.ravel()
    flat_fuel = fuels.ravel()
    for x, y, fuel in zip(flat_x, flat_y, flat_fuel):
        polygons.append(
            mpatches.RegularPolygon(
                (x, y), numVertices=6, radius=cell_size, orientation=0
            )
        )
        hex_color = fc.fuel_color_mapping.get(int(fuel), _UNKNOWN_FUEL_COLOR)
        colors.append(mcolors.to_rgba(hex_color))

    coll = PatchCollection(polygons, facecolors=colors, zorder=1)
    return coll, num_rows, num_cols


def _draw_static_elements(ax, map_params: MapParams):
    """Draw roads, fire breaks, and the initial ignition region."""
    # === Roads ===
    if map_params.roads:
        for road, road_type, _road_width in map_params.roads:
            x, y = road[0], road[1]
            road_color = rc.road_color_mapping[road_type]
            ax.plot(x, y, color=road_color, linewidth=1.5, zorder=2)

    scenario = map_params.scenario_data
    if scenario is None:
        return

    # === Fire breaks ===
    for fire_break in scenario.fire_breaks:
        if isinstance(fire_break, LineString):
            x, y = fire_break.xy
            ax.plot(x, y, color="blue", linewidth=2, zorder=3)

    # === Initial ignition region ===
    for geom in scenario.initial_ign:
        if isinstance(geom, Polygon):
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, facecolor="#F97306", edgecolor="red",
                    linewidth=1.5, alpha=0.8, zorder=4)
        elif isinstance(geom, LineString):
            xs, ys = geom.xy
            ax.plot(xs, ys, color="red", linewidth=2, zorder=4)
        elif isinstance(geom, Point):
            ax.plot(geom.x, geom.y, marker="*", color="red",
                    markersize=14, zorder=4)


def _add_grid_overlay(ax, width_m, height_m, spacing):
    """Add a labeled coordinate grid (in meters) over the map."""
    xticks = np.arange(0, width_m + spacing, spacing)
    yticks = np.arange(0, height_m + spacing, spacing)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Re-enable the tick labels the base visualizer suppresses; this is the
    # whole point of the tool.
    ax.tick_params(left=True, right=False, bottom=True,
                   labelleft=True, labelbottom=True, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(90)

    ax.grid(True, which="major", color="0.3", linestyle="--",
            linewidth=0.6, alpha=0.7, zorder=6)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def visualize_map(folder: str, cell_size: float, grid_spacing: float,
                  bare: bool, save_path: str | None):
    """Render the map with a coordinate grid overlay."""
    map_params = MapParams.load(folder)
    lcp = map_params.lcp_data
    width_m, height_m = lcp.width_m, lcp.height_m

    if save_path and not _have_display():
        matplotlib.use("Agg")

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_axes([0.08, 0.08, 0.88, 0.88])
    ax.set_aspect("equal")
    ax.axis([0, width_m, 0, height_m])

    print(f"Building hex grid for '{os.path.basename(folder)}' "
          f"(cell_size={cell_size} m)...", file=sys.stderr)
    coll, num_rows, num_cols = _build_cell_collection(map_params, cell_size)
    ax.add_collection(coll)
    print(f"  grid: {num_rows} rows x {num_cols} cols, "
          f"map size {width_m:.0f} x {height_m:.0f} m", file=sys.stderr)

    if not bare:
        _draw_static_elements(ax, map_params)

    _add_grid_overlay(ax, width_m, height_m, grid_spacing)

    ax.set_title(f"{os.path.basename(folder)} "
                 f"({width_m:.0f} x {height_m:.0f} m)")

    # Live coordinate readout in the toolbar.
    def _format_coord(x, y):
        return f"x={x:.1f} m, y={y:.1f} m"
    ax.format_coord = _format_coord

    # Click-to-print coordinates (and fuel under the cursor).
    flipped_fuel = np.flipud(lcp.fuel_map)
    data_rows, data_cols = flipped_fuel.shape

    def _on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        col = int(np.clip(x // lcp.resolution, 0, data_cols - 1))
        row = int(np.clip(y // lcp.resolution, 0, data_rows - 1))
        fuel_id = int(flipped_fuel[row, col])
        fuel_name = fc.fuel_names.get(fuel_id, "unknown")
        print(f"clicked: x={x:.1f} m, y={y:.1f} m  |  fuel {fuel_id} "
              f"({fuel_name})")

    fig.canvas.mpl_connect("button_press_event", _on_click)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {os.path.abspath(save_path)}", file=sys.stderr)

    if _have_display():
        print("Left-click on the map to print coordinates to this terminal.",
              file=sys.stderr)
        plt.show()


def _have_display() -> bool:
    """Best-effort check for an interactive display."""
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY"))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize an EMBRS map with a coordinate grid overlay "
                    "for reading off key locations."
    )
    parser.add_argument(
        "map",
        help="Path to an EMBRS map folder (containing map_params.pkl), or a "
             "bare map name resolved under the sibling embrs_map/ directory.",
    )
    parser.add_argument(
        "--cell-size", type=float, default=30.0,
        help="Hexagon side length in meters. Larger renders faster / coarser. "
             "Default: 30.",
    )
    parser.add_argument(
        "--grid-spacing", type=float, default=1000.0,
        help="Spacing between coordinate grid lines in meters. Default: 1000.",
    )
    parser.add_argument(
        "--bare", action="store_true",
        help="Hide roads, fire breaks, and the initial ignition region.",
    )
    parser.add_argument(
        "--save", metavar="PATH", default=None,
        help="Save the figure to this path (e.g. map_grid.png) in addition to "
             "(or instead of, when headless) displaying it.",
    )
    args = parser.parse_args()

    folder = _resolve_map_folder(args.map)
    visualize_map(folder, args.cell_size, args.grid_spacing, args.bare,
                  args.save)


if __name__ == "__main__":
    main()
