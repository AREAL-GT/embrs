"""
Ensemble Prediction Video Generator

Creates professional video visualizations of ensemble fire prediction output,
showing burn probability evolution over time with hexagonal cell polygons.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.animation import FFMpegWriter
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon
from datetime import timedelta
from typing import Optional, Tuple
import os

from embrs.utilities.data_classes import EnsemblePredictionOutput


def create_hexagon_polygon(x: float, y: float, cell_size: float) -> Polygon:
    """Create a hexagonal polygon in point-up orientation.

    Args:
        x: X coordinate of cell center (meters)
        y: Y coordinate of cell center (meters)
        cell_size: Side length of hexagon (meters)

    Returns:
        Shapely Polygon representing the hexagonal cell
    """
    l = cell_size
    sqrt3_2 = np.sqrt(3) / 2

    hex_coords = [
        (x, y + l),
        (x + sqrt3_2 * l, y + l / 2),
        (x + sqrt3_2 * l, y - l / 2),
        (x, y - l),
        (x - sqrt3_2 * l, y - l / 2),
        (x - sqrt3_2 * l, y + l / 2),
        (x, y + l)
    ]

    return Polygon(hex_coords)


def create_ensemble_video(
    ensemble_output: EnsemblePredictionOutput,
    cell_size: float,
    output_path: str = "ensemble_prediction.mp4",
    map_size: Optional[Tuple[float, float]] = None,
    fps: int = 10,
    dpi: int = 150,
    title: str = "Ensemble Fire Spread Prediction",
    figsize: Tuple[float, float] = (12, 10),
    colormap: str = "YlOrRd",
    show_progress: bool = True
) -> str:
    """
    Create a video visualization of ensemble prediction burn probability over time.

    For each time step, plots all cells that have been predicted to burn by any
    ensemble member, colored by their burn probability. Suitable for presentations.

    Args:
        ensemble_output: EnsemblePredictionOutput from FirePredictor.run_ensemble()
        cell_size: Size of hexagonal cells in meters (side length)
        output_path: Path to save the video file (default: "ensemble_prediction.mp4")
        map_size: Tuple of (width_m, height_m) for the map. If None, computed from data.
        fps: Frames per second for the video (default: 10)
        dpi: Resolution of the video (default: 150)
        title: Title displayed on the video (default: "Ensemble Fire Spread Prediction")
        figsize: Figure size in inches (default: (12, 10))
        colormap: Matplotlib colormap name (default: "YlOrRd" - yellow to orange to red)
        show_progress: Print progress updates during video creation (default: True)

    Returns:
        Path to the saved video file

    Example:
        >>> from embrs.tools.ensemble_video import create_ensemble_video
        >>> result = predictor.run_ensemble(state_estimates, ...)
        >>> video_path = create_ensemble_video(
        ...     result,
        ...     cell_size=45.0,
        ...     output_path="my_prediction.mp4"
        ... )
    """
    # Use non-interactive backend for video creation
    mpl.use('Agg')

    burn_probability = ensemble_output.burn_probability
    n_ensemble = ensemble_output.n_ensemble

    # Get sorted time steps
    time_steps = sorted(burn_probability.keys())
    if not time_steps:
        raise ValueError("No time steps in ensemble output")

    if show_progress:
        print(f"Creating ensemble video with {len(time_steps)} time steps...")

    # Compute map bounds from all cell positions across all time steps
    all_positions = set()
    for time_s in time_steps:
        all_positions.update(burn_probability[time_s].keys())

    if not all_positions:
        raise ValueError("No cells with burn probability data")

    xs, ys = zip(*all_positions)

    # Add padding for hexagon extent
    hex_padding = cell_size * 1.5

    if map_size is not None:
        x_min, x_max = 0, map_size[0]
        y_min, y_max = 0, map_size[1]
    else:
        x_min = min(xs) - hex_padding
        x_max = max(xs) + hex_padding
        y_min = min(ys) - hex_padding
        y_max = max(ys) + hex_padding

    # Pre-compute all hexagon patches
    if show_progress:
        print("Pre-computing hexagon geometries...")

    hex_cache = {}
    for pos in all_positions:
        x, y = pos
        hex_poly = create_hexagon_polygon(x, y, cell_size)
        hex_coords = list(hex_poly.exterior.coords)
        hex_cache[pos] = mpatches.Polygon(hex_coords, closed=True)

    # Setup figure with professional styling
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Set fixed axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')

    # Professional axis styling
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add light gray background
    ax.set_facecolor('#f5f5f5')

    # Setup colormap and normalization
    cmap = plt.cm.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Burn Probability', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add ensemble info text
    info_text = ax.text(
        0.02, 0.98,
        f'Ensemble: {n_ensemble} members',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    )

    # Title and time text (will be updated)
    title_text = ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    time_text = ax.text(
        0.98, 0.98,
        '',
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    )

    # Statistics text box
    stats_text = ax.text(
        0.02, 0.02,
        '',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    )

    plt.tight_layout()

    # Create video writer
    metadata = {
        'title': title,
        'artist': 'EMBRS Fire Simulation',
        'comment': f'Ensemble prediction with {n_ensemble} members'
    }
    writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=5000)

    # Reference to current patch collection
    current_collection = None

    # Start time for relative time display
    start_time_s = time_steps[0]

    if show_progress:
        print(f"Rendering {len(time_steps)} frames...")

    with writer.saving(fig, output_path, dpi=dpi):
        for i, time_s in enumerate(time_steps):
            # Get probabilities for this time step
            probs = burn_probability[time_s]

            if not probs:
                continue

            # Remove previous collection
            if current_collection is not None:
                current_collection.remove()

            # Create patches for cells with burn probability
            patches = []
            colors = []

            for pos, prob in probs.items():
                if pos in hex_cache:
                    # Create a new patch (can't reuse patches in collections)
                    hex_coords = list(hex_cache[pos].get_xy())
                    patch = mpatches.Polygon(hex_coords, closed=True)
                    patches.append(patch)
                    colors.append(prob)

            if patches:
                # Create patch collection with colors
                collection = PatchCollection(
                    patches,
                    cmap=cmap,
                    norm=norm,
                    edgecolors='black',
                    linewidths=0.3,
                    alpha=0.9
                )
                collection.set_array(np.array(colors))
                current_collection = ax.add_collection(collection)

            # Update time display
            elapsed_s = time_s - start_time_s
            elapsed_hr = elapsed_s / 3600
            time_text.set_text(f'Time: {elapsed_hr:.2f} hr\n({elapsed_s:.0f} s)')

            # Update statistics
            if probs:
                prob_values = list(probs.values())
                high_prob = sum(1 for p in prob_values if p >= 0.8)
                med_prob = sum(1 for p in prob_values if 0.5 <= p < 0.8)
                low_prob = sum(1 for p in prob_values if p < 0.5)

                stats_str = (
                    f'Cells: {len(probs):,}\n'
                    f'High prob (≥80%): {high_prob:,}\n'
                    f'Med prob (50-80%): {med_prob:,}\n'
                    f'Low prob (<50%): {low_prob:,}'
                )
                stats_text.set_text(stats_str)

            # Grab frame
            writer.grab_frame()

            # Progress update
            if show_progress and (i + 1) % max(1, len(time_steps) // 10) == 0:
                print(f"  Progress: {i + 1}/{len(time_steps)} frames ({100*(i+1)/len(time_steps):.0f}%)")

    plt.close(fig)

    # Get absolute path for output
    abs_path = os.path.abspath(output_path)

    if show_progress:
        print(f"\nVideo saved to: {abs_path}")
        print(f"  Duration: {len(time_steps)/fps:.1f} seconds")
        print(f"  Resolution: {int(figsize[0]*dpi)} x {int(figsize[1]*dpi)}")

    return abs_path


def create_ensemble_video_from_predictor(
    predictor,
    ensemble_output: EnsemblePredictionOutput,
    output_path: str = "ensemble_prediction.mp4",
    **kwargs
) -> str:
    """
    Convenience function to create ensemble video using predictor's cell size and map info.

    Args:
        predictor: FirePredictor instance
        ensemble_output: EnsemblePredictionOutput from run_ensemble()
        output_path: Path to save the video
        **kwargs: Additional arguments passed to create_ensemble_video()

    Returns:
        Path to the saved video file
    """
    # Get cell size from predictor
    cell_size = predictor.c_size

    # Try to get map size from predictor
    map_size = None
    if hasattr(predictor, '_size'):
        map_size = predictor._size

    return create_ensemble_video(
        ensemble_output=ensemble_output,
        cell_size=cell_size,
        output_path=output_path,
        map_size=map_size,
        **kwargs
    )
