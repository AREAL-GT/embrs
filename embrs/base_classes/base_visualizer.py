"""Base visualization functionality for fire simulation display.

Provides common visualization components including grid rendering, weather
display, static map elements (roads, firebreaks), and prediction overlays.

Classes:
    - BaseVisualizer: Base class for simulation visualization.

.. autoclass:: BaseVisualizer
    :members:
"""

from embrs.utilities.logger_schemas import CellLogEntry, AgentLogEntry, ActionsEntry
from embrs.utilities.data_classes import VisualizerInputs
from embrs.utilities.fire_util import RoadConstants as rc, CellStates, CrownStatus
from embrs.models.fuel_models import FuelConstants as fc
from embrs.utilities.fire_util import UtilFuncs as util
from embrs.utilities.unit_conversions import F_to_C

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, TextArea
from matplotlib.lines import Line2D

from matplotlib.collections import PatchCollection
import numpy as np
from shapely.geometry import LineString

from datetime import timedelta

# Target number of wind arrows per axis (total arrows = WIND_ARROWS_PER_AXIS^2)
WIND_ARROWS_PER_AXIS = 20

# Fixed quiver sizing (inches) — constant across all simulations
WIND_ARROW_SCALE = 4.0       # data units per inch (arrow length = 1/SCALE inches)
WIND_ARROW_WIDTH = 0.002     # shaft width in inches


class BaseVisualizer:
    """Base class for fire simulation visualization.

    Provides common visualization functionality including hexagonal grid
    rendering, weather data display, static elements (roads, firebreaks,
    elevation contours), and prediction overlays.

    Attributes:
        fig: Matplotlib figure object.
        h_ax: Main axes for the hexagonal grid display.
        render (bool): Whether to render to screen (False for headless).
        cell_size (float): Size of hexagonal cells in meters.
        width_m (float): Simulation width in meters.
        height_m (float): Simulation height in meters.
    """

    def __init__(self, params: VisualizerInputs, render=True):
        """Initialize the visualizer with simulation parameters.

        Args:
            params (VisualizerInputs): Configuration parameters for visualization.
            render (bool, optional): Whether to render to screen. Use False
                for headless operation. Defaults to True.
        """
        self.render = render

        if not self.render:
            mpl.use('Agg')  # Use a non-interactive backend if not rendering

        else:
            mpl.use('tkAgg')

        self.grid_height = params.sim_shape[0]
        self.grid_width = params.sim_shape[1]
        self.cell_size = params.cell_size
        self.coarse_elevation = params.elevation
        self.width_m = params.sim_size[0]
        self.height_m = params.sim_size[1]

        self.roads = params.roads
        self.fire_breaks = params.fire_breaks

        self.wind_forecast = params.wind_forecast
        self.wind_res = params.wind_resolution
        self.wind_t_step = params.wind_t_step
        self.wind_idx = -1
        self.wind_quiver = None
        self.wind_xpad = params.wind_xpad
        self.wind_ypad = params.wind_ypad

        self.temp_forecast = params.temp_forecast
        self.rh_forecast = params.rh_forecast
        self.forecast_t_step = params.forecast_t_step
        self.forecast_idx = -1

        self.north_dir_deg = params.north_dir_deg
        self._start_datetime = params.start_datetime

        self.scale_bar_km = params.scale_bar_km
        self.show_legend = params.show_legend
        self.show_wind_cbar = params.show_wind_cbar
        self.show_wind_field = params.show_wind_field
        self.show_weather_data = params.show_weather_data
        self.show_temp_in_F = params.show_temp_in_F

        self.retardant_art = None
        self.water_drop_art = None
        self.agent_art = []
        self.agent_labels = []
        self.legend_elements = []

        self.show_compass = params.show_compass
        init_entries = params.init_entries

        self._process_weather()
        self._setup_figure()
        self._setup_grid(init_entries)

        if render:
            self.fig.canvas.draw()
            plt.pause(1)

        self.initial_state = self.fig.canvas.copy_from_bbox(self.h_ax.bbox)

    def _process_weather(self):
        """Process weather data for visualization.

        Calculates global wind speed normalization, wind grid coordinates,
        and converts temperature units if needed.
        """
        if self.show_wind_field:
            # Calculate global max speed across all time steps for consistent coloring
            all_speeds = [forecast[:, :, 0] for forecast in self.wind_forecast]
            self.global_max_speed = max(np.max(s) for s in all_speeds)
            self.wind_norm = mcolors.Normalize(vmin=0, vmax=self.global_max_speed)

            # Pre-compute the downsampled grid coordinates
            self._compute_wind_grid()

        if self.show_weather_data:
            if not self.show_temp_in_F:
                self.temp_forecast = [np.round(F_to_C(temp), 1) for temp in self.temp_forecast]

    def _compute_wind_grid(self):
        """Compute the downsampled wind grid coordinates for quiver plot.

        Creates a fixed-size grid of arrow positions regardless of domain size,
        ensuring consistent arrow density across different map sizes.
        """
        # Get wind forecast dimensions (rows, cols) from first time step
        wind_rows = self.wind_forecast[0].shape[0]
        wind_cols = self.wind_forecast[0].shape[1]

        # Calculate the actual extent of the wind forecast in the simulation domain
        wind_width = wind_cols * self.wind_res
        wind_height = wind_rows * self.wind_res

        # Create coordinate arrays for the full wind grid
        # Wind grid starts at (wind_xpad, wind_ypad) and has resolution wind_res
        x_coords = np.linspace(
            self.wind_xpad + self.wind_res / 2,
            self.wind_xpad + wind_width - self.wind_res / 2,
            wind_cols
        )
        y_coords = np.linspace(
            self.wind_ypad + self.wind_res / 2,
            self.wind_ypad + wind_height - self.wind_res / 2,
            wind_rows
        )

        # Calculate downsampling step to achieve target arrow count
        # We want approximately WIND_ARROWS_PER_AXIS arrows in each direction
        x_step = max(1, wind_cols // WIND_ARROWS_PER_AXIS)
        y_step = max(1, wind_rows // WIND_ARROWS_PER_AXIS)

        # Downsample coordinates
        self.wind_x_display = x_coords[::x_step]
        self.wind_y_display = y_coords[::y_step]

        # Store step sizes for extracting matching data
        self.wind_x_step = x_step
        self.wind_y_step = y_step

        # Create meshgrid for quiver plot
        self.wind_X, self.wind_Y = np.meshgrid(self.wind_x_display, self.wind_y_display)

    def _init_wind_field(self):
        """Initialize the wind field quiver plot.

        Creates the initial quiver plot with arrows colored by wind speed.
        Called during static element initialization if show_wind_field is True.
        """
        if not self.show_wind_field or self.wind_forecast is None:
            return

        # Get initial wind data (time index 0)
        speed_grid, u_grid, v_grid = self._get_downsampled_wind_data(0)

        # Get colors from colormap based on wind speed
        cmap = plt.cm.turbo
        colors = cmap(self.wind_norm(speed_grid.ravel()))

        # Create quiver plot with color-coded arrows
        # scale_units='inches' makes arrow size constant in screen space,
        # independent of domain size or weather conditions
        self.wind_quiver = self.h_ax.quiver(
            self.wind_X, self.wind_Y,
            u_grid, v_grid,
            color=colors,
            scale=WIND_ARROW_SCALE,
            scale_units='inches',
            width=WIND_ARROW_WIDTH,
            headwidth=4,
            headlength=5,
            headaxislength=4,
            zorder=2,
            alpha=0.8
        )

        self.wind_idx = 0

    def _get_downsampled_wind_data(self, time_idx: int):
        """Extract downsampled wind speed and direction components.

        Args:
            time_idx: Time index into the wind forecast array.

        Returns:
            tuple: (speed_grid, u_grid, v_grid) - downsampled speed and
                velocity components (u=east, v=north).
        """
        # Get full wind data for this time step
        # wind_forecast shape: (time_steps, rows, cols, 2) where [.., 0]=speed, [.., 1]=direction
        speed_full = self.wind_forecast[time_idx][:, :, 0]
        direction_full = self.wind_forecast[time_idx][:, :, 1]

        # Downsample to match display grid
        speed_grid = speed_full[::self.wind_y_step, ::self.wind_x_step]
        direction_grid = direction_full[::self.wind_y_step, ::self.wind_x_step]

        # Wrap wind angle
        direction_to = direction_grid % 360
        math_angle_rad = np.deg2rad(direction_to)

        # Calculate u (east) and v (north) components
        # Normalize so arrow lengths are based on speed relative to max
        u_grid = np.sin(math_angle_rad)
        v_grid = np.cos(math_angle_rad)

        return speed_grid, u_grid, v_grid

    def _update_wind_field(self, sim_time_s: float):
        """Update the wind field visualization for the current time step.

        Args:
            sim_time_s: Current simulation time in seconds.

        Returns:
            bool: True if the wind field was updated, False otherwise.
        """
        if not self.show_wind_field or self.wind_quiver is None:
            return False

        # Calculate current wind time index
        new_wind_idx = int(np.floor(sim_time_s / self.wind_t_step))

        # Clamp to available forecast range
        new_wind_idx = min(new_wind_idx, len(self.wind_forecast) - 1)

        if new_wind_idx == self.wind_idx:
            return False

        self.wind_idx = new_wind_idx

        # Get updated wind data
        speed_grid, u_grid, v_grid = self._get_downsampled_wind_data(new_wind_idx)

        # Update quiver arrows
        self.wind_quiver.set_UVC(u_grid, v_grid)

        # Update colors based on new speeds
        cmap = plt.cm.turbo
        colors = cmap(self.wind_norm(speed_grid.ravel()))
        self.wind_quiver.set_color(colors)

        return True

    def _setup_figure(self):
        """Set up the matplotlib figure and axes."""
        if self.render:
            plt.ion()

        self.fig = plt.figure(figsize=(9, 8))
        self.h_ax = self.fig.add_axes([0.05, 0.05, 0.9, 0.9])

        self.h_ax.set_aspect('equal')
        self.h_ax.axis([0, self.width_m, 0, self.height_m])
        plt.tick_params(left=False, right=False, bottom=False,
                        labelleft=False, labelbottom=False)

    def _setup_grid(self, init_entries: list[CellLogEntry]) -> None:
        """Initialize the hexagonal grid display.

        Creates polygon patches for all cells and sets initial colors
        based on cell state and fuel type.

        Args:
            init_entries (list[CellLogEntry]): Initial cell state entries.
        """
        # Pre-create all polygons and store them
        self.all_polygons = []
        fuel_types_seen = set()
        for entry in init_entries:
            polygon = mpatches.RegularPolygon((entry.x, entry.y),
                                              numVertices=6, radius=self.cell_size, orientation=0)
            self.all_polygons.append(polygon)

            if entry.state == CellStates.FUEL:
                fuel_color = fc.fuel_color_mapping[entry.fuel]
                if fuel_color not in fuel_types_seen:
                    fuel_types_seen.add(fuel_color)
                    legend_patch = mpatches.Patch(color=fuel_color, label=fc.fuel_names[entry.fuel])
                    self.legend_elements.append((entry.fuel, legend_patch))

        # Assign initial facecolors based on cell state
        self.cell_colors = [self._get_cell_color(entry) for entry in init_entries]
        self.cell_id_to_index = {
            entry.id: i for i, entry in enumerate(init_entries)
        }

        # Create a single PatchCollection for all cells
        self.all_cells_coll = PatchCollection(self.all_polygons, facecolors=self.cell_colors, zorder=1)
        self.h_ax.add_collection(self.all_cells_coll)

        self._init_static_elements()

    def _get_cell_color(self, entry: CellLogEntry):
        """Get the display color for a cell based on its state.

        Args:
            entry (CellLogEntry): Cell state entry.

        Returns:
            tuple: RGBA color tuple for the cell.
        """
        if entry.state == CellStates.FUEL:
            base_color = mcolors.to_rgba(fc.fuel_color_mapping[entry.fuel])
            fuel_frac = entry.w_n_dead / entry.w_n_dead_start if entry.w_n_dead_start > 0 else 1.0
            return tuple(np.array(base_color) * fuel_frac)
        elif entry.state == CellStates.FIRE:
            return mcolors.to_rgba('#F97306')
        elif entry.state == CellStates.BURNT:
            return mcolors.to_rgba('k')
        elif entry.state == CellStates.CROWN:
            return mcolors.to_rgba('magenta')
        else:
            return (0, 0, 0, 0)  # transparent for inactive cells

    def update_grid(self, sim_time_s: float, entries: list[CellLogEntry], agents: list[AgentLogEntry] = [], actions: list[ActionsEntry] = []) -> None:
        """Update the grid display with new cell states.

        Args:
            sim_time_s (float): Current simulation time in seconds.
            entries (list[CellLogEntry]): Updated cell state entries.
            agents (list[AgentLogEntry], optional): Agent positions. Defaults to [].
            actions (list[ActionsEntry], optional): Active control actions. Defaults to [].
        """
        # Update wind field if needed
        self._update_wind_field(sim_time_s)

        # Update weather data if needed
        weather_idx = int(np.floor(sim_time_s / self.forecast_t_step))
        if self.show_weather_data and weather_idx != self.forecast_idx and weather_idx < len(self.temp_forecast):
            self.forecast_idx = weather_idx
            temp_unit = "F" if self.show_temp_in_F else "C"
            t_rounded = np.round(float(self.temp_forecast[self.forecast_idx]), 1)
            rh_rounded = np.round(float(self.rh_forecast[self.forecast_idx]), 1)
            weather_str = f"Temp: {t_rounded} °{temp_unit}, RH: {rh_rounded} %"
            self.weather_text.set_text(weather_str)

        # Update only changed cells
        for entry in entries:
            idx = self.cell_id_to_index[entry.id]
            self.cell_colors[idx] = self._get_cell_color(entry)

        self.all_cells_coll.set_facecolors(self.cell_colors)

         # Draw dynamic elements
        if agents:
            for agent_art in self.agent_art:
                agent_art.remove()
            self.agent_art.clear()

            for agent in agents:
                scatter = self.h_ax.scatter(agent.x, agent.y, marker=agent.marker, color=agent.color, zorder=5)
                self.agent_art.append(scatter)

        if actions:
            for action in actions:
                if action.action_type == 'fireline_construction':
                    self.h_ax.plot(action.x_coords, action.y_coords, color='blue', linewidth=self.meters_to_points(action.width) * 5, zorder=4)
                elif action.action_type == 'long_term_retardant':
                    if self.retardant_art:
                        self.retardant_art.remove()
                    self.retardant_art = self.h_ax.scatter(action.x_coords, action.y_coords, marker='h', s=self.meters_to_points(6*self.cell_size), c=action.effectiveness, cmap='Reds_r', vmin=0, vmax=1, zorder=4)
                elif action.action_type == 'short_term_suppressant':
                    if self.water_drop_art:
                        self.water_drop_art.remove()
                    self.water_drop_art = self.h_ax.scatter(action.x_coords, action.y_coords, marker='h', s=self.meters_to_points(6*self.cell_size), c=action.effectiveness, cmap='Blues', vmin=0.5, vmax=1, zorder=4)


        # Update time displays
        sim_datetime = self._start_datetime + timedelta(seconds=sim_time_s)
        self.datetime_text.set_text(sim_datetime.strftime("%Y-%m-%d %H:%M"))
        self.elapsed_text.set_text(util.get_time_str(sim_time_s))

        if self.render:
            self.fig.canvas.restore_region(self.initial_state)
            self.h_ax.draw_artist(self.all_cells_coll)
            self.fig.canvas.blit(self.h_ax.bbox)
            self.fig.canvas.flush_events()
            plt.pause(0.001)


    def close(self):
        """Close the visualization figure."""
        if self.fig and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
            self.fig = None

    def reset_figure(self, done=False):
        """Reset the visualization figure.

        Args:
            done (bool, optional): If True, only closes without reinitializing.
                Defaults to False.
        """
        self.close()
        if done:
            return
        self.h_ax.clear()
        self._setup_figure()
        self._init_static_elements()
        self.h_ax.add_collection(self.all_cells_coll)
        self.fig.canvas.draw()
        self.initial_state = self.fig.canvas.copy_from_bbox(self.h_ax.bbox)
        self.fig.canvas.blit(self.h_ax.bbox)
        self.fig.canvas.flush_events()

    def _init_static_elements(self):
        """Initialize static visualization elements.

        Draws elevation contours, weather display, compass, time displays,
        roads, fire breaks, legend, scale bar, and wind field.
        """
        # === Wind field (draw first so it's behind other elements) ===
        if self.show_wind_field:
            self._init_wind_field()

        # === Elevation contour ===
        x = np.arange(0, self.grid_width)
        y = np.arange(0, self.grid_height)
        X, Y = np.meshgrid(x, y)
        cont = self.h_ax.contour(X * self.cell_size * np.sqrt(3), Y * self.cell_size * 1.5,
                                self.coarse_elevation, colors='k')
        self.h_ax.clabel(cont, inline=True, fontsize=10, zorder=2)

        if self.show_weather_data:
            self.weather_box = mpatches.FancyBboxPatch((0.02, 0.90), 0.25, 0.0000125/2, transform=self.h_ax.transAxes,
                                                   boxstyle='square,pad=0.02',
                                                    facecolor='white', edgecolor='black',
                                                    linewidth=1, zorder=3, alpha=0.75)

            self.weather_text = self.h_ax.text(0.01, 0.90, '',
                                        transform=self.h_ax.transAxes,
                                        ha='left', va='center',
                                        fontsize=10, zorder=4)
            
            self.h_ax.add_patch(self.weather_box)

        if self.show_compass:
            # === Compass ===
            lx = 0.02
            ly = 0.84

            if self.show_weather_data:
                ly -= 0.04

            self.compass_box = mpatches.FancyBboxPatch((lx, ly), 0.06, 0.06,
                                                    transform=self.h_ax.transAxes,
                                                    boxstyle='square,pad=0.02',
                                                    facecolor='white', edgecolor='black',
                                                    linewidth=1, zorder=3, alpha=0.75)
            self.h_ax.add_patch(self.compass_box)
            cx, cy = lx + 0.03, ly + 0.03  # center of box
        
            # Compass arrow
            arrow_len = 0.025
            dx = np.sin(np.deg2rad(self.north_dir_deg)) * arrow_len
            dy = np.cos(np.deg2rad(self.north_dir_deg)) * arrow_len
            self.arrow_obj = self.h_ax.arrow(cx, cy - 0.035, dx, dy,
                                            transform=self.h_ax.transAxes,
                                            width=0.004, head_width=0.015,
                                            color='red', zorder=4)
            self.compassheader = self.h_ax.text(cx, cy + 0.03, 'N',
                                                transform=self.h_ax.transAxes,
                                                ha='center', va='center',
                                                fontsize=10, weight='bold', color='red')

        self.datetime_box = mpatches.FancyBboxPatch((0.02, 0.98), 0.25, 0.0000125/2, transform=self.h_ax.transAxes,
                                                    boxstyle='square,pad=0.02',
                                                    facecolor='white', edgecolor='black',
                                                    linewidth=1, zorder=3, alpha=0.75)

        self.h_ax.add_patch(self.datetime_box)
        
        # # === Date/time box ===
        self.datetime_text = self.h_ax.text(0.045, 0.98, '', transform=self.h_ax.transAxes,
                                            ha='left', va='center',
                                            fontsize=10, zorder=4)
        
        # === Elapsed time ===
        self.elapsed_box = mpatches.FancyBboxPatch((0.02, 0.94), 0.25, 0.0000125/2, transform=self.h_ax.transAxes,
                                                   boxstyle='square,pad=0.02',
                                                    facecolor='white', edgecolor='black',
                                                    linewidth=1, zorder=3, alpha=0.75)

        self.h_ax.add_patch(self.elapsed_box)

        self.timeheader = self.h_ax.text(0.01, 0.94, 'elapsed:',
                                        transform=self.h_ax.transAxes,
                                        ha='left', va='center', fontsize=10, zorder=4)
        self.elapsed_text = self.h_ax.text(0.125, 0.94, '',
                                        transform=self.h_ax.transAxes,
                                        ha='left', va='center',
                                        fontsize=10, zorder=4)

        # === Roads ===
        if self.roads is not None:
            added_colors = set()
            for road, road_type, road_width in self.roads:
                x, y = road[0], road[1]
                road_color = rc.road_color_mapping[road_type]
                self.h_ax.plot(x, y, color=road_color, linewidth=self.meters_to_points(road_width) * 5, zorder=2)
                if road_color not in added_colors and self.show_legend:
                    added_colors.add(road_color)
                    legend_patch = mpatches.Patch(color=road_color, label=f"Road - {road_type}")
                    self.legend_elements.append((204 + np.where(np.array(rc.major_road_types) == road_type)[0][0], legend_patch))

        # === Firebreaks ===
        for fire_break, break_width, _ in self.fire_breaks:
            if isinstance(fire_break, LineString):
                x, y = fire_break.xy
                self.h_ax.plot(x, y, color='blue', linewidth=self.meters_to_points(break_width) * 5, zorder=2)

        # === Legend ===
        if self.legend_elements and self.show_legend:
            sorted_patches = [patch for _, patch in sorted(self.legend_elements, key=lambda x: x[0])]
            self.h_ax.legend(handles=sorted_patches, loc='upper right', borderaxespad=0)

        # === Scale bar ===
        bar_length = self.scale_bar_km * 1000  # meters
        if self.scale_bar_km < 1:
            scale_label = f"{int(bar_length)} m"
        else:
            scale_label = f"{self.scale_bar_km:.1f} km"

        # Line for the scale bar
        line = Line2D([0, bar_length], [0, 0], color='black', linewidth=2, solid_capstyle='butt')
        line_box = AuxTransformBox(self.h_ax.transData)
        line_box.add_artist(line)

        # Text label
        text = TextArea(scale_label, textprops=dict(color='black', fontsize=10))

        # Stack line and label vertically
        packed = VPacker(children=[line_box, text], align="center", pad=0, sep=2)

        # Anchor with background patch (frameon=True makes a white box)
        scalebar_box = AnchoredOffsetbox(loc='lower left',
                                        child=packed,
                                        pad=0.4,
                                        frameon=True,
                                        borderpad=0.5)
        scalebar_box.patch.set_facecolor('white')
        scalebar_box.patch.set_alpha(0.75)
        scalebar_box.zorder = 4

        self.h_ax.add_artist(scalebar_box)

        if self.show_wind_field and self.show_wind_cbar:
            sm = ScalarMappable(norm=self.wind_norm, cmap='turbo')
            sm.set_array([])
            self.wind_cbar = self.fig.colorbar(
                sm, ax=self.h_ax, orientation='vertical', shrink=0.7,
                pad=0.02, label='Wind speed (m/s)'
            )

            self.wind_cbar.ax.tick_params(labelsize=8)


    def meters_to_points(self, meters: float) -> float:
        """Convert meters to matplotlib points for sizing elements.

        Args:
            meters (float): Distance in meters.

        Returns:
            float: Equivalent size in matplotlib points.
        """
        fig_width_inch, _ = self.fig.get_size_inches()
        meters_per_inch = self.width_m / fig_width_inch
        meters_per_point = meters_per_inch / 72
        return meters / meters_per_point


    def visualize_prediction(self, prediction):
        """Visualize a fire spread prediction overlay.

        Displays predicted fire arrival times as colored points on the grid,
        with colors indicating arrival time.

        Args:
            prediction (dict): Dictionary mapping timestamps (seconds) to lists
                of (x, y) coordinate tuples where fire is predicted to arrive.
        """
        # Clear any existing prediction visualization
        if hasattr(self, 'prediction_scatter'):
            self.prediction_scatter.remove()
            delattr(self, 'prediction_scatter')

        time_steps = sorted(prediction.keys())
        if not time_steps:
            return

        cmap = mpl.cm.get_cmap("viridis")
        norm = plt.Normalize(time_steps[0], time_steps[-1])

        # Collect all points and their corresponding times
        all_points = []
        all_times = []
        for time in time_steps:
            points = prediction[time]
            all_points.extend(points)
            all_times.extend([time] * len(points))

        if all_points:
            x, y = zip(*all_points)
            self.prediction_scatter = self.h_ax.scatter(x, y, c=all_times, cmap=cmap, norm=norm, 
                                                        s=1, zorder=1)
            self.fig.canvas.draw()


    def visualize_ensemble_prediction(self, burn_probability):
        """Visualize ensemble burn probability overlay.

        Displays the final burn probability from an ensemble prediction,
        with color intensity indicating probability of burning.

        Args:
            burn_probability (dict): Dictionary mapping timestamps to
                dictionaries of {(x, y): probability} representing cumulative
                burn probability at each time step. Uses the final time step.
        """
        # Clear any existing prediction visualization
        if hasattr(self, 'prediction_scatter'):
            self.prediction_scatter.remove()
            delattr(self, 'prediction_scatter')
        
        # Get the final time step (latest prediction)
        time_steps = sorted(burn_probability.keys())
        if not time_steps:
            return
        
        final_time = time_steps[-1]
        final_probs = burn_probability[final_time]
        
        if not final_probs:
            return
        
        # Extract points and their probabilities
        points = list(final_probs.keys())
        probs = [final_probs[point] for point in points]
        
        # Create colormap for probabilities (0 to 1)
        cmap = mpl.cm.get_cmap("hot")  # 'hot' colormap: dark to bright
        norm = plt.Normalize(0, 1)
        
        x, y = zip(*points)
        self.prediction_scatter = self.h_ax.scatter(
            x, y, 
            c=probs, 
            cmap=cmap, 
            norm=norm, 
            s=1, 
            zorder=1,
            alpha=0.8
        )
        
        # Add colorbar if not already present
        if not hasattr(self, 'prediction_colorbar'):
            self.prediction_colorbar = self.fig.colorbar(
                self.prediction_scatter, 
                ax=self.h_ax, 
                label='Burn Probability'
            )
        else:
            self.prediction_colorbar.update_normal(self.prediction_scatter)
        
        self.fig.canvas.draw()