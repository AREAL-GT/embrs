
from embrs.utilities.logger_schemas import CellLogEntry, AgentLogEntry
from embrs.utilities.data_classes import VisualizerInputs
from embrs.utilities.fire_util import RoadConstants as rc, CellStates, FuelConstants as fc, CrownStatus
from embrs.utilities.fire_util import UtilFuncs as util


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
import copy

# TODO: Improve crown visualization? 
# TODO: Implement and improve fully burning visualization?
# TODO: Worth adding temperature and other weather conditions?

class BaseVisualizer:
    def __init__(self, params: VisualizerInputs, render=True):
        self.render = render

        if not self.render:
            mpl.use('Agg')  # Use a non-interactive backend if not rendering
        
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
        self.wind_grid = None
        self.wind_xpad = params.wind_xpad
        self.wind_ypad = params.wind_ypad

        self.north_dir_deg = params.north_dir_deg
        self._start_datetime = params.start_datetime

        self.scale_bar_km = params.scale_bar_km
        self.show_legend = params.show_legend
        self.show_wind_cbar = params.show_wind_cbar
        self.show_wind_field = params.show_wind_field

        self.show_compass = params.show_compass

        init_entries = params.init_entries

        self._process_wind()
        self._setup_figure()
        self._setup_grid(init_entries)

        if render:
            self.fig.canvas.draw()
            plt.pause(1)

        self.initial_state = self.fig.canvas.copy_from_bbox(self.h_ax.bbox)

    def _process_wind(self):
        if self.show_wind_field:
            all_speeds = [forecast[:, :, 0] for forecast in self.wind_forecast]
            self.global_max_speed = max(np.max(s) for s in all_speeds)

            self.wind_norm = mcolors.Normalize(vmin=0, vmax=self.global_max_speed)

    def _setup_figure(self):
        if self.render:
            plt.ion()

        self.fig = plt.figure(figsize=(9, 8))
        self.h_ax = self.fig.add_axes([0.05, 0.05, 0.9, 0.9])

        self.h_ax.set_aspect('equal')
        self.h_ax.axis([0, self.width_m, 0, self.height_m])
        self._scale_factor_x = self.cell_size * np.sqrt(3)
        self._scale_factor_y = self.cell_size * 1.5

        plt.tick_params(left = False, right = False, bottom = False,
                labelleft = False, labelbottom = False)

    def _init_static_elements(self):

        # === Elevation contour ===
        x = np.arange(0, self.grid_width)
        y = np.arange(0, self.grid_height)
        X, Y = np.meshgrid(x, y)
        cont = self.h_ax.contour(X * self.cell_size * np.sqrt(3), Y * self.cell_size * 1.5,
                                self.coarse_elevation, colors='k')
        self.h_ax.clabel(cont, inline=True, fontsize=10, zorder=2)

        if self.show_compass:
            # === Compass ===
            self.compass_box = mpatches.FancyBboxPatch((0.02, 0.84), 0.06, 0.06,
                                                    transform=self.h_ax.transAxes,
                                                    boxstyle='square,pad=0.02',
                                                    facecolor='white', edgecolor='black',
                                                    linewidth=1, zorder=3, alpha=0.75)
            self.h_ax.add_patch(self.compass_box)
            cx, cy = 0.02 + 0.03, 0.84 + 0.03  # center of box
        
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

        self.datetime_box = mpatches.FancyBboxPatch((0.02, 0.98), 0.2, 0.0000125/2, transform=self.h_ax.transAxes,
                                                    boxstyle='square,pad=0.02',
                                                    facecolor='white', edgecolor='black',
                                                    linewidth=1, zorder=3, alpha=0.75)

        self.h_ax.add_patch(self.datetime_box)


        # # === Date/time box ===
        self.datetime_text = self.h_ax.text(0.02, 0.98, '', transform=self.h_ax.transAxes,
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
            for road, road_type, road_width in self.roads:
                x, y = road[0], road[1]
                road_color = rc.road_color_mapping[road_type]
                self.h_ax.plot(x, y, color=road_color, linewidth=self.meters_to_points(road_width))
                if road_color not in self.added_colors and self.show_legend:
                    self.added_colors.append(road_color)
                    self.legend_elements.append(mpatches.Patch(color=road_color,
                                                            label=f"Road - {road_type}"))

        # === Firebreaks ===
        for fire_break, break_width in self.fire_breaks:
            if isinstance(fire_break, LineString):
                x, y = fire_break.xy
                self.h_ax.plot(x, y, color='blue', linewidth=self.meters_to_points(break_width))

        # === Legend ===
        if self.legend_elements and self.show_legend:
            self.h_ax.legend(handles=self.legend_elements, loc='upper right', borderaxespad=0)

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
            sm = ScalarMappable(norm=self.wind_norm, cmap='jet')
            sm.set_array([])
            self.wind_cbar = self.fig.colorbar(
                sm, ax=self.h_ax, orientation='vertical', shrink=0.7,
                pad=0.02, label='Wind speed (m/s)'
            )

            self.wind_cbar.ax.tick_params(labelsize=8)

    def _setup_grid(self, init_entries: list[CellLogEntry]) -> None:

        tree_patches = []
        fire_patches = []
        burnt_patches = []
        alpha_arr = []

        self.added_colors = []
        self.legend_elements = []

        for entry in init_entries:

            polygon = mpatches.RegularPolygon((entry.x, entry.y), 
                                              numVertices=6, radius=self.cell_size, orientation=0, zorder=4)
            
            if entry.state == CellStates.FUEL:
                color = fc.fuel_color_mapping[entry.fuel]

                if color not in self.added_colors and self.show_legend:
                    self.added_colors.append(color)
                    fuel_name = fc.fuel_names[entry.fuel]
                    self.legend_elements.append(mpatches.Patch(color=color,
                                                               label=fuel_name))

                polygon.set(color=color)
                tree_patches.append(polygon)

            elif entry.state == CellStates.FIRE:
                fire_patches.append(polygon)
                alpha_arr.append(entry.I_ss)


            else:
                burnt_patches.append(polygon)

        fuel_coll = PatchCollection(tree_patches, match_original=True)

        if fire_patches:
            fire_coll = PatchCollection(fire_patches, edgecolor='none',  facecolor='#F97306')
            norm = mcolors.LogNorm(vmin=max(min(alpha_arr), 1e-3), vmax=max(alpha_arr))
            fire_coll.set_array(alpha_arr)
            fire_coll.set_cmap(mpl.colormaps["gist_heat"])
            fire_coll.set_norm(norm)

        burnt_coll = PatchCollection(burnt_patches, edgecolor='none', facecolor='k')

        self.collections = [copy.copy(fuel_coll), copy.copy(fire_coll), copy.copy(burnt_coll)]

        self.h_ax.add_collection(fuel_coll)
        self.h_ax.add_collection(fire_coll)
        self.h_ax.add_collection(burnt_coll)

        self._init_static_elements()

    
    def update_grid(self, sim_time_s: float, entries: list[CellLogEntry], agents: list[AgentLogEntry] = []) -> None:
        """_summary_

        Args:
            entries (list): _description_
        """

        fire_patches = []
        tree_patches = []
        burnt_patches = []
        crown_patches = []
        alpha_arr = []

        soak_xs = []
        soak_ys = []
        c_vals = []

        wind_idx = int(np.floor((sim_time_s / self.wind_t_step)))

        if self.show_wind_field and wind_idx != self.wind_idx and wind_idx < len(self.wind_forecast):
            self.wind_idx = wind_idx

            if self.wind_grid is not None:
                self.wind_grid.remove()

            curr_forecast = self.wind_forecast[self.wind_idx]

            # Determine number of samples in each dimension based on desired spacing
            n_rows, n_cols = curr_forecast.shape[:2]

            # TODO: Decide if we would like to downsample the wind grid or make it user configurable
            # desired_spacing = self.width_m / 6
            # desired_num_rows = max(int(np.round(n_rows * self.wind_res / desired_spacing)), 2)
            # desired_num_cols = max(int(np.round(n_cols * self.wind_res / desired_spacing)), 2)

            row_indices = np.linspace(0, n_rows - 1, n_rows, dtype=int)
            col_indices = np.linspace(0, n_cols - 1, n_cols, dtype=int)

            # Index wind fields
            wind_speed = curr_forecast[np.ix_(row_indices, col_indices, [0])][:, :, 0]
            wind_dir_deg = curr_forecast[np.ix_(row_indices, col_indices, [1])][:, :, 0]

            X, Y = np.meshgrid(
                (col_indices + 0.5) * self.wind_res,
                (row_indices + 0.5) * self.wind_res
            )

            # Convert wind speed and direction to u and v components
            U = np.sin(np.deg2rad(wind_dir_deg))
            V = np.cos(np.deg2rad(wind_dir_deg))

            mag = np.sqrt(U**2 + V**2)

            U_norm = U / mag
            V_norm = V / mag

            # Plot the wind vectors
            self.wind_grid = self.h_ax.quiver(
                X + self.wind_xpad, Y + self.wind_ypad, U_norm, V_norm, wind_speed,
                scale=20, cmap='jet', norm=self.wind_norm,
                width=0.003, zorder=2, alpha=0.5)

        for entry in entries:
            polygon = mpatches.RegularPolygon((entry.x, entry.y), numVertices=6,
                                              radius=self.cell_size, orientation=0)
            
            if entry.state == CellStates.FUEL:
                # TODO: add scaling of rbga color for fuel content
                
                color = fc.fuel_color_mapping[entry.fuel]
                polygon.set(color=color)
                tree_patches.append(polygon)

            elif entry.state == CellStates.FIRE and entry.crown_state != CrownStatus.NONE:
                crown_patches.append(polygon)

            elif entry.state == CellStates.FIRE:
                fire_patches.append(polygon)
                alpha_arr.append(entry.I_ss)

            else:
                burnt_patches.append(polygon)


        # TODO: handle soak_xs and ys for when suppressant has been dropped in a cell

        fuel_coll = PatchCollection(tree_patches, match_original=True)

        if fire_patches:
            fire_coll = PatchCollection(fire_patches, edgecolor='none', facecolor='#F97306')
            norm = mcolors.LogNorm(vmin=max(min(alpha_arr), 1e-3), vmax=max(alpha_arr))
            fire_coll.set_array(alpha_arr)
            fire_coll.set_cmap(mpl.colormaps["gist_heat"])
            fire_coll.set_norm(norm)

        crown_coll = PatchCollection(crown_patches, edgecolor ='none', facecolor ='magenta')

        burnt_coll = PatchCollection(burnt_patches, edgecolor='none', facecolor='k')

        self.h_ax.add_collection(fuel_coll)
        self.h_ax.add_collection(fire_coll)
        self.h_ax.add_collection(burnt_coll)
        self.h_ax.add_collection(crown_coll)

        # Set time displays based on sim time
        sim_datetime = self._start_datetime + timedelta(seconds=sim_time_s)
        datetime_str = sim_datetime.strftime("%Y-%m-%d %H:%M")
        self.datetime_text.set_text(datetime_str)

        time_str = util.get_time_str(sim_time_s)
        self.elapsed_text.set_text(time_str)

        # Plot agents at current time if they exist
        if agents:
            if self.agent_art is not None:
                for a in self.agent_art:
                    a.remove()
            if self.agent_labels is not None:
                for label in self.agent_labels:
                    label.remove()

            self.agent_art = []
            for agent_entry in agents:
                a = self.h_ax.scatter(agent_entry.x, agent_entry.y, marker=agent_entry.marker, color=agent_entry.color)
                self.agents.append(a)

            self.agent_labels = []
            for agent_entry in agents:
                if agent_entry.label is not None:
                    label = self.h_ax.annotate(agent_entry.label, (agent_entry.x, agent_entry.y))
                    self.agent_labels.append(label)

        if self.render:
            self.fig.canvas.blit(self.h_ax.bbox)
            self.fig.canvas.flush_events()

    def visualize_prediction(self, prediction):
        """Visualizes a prediction grid on top of the current simulation visualization.
        
        Args:
            prediction_grid (dict): Dictionary mapping timestamps to lists of (x,y) coordinates
                                  representing predicted fire spread
        """
        # Clear any existing prediction visualization
        if hasattr(self, 'prediction_scatter'):
            self.prediction_scatter.remove()
            delattr(self, 'prediction_scatter')

        time_steps = sorted(prediction.keys())
        if not time_steps:
            return

        cmap = mpl.cm.get_cmap("Oranges_r")
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
                                                      alpha=0.3, zorder=1)
            self.fig.canvas.draw()
    
    def close(self):
        """Closes the visualizer window and cleans up the figure."""
        if self.fig and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
            self.fig = None

    def reset_figure(self, done=False):
        """Resets the visualizer to its initial state, optionally closing if simulation is done."""
        if done:
            self.close()
            return

        # Clear current figure
        self.h_ax.clear()

        # Re-set up axes and elements
        self._setup_figure()
        self._init_static_elements()

        # Re-add all initial patches stored in self.collections
        for coll in self.collections:
            self.h_ax.add_collection(copy.copy(coll))

        # Reset wind field
        self.wind_grid = None
        self.wind_idx = -1
        self._process_wind()

        # Redraw canvas from initial background
        self.fig.canvas.draw()
        self.initial_state = self.fig.canvas.copy_from_bbox(self.h_ax.bbox)
        self.fig.canvas.blit(self.h_ax.bbox)
        self.fig.canvas.flush_events()
 
    def meters_to_points(self, meters):
        fig_width_inch, _ = self.fig.get_size_inches()
        meters_per_inch = self.width_m / fig_width_inch
        meters_per_point = meters_per_inch / 72
        return meters / meters_per_point