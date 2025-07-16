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

class BaseVisualizer:
    def __init__(self, params: VisualizerInputs, render=True):
        self.render = render

        if not self.render:
            mpl.use('Agg')  # Use a non-interactive backend if not rendering

        else:
            mpl.use('QtAgg')

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

        self.temp_forecast = params.temp_forecast
        self.rh_forecast = params.rh_forecast
        self.forecast_t_step = params.forecast_t_step
        self.forecast_idx = -1

        self.north_dir_deg = params.north_dir_deg
        self._start_datetime = params.start_datetime

        self.scale_bar_km = params.scale_bar_km
        self.show_legend = params.show_legend
        self.show_wind_cbar = False
        self.show_wind_field = False
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
        if self.show_wind_field:
            all_speeds = [forecast[:, :, 0] for forecast in self.wind_forecast]
            self.global_max_speed = max(np.max(s) for s in all_speeds)
            self.wind_norm = mcolors.Normalize(vmin=0, vmax=self.global_max_speed)

        if self.show_weather_data:
            if not self.show_temp_in_F:
                self.temp_forecast = [np.round(F_to_C(temp), 1) for temp in self.temp_forecast]

    def _setup_figure(self):
        if self.render:
            plt.ion()

        self.fig = plt.figure(figsize=(9, 8))
        self.h_ax = self.fig.add_axes([0.05, 0.05, 0.9, 0.9])

        self.h_ax.set_aspect('equal')
        self.h_ax.axis([0, self.width_m, 0, self.height_m])
        plt.tick_params(left=False, right=False, bottom=False,
                        labelleft=False, labelbottom=False)

    def _setup_grid(self, init_entries: list[CellLogEntry]) -> None:

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
                    self.legend_elements.append(mpatches.Patch(color=fuel_color, label=fc.fuel_names[entry.fuel]))

        # Assign initial facecolors based on cell state
        self.cell_colors = [self._get_cell_color(entry) for entry in init_entries]
        self.cell_id_to_index = {
            entry.id: i for i, entry in enumerate(init_entries)
        }

        # Create a single PatchCollection for all cells
        self.all_cells_coll = PatchCollection(self.all_polygons, facecolors=self.cell_colors, match_original=True, zorder=1)
        self.h_ax.add_collection(self.all_cells_coll)

        self._init_static_elements()

    def _get_cell_color(self, entry: CellLogEntry):
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
        # Update weather data if needed
        weather_idx = int(np.floor(sim_time_s / self.forecast_t_step))
        if self.show_weather_data and weather_idx != self.forecast_idx and weather_idx < len(self.temp_forecast):
            self.forecast_idx = weather_idx
            temp_unit = "F" if self.show_temp_in_F else "C"
            weather_str = f"Temp: {self.temp_forecast[self.forecast_idx]} °{temp_unit}, RH: {self.rh_forecast[self.forecast_idx]} %"
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
                    self.h_ax.plot(action.x_coords, action.y_coords, color='blue', linewidth=self.meters_to_points(action.width), zorder=4)
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
        if self.fig and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
            self.fig = None

    def reset_figure(self, done=False):
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
                self.h_ax.plot(x, y, color=road_color, linewidth=self.meters_to_points(road_width), zorder=2)
                if road_color not in added_colors and self.show_legend:
                    added_colors.add(road_color)
                    self.legend_elements.append(mpatches.Patch(color=road_color,
                                                            label=f"Road - {road_type}"))

        # === Firebreaks ===
        for fire_break, break_width, _ in self.fire_breaks:
            if isinstance(fire_break, LineString):
                x, y = fire_break.xy
                self.h_ax.plot(x, y, color='blue', linewidth=self.meters_to_points(break_width), zorder=2)

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
            sm = ScalarMappable(norm=self.wind_norm, cmap='turbo')
            sm.set_array([])
            self.wind_cbar = self.fig.colorbar(
                sm, ax=self.h_ax, orientation='vertical', shrink=0.7,
                pad=0.02, label='Wind speed (m/s)'
            )

            self.wind_cbar.ax.tick_params(labelsize=8)


    def meters_to_points(self, meters):
        fig_width_inch, _ = self.fig.get_size_inches()
        meters_per_inch = self.width_m / fig_width_inch
        meters_per_point = meters_per_inch / 72
        return meters / meters_per_point


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