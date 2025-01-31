"""Module responsible for visualization of simulations in real-time

.. autoclass:: Visualizer
    :members:
"""

import copy
from shapely.geometry import LineString
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib import cm
import numpy as np

from embrs.utilities.fire_util import CellStates, FireTypes
from embrs.utilities.fire_util import FuelConstants as fc
from embrs.utilities.fire_util import RoadConstants as rc
from embrs.utilities.fire_util import UtilFuncs as util
from embrs.fire_simulator.fire import FireSim

mpl.use('TkAgg')

class Visualizer:
    """Class that visualizes simulations in real-time

    :param sim: :class:`~fire_simulator.fire.FireSim` instance to visualize
    :type sim: FireSim
    :param artists: list of artists that should be drawn initially, useful for quickly starting
                    numerous visualizations one after another, defaults to None
    :type artists: list, optional
    :param collections: list of collection objects that should be drawn initially, useful for
                        quickly starting numerous visualizations one after another, defaults to None
    :type collections: list, optional
    :param saved_legend: saved legend box, useful for quickly starting
                         numerous visualizations one after another, defaults to None
    :type saved_legend: Axes.legend, optional
    :param scale_bar_km: Determines how much distance the scale bar should represent in km,
                         defaults to 1.0
    :type scale_bar_km: int, optional
        """
    def __init__(self, sim: FireSim, artists:list=None, collections:list=None,
                 saved_legend:Axes.legend= None, scale_bar_km:float = 1.0):
        """Constructor method that initializes a visualization by populating all patches for cells
        and drawing all initial artists"""
        self.sim = sim

        width_m = sim.cell_size * np.sqrt(3) * sim.grid_width
        height_m = sim.cell_size * 1.5 * sim.grid_height

        self.agents = None
        self.agent_labels = None

        plt.ion()
        h_fig = plt.figure(figsize=(10, 10))
        h_ax = h_fig.add_axes([0.05, 0.05, 0.9, 0.9])

        # Create meshgrid for plotting contours
        x = np.arange(0, sim.shape[1])
        y = np.arange(0, sim.shape[0])
        X, Y = np.meshgrid(x, y)

        cont = h_ax.contour(X*sim.cell_size*np.sqrt(3),Y*sim.cell_size*1.5,
                            sim.coarse_topography, colors='k')

        h_ax.clabel(cont, inline=True, fontsize=10, zorder=2)

        if artists is None or collections is None:
            print("Initializing visualization... ")
            burnt_patches = []
            # alpha_arr = [0, 1]
            break_fuel_arr = [0, 1]

            # Add low and high polygons to prevent weird color mapping
            r = 1/np.sqrt(3)
            low_poly = mpatches.RegularPolygon((-10,-10), numVertices=6, radius=r, orientation=0)
            high_poly = mpatches.RegularPolygon((-10,-10), numVertices=6, radius=r, orientation=0)
            fire_patches = [low_poly, high_poly]
            prescribe_patches = [low_poly, high_poly]
            tree_patches = [low_poly, high_poly]
            fire_breaks = [low_poly, high_poly]

            legend_elements = []
            added_colors = []

            # Add patches for each cell
            for i in range(sim.shape[0]):
                for j in range(sim.shape[1]):
                    curr_cell = sim.cell_grid[i][j]

                    polygon = mpatches.RegularPolygon((curr_cell.x_pos, curr_cell.y_pos),
                                     numVertices=6, radius=sim.cell_size, orientation=0)

                    if curr_cell.state == CellStates.FUEL:
                        color = fc.fuel_color_mapping[curr_cell.fuel_type.model_num]
                        if color not in added_colors:
                            added_colors.append(color)
                            legend_elements.append(mpatches.Patch(color = color,
                                                label = curr_cell.fuel_type.name))

                        # if curr_cell.fuel_content < 1 and curr_cell.fuel_type.burnable:
                        #     fire_breaks.append(polygon)
                        #     break_fuel_arr.append(curr_cell.fuel_content)

                        # else:
                        polygon.set(color = color)
                        tree_patches.append(polygon)

                    elif curr_cell.state == CellStates.FIRE:
                        if curr_cell.fire_type == FireTypes.WILD:
                            fire_patches.append(polygon)
                            # alpha_arr.append(curr_cell.fuel_content)
                        else:
                            prescribe_patches.append(polygon)

                    else:
                        burnt_patches.append(polygon)

            # Create collections grouping cells in each of their states
            tree_coll =  PatchCollection(tree_patches, match_original = True)

            if len(fire_breaks) > 0:
                breaks_coll = PatchCollection(fire_breaks, edgecolor='none')
                breaks_coll.set(array= break_fuel_arr, cmap=mpl.colormaps["gist_gray"])

            fire_coll = PatchCollection(fire_patches, edgecolor='none', facecolor='#F97306')
            # if len(alpha_arr) > 0:
            #     alpha_arr = [float(i)/sum(alpha_arr) for i in alpha_arr]
            #     fire_coll.set(array=alpha_arr, cmap=mpl.colormaps["gist_heat"])

            prescribe_coll = PatchCollection(prescribe_patches, edgecolor='none', facecolor='b')

            burnt_coll = PatchCollection(burnt_patches, edgecolor='none', facecolor='k')

            self.collections = [copy.copy(breaks_coll), copy.copy(tree_coll),
                                copy.copy(fire_coll), copy.copy(burnt_coll),
                                copy.copy(prescribe_coll)]

            # Add collections to plot
            h_ax.add_collection(breaks_coll)
            h_ax.add_collection(tree_coll)
            h_ax.add_collection(fire_coll)
            h_ax.add_collection(burnt_coll)
            h_ax.add_collection(prescribe_coll)

            # Create time display
            time_box_x = 0
            time_box_y = sim.grid_height*1.5*sim.cell_size-(15/600) * height_m
            time_box_w = (1/6)*width_m
            time_box_h = (15/600)*height_m

            self.time_box = mpatches.Rectangle((time_box_x, time_box_y), time_box_w, time_box_h,
                                                facecolor='white', edgecolor='black', linewidth=1,
                                                zorder=3, alpha = 0.75)

            # Create scale display
            self.scale_box = mpatches.Rectangle((0, 10), 1100, (2/60)*height_m, facecolor='white',
                                                edgecolor='k', linewidth= 1, alpha=0.75, zorder= 3)
            
            # Add display items to artists
            self.artists = [copy.copy(self.time_box),
                            copy.copy(self.scale_box)]

            # Add display items to plot
            h_ax.add_patch(self.scale_box)
            h_ax.add_patch(self.time_box)

            # Plot roads if they exist
            if sim.roads is not None:
                for road in sim.roads:
                    x_trimmed = []
                    y_trimmed = []

                    x, y = zip(*road[0])
                    for i, x_i in enumerate(x):
                        if 0 < x_i/30 < sim.grid_width and 0 < y[i]/30 < sim.grid_height:
                            x_trimmed.append(x_i)
                            y_trimmed.append(y[i])

                    x = tuple(xi for xi in x_trimmed)
                    y = tuple(yi for yi in y_trimmed)

                    road_color = rc.road_color_mapping[road[1]]
                    h_ax.plot(x, y, color= road_color)

                    if road_color not in added_colors:
                        added_colors.append(road_color)
                        legend_elements.append(mpatches.Patch(color=road_color,
                                               label = f"Road - {road[1]}"))

            # Plot firebreaks if they exist
            if sim.fire_breaks is not None:
                # Create a colormap for grey shades
                cmap = mpl.colormaps["Greys_r"]

                for fire_break in sim.fire_breaks:
                    line = fire_break['geometry']
                    fuel_val = fire_break['fuel_value']
                    if isinstance(line, LineString):
                        # Normalize the fuel_val between 0 and 1
                        normalized_fuel_val = fuel_val / 100.0
                        color = cmap(normalized_fuel_val)
                        x, y = line.xy
                        h_ax.plot(x, y, color=color)

            h_ax.legend(handles=legend_elements, loc='upper right', borderaxespad=0)
            self.legend_elements = legend_elements

            # Plot wind vector field
            curr_forecast = sim.wind_forecast[sim._curr_wind_idx]
            # Downsample the wind data for plotting
            downsample_factor = 5
            wind_speed = curr_forecast[::downsample_factor, ::downsample_factor, 0]
            wind_dir_deg = curr_forecast[::downsample_factor, ::downsample_factor, 1]
            X, Y = np.meshgrid(np.arange(0, wind_speed.shape[1]) * sim._wind_res * downsample_factor,
                               np.arange(0, wind_speed.shape[0]) * sim._wind_res * downsample_factor)

            # Convert wind speed and direction to u and v components
            U = np.sin(np.deg2rad(wind_dir_deg))
            V = np.cos(np.deg2rad(wind_dir_deg))

            # Plot the wind vectors
            self.wind_grid = h_ax.quiver(X, Y, U, V, wind_speed, scale=None, cmap= 'jet', width=0.002, zorder=3)
            self.wind_idx = sim._curr_wind_idx

        # Reload visualizer from initial state
        else:
            for coll in collections:
                coll = copy.copy(coll)
                h_ax.add_collection(coll)

            for artist in artists:
                artist = copy.copy(artist)
                h_ax.add_patch(artist)

            # Plot roads if they exist
            if sim.roads is not None:
                for road in sim.roads:
                    x_trimmed = []
                    y_trimmed = []

                    x, y = zip(*road[0])
                    for i, x_i in enumerate(x):
                        if 0 < x_i/30 < sim.grid_width and 0 < y[i]/30 < sim.grid_height:
                            x_trimmed.append(x_i)
                            y_trimmed.append(y[i])

                    x = tuple(xi for xi in x_trimmed)
                    y = tuple(yi for yi in y_trimmed)

                    road_color = fc.fuel_color_mapping[91]
                    h_ax.plot(x, y, color= road_color)

            h_ax.legend(handles=saved_legend, loc='upper right', borderaxespad=0)

        time_str = util.get_time_str(sim.curr_time_s)

        rx, ry = self.time_box.get_xy()
        cx = rx + self.time_box.get_width()/2
        cy = ry + self.time_box.get_height()/2

        self.timeheader = h_ax.text(20, cy, 'time:', ha='left', va='center')

        self.simtext = h_ax.text(2*cx - 20, cy, time_str, ha='right', va='center')

        h_ax.set_aspect('equal')
        h_ax.axis([0, sim.cell_size*sim.shape[1]*np.sqrt(3) - (sim.cell_size*np.sqrt(3)/2),
                   0, sim.cell_size*1.5*sim.shape[0] - (sim.cell_size*1.5)])

        plt.tick_params(left = False, right = False, bottom = False,
                        labelleft = False, labelbottom = False)

        num_cells_scale = scale_bar_km * 1000

        if scale_bar_km < 1:
            scale_size = str(num_cells_scale) + "m"
        else:
            scale_size = str(scale_bar_km) + "km"

        scalebar = AnchoredSizeBar(h_ax.transData, num_cells_scale, scale_size, 'lower left',
                                   color ='k', pad = 0.1, frameon=False)
        h_ax.add_artist(scalebar)


        self.h_ax = h_ax
        self.fig = h_fig

        self.fig.canvas.draw()
        self.initial_state = self.fig.canvas.copy_from_bbox(self.h_ax.bbox)

        plt.pause(1)

    def update_grid(self, sim: FireSim):
        """Updates the grid based on the current state of the simulation, this function is called
        at a frequency of sim.display_freq_s set in the FireSim constructor.

        :param sim: FireSim instance to display
        :type sim: FireSim
        """
        self.simtext.set_visible(False)

        if self.wind_idx != sim._curr_wind_idx:
            self.wind_grid.remove()
            curr_forecast = sim.wind_forecast[sim._curr_wind_idx]

            # Downsample the wind data for plotting
            downsample_factor = 5
            wind_speed = curr_forecast[::downsample_factor, ::downsample_factor, 0]
            wind_dir_deg = curr_forecast[::downsample_factor, ::downsample_factor, 1]
            X, Y = np.meshgrid(np.arange(0, wind_speed.shape[1]) * sim._wind_res * downsample_factor,
                                np.arange(0, wind_speed.shape[0]) * sim._wind_res * downsample_factor)

            # Convert wind speed and direction to u and v components
            U = np.sin(np.deg2rad(wind_dir_deg))
            V = np.cos(np.deg2rad(wind_dir_deg))

            # Plot the wind vectors
            self.wind_grid = self.h_ax.quiver(X, Y, U, V, wind_speed, scale=None, cmap= 'jet', width=0.002, zorder=3)
            self.wind_idx = sim._curr_wind_idx

        fire_patches = []
        tree_patches = []
        burnt_patches = []
        # alpha_arr = [0, 1]

        # Add low and high polygons to prevent weird color mapping
        r = 1/np.sqrt(3)
        low_poly = mpatches.RegularPolygon((-10, -10), numVertices=6, radius=r,orientation=0)
        high_poly = mpatches.RegularPolygon((-10, -10), numVertices=6, radius=r,orientation=0)
        fire_patches = [low_poly, high_poly]
        tree_patches = [low_poly, high_poly]
        prescribe_patches = [low_poly, high_poly]

        soak_xs = []
        soak_ys = []
        c_vals = []

        for c in sim.updated_cells.values():
            polygon = mpatches.RegularPolygon((c.x_pos, c.y_pos), numVertices=6,
                                              radius=sim.cell_size, orientation=0)

            if c.state == CellStates.FUEL:
                color = np.array(list(mcolors.to_rgba(fc.fuel_color_mapping[c.fuel_type.model_num])))
                # Scale color based on cell's fuel content
                color = color # *  c.fuel_content
                polygon.set_facecolor(color)
                tree_patches.append(polygon)

                if c.dead_m > 0.08: # fuel moisture not nominal
                    soak_xs.append(c.x_pos)
                    soak_ys.append(c.y_pos)
                    c_val = c.dead_m/fc.dead_fuel_moisture_ext_table[c.fuel_type.model_num] # TODO: Use FuelModel value
                    c_val = np.min([1, c_val])
                    c_vals.append(c_val)

            elif c.state == CellStates.FIRE:
                if c.fire_type == FireTypes.WILD:
                    fire_patches.append(polygon)
                    # alpha_arr.append(c.fuel_content)
                else:
                    prescribe_patches.append(polygon)

            else:
                burnt_patches.append(polygon)

        color_map = cm.get_cmap('Blues')
        norm = mcolors.Normalize(vmin=0, vmax=1)
        self.h_ax.scatter(soak_xs, soak_ys, c=c_vals, cmap = color_map, marker='2', norm=norm)

        tree_patches = np.array(tree_patches)
        fire_patches = np.array(fire_patches)
        prescribe_patches = np.array(prescribe_patches)
        burnt_patches = np.array(burnt_patches)
        # alpha_arr = np.array(alpha_arr)

        tree_coll =  PatchCollection(tree_patches, match_original=True)

        fire_coll = PatchCollection(fire_patches, edgecolor='none', facecolor='#F97306')
        # if len(alpha_arr) > 0:
        #     fire_coll.set(array=alpha_arr, cmap=mpl.colormaps["gist_heat"])

        prescribe_coll = PatchCollection(prescribe_patches, edgecolor='none', facecolor='pink')

        burnt_coll = PatchCollection(burnt_patches, edgecolor='none', facecolor='k')

        self.h_ax.add_collection(tree_coll)
        self.h_ax.add_collection(fire_coll)
        self.h_ax.add_collection(burnt_coll)
        self.h_ax.add_collection(prescribe_coll)


        sim_time_s = sim.time_step*sim.iters
        time_str = util.get_time_str(sim_time_s)

        rx, ry = self.time_box.get_xy()
        cx = rx + self.time_box.get_width()/2
        cy = ry + self.time_box.get_height()/2

        self.simtext = self.h_ax.text(2*cx - 20, cy, time_str, ha='right', va='center')



        # Plot agents at current time if they exist
        if len(sim.agent_list) > 0:
            if self.agents is not None:
                for a in self.agents:
                    a.remove()
            if self.agent_labels is not None:
                for label in self.agent_labels:
                    label.remove()

            self.agents = []
            for agent in sim.agent_list:
                a = self.h_ax.scatter(agent.x, agent.y, marker=agent.marker, color=agent.color)
                self.agents.append(a)

            self.agent_labels = []
            for agent in sim.agent_list:
                if agent.label is not None:
                    label = self.h_ax.annotate(agent.label, (agent.x, agent.y))
                    self.agent_labels.append(label)

        sim.updated_cells.clear()

        self.fig.canvas.blit(self.h_ax.bbox)
        self.fig.canvas.flush_events()

    def reset_figure(self, done=False):
        """Resets the figure back to its initial state. Used to reset the figure between different
        simulation runs

        :param done: Closes the figure when set to True, resets and continues to next visualization
                     if False, defaults to False
        :type done: bool, optional
        """
        # Close figure
        plt.close(self.fig)

        if not done:
            self.__init__(self.sim, self.artists,
                          self.collections, self.legend_elements)

    def visualize_prediction(self, prediction_grid):
        print("visualzing prediction...")

        time_steps = sorted(prediction_grid.keys())

        cmap = mpl.colormaps["viridis"]
        norm = plt.Normalize(time_steps[0], time_steps[-1])

        for time in prediction_grid.keys():
            x,y = zip(*prediction_grid[time])
            self.h_ax.scatter(x,y, c=[cmap(norm(time))])

        print("Finished visualizing....")
