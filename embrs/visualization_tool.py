"""Module responsible for running the visualization tool that allows users to load log files and
visualize them.
"""
import pickle
import sys
import os
import matplotlib as mpl
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from matplotlib import cm
from shapely.geometry import LineString
import pandas as pd
import numpy as np
import msgpack

from embrs.utilities.file_io import VizFolderSelector, LoaderWindow
from embrs.utilities.fire_util import CellStates
from embrs.utilities.fire_util import FuelConstants as fc, RoadConstants as rc, UtilFuncs as util


class VisualizationTool:
    """Class for visualizing fire simulations from their log files

    :param data: dictionary of data retrieved from visualization tool's file IO interface
    :type data: dict
    """
    def __init__(self, data: dict):
        """Constructor method that takes in dictionary of visualization data and generates the
        initial figure.
        """
        if data is None:
            sys.exit(0)

        self.loader_window = LoaderWindow('Loading...', max_value=8)
        self.loader_window.set_text("Loading sim data...")

        log_file = data['file']
        self.update_freq_s = data['freq']
        self.scale_km = data['scale_km']
        self.show_legend = data['legend']
        init_location = data['init_location']
        self.has_agents = data['has_agents']

        if self.has_agents:
            agent_file = data['agent_file']

        filename = os.path.basename(log_file)
        run_folder = os.path.dirname(log_file)
        log_folder = os.path.dirname(run_folder)

        # Get initial state of the system
        if init_location:
            init_path = f"{log_folder}/init_fire_state.pkl"
        else:
            init_path = f"{os.path.dirname(log_folder)}/init_fire_state.pkl"

        with open(init_path, 'rb') as f:
            self.sim = pickle.load(f)

        # Get wind object
        self.wind = self.sim['wind_vec']

        grid_width = self.sim['grid_width']
        grid_height = self.sim['grid_height']
        cell_size = self.sim['cell_size']

        width_m = grid_width * np.sqrt(3) * cell_size
        height_m = grid_height * 1.5 * cell_size

        # Get all sim data
        with open(run_folder + '/' + filename, 'rb') as f:
            self.data = msgpack.unpack(f, strict_map_key = False)

        self.loader_window.set_text("Clustering data based on display frequency...")
        # Cluster the data based on update frequency
        self.cluster_data()

        # Get agent data if present
        if self.has_agents:
            with open(agent_file, 'rb') as f:
                self.agent_data = pickle.load(f)

        self.loader_window.increment_progress()
        self.loader_window.set_text("Initializing figure...")

        # Set axis parameters
        h_fig = plt.figure(figsize=(10, 10))
        h_ax = h_fig.add_axes([0.05, 0.05, 0.9, 0.9])
        h_ax.set_aspect('equal')
        h_ax.axis([0, self.sim['cell_size']*self.sim['grid_width']*np.sqrt(3) -
                 (self.sim['cell_size']*np.sqrt(3)/2), 0,
                  self.sim['cell_size']*1.5*self.sim['grid_height'] - (self.sim['cell_size']*1.5)])

        plt.tick_params(left = False,
                        right = False,
                        bottom = False,
                        labelleft = False,
                        labelbottom = False)

        self.loader_window.increment_progress()
        self.loader_window.set_text("Generating contour map...")

        # Generate contour map
        x = np.arange(0, self.sim['grid_width']+1)
        y = np.arange(0, self.sim['grid_height']+1)
        X, Y = np.meshgrid(x, y)

        cont = h_ax.contour(X*self.sim['cell_size']*np.sqrt(3),Y*self.sim['cell_size']*1.5,
                            self.sim['coarse_elevation'], colors='k', zorder=2)
        h_ax.clabel(cont, inline=True, fontsize=10, zorder=3)

        self.loader_window.increment_progress()
        self.loader_window.set_text("Adding patches for each cell...")

        # Add low and high polygons to prevent weird color mapping
        low_poly = mpatches.RegularPolygon((-10, -10), numVertices=6,
                                           radius=1/np.sqrt(3),orientation=0)

        high_poly = mpatches.RegularPolygon((-10, -10), numVertices=6,
                                            radius=1/np.sqrt(3),orientation=0)

        fire_patches = [low_poly, high_poly]
        tree_patches = [low_poly, high_poly]
        burnt_patches = []
        fire_breaks = [low_poly, high_poly]
        alpha_arr = [0, 1]
        break_fuel_arr = [0, 1]

        added_colors = set()
        legend_elements = []

        # Draw each cell based on initial states
        for curr_cell in self.sim['cell_dict'].values():
            polygon = mpatches.RegularPolygon((curr_cell.x_pos, curr_cell.y_pos),
                                               numVertices=6, radius=self.sim['cell_size'],
                                               orientation=0)

            if curr_cell.state == CellStates.FUEL:
                color = fc.fuel_color_mapping[curr_cell.fuel_type.fuel_type]
                if color not in added_colors:
                    added_colors.add(color)
                    legend_elements.append(mpatches.Patch(color = color,
                                                          label = curr_cell.fuel_type.name))

                if curr_cell.fuel_content < 1 and curr_cell.fuel_type.fuel_type < 90:
                    fire_breaks.append(polygon)
                    break_fuel_arr.append(curr_cell.fuel_content)

                else:
                    polygon.set(color = color)
                    tree_patches.append(polygon)

            elif curr_cell.state == CellStates.FIRE:
                fire_patches.append(polygon)
                alpha_arr.append(curr_cell.fuel_content)
                
            else:
                burnt_patches.append(polygon)

        # Add all patches to collections
        if tree_patches:
            tree_patches = np.array(tree_patches)
            tree_coll =  PatchCollection(tree_patches, match_original=True)
            h_ax.add_collection(tree_coll)
        if fire_patches:
            fire_patches = np.array(fire_patches)
            fire_coll = PatchCollection(fire_patches, edgecolor='none', facecolor='#F97306')
            fire_coll.set(array=np.array(alpha_arr), cmap=mpl.colormaps["gist_heat"])
            h_ax.add_collection(fire_coll)
        if burnt_patches:
            burnt_patches = np.array(burnt_patches)
            burnt_coll = PatchCollection(burnt_patches, edgecolor='none', facecolor='k')
            h_ax.add_collection(burnt_coll)

        if len(fire_breaks) > 0:
            breaks_coll = PatchCollection(fire_breaks, edgecolor='none')
            breaks_coll.set(array= break_fuel_arr, cmap=mpl.colormaps["gist_gray"])
            h_ax.add_collection(breaks_coll)

        self.loader_window.increment_progress()
        self.loader_window.set_text("Adding info artists...")

        # Generate time display area
        self.time_box = mpatches.Rectangle((0,self.sim['grid_height']*1.5*self.sim['cell_size']
                                            -(150/6000)*height_m), (1000/6000)*width_m,
                                            (150/6000)*height_m, facecolor='white',
                                            edgecolor='black', linewidth=1, zorder=3, alpha = 0.75)
        h_ax.add_patch(self.time_box)

        sim_time_s = 0
        time_str = util.get_time_str(sim_time_s)

        rx, ry = self.time_box.get_xy()
        cx = rx + self.time_box.get_width()/2
        cy = ry + self.time_box.get_height()/2

        self.simtext = h_ax.text(2*cx - 20, cy, time_str, ha='right', va='center')
        timeheader = h_ax.text(20, cy, 'Time: ', ha='left',va ='center')
        h_ax.add_artist(timeheader)

        # Generate wind display area
        self.wind_box = mpatches.Rectangle((0, self.sim['grid_height']*1.5*self.sim['cell_size']
                                            -(650/6000)*height_m), (500/6000)*width_m,
                                            (500/6000)*height_m, facecolor='white',
                                            edgecolor = 'black', linewidth=1, zorder = 3,
                                            alpha = 0.75)

        h_ax.add_patch(self.wind_box)

        wx, wy = self.wind_box.get_xy()
        cx = wx + self.wind_box.get_width()/2

        windheader = h_ax.text(cx, wy + 0.85 * self.wind_box.get_height(), 'Wind:', ha = 'center',
                               va = 'center')

        h_ax.add_artist(windheader)

        # Generate scale area
        scale_box = mpatches.Rectangle((0, 10), self.scale_km*1000 + 100, (200/6000)*height_m,
                                       facecolor='white', edgecolor='k', linewidth= 1, alpha=0.75,
                                       zorder= 3)
        h_ax.add_patch(scale_box)

        num_cells_scale = self.scale_km * 1000

        if self.scale_km < 1:
            scale_size = str(self.scale_km * 1000) + "m"
        else:
            scale_size = str(self.scale_km) + "km"

        scalebar = AnchoredSizeBar(h_ax.transData, num_cells_scale, scale_size, 'lower left',
                                   color ='k', pad = 0.1, frameon=False)

        h_ax.add_artist(scalebar)

        self.loader_window.increment_progress()

        # Plot roads if they exist
        if 'roads' in self.sim.keys():
            self.loader_window.set_text("Plotting roads...")
            roads = self.sim['roads']

            for road in roads:
                x_trimmed = []
                y_trimmed = []

                x, y = zip(*road[0])
                for i in range(len(x)):
                    if 0 <x[i]/30 <self.sim['grid_width'] and 0 < y[i]/30 <self.sim['grid_height']:
                        x_trimmed.append(x[i])
                        y_trimmed.append(y[i])

                x = tuple(xi for xi in x_trimmed)
                y = tuple(yi for yi in y_trimmed)

                road_color = rc.road_color_mapping[road[1]]

                h_ax.plot(x, y, color= road_color)

                if road_color not in added_colors:
                    added_colors.add(road_color)
                    legend_elements.append(mpatches.Patch(color=road_color,
                                                          label = f"Road - {road[1]}"))


        fire_breaks = self.sim['fire_breaks']

        # Create a colormap for grey shades
        cmap = mpl.colormaps["Greys_r"]

        for fire_break in fire_breaks:
            line = fire_break['geometry']
            fuel_val = fire_break['fuel_value']
            if isinstance(line, LineString):
                # Normalize the fuel_val between 0 and 1
                normalized_fuel_val = fuel_val / 100.0
                color = cmap(normalized_fuel_val)
                x, y = line.xy
                h_ax.plot(x, y, color=color)

        self.loader_window.increment_progress()

        # Add legend area
        if self.show_legend:
            h_ax.legend(handles=legend_elements, loc='upper right', borderaxespad=0)

        # Create class objects
        self.arrow_obj = None
        self.windtext = None
        self.anim = None
        self.static_artists = []
        self.h_ax = h_ax
        self.fig = h_fig

        self.loader_window.increment_progress()
        self.loader_window.set_text("Rendering...")

        # Pause to allow rendering to complete
        plt.pause(1)

        self.loader_window.close()

    def update_viz(self, frame: int) -> list:
        """Function that updates the visualization based on all the changes to the state since the
        last frame

        :param frame: current frame the visualization is on
        :type frame: int
        :return: list of artists drawn
        :rtype: list
        """
        # Create an array to hold elements to be changed
        artists = []

        # Reset static artists if viz has restarted
        if frame == 0:
            self.static_artists = []

        # Get current time in seconds
        curr_time = frame * self.update_freq_s

        # Get updated cells for this iteration
        updated_cells = self.data[frame]

        # Update wind vector
        self.wind._update_wind(curr_time)

        # Add low and high polygons to prevent weird color mapping
        low_poly = mpatches.RegularPolygon((-10, -10), numVertices=6, radius=1/np.sqrt(3),
                                            orientation=0)

        high_poly = mpatches.RegularPolygon((-10, -10), numVertices=6, radius=1/np.sqrt(3),
                                            orientation=0)

        fire_patches = [low_poly, high_poly]
        tree_patches = [low_poly, high_poly]
        burnt_patches = []

        alpha_arr = [0,1]

        soak_xs = []
        soak_ys = []
        c_vals = []

        # Get initial state to apply static properties of cells
        cell_dict = self.sim['cell_dict']

        # Draw all the cells that need to be updated
        for cell in updated_cells:
            c = cell_dict[cell['id']]

            polygon = mpatches.RegularPolygon((c.x_pos, c.y_pos), numVertices=6,
                                              radius=self.sim['cell_size'], orientation=0)

            if cell['state'] == CellStates.FUEL:
                color=np.array(list(mcolors.to_rgba(fc.fuel_color_mapping[c.fuel_type.fuel_type])))
                # Scale color based on cell's fuel content
                color = color *  cell['fuel_content']
                polygon.set_facecolor(color)
                tree_patches.append(polygon)

                if cell['dead_m'] > 0.08: # fuel moisture not nominal
                    soak_xs.append(c.x_pos)
                    soak_ys.append(c.y_pos)
                    c_val = cell['dead_m']/fc.dead_fuel_moisture_ext_table[c.fuel_type.fuel_type]
                    c_val = np.min([1, c_val])
                    c_vals.append(c_val)

            elif cell['state'] == CellStates.FIRE:
                fire_patches.append(polygon)
                alpha_arr.append(cell['fuel_content'])
            else:
                burnt_patches.append(polygon)

        color_map = cm.get_cmap('Blues')
        norm = mcolors.Normalize(vmin=0, vmax=1)
        moisture_viz = self.h_ax.scatter(soak_xs, soak_ys, c=c_vals, cmap=color_map, marker='2',
                                         norm=norm)

        self.static_artists.append(moisture_viz)

        # Add patches to collections
        if tree_patches:
            tree_patches = np.array(tree_patches)
            tree_coll =  PatchCollection(tree_patches, match_original=True)
            self.h_ax.add_collection(tree_coll)
            artists.append(tree_coll)
        if fire_patches:
            fire_patches = np.array(fire_patches)
            fire_coll = PatchCollection(fire_patches, edgecolor='none', facecolor='#F97306')
            fire_coll.set(array=np.array(alpha_arr), cmap=mpl.colormaps["gist_heat"])
            self.h_ax.add_collection(fire_coll)
            artists.append(fire_coll)
        if burnt_patches:
            burnt_patches = np.array(burnt_patches)
            burnt_coll = PatchCollection(burnt_patches, edgecolor='none', facecolor='k')
            artists.append(burnt_coll)
            self.h_ax.add_collection(burnt_coll)
            self.static_artists.append(burnt_coll)

        # Update wind text
        wx, wy = self.wind_box.get_xy()
        cx = wx + self.wind_box.get_width()/2
        cy = wy + self.wind_box.get_height()/2

        self.windtext = self.h_ax.text(cx, wy + 0.1 * self.wind_box.get_height(),
                                       str(np.round(self.wind.wind_speed,2)) + " m/s",
                                       ha = 'center', va = 'center')

        artists.append(self.windtext)

        # Update wind vector
        if self.wind.wind_speed != 0:
            wind_dir_vec = self.wind.vec/np.linalg.norm(self.wind.vec)
            dx = wind_dir_vec[0]
            dy = wind_dir_vec[1]
            arrow_len = self.wind_box.get_height()/3

            self.arrow_obj = self.h_ax.arrow(cx-(arrow_len*dx) , cy-(arrow_len*dy), dx*arrow_len,
                                             dy*arrow_len, width=10, head_width = 50, color = 'r',
                                             zorder= 3)

        else:
            self.arrow_obj = self.h_ax.text(cx, cy, 'X', fontsize = 20, color = 'r', ha='center',
                                            va = 'center')

        artists.append(self.arrow_obj)

        # Update time string
        time_str = util.get_time_str(curr_time)

        rx, ry = self.time_box.get_xy()
        cx = rx + self.time_box.get_width()/2
        cy = ry + self.time_box.get_height()/2

        # Plot agents for this iteration
        if self.has_agents:
            if curr_time in self.agent_data:
                agents = self.agent_data[curr_time]

                for a in agents:
                    agent_disp = self.h_ax.scatter(a["x"], a["y"], marker=a["marker"],
                                                   color=a["color"])
                    if a["label"] is not None:
                        agent_label = self.h_ax.annotate(str(a["label"]), (a["x"], a["y"]))
                        artists.append(agent_label)
                    artists.append(agent_disp)

        self.simtext.set_visible(False)
        self.simtext = self.h_ax.text(2*cx - 20 , cy, time_str, ha='right', va='center')
        artists.append(self.simtext)

        return artists + self.static_artists

    def run_animation(self):
        """Runs animation based on the time steps of the simulation
        """
        time_steps = self.data.keys()
        self.anim = FuncAnimation(self.fig, self.update_viz, frames=time_steps, interval=1,
                                  blit=True)

        plt.show()

    def cluster_data(self):
        """Function that clusters data into groups of time steps based on the user's selection
        of sim time per frame to be displayed
        """
        # Convert the dictionary into a DataFrame
        df = pd.concat({k: pd.DataFrame(v) for k, v in self.data.items()}, names=['timestamp'])

        # Reset the index to get timestamp as a column
        df.reset_index(level=0, inplace=True)

        # Create a new column 'bins' that will represent the new intervals
        df['bins'] = df['timestamp'] // self.update_freq_s

        # Group by 'id' and 'bins', and keep the last entry in each group
        df = df.groupby(['id', 'bins']).last()

        # Reset index to get back original structure
        df.reset_index(inplace=True)

        # Group by 'bins' and convert each group to a dictionary, then convert to a dictionary
        result = df.groupby('bins').apply(lambda x: x.drop(columns=['bins']).to_dict('records')).to_dict()

        self.data = result

def run(data: dict):
    """Function to run after a user selects the visualization parameters to run. Constructs a 
    VisualizationTool instance with input data and calls the 'run_animation' function

    :param data: _description_
    :type data: dict
    """
    viz = VisualizationTool(data)
    viz.run_animation()

def main():
    folder_selector = VizFolderSelector(run)
    folder_selector.run()

if __name__ == "__main__":
    main()
