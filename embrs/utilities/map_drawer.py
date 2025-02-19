"""Module that handles user drawing on top of sim map when specifying map parameters.
"""

from embrs.utilities.fire_util import FuelConstants as fc

from typing import Tuple
import rasterio
import numpy as np
from PyQt5.QtWidgets import QApplication, QInputDialog
from matplotlib.widgets import Button, RectangleSelector
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

class CropTiffTool:
    """
    Class for cropping a TIFF file using a bounding box drawn by the user.
    Uses the RectangleSelector widget for selecting the area to crop.
    """
    def __init__(self, fig: plt.Figure, tiff_path: str):
        self.fig = fig
        self.ax = fig.add_subplot(111)
        self.tiff_path = tiff_path
        self.coords = None  # Store bounding box coordinates
        
        # Load TIFF data
        self.load_tiff()
        
        # Create RectangleSelector for user selection
        self.selector = RectangleSelector(
            self.ax, self.on_select, useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )
        
        # Create accept and decline buttons
        self.accept_button = Button(plt.axes([0.78, 0.05, 0.1, 0.075]), 'Accept')
        self.accept_button.on_clicked(self.accept)
        self.accept_button.ax.set_visible(False)
        
        self.decline_button = Button(plt.axes([0.885, 0.05, 0.1, 0.075]), 'Decline')
        self.decline_button.on_clicked(self.decline)
        self.decline_button.ax.set_visible(False)
        
        self.fig.canvas.draw()
    
    def load_tiff(self):
        """Loads and plots the TIFF data."""
        with rasterio.open(self.tiff_path) as dataset:
            self.extent = [dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top]
            self.fuel_data = dataset.read(4) # Read fuel data from raster
            
            nodata_value = -9999

            print(f"nodata_value: {nodata_value}")

            if nodata_value is not None:
                self.fuel_data[self.fuel_data == nodata_value] = -100

        # Create a color list in the right order
        colors = [fc.fuel_color_mapping[key] for key in sorted(fc.fuel_color_mapping.keys())]

        # Create a colormap from the list
        cmap = ListedColormap(colors)

        # Create a norm object to map your data points to the colormap
        norm = BoundaryNorm(list(sorted(fc.fuel_color_mapping.keys())) + [100], cmap.N)

        self.ax.imshow(self.fuel_data, extent=self.extent, cmap=cmap, norm=norm)
        self.ax.set_title("Draw a bounding box to crop the TIFF file")
        
    def on_select(self, eclick, erelease):
        """Handles the selection of a bounding box."""
        self.coords = [(eclick.xdata, eclick.ydata), (erelease.xdata, erelease.ydata)]
        self.accept_button.ax.set_visible(True)
        self.decline_button.ax.set_visible(True)
        self.fig.canvas.draw()
    
    def accept(self, event):
        """Confirms the selection and closes the figure."""
        print(f"Bounding Box Selected: {self.coords}")
        plt.close(self.fig)
    
    def decline(self, event):
        """Clears the selection and resets the figure."""
        self.coords = None
        self.accept_button.ax.set_visible(False)
        self.decline_button.ax.set_visible(False)
        self.selector.set_visible(False)
        self.fig.canvas.draw()
    
    def get_coords(self):
        """Returns the selected bounding box coordinates."""
        return self.coords


# TODO: this class needs to be fixed
class PolygonDrawer:
    """Class used for drawing polygons on top of sim map for specifying locations of initial
    ignitions and fire-breaks.

    :param fig: matplotlib figure object used to draw on top of
    :type fig: matplotlib.figure.Figure
    """
    def __init__(self, fig: matplotlib.figure.Figure):
        """Constructor method that initializes all variables and sets up the GUI
        """
        self.fig = fig

        if fig.axes:
            self.ax = fig.axes[0]  # Get the existing Axes object if available
        else:
            self.ax = fig.subplots()

        self.num_road_lines = len(self.ax.lines)

        self.xlims = self.ax.get_xlim()
        self.ylims = self.ax.get_ylim()

        self.ax.set_title("Click on the map to draw a polygon to specify the initial fire region")

        self.ax.invert_yaxis()
        self.line, = self.ax.plot([], [], 'r-')  # Create a line for confirmed segments
        self.preview_line, = self.ax.plot([], [], 'r--')  # Create a line for the preview segment
        self.xs = []
        self.ys = []
        self.polygons = []  # List of polygons
        self.temp_polygon = None  # For storing the temporary polygon
        self.decision_pending = False

        # Create accept/decline buttons but keep them invisible until a polygon is closed
        self.accept_button = Button(plt.axes([0.78, 0.05, 0.1, 0.075]), 'Accept')
        self.accept_button.on_clicked(self.accept)
        self.accept_button.ax.set_visible(False)

        self.decline_button = Button(plt.axes([0.885, 0.05, 0.1, 0.075]), 'Decline')
        self.decline_button.on_clicked(self.decline)
        self.decline_button.ax.set_visible(False)

        # Create apply/clear buttons but keep them inactive until a polygon or line is accepted
        self.apply_button = Button(plt.axes([0.785, 0.95, 0.1, 0.04]), 'Apply')
        self.apply_button.on_clicked(self.apply)
        self.apply_button.set_active(False)
        self.apply_button.color = '0.85'
        self.apply_button.hovercolor = self.apply_button.color

        self.clear_button = Button(plt.axes([0.89, 0.95, 0.1, 0.04]), 'Clear')
        self.clear_button.on_clicked(self.clear)
        self.clear_button.set_active(False)
        self.clear_button.color = '0.85'
        self.clear_button.hovercolor = self.clear_button.color

        # Create no fire breaks button but keep it invisible until polygons are specified
        self.no_fire_breaks_button = Button(plt.axes([0.78, 0.05, 0.1, 0.075]), 'No Fire Breaks')
        self.no_fire_breaks_button.on_clicked(self.skip_fire_breaks)
        self.no_fire_breaks_button.ax.set_visible(False)

        # Create reset view button that resets view to original
        self.reset_view_button = Button(plt.axes([0.12, 0.05, 0.1, 0.075]), 'Reset View')
        self.reset_view_button.on_clicked(self.reset_view)

        # Set up event handlers
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        # Additional initialization for line segments
        self.line_segments = []  # List of line segments
        self.current_line = None  # For storing the current line segment
        self.temp_line_segments = []  # For storing temporary line segments
        self.lines = []
        self.fire_break_fuel_vals = []

        # Parameter to track whether in ignition or fire-break mode
        self.mode = 'ignition'

        self.valid = False

    def on_press(self, event: matplotlib.backend_bases.MouseEvent):
        """Callback for handling button presses in the figure, left click places either initial
        ignitions or fire-breaks, right click can be used to pan the view.

        :param event: MouseEvent triggered from the click
        :type event: matplotlib.backend_bases.MouseEvent
        """
        if event.inaxes in [self.accept_button.ax,
                            self.decline_button.ax,
                            self.apply_button.ax,
                            self.clear_button.ax,
                            self.reset_view_button.ax]:
            return

        # Left click allows user to draw on map
        if event.button == 1:
            if self.decision_pending:
                return

            if event.xdata is None or event.ydata is None:
                return

            if self.mode == 'ignition':
                # If there are already points, check if the new point closes a polygon
                if len(self.xs) > 2:
                    dx = self.xs[0] - event.xdata
                    dy = self.ys[0] - event.ydata
                    # If the point is close to the first point, close the polygon
                    if np.hypot(dx, dy) < 1:
                        self.ax.set_title("Press 'Accept' to confirm or 'Decline' to discard")

                        # Store polygon vertices
                        self.temp_polygon = list(zip(self.xs, self.ys))

                        # Append the first point to the lists of points
                        self.xs.append(self.xs[0])
                        self.ys.append(self.ys[0])

                        # Fill the polygon
                        self.ax.fill(self.xs, self.ys, 'r', alpha=0.5)
                        self.line.set_data(self.xs, self.ys)

                        # Clear the x an y values, and preview line
                        self.xs = []
                        self.ys = []
                        self.preview_line.set_data([], [])

                        # Make buttons visible once polygon is closed
                        self.accept_button.ax.set_visible(True)
                        self.decline_button.ax.set_visible(True)
                        self.decision_pending = True
                        self.fig.canvas.draw()

                        return

                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.line.set_data(self.xs, self.ys)

                self.set_button_status(False, False)

                self.fig.canvas.draw()

            elif self.mode == 'fire-breaks':
                self.no_fire_breaks_button.ax.set_visible(False)

                if len(self.xs) > 0:
                    self.temp_line_segments.append([self.xs[-1], self.ys[-1]])
                    self.temp_line_segments.append([event.xdata, event.ydata])
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.line.set_data(self.xs, self.ys)

                if len(self.xs) > 1:
                    self.accept_button.ax.set_visible(True)
                    self.decline_button.ax.set_visible(True)
                    title = """Click 'Accept' to confirm or 'Decline' to discard. Or continue
                               to draw fire-break"""

                    self.ax.set_title(title)
                else:
                    self.accept_button.ax.set_visible(False)
                    self.decline_button.ax.set_visible(False)

                self.set_button_status(False, False)

                self.fig.canvas.draw()

        # Right click allows user to pan the view
        elif event.button == 3:
            self.ax._pan_start = [event.xdata, event.ydata]

    def on_release(self, event: matplotlib.backend_bases.MouseEvent):
        """Callback function for handling mouse release, only used when panning the view

        :param event: MouseEvent triggered by releasing the mouse
        :type event: matplotlib.backend_bases.MouseEvent
        """
        if event.button == 3:
            self.ax._pan_start = None

    def on_motion(self, event: matplotlib.backend_bases.MouseEvent):
        """Callback function for handling mouse motion. Pans the view when right-click active,
        previews lines to be drawn otherwise

        :param event: MouseEvent triggered by moving mouse
        :type event: matplotlib.backend_bases.MouseEvent
        """
        if event.inaxes in [self.accept_button.ax,
                            self.decline_button.ax,
                            self.apply_button.ax,
                            self.clear_button.ax]:
            return

        if event.button == 3 and self.ax._pan_start is not None:
            dx = event.xdata - self.ax._pan_start[0]
            dy = event.ydata - self.ax._pan_start[1]
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)

            self.fig.canvas.draw()

        else:
            if self.decision_pending:
                return

            if event.xdata is None or event.ydata is None:
                return
            if not self.xs:  # If no points have been confirmed, don't draw a preview
                return

            if self.mode == 'ignition':
                dx = self.xs[0] - event.xdata
                dy = self.ys[0] - event.ydata

                # If the mouse is close to the first point, snap to the first point
                if np.hypot(dx, dy) < 1:
                    preview_xs = [self.xs[-1], self.xs[0]]
                    preview_ys = [self.ys[-1], self.ys[0]]

                # If the mouse is not close to the first point, draw to the current mouse position
                else:
                    preview_xs = [self.xs[-1], event.xdata]
                    preview_ys = [self.ys[-1], event.ydata]

                self.preview_line.set_data(preview_xs, preview_ys)
                self.fig.canvas.draw()

            elif self.mode == 'fire-breaks':
                # If no points have been confirmed, don't draw a preview
                if not self.xs:
                    return
                preview_xs = [self.xs[-1], event.xdata]
                preview_ys = [self.ys[-1], event.ydata]
                self.preview_line.set_data(preview_xs, preview_ys)

            self.fig.canvas.draw()

    def on_scroll(self, event: matplotlib.backend_bases.MouseEvent):
        """Callback function for handling scrolling in the figure. Zooms the view in and out
        focused wherever the mouse pointer is.

        :param event: MouseEvent triggered by scrolling
        :type event: matplotlib.backend_bases.MouseEvent
        """
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location

        if xdata is None or ydata is None:
            return

        xleft = xdata - cur_xlim[0]
        xright = cur_xlim[1] - xdata
        ybottom = ydata - cur_ylim[0]
        ytop = cur_ylim[1] - ydata

        if event.button == 'up':
            scale_factor = 1/1.1
        elif event.button == 'down':
            scale_factor = 1.1
        else:
            scale_factor = 1

        self.ax.set_xlim([xdata - xleft*scale_factor,
                    xdata + xright*scale_factor])
        self.ax.set_ylim([ydata - ybottom*scale_factor,
                    ydata + ytop*scale_factor])

        self.fig.canvas.draw()

    def set_button_status(self, apply_active: bool, clear_active: bool):
        """Set the status of the 'apply' and 'clear' buttons to set whether they are active or not

        :param apply_active: boolean to set 'apply' button status, True = active, False = inactive
        :type apply_active: bool
        :param clear_active: boolean to set 'clear' button status, True = active, False = inactive
        :type clear_active: bool
        """
        self.apply_button.set_active(apply_active)
        self.apply_button.color = '0.75' if apply_active else '0.85'
        self.apply_button.hovercolor = '0.95' if apply_active else '0.85'
        self.apply_button.ax.set_facecolor(self.apply_button.color)

        self.clear_button.set_active(clear_active)
        self.clear_button.color = '0.75' if clear_active else '0.85'
        self.clear_button.hovercolor = '0.95' if clear_active else '0.85'
        self.clear_button.ax.set_facecolor(self.clear_button.color)

    def reset_current_polygon(self):
        """Remove all drawn polygons
        """
        self.xs = []
        self.ys = []
        for patch in self.ax.patches:
            patch.remove()

    def reset_current_lines(self):
        """Remove all drawn lines
        """
        self.xs = []
        self.ys = []
        for line in self.ax.lines[self.num_road_lines:]:
            if line not in (self.line, self.preview_line):
                line.remove()

        self.line.set_data([], [])
        self.preview_line.set_data([],[])

    def decline(self, event: matplotlib.backend_bases.MouseEvent):
        """Callback function for handling the decline button being pressed. Clears the most recent
        polygon or line drawn

        :param event: MouseEvent triggered from clicking 'decline'
        :type event: matplotlib.backend_bases.MouseEvent
        """
        if self.mode == 'ignition':
            self.xs = []
            self.ys = []
            self.ax.patches.pop()
            self.hide_buttons()
            title = "Click on the map to draw a polygon to specify the initial fire region"
            self.ax.set_title(title)
            self.line.set_data([], [])
            self.preview_line.set_data([], [])
            self.decision_pending = False

            if self.polygons:
                self.set_button_status(True, True)

            self.fig.canvas.draw()
        elif self.mode == 'fire-breaks':
            self.temp_line_segments = []
            self.current_line = None
            for line in self.lines:
                line.remove()
            self.lines = []
            self.preview_line.set_data([], [])
            self.line.set_data([], [])
            self.xs = []
            self.ys = []

            self.accept_button.ax.set_visible(False)
            self.decline_button.ax.set_visible(False)

            if len(self.line_segments) == 0:
                self.no_fire_breaks_button.ax.set_visible(True)

            self.fig.canvas.draw()

    def clear(self, event: matplotlib.backend_bases.MouseEvent):
        """Callback function for handling the clear button being pressed. Clears all polygons or
        all lines depending on the current mode

        :param event: MouseEvent triggered from clicking 'clear'
        :type event: matplotlib.backend_bases.MouseEvent
        """
        if self.mode == 'ignition':
            self.polygons = []
            self.reset_current_polygon()
            title = "Click on the map to draw a polygon to specify the initial fire region"
            self.ax.set_title(title)

        elif self.mode == 'fire-breaks':
            self.line_segments = []
            self.fire_break_fuel_vals = []
            self.reset_current_lines()
            self.ax.set_title("Draw line segments to specify fire-breaks")
            self.no_fire_breaks_button.ax.set_visible(True)

        self.set_button_status(False, False)
        self.fig.canvas.draw()

    def accept(self, event: matplotlib.backend_bases.MouseEvent):
        """Callback function for handling the accept button being pressed. Confirms the most recent
        polygon or line.

        :param event: MouseEvent triggered from clicking 'accept'
        :type event: matplotlib.backend_bases.MouseEvent
        """
        if self.mode == 'ignition':
            self.polygons.append(self.temp_polygon)
            self.temp_polygon = None
            self.hide_buttons()
            self.ax.plot()

            if self.polygons:
                self.set_button_status(True, True)

            title = "Draw another fire region or click 'Apply' to save changes and move on"
            self.ax.set_title(title)

        elif self.mode == 'fire-breaks':
            if self.temp_line_segments:
                self.line_segments.append(self.temp_line_segments)
                self.temp_line_segments = []
                self.set_button_status(True, True)

                val = self.get_fuel_value()

                self.fire_break_fuel_vals.append(val)

            self.hide_buttons()
            title = "Draw another fire-break line or click 'Apply' to save changes and finish"
            self.ax.set_title(title)
            self.ax.plot(self.xs, self.ys, 'b')  # plot the line segment

        self.xs = []
        self.ys = []
        self.line.set_data([], [])
        self.preview_line.set_data([], [])
        self.decision_pending = False

        self.fig.canvas.draw()

    def reset_view(self, event: matplotlib.backend_bases.MouseEvent):
        """Resets the view to the original display

        :param event: MouseEvent triggered from pressing 'reset_view'
        :type event: matplotlib.backend_bases.MouseEvent
        """
        self.ax.set_xlim(self.xlims)
        self.ax.set_ylim(self.ylims)
        self.ax.invert_yaxis()

        self.fig.canvas.draw()

    def get_fuel_value(self) -> float:
        """Prompts user for the fuel value of a just drawn fire-break

        :return: float fuel value entered by the user
        :rtype: float
        """
        app = QApplication([])
        request  = "Enter percent fuel remaining in fire break:"
        value, ok = QInputDialog.getDouble(None, "Input Dialog", request)
        if ok:
            return value

        return None

    def apply(self, event: matplotlib.backend_bases.MouseEvent):
        """Callback function for handling the apply button being pressed. Saves the polygons or
        lines drawn in permanent data structures, closes the figure if process is complete

        :param event: MouseEvent triggered from clicking 'apply'
        :type event: matplotlib.backend_bases.MouseEvent
        """
        if self.mode == 'ignition':
            self.mode = 'fire-breaks'
            self.ax.set_title("Draw line segments to specify fire-breaks")
            self.no_fire_breaks_button.ax.set_visible(True)
            self.fig.canvas.draw()
            self.preview_line, = self.ax.plot([], [], 'b--')
            self.line, = self.ax.plot([], [], 'b-')

        elif self.mode == 'fire-breaks':
            self.valid = True
            plt.close(self.fig)

    def skip_fire_breaks(self, event: matplotlib.backend_bases.MouseEvent):
        """Skips the drawing of the fire-breaks and closes the figure

        :param event: MouseEvent triggered from clicking 'skip fire breaks'
        :type event: matplotlib.backend_bases.MouseEvent
        """
        self.line_segments = []
        self.fire_break_fuel_vals = []
        self.valid = True
        plt.close(self.fig)

    def hide_buttons(self):
        """Hide the accept and decline buttons from view
        """
        self.accept_button.ax.set_visible(False)
        self.decline_button.ax.set_visible(False)
        self.fig.canvas.draw()

    def get_ignitions(self) -> list:
        """Get the ignition polygons that have been finalized

        :return: list of polygon coordinates representing ignition areas
        :rtype: list
        """

        return self.polygons

    def get_fire_breaks(self) -> Tuple[list, list]:
        """Get the fire breaks drawn and finalized along with their fuel values

        :return: Returns a list with the coordinates of the fire-break line segments, and a list
                 corresponding to each of their fuel values
        :rtype: Tuple[list, list]
        """

        return self.line_segments, self.fire_break_fuel_vals
