"""Module used to create GUIs and handle backend processing for user file input and output.
"""

from tkinter import BOTH, filedialog
from datetime import datetime, time, timedelta
from typing import Callable, Tuple
import tkinter.simpledialog as sd
from tkcalendar import DateEntry
from tkinter import ttk
from time import sleep
import tkinter as tk
import numpy as np
import importlib
import inspect
import pickle
import json
import sys
import os

from embrs.utilities.data_classes import MapParams, SimParams, WeatherParams, UniformMapParams, PlaybackVisualizerParams
from embrs.utilities.fire_util import CanopySpecies
from embrs.models.fuel_models import FuelConstants
from embrs.base_classes.control_base import ControlClass

class FileSelectBase:
    """Base class for creating tkinter file and folder selector interfaces

    :param title: tile to be displayed at the top of the window
    :type title: str
    """
    def __init__(self, title: str):
        """Constructor method that creates a tk root and sets the title
        """
        self.root = tk.Tk()
        self.root.title(title)
        self.result = None

    def create_frame(self, tar: tk.Frame) -> tk.Frame:
        """Create a new tkinter frame within 'tar'

        :param tar: target tk.Frame that the new frame will be created within
        :type tar: tk.Frame
        :return: tk.Frame within 'tar' that can be used to add tkinter elements to
        :rtype: tk.Frame
        """
        frame = tk.Frame(tar)
        frame.grid(sticky='nsew', padx=5, pady=5)
        return frame

    def create_entry_with_label_and_button(self, frame: tk.Frame, text: str, text_var: any,
                                           button_text: str,button_command: Callable
                                           ) -> Tuple[tk.Label,tk.Entry, tk.Button, tk.Frame]:
        """Creates a tk.Frame with a label, entry, and button.

        :param frame: Root frame where the new frame should be located
        :type frame: tk.Frame
        :param text: Text that will displayed on the label
        :type text: str
        :param text_var: Variable where the data entered in the entry will be stored
        :type text_var: any
        :param button_text: Button text to let user know what the button does
        :type button_text: str
        :param button_command: Callable function that should be triggered when the button is
                               pressed
        :type button_command: Callable
        :return: The resulting tk.Label, tk.Entry, tk.Button, and tk.Frame objects
        :rtype: Tuple[tk.Label, tk.Entry, tk.Button, tk.Frame]
        """
        new_frame = self.create_frame(frame)

        label = tk.Label(new_frame, text=text, anchor="w")
        label.grid(row=0, column=0)

        entry = tk.Entry(new_frame, textvariable=text_var, width=50)
        entry.grid(row=0, column=1, sticky="w")

        button = tk.Button(new_frame, text=button_text, command=button_command)
        button.grid(row=0, column=2)

        return label, entry, button, new_frame

    def create_file_selector(self, frame: tk.Frame, text: str, text_var: any, file_type:list=None
                            ) -> Tuple[tk.Label, tk.Entry, tk.Button, tk.Frame]:
        """Creates an entry, label, and button that operate as a way to select a specific file
        path from user's device

        :param frame: Root frame where the new frame should be located
        :type frame: tk.Frame
        :param text: Text that will displayed on the label
        :type text: str
        :param text_var: Variable where the path entered in the entry will be stored
        :type text_var: any
        :param file_type: List of acceptable file types to be selected in the form
                          [(description, file extension)], if None all files will be selectable,
                          defaults to None
        :type file_type: list, optional
        :return: The resulting tk.Label, tk.Entry, tk.Button, and tk.Frame objects
        :rtype: Tuple[tk.Label, tk.Entry, tk.Button, tk.Frame]
        """
        label, entry, button, frame = self.create_entry_with_label_and_button(frame, text,
                                                text_var, "Browse",
                                                lambda: self.select_file(text_var, file_type))

        return label, entry, button, frame
    def create_folder_selector(self, frame: tk.Frame, text: str, text_var: any
                              ) -> Tuple[tk.Label, tk.Entry, tk.Button, tk.Frame]:
        """Creates an entry, label, and button that operate as a way to select a specific folder
        path from user's device

        :param frame: Root frame where the new frame should be located
        :type frame: tk.Frame
        :param text: Text that will displayed on the label
        :type text: str
        :param text_var: Variable where the path entered in the entry will be stored
        :type text_var: any
        :return: The resulting tk.Label, tk.Entry, tk.Button, and tk.Frame objects
        :rtype: Tuple[tk.Label, tk.Entry, tk.Button, tk.Frame]
        """
        label, entry, button, frame = self.create_entry_with_label_and_button(frame, text,
                                                    text_var, "Browse",
                                                    lambda: self.select_folder(text_var))

        return label, entry, button, frame

    def create_spinbox_with_two_labels(self, frame: tk.Frame, left_label: str, max_val: float,
                                    var: any, right_label: str,
                                    row=0, column=0):
        """Creates a frame containing a spinbox with one or two labels."""
        new_frame = tk.Frame(frame)
        new_frame.grid(row=row, column=column, sticky='w', padx=5, pady=2)

        tk.Label(new_frame, text=left_label).grid(row=0, column=0, sticky='w')

        if isinstance(var, tk.DoubleVar):
            spinbox = tk.Spinbox(new_frame, from_=0, to=max_val, increment=0.01,
                                textvariable=var, width=6, format="%.2f")
        else:
            spinbox = tk.Spinbox(new_frame, from_=0, to=max_val, textvariable=var, width=6)

        spinbox.grid(row=0, column=1, sticky='w')

        if right_label is not None:
            tk.Label(new_frame, text=right_label).grid(row=0, column=2, sticky='w')

        return spinbox

    def select_file(self, text_var: any, file_type: list):
        """Function that opens a filedialog window and prompts user to select a file from their
        device

        :param text_var: Variable where the path selected will be stored
        :type text_var: any
        :param file_type: List of acceptable file types to be selected in the form
                          [(description, file extension)], if None all files will be selectable
        :type file_type: list
        """
        filepath = filedialog.askopenfilename(filetypes=file_type)

        if filepath != "":
            text_var.set(filepath)
            self.validate_fields()

    def select_folder(self, text_var: any):
        """Function that opens a filedialog window and prompts user to select a folder from their
        device

        :param text_var: Variable where the path selected will be stored
        :type text_var: any
        """
        folderpath = filedialog.askdirectory()

        if folderpath != "":
            text_var.set(folderpath)
            self.validate_fields()

    def run(self):
        """Runs the FileSelectBase instance and returns the result variable

        :return: Results of the run, typically a dictionary containing input data from user
        :rtype: any
        """
        self.root.mainloop()
        return self.result

    def validate_fields(self):
        """Function to validate inputs that all instances should implement

        :raises NotImplementedError: not implemented here
        """
        raise NotImplementedError

    def submit(self):
        """Function to link to submit button to signal the end of the input process that all
        instances should implement

        :raises NotImplementedError: not implemented here
        """
        raise NotImplementedError
    

class UniformMapCreator(FileSelectBase):
    def __init__(self):

        super().__init__("Uniform Map Creator")

        # Define variables
        self.map_folder = tk.StringVar()

        self.map_folder.trace_add("write", self.validate_fields)
        
        self.slope = tk.DoubleVar()
        self.aspect = tk.DoubleVar()
        self.fuel_selection = tk.StringVar()
        self.fuel_selection.set("Short grass")
        self.fuel_selection_val = 1
        self.canopy_cover = tk.DoubleVar()
        self.canopy_height = tk.DoubleVar()
        self.canopy_base_height = tk.DoubleVar()
        self.canopy_bulk_density = tk.DoubleVar()
        self.fccs_id = tk.IntVar()

        self.width_m = tk.DoubleVar()
        self.height_m = tk.DoubleVar()

        frame = self.create_frame(self.root)

        # Create field to select save destination
        self.create_folder_selector(frame, "Save map to:   ", self.map_folder)

        # Create frame for slope selection
        self.create_spinbox_with_two_labels(frame, "Slope:       ", 90, self.slope, "degrees", row = 1)

        # Create frame for aspect selection
        self.create_spinbox_with_two_labels(frame, "Aspect:       ", 360, self.aspect, "degrees", row = 2)

        # Create frame for fuel selection
        fuel_frame = tk.Frame(frame)
        fuel_frame.grid(padx=10,pady=5)
        tk.Label(fuel_frame, text="Uniform Fuel Type:",
                 anchor="center").grid(row=2, column=0)

        tk.OptionMenu(fuel_frame, self.fuel_selection,
                      *FuelConstants.fuel_names.values()).grid(row=2, column=1)


        # Create frame for canopy height selection
        self.create_spinbox_with_two_labels(frame, "Canopy Height:       ", 1e7, self.canopy_height, "meters", row =4)

        # Create frame for canopy cover selection
        self.create_spinbox_with_two_labels(frame, "Canopy Cover:       ", 100, self.canopy_cover, "%", row=5)

        # Create frame for canopy base height selection
        self.create_spinbox_with_two_labels(frame, "Canopy Base Height:       ", 1e7, self.canopy_base_height, "meters", row=6)

        # Create frame for canopy bulk density selection
        self.create_spinbox_with_two_labels(frame, "Canopy Bulk Density:       ", 1e7, self.canopy_bulk_density, "kg/m^3", row=7)

        # Create frame for duff loading selection
        self.create_spinbox_with_two_labels(frame, "FCCS Type:       ", 1e7, self.fccs_id, "", row=8)

        # Create frame for width selection
        self.create_spinbox_with_two_labels(frame, "Width:       ", 1e7, self.width_m, "meters", row=9)

        # Create frame for height selection
        self.create_spinbox_with_two_labels(frame, "Height:       ", 1e7, self.height_m, "meters", row=10)

        # Create a submit button
        self.submit_button = tk.Button(frame, text="Submit", command=self.submit, state='disabled')
        self.submit_button.grid(pady=10)

        self.width_m.trace_add("write", self.validate_fields)
        self.height_m.trace_add("write", self.validate_fields)
        self.fuel_selection.trace_add("write", self.fuel_selection_changed)


    def fuel_selection_changed(self, *args):
        """Callback function to handle the uniform fuel type value being changed
        """
        self.fuel_selection_val = FuelConstants.fuel_type_reverse_lookup[self.fuel_selection.get()]

    def validate_fields(self, *args):
        """Function used to validate the inputs, primarily responsible for activating/disabling
        the submit button based on if all necessary input has been provided.
        """
        # Check that all fields are filled before enabling submit button
        if self.map_folder.get() and self.width_m.get() != 0 and self.height_m.get() != 0:
            self.submit_button.config(state='normal')

        else:
            self.submit_button.config(state='disabled')

    def submit(self):

        uniform_data = UniformMapParams()

        uniform_data.slope = self.slope.get()
        uniform_data.aspect = self.aspect.get()
        uniform_data.fuel = self.fuel_selection_val
        uniform_data.canopy_cover = self.canopy_cover.get()
        uniform_data.canopy_height = self.canopy_height.get()
        uniform_data.canopy_base_height = self.canopy_base_height.get()
        uniform_data.canopy_bulk_density = self.canopy_bulk_density.get()
        uniform_data.fccs_id = self.fccs_id.get()
        uniform_data.height = self.height_m.get()
        uniform_data.width = self.width_m.get()

        params = MapParams()

        params.uniform_map = True
        params.uniform_data = uniform_data
        params.folder = self.map_folder.get()
        params.import_roads = False

        self.result = params

        # Close the window
        self.root.withdraw()
        self.root.quit()


class MapGenFileSelector(FileSelectBase):
    """Class used to prompt user for map generation files
    """
    def __init__(self):
        """Constructor method, populates tk window with all necessary elements and initializes
        necessary variables
        """

        super().__init__("Select Input Data Files and Map Destination Folder")

        # Define variables
        self.output_map_folder = tk.StringVar()
        self.lcp_filename = tk.StringVar()
        self.fccs_filename = tk.StringVar()
        self.include_fccs = tk.BooleanVar()
        self.include_fccs.set(True)
        self.import_roads = tk.BooleanVar()
        self.import_roads.set(False)


        frame = self.create_frame(self.root)

        # Create field to select save destination
        self.create_folder_selector(frame, "Save map to:   ", self.output_map_folder)

        _, _, self.lcp_button, self.lcp_frame = self.create_file_selector(frame, "Landscape File:     ",
                                                self.lcp_filename,
                                                [("Tagged Image File Format","*.tif"),
                                                ("Tagged Image File Format","*.tiff")])
        
        _, self.fccs_entry, self.fccs_button, self.fccs_frame = self.create_file_selector(frame, "FCCS File:         ",
                                                self.fccs_filename,
                                                [("Tagged Image File Format","*.tif"),
                                                ("Tagged Image File Format","*.tiff")])
        
        # Create frame for importing roads
        import_road_frame = tk.Frame(frame)
        import_road_frame.grid(pady=10)

        self.import_roads_button = tk.Checkbutton(import_road_frame,
                                   text='Import Roads from OpenStreetMap',
                                   variable = self.import_roads, anchor="center")

        self.import_roads_button.grid(row=5, column=0)

        # Create frame for fccs option
        include_fccs_frame = tk.Frame(frame)
        include_fccs_frame.grid(pady=10)

        self.include_fccs_button = tk.Checkbutton(include_fccs_frame,
                                   text='Include FCCS for Duff Loading',
                                   variable = self.include_fccs, anchor="center")

        self.include_fccs_button.grid(row=6, column=0)

        # Create a submit button
        self.submit_button = tk.Button(frame, text="Submit", command=self.submit, state='disabled')
        self.submit_button.grid(pady=10)

        
        self.output_map_folder.trace_add("write", self.validate_fields)
        self.lcp_filename.trace_add("write", self.validate_fields)
        self.include_fccs.trace_add("write", self.toggle_fccs)


    def toggle_fccs(self, *args):
        if self.include_fccs.get():
            self.fccs_button.configure(state='normal')
            self.fccs_entry.configure(state='normal')
        else:
            self.fccs_button.configure(state='disabled')
            self.fccs_entry.configure(state='disabled')

        self.validate_fields()

    def validate_fields(self, *args):
        """Function used to validate the inputs, primarily responsible for activating/disabling
        the submit button based on if all necessary input has been provided.
        """
        # Check that all fields are filled before enabling submit button
        if self.lcp_filename.get() and self.output_map_folder.get() and (self.fccs_filename.get() or not self.include_fccs.get()):
            self.submit_button.config(state='normal')

        else:
            self.submit_button.config(state='disabled')

    def submit(self):
        """Callback when the submit button is pressed. Stores all the relevant data in the result
        variable so it can be retrieved
        """
        map_params = MapParams()

        map_params.uniform_map = False
        map_params.folder = self.output_map_folder.get()
        map_params.lcp_filepath = self.lcp_filename.get()
        map_params.include_fccs = self.include_fccs.get()

        if not self.include_fccs.get():
            map_params.fccs_filepath = ''

        else:
            map_params.fccs_filepath = self.fccs_filename.get()
        
        map_params.import_roads = self.import_roads.get()

        self.result = map_params

        self.root.withdraw()
        self.root.quit()


class SimFolderSelector(FileSelectBase):
    def __init__(self, submit_callback: Callable):
        super().__init__("EMBRS")
        self.submit_callback = submit_callback

        # Define Variables
        self.map_folder = tk.StringVar()
        self.log_folder = tk.StringVar()
        self.weather_file = tk.StringVar()
        self.init_mf_1hr = tk.DoubleVar(value=6)
        self.init_mf_10hr = tk.DoubleVar(value=7)
        self.init_mf_100hr = tk.DoubleVar(value=8)
        self.time_step = tk.IntVar(value=5)
        self.cell_size = tk.IntVar(value=10)
        self.model_spotting = tk.BooleanVar(value=True)
        self.canopy_species_name = tk.StringVar(value="Engelmann spruce")
        self.canopy_species = 0
        self.dbh_cm = tk.DoubleVar(value=20)
        self.spot_ign_prob = tk.DoubleVar(value=0.05)
        self.min_spot_dist = tk.DoubleVar(value=50)
        self.spot_delay = tk.IntVar(value=30)
        self.duration = tk.DoubleVar(value=1.0)
        self.num_runs = tk.IntVar(value=1)
        self.user_path = tk.StringVar()
        self.user_class_name = tk.StringVar()
        self.viz_on = tk.BooleanVar(value=True)
        self.skip_log = tk.BooleanVar(value=False)
        self.use_open_meteo = tk.BooleanVar(value=True)
        self.use_weather_file = tk.BooleanVar(value=False)
        self.mesh_resolution = tk.IntVar(value=250)
        self.start_date = tk.StringVar()
        self.end_date = tk.StringVar()
        self.start_hour = tk.IntVar(value=12)
        self.start_min = tk.IntVar(value=0)
        self.end_hour = tk.IntVar(value=12)
        self.end_min = tk.IntVar(value=0)
        self.start_ampm = tk.StringVar(value="AM")
        self.end_ampm = tk.StringVar(value="AM")

        self.prev_t_unit = "hours"
        self.max_duration = np.inf
        self.class_options = []
        self.user_module = None

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both')

        self.weather_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)
        self.user_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.weather_tab, text='Weather Inputs')
        self.notebook.add(self.model_tab, text='Model Settings')
        self.notebook.add(self.user_tab, text='User Module')

        self.setup_weather_tab()
        self.setup_model_tab()
        self.setup_user_tab()

        self.submit_button = tk.Button(self.root, text="Submit", command=self.submit, state='disabled')
        self.submit_button.pack(pady=10)

        self.update_datetime()
        self.open_meteo_toggled()

    def setup_weather_tab(self):
        self.create_folder_selector(self.weather_tab, "Map folder:", self.map_folder)
        _, self.log_entry, self.log_button, self.log_frame = self.create_folder_selector(self.weather_tab, "Log folder:", self.log_folder)
        tk.Checkbutton(self.log_frame, text='Disable logging', variable=self.skip_log, command=self.write_logs_toggled).grid(row=0, column=3)
        weather_option_frame = self.create_frame(self.weather_tab)
        tk.Label(weather_option_frame, text="Weather Option:").grid(row=0, column=0)
        tk.Checkbutton(weather_option_frame, text='Import OpenMeteo Weather', variable=self.use_open_meteo, command=self.open_meteo_toggled).grid(row=0, column=1)
        tk.Checkbutton(weather_option_frame, text='Import Weather Stream File (.wxs)', variable=self.use_weather_file, command=self.weather_file_toggled).grid(row=0, column=2)

        self.open_meteo_frame = self.create_frame(self.weather_tab)
        tk.Label(self.open_meteo_frame, text="Start Date:").grid(row=0, column=0)
        self.start_cal = DateEntry(self.open_meteo_frame, date_pattern="y-mm-dd", background='white')
        self.start_cal.grid(row=0, column=1)
        self.start_cal.bind("<<DateEntrySelected>>", lambda e: self.update_datetime())
        tk.Label(self.open_meteo_frame, text="Start Time:").grid(row=0, column=2)
        tk.Spinbox(self.open_meteo_frame, from_=1, to=12, width=5, textvariable=self.start_hour).grid(row=0, column=3)
        tk.Spinbox(self.open_meteo_frame, from_=0, to=59, width=5, textvariable=self.start_min).grid(row=0, column=4)
        ttk.Combobox(self.open_meteo_frame, values=["AM", "PM"], width=5, state='readonly', textvariable=self.start_ampm).grid(row=0, column=5)

        tk.Label(self.open_meteo_frame, text="End Date:").grid(row=1, column=0)
        self.end_cal = DateEntry(self.open_meteo_frame, date_pattern="y-mm-dd", background='white')
        self.end_cal.grid(row=1, column=1)
        self.end_cal.bind("<<DateEntrySelected>>", lambda e: self.update_datetime())
        tk.Label(self.open_meteo_frame, text="End Time:").grid(row=1, column=2)
        tk.Spinbox(self.open_meteo_frame, from_=1, to=12, width=5, textvariable=self.end_hour).grid(row=1, column=3)
        tk.Spinbox(self.open_meteo_frame, from_=0, to=59, width=5, textvariable=self.end_min).grid(row=1, column=4)
        ttk.Combobox(self.open_meteo_frame, values=["AM", "PM"], width=5, state='readonly', textvariable=self.end_ampm).grid(row=1, column=5)

        _, self.weather_entry, self.weather_button, self.weather_file_frame = self.create_file_selector(self.weather_tab, "Weather file:", self.weather_file, [("Weather Stream Files", "*.wxs")])

        weather_settings = self.create_frame(self.weather_tab)
        self.create_spinbox_with_two_labels(weather_settings, "Wind Mesh Resolution:", np.inf, self.mesh_resolution, "meters", row=0, column=0)
        self.create_spinbox_with_two_labels(weather_settings, "Initial Fuel Moisture: 1 hr:", 100, self.init_mf_1hr, "%", row=1, column=0)
        self.create_spinbox_with_two_labels(weather_settings, "10 hr:", 100, self.init_mf_10hr, "%", row=1, column=1)
        self.create_spinbox_with_two_labels(weather_settings, "100 hr:", 100, self.init_mf_100hr, "%", row=1, column=2)

        self.map_folder.trace_add("write", self.map_folder_changed)
        self.log_folder.trace_add("write", self.log_folder_changed)
        self.weather_file.trace_add("write", self.weather_file_changed)

        self.start_hour.trace_add("write", self.update_datetime)
        self.start_min.trace_add("write", self.update_datetime)
        self.start_ampm.trace_add("write", self.update_datetime)
        self.end_hour.trace_add("write", self.update_datetime)
        self.end_min.trace_add("write", self.update_datetime)
        self.end_ampm.trace_add("write", self.update_datetime)

    def setup_model_tab(self):
        # Create a big frame that will hold two columns
        self.model_content_frame = self.create_frame(self.model_tab)

        # Now place spinboxes: left side (col=0), right side (col=1)
        self.create_spinbox_with_two_labels(self.model_content_frame, "Time Step (s):", np.inf, self.time_step, "seconds", row=0, column=0)
        self.create_spinbox_with_two_labels(self.model_content_frame, "Cell Size (m):", np.inf, self.cell_size, "meters", row=0, column=1)

        self.create_spinbox_with_two_labels(self.model_content_frame, "Iterations:", np.inf, self.num_runs, None, row=1, column=0)

        self.viz_frame = self.create_frame(self.model_tab)
        tk.Checkbutton(self.viz_frame, text = "Visualize in Real-time", variable=self.viz_on, onvalue=True, offvalue=False).grid(row = 0, column=0, pady=10)

        # Spotting toggle
        self.spotting_frame = self.create_frame(self.model_tab)
        spotting_toggle = tk.Checkbutton(self.spotting_frame, text="Model Spotting", variable=self.model_spotting, onvalue=True, offvalue=False, command=self.model_spotting_toggled)
        spotting_toggle.grid(row=0, column=0, pady=10)

        self.spotting_options_frame = self.create_frame(self.spotting_frame)
        tk.Label(self.spotting_options_frame, text="Canopy Species:").grid(row=0, column=0)
        tk.OptionMenu(self.spotting_options_frame, self.canopy_species_name, *CanopySpecies.species_names.values()).grid(row=0, column=1)
        self.create_spinbox_with_two_labels(self.spotting_options_frame, "Diam. at Breast Height:", 100, self.dbh_cm, 'cm', row=1, column=0)
        self.create_spinbox_with_two_labels(self.spotting_options_frame, "Spot Ignition Probability:", 1.0, self.spot_ign_prob, "", row=2, column=0)
        self.create_spinbox_with_two_labels(self.spotting_options_frame, "Min. Spot Distance. (m):", np.inf, self.min_spot_dist, "meters", row=3, column=0)
        self.create_spinbox_with_two_labels(self.spotting_options_frame, "Spot Delay (s)", np.inf, self.spot_delay, "seconds", row=4, column=0)

        self.canopy_species_name.trace_add("write", self.canopy_species_changed)


    def setup_user_tab(self):
        self.create_file_selector(self.user_tab, "User module:", self.user_path, [("Python Files", "*.py")])
        
        class_name_frame = tk.Frame(self.user_tab)
        class_name_frame.grid(row=1, column=0, sticky='w', padx=10, pady=5)

        tk.Label(class_name_frame, text="User class name:").grid(row=0, column=0)
        self.class_opt_menu = tk.OptionMenu(class_name_frame, self.user_class_name, self.class_options)
        self.class_opt_menu.grid(row=0, column=1)

        self.user_path.trace_add("write", self.user_path_changed)
        self.user_class_name.trace_add("write", self.class_changed)

    def write_logs_toggled(self, *args):
        if self.skip_log.get():
            self.log_button.configure(state='disabled')
            self.log_entry.configure(state='disabled')
            self.log_folder.set("")
        
        else:
            self.log_button.configure(state='normal')
            self.log_entry.configure(state='normal')

        self.validate_fields()

    def model_spotting_toggled(self):
        if self.model_spotting.get():
            self.spotting_options_frame.grid(sticky='w', padx= 5, pady=5)
        else:
            self.spotting_options_frame.grid_remove()

    def open_meteo_toggled(self, *args):
        """Callback for the OpenMeteo toggle button. Ensures only OpenMeteo is selected."""
        if self.use_open_meteo.get():
            # Turn off weather file option
            self.use_weather_file.set(False)
            # Disable weather file widgets
            self.weather_file.set("")
            self.weather_button.configure(state='disabled')
            self.weather_entry.configure(state='disabled')
        else:
            self.use_weather_file.set(True)
            self.weather_button.configure(state='normal')
            self.weather_entry.configure(state='normal')

        self.validate_fields()

    def weather_file_toggled(self, *args):
        """Callback for the Weather File toggle button. Ensures only Weather File is selected."""
        if self.use_weather_file.get():
            # Turn off OpenMeteo option
            self.use_open_meteo.set(False)
            # Enable weather file widgets
            self.weather_button.configure(state='normal')
            self.weather_entry.configure(state='normal')
        else:
            # Prevent both options from being off
            self.use_open_meteo.set(True)
            self.use_weather_file.set(False)
            # Disable weather file widgets
            self.weather_file.set("")
            self.weather_button.configure(state='disabled')
            self.weather_entry.configure(state='disabled')

    def canopy_species_changed(self, *args):        
        self.canopy_species = CanopySpecies.species_ids[self.canopy_species_name.get()]

    def map_folder_changed(self, *args):
        """Callback function that handles the map folder selection being changed, sets the max
        duration allowed based on the length of the wind forecast in the map
        """
        # open json
        folderpath = self.map_folder.get()
        map_filepath = folderpath + "/map_params.pkl"

        if os.path.exists(map_filepath):
            pass

        elif folderpath == "":
            pass

        else:
            self.map_folder.set("")
            window = tk.Tk()
            window.withdraw()
            tk.messagebox.showwarning("Error", "Selected folder does not contain a valid map!")
            window.destroy()

    def log_folder_changed(self, *args):
        """Callback function that handles the log folder selection being changed, warns the user
        if the log folder is already populated with logs so they are aware that those logs will
        be overwritten
        """
        folderpath = self.log_folder.get()

        if os.path.exists(folderpath + '/init_state.parquet') or os.path.exists(folderpath + '/run_0'):
            window = tk.Tk()
            window.withdraw()
            msg = """Warning: Selected folder already contains log files, these will be overwritten
                     if you keep this selection"""
            tk.messagebox.showwarning("Warning", msg)
            window.destroy()

    def weather_file_changed(self, *args):
        """Callback function to handle weather file input being changed. Sets the max duration of
        the simulation to the duration of the forecast.
        """

        filepath = self.weather_file.get()

        if os.path.exists(filepath):
            # lightweight line-by-line parse to get first and last timestamps
            times = []
            with open(filepath, "r") as f:
                header_found = False
                for line in f:
                    line = line.strip()
                    if line.startswith("Year") and not header_found:
                        header_found = True
                        continue
                    if not header_found or not line or line.startswith("RAWS_"):
                        continue

                    parts = line.split()
                    if len(parts) >= 4:
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                        hour = int(parts[3].zfill(4)[:2])
                        dt = datetime(year, month, day, hour)
                        times.append(dt)

            if len(times) >= 2:
                start_dt = times[0]
                end_dt = times[-1]
                self.start_cal.set_date(start_dt.date())
                self.start_hour.set(start_dt.hour % 12 or 12)
                self.start_min.set(0)
                self.start_ampm.set("AM" if start_dt.hour < 12 else "PM")

                self.end_cal.set_date(end_dt.date())
                self.end_hour.set(end_dt.hour % 12 or 12)
                self.end_min.set(0)
                self.end_ampm.set("AM" if end_dt.hour < 12 else "PM")

                # Update internal datetime values
                self.update_datetime()

            else:
                print("WXS file has too few entries to determine time bounds.")


    def user_path_changed(self, *args):
        """Callback function that handles the user module path selection being changed, it gets
        a list of the classes within the module so that the class option menu can be refreshed.
        """

        user_module_file = self.user_path.get()
        user_module_dir = os.path.dirname(user_module_file)
        user_module_name = os.path.splitext(os.path.basename(user_module_file))[0]

        # Add the directory of the module to sys.path
        sys.path.insert(0, user_module_dir)

        try:
            # Dynamically import the module
            user_module = importlib.import_module(user_module_name)
            self.user_module = user_module

            self.class_options = [name for name, obj in inspect.getmembers(user_module)
                                if inspect.isclass(obj)
                                and obj.__module__ == user_module.__name__]

            self.update_option_menu()

        finally:
            # Remove the directory from sys.path
            sys.path.pop(0)

    def update_option_menu(self):
        """Function that changes the class option menu based on the module that is currently
        selected.
        """

        self.user_class_name.set(self.class_options[0])
        # Clear the current menu
        menu = self.class_opt_menu["menu"]
        menu.delete(0, "end")

        # Add the new options
        for class_option in self.class_options:
            menu.add_command(label=class_option,
                            command=lambda value=class_option: self.user_class_name.set(value))

    def class_changed(self, *args):
        """Callback function that handles the user class selection being changed, it checks to
        ensure the class selected is a valid subclass of ControlClass.
        """

        # Access the class within the module
        UserCodeClass = getattr(self.user_module, self.user_class_name.get())

        # Check if the user's class is a subclass of the abstract base class
        if not issubclass(UserCodeClass, ControlClass):
            menu = self.class_opt_menu["menu"]
            menu.delete(0, "end")
            window = tk.Tk()
            window.withdraw()
            tk.messagebox.showwarning("Warning", "User class must be instance of ControlClass!")
            window.destroy()

    def update_datetime(self, *args):
        """Update datetime values whenever any input field changes"""
        start_hr = self.convert_to_24_hr_time(self.start_hour.get(), self.start_ampm.get())
        start_time = time(start_hr, self.start_min.get())
        self.start_datetime = datetime.combine(self.start_cal.get_date(), start_time)

        end_hr = self.convert_to_24_hr_time(self.end_hour.get(), self.end_ampm.get())
        end_time = time(end_hr, self.end_min.get())
        self.end_datetime = datetime.combine(self.end_cal.get_date(), end_time)

        if self.end_datetime < self.start_datetime:
            # User is likely still working on entering dates, don't limit duration
            self.max_duration = np.inf

        else:
            # Update max duration based on length of forecast
            forecast_len = self.end_datetime - self.start_datetime
            forecast_len_hr = forecast_len.total_seconds() / 3600
            self.max_duration = forecast_len_hr
            self.duration.set(self.max_duration)

    def convert_to_24_hr_time(self, hour, ampm):
        if ampm == "PM" and hour != 12:
                return hour + 12

        elif ampm == "AM" and hour == 12:
                return 0

        return hour

    def validate_fields(self, *args):
        """Function used to validate the inputs, primarily responsible for activating/disabling
        the submit button based on if all necessary input has been provided.
        """
        if self.map_folder.get() and  (self.skip_log.get() or self.log_folder.get()) and (self.use_open_meteo.get() or self.weather_file.get()):
            self.submit_button.config(state='normal')
        else:
            self.submit_button.config(state='disabled')

    def submit(self):
        """Callback when the submit button is pressed. Stores all the relevant data in the result
        variable so it can be retrieved
        """
        duration_raw = self.duration.get()
        duration_s = duration_raw * 3600

        if not (self.start_datetime < self.end_datetime < datetime.now()):
            tk.messagebox.showwarning("Invalid Date Selection", 
                                "Start date and time must be before the end date and time, and the end date must be in the past.")

        else:
            with open(os.path.join(self.map_folder.get(), "map_params.pkl"), "rb") as f:
                map_params = pickle.load(f)

            weather_input = WeatherParams(
                input_type = "OpenMeteo" if self.use_open_meteo.get() else "File",
                file = self.weather_file.get(),
                mesh_resolution = self.mesh_resolution.get(),
                start_datetime = self.start_datetime,
                end_datetime = self.end_datetime
             )

            sim_params = SimParams(
                map_params = map_params,
                log_folder = self.log_folder.get(),
                weather_input = weather_input,
                t_step_s = self.time_step.get(),
                init_mf = [self.init_mf_1hr.get()/100, self.init_mf_10hr.get()/100, self.init_mf_100hr.get()/100],
                model_spotting = self.model_spotting.get(),
                canopy_species = self.canopy_species,
                dbh_cm = self.dbh_cm.get(),
                spot_ign_prob = self.spot_ign_prob.get(),
                min_spot_dist = self.min_spot_dist.get(),
                spot_delay_s = self.spot_delay.get(),
                cell_size = self.cell_size.get(),
                duration_s = duration_s,
                visualize = self.viz_on.get(),
                num_runs = self.num_runs.get(),
                user_path = self.user_path.get(),
                user_class = self.user_class_name.get(),
                write_logs = not self.skip_log.get()
            )

            self.submit_callback(sim_params)

class VizFolderSelector(FileSelectBase):
    """Class used to prompt user for log files to be visualized

    :param submit_callback: Function that should be called when the submit button is pressed
    :type submit_callback: Callable
    """
    def __init__(self, normal_callback: Callable, arrival_callback: Callable):
        """Constructor method, populates tk window with all necessary elements and initializes
        necessary variables
        """
        super().__init__("Visualization Tool")

        # Define variables
        self.viz_folder = tk.StringVar()
        self.viz_folder.trace_add("write", self.viz_folder_changed)
        self.run_folder = tk.StringVar()
        self.run_folder.trace_add("write", self.run_folder_changed)
        self.viz_freq = tk.IntVar()
        self.viz_freq.set(60)
        self.scale_km = tk.DoubleVar()
        self.scale_km.set(1.0)
        self.legend = tk.BooleanVar()
        self.legend.set(True)
        self.show_wind_cbar = tk.BooleanVar()
        self.show_wind_cbar.set(True)
        self.show_wind_field = tk.BooleanVar()
        self.show_wind_field.set(True)
        self.show_wind_field.trace_add("write", self.wind_field_toggled)
        self.show_weather_data = tk.BooleanVar()
        self.show_weather_data.set(True)
        self.show_weather_data.trace_add("write", self.show_weather_data_toggled)
        self.temp_units = tk.StringVar()
        self.temp_units.set("Fahrenheit")
        self.show_compass = tk.BooleanVar()
        self.show_compass.set(True)
        self.save_video = tk.BooleanVar()
        self.save_video.set(False)
        self.save_video.trace_add("write", self.toggle_video_options)
        self.video_folder = tk.StringVar()
        self.video_name = tk.StringVar()
        self.video_fps = tk.IntVar()
        self.video_fps.set(10)
        self.render_visualization = tk.BooleanVar()
        self.render_visualization.set(True)

        self.arrival_time = tk.BooleanVar()
        self.arrival_time.set(False)
        self.arrival_time.trace_add("write", self.toggle_arrival_time)

        self.has_agents = False
        self.run_folders = ["No runs available, select a folder"]

        self.init_location = False

        # Save submit callback function
        self.normal_callback = normal_callback
        self.arrival_callback = arrival_callback

        frame = self.create_frame(self.root)

        # === Folder Selection Frame ===
        folder_frame = tk.LabelFrame(frame, text="Folder Selection", padx=10, pady=5)
        folder_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=5)

        self.create_folder_selector(folder_frame, "Log folder:    ", self.viz_folder)

        run_selection_frame = tk.Frame(folder_frame)
        run_selection_frame.grid(padx=10, pady=5)
        tk.Label(run_selection_frame, text="Run to Visualize:").grid(row=0, column=0)
        self.run_options = tk.OptionMenu(run_selection_frame, self.run_folder, *self.run_folders)
        self.run_options.grid(row=0, column=1)

        # === Visualization Settings Frame ===
        settings_frame = tk.LabelFrame(frame, text="Visualization Settings", padx=10, pady=5)
        settings_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=5)

        self.create_spinbox_with_two_labels(settings_frame, "Update Frequency:", np.inf, self.viz_freq, "seconds", row=0, column=0)
        self.create_spinbox_with_two_labels(settings_frame, "Scale bar size:        ", 100, self.scale_km, "km", row=0, column=1)

        # === Video Recording Options Frame ===
        video_frame = tk.LabelFrame(frame, text="Video Recording Options", padx=10, pady=5)
        video_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        video_frame.columnconfigure(1, weight=1)

        self.save_video_checkbox = tk.Checkbutton(video_frame, text="Save visualization as MP4", variable=self.save_video)
        self.save_video_checkbox.grid(row=0, column=0, columnspan=2, sticky='w')

        _, self.video_path_field, self.vid_path_button, _ = self.create_folder_selector(video_frame, "Video Save Folder: ", self.video_folder)
        _, self.videoname_field, self.videoname_button, _ = self.create_file_selector(video_frame,   "Video filename:       ", self.video_name)

        self.videoname_button.grid_remove()

        self.frame_rate_spin = self.create_spinbox_with_two_labels(video_frame, "Video Frame Rate:", np.inf, self.video_fps, "FPS", row=3, column=0)

        self.show_viz_checkbox = tk.Checkbutton(video_frame, text="Show Visualization while saving video", variable=self.render_visualization)
        self.show_viz_checkbox.grid(row=4, column=0, sticky='w')

        # === Display Options Frame ===
        display_frame = tk.LabelFrame(frame, text="Display Options", padx=10, pady=5)
        display_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=5)

        tk.Checkbutton(display_frame, text="Display fuel legend", variable=self.legend).grid(row=0, column=0, sticky='w')
        self.wind_cbar_checkbox = tk.Checkbutton(display_frame, text="Show wind colorbar", variable=self.show_wind_cbar)
        self.wind_cbar_checkbox.grid(row=0, column=1, sticky='w')

        tk.Checkbutton(display_frame, text="Show wind field", variable=self.show_wind_field).grid(row=1, column=0, sticky='w')
        tk.Checkbutton(display_frame, text="Show compass", variable=self.show_compass).grid(row=1, column=1, sticky='w')
        tk.Checkbutton(display_frame, text="Show weather data", variable=self.show_weather_data).grid(row=2, column=0, sticky='w')

        tk.Label(display_frame, text="Temperature units:").grid(row=2, column=1, sticky='w')
        self.temp_units = tk.StringVar(value="Fahrenheit")
        self.temp_units_menu = tk.OptionMenu(display_frame, self.temp_units, "Fahrenheit", "Celsius")
        self.temp_units_menu.grid(row=2, column=2, sticky='w')

        # BETA Arrival time
        arrival_frame = tk.LabelFrame(frame, text="BETA Alternate Visualization", padx=10, pady=5)
        arrival_frame.grid(row=4, column=0, columnspan=3, sticky='ew', padx=10, pady=5)

        self.arrival_time_checkbox = tk.Checkbutton(
            arrival_frame,
            text="Visualize Arrival Time",
            variable=self.arrival_time
        )

        self.arrival_time_checkbox.grid(row=0, column=0, sticky='w')

        # === Submit Button ===
        self.submit_button = tk.Button(frame, text='Submit', command=self.submit, state='disabled')
        self.submit_button.grid(row=5, column=0, columnspan=3, pady=10)

        self.toggle_video_options()

    def toggle_arrival_time(self, *args):
        """Disables video options if arrival time plot visualization is selected."""
        if self.arrival_time.get():
            # Disable video options completely
            self.save_video_checkbox.config(state='disabled')
            self.video_path_field.config(state='disabled')
            self.vid_path_button.config(state='disabled')
            self.videoname_field.config(state='disabled')
            self.frame_rate_spin.config(state='disabled')
            self.show_viz_checkbox.config(state='disabled')
        else:
            # Re-enable video options based on save_video checkbox
            self.save_video_checkbox.config(state='normal')
            self.toggle_video_options()


    def viz_folder_changed(self, *args):
        """Callback function for selecting the log file to be displayed. Checks the file selected
        to make sure that all the required file types are there to visualize the data.
        # """
        folderpath = self.viz_folder.get()
        self.run_folders = self.get_run_sub_folders(folderpath)

        if self.run_folders:
            menu = self.run_options["menu"]
            menu.delete(0, "end")
            for run in self.run_folders:
                menu.add_command(label=run, command=lambda value=run: self.run_folder.set(value))

            self.run_folder.set(self.run_folders[0])

    def run_folder_changed(self, *args):
        """Callback function to handle the run folder input being changed. Ensures the selection
        contains valid log files.
        """
        run_foldername = self.run_folder.get()
        run_folderpath = os.path.join(self.viz_folder.get(), run_foldername)

        if os.path.exists(os.path.join(run_folderpath, 'agent_logs.parquet')):
            self.has_agents = True
            self.agent_file = os.path.join(run_folderpath, 'agent_logs.parquet')

        else:
            self.has_agents = False

        if os.path.exists(os.path.join(run_folderpath, 'action_logs.parquet')):
            self.has_actions = True
            self.action_file = os.path.join(run_folderpath, 'action_logs.parquet')
        else:
            self.has_actions = False

        if os.path.exists(os.path.join(run_folderpath, 'prediction_logs.parquet')):
            self.has_predictions = True
            self.prediction_file = os.path.join(run_folderpath, 'prediction_logs.parquet')
        else:  
            self.has_predictions = False

        if os.path.exists(os.path.join(run_folderpath, 'cell_logs.parquet')):
            self.viz_file = os.path.join(run_folderpath, 'cell_logs.parquet')

        else:
            window = tk.Tk()
            window.withdraw()
            tk.messagebox.showwarning("Error",
            "Selected folder does not have a valid log file!")

            window.destroy()

            self.run_folder.set("")

    def toggle_video_options(self, *args):
        """Enable or disable video path and FPS fields based on the save_video checkbox."""
        if self.save_video.get():
            self.video_path_field.config(state='normal')
            self.vid_path_button.config(state='normal')
            self.videoname_field.config(state='normal')
            self.frame_rate_spin.config(state='normal')
            self.show_viz_checkbox.config(state='normal')
        else:
            self.video_path_field.config(state='disabled')
            self.vid_path_button.config(state='disabled')
            self.videoname_field.config(state='disabled')
            self.frame_rate_spin.config(state='disabled')
            self.show_viz_checkbox.config(state='disabled')

    def wind_field_toggled(self, *args):
        """Ensures wind colorbar is only shown if wind field is shown."""
        if not self.show_wind_field.get():
            self.show_wind_cbar.set(False)
            self.wind_cbar_checkbox.config(state='disabled')
        else:
            self.show_wind_cbar.set(True)
            self.wind_cbar_checkbox.config(state='normal')

    def show_weather_data_toggled(self, *args):
        if not self.show_weather_data.get():
            self.temp_units_menu.config(state='disabled')
        else:
            self.temp_units_menu.config(state='normal')

    def get_run_sub_folders(self, folderpath: str) -> list:
        """Function that retrieves all the runs contained within a log folder. Returns a list of
        the folders to populate the option menu

        :param folderpath: string representing the path to the selected folder
        :type folderpath: str
        :return: list of run folder names inside the log folder with valid log files
        :rtype: list
        """

        if not os.path.exists(folderpath):
            return []

        if not os.path.exists(os.path.join(folderpath, 'init_state.parquet')):
            window = tk.Tk()
            window.withdraw()
            tk.messagebox.showwarning("Error",
            "Selected file not in a folder with an initial state file!")

            window.destroy()

            return []

        self.init_location = True

        contents = os.listdir(folderpath)
        subfolders = [d for d in contents if os.path.isdir(os.path.join(folderpath, d)) and d.startswith("run")]

        return subfolders

    def submit(self):
        """Callback when the submit button is pressed. Stores all the relevant data in the result
        variable so it can be retrieved
        """

        if self.save_video.get():
            video_filename = self.video_name.get().strip()

            if not video_filename.lower().endswith(".mp4"):
                video_filename += ".mp4"

            video_path = os.path.join(self.video_folder.get(), video_filename)

            if os.path.exists(video_path):
                result = tk.messagebox.askyesno(
                    "File Exists",
                    f"A video named '{video_filename}' already exists.\nDo you want to overwrite it?"
                )
                if not result:
                    return  # Cancel and return to GUI without submitting

        else:
            self.render_visualization.set(True)
            video_filename = ""

        show_temp_in_F = self.temp_units.get() == "Fahrenheit"


        self.result = PlaybackVisualizerParams(
            cell_file= self.viz_file,
            init_location=self.init_location,
            save_video=self.save_video.get(),
            video_folder=self.video_folder.get(),
            video_name=video_filename,
            has_agents=self.has_agents,
            has_actions=self.has_actions,
            has_predictions=self.has_predictions,
            video_fps=self.video_fps.get(),
            freq=self.viz_freq.get(),
            scale_km=self.scale_km.get(),
            show_legend=self.legend.get(),
            show_wind_cbar=self.show_wind_cbar.get(),
            show_wind_field=self.show_wind_field.get(),
            show_weather_data=self.show_weather_data.get(),
            show_compass=self.show_compass.get(),
            show_visualization=self.render_visualization.get(),
            show_temp_in_F=show_temp_in_F
        )

        if self.has_agents:
            self.result.agent_file = self.agent_file

        if self.has_actions:
            self.result.action_file = self.action_file

        if self.has_predictions:
            self.result.prediction_file = self.prediction_file


        if self.arrival_time.get():
            self.arrival_callback(self.result)

        else:
            self.normal_callback(self.result)

    def validate_fields(self, *args):
        """Function used to validate the inputs, primarily responsible for activating/disabling
        the submit button based on if all necessary input has been provided.
        """
        if self.viz_folder.get() and self.viz_freq.get() > 0:
            self.submit_button.config(state='normal')
        else:
            self.submit_button.config(state='disabled')

class LoaderWindow:
    """Class used to created loading bars for progress updates to user while programs are running
    in backend

    :param title: title to be displayed at the top of the window
    :type title: str
    :param max_value: number of increments to complete the task
    :type max_value: int
    """
    def __init__(self, title: str, max_value: int):
        """Constructor method, populates tk window with all necessary elements and initializes
        necessary variables
        """
        self.root = tk.Toplevel()
        self.root.geometry('300x100')
        self.root.title(title)

        self.text_var = tk.StringVar()
        self.text_label = tk.Label(self.root, textvariable=self.text_var)
        self.text_label.pack()

        self.progressbar = ttk.Progressbar(self.root, maximum=max_value)
        self.progressbar.pack(fill='x')
        sleep(0.5)

    def set_text(self, text:str):
        """Sets the text of the window

        :param text: text to be displayed along the progress bar to inform user what task is being
                     completed
        :type text: str
        """
        sleep(0.5)
        self.text_var.set(text)
        self.root.update_idletasks()  # Ensure the window updates

    def increment_progress(self, increment_value:int=1):
        """Increments the progress forward by 'increment_value'

        :param increment_value: amount to increment forward, defaults to 1
        :type increment_value: int, optional
        """
        self.progressbar.step(increment_value)
        self.root.update_idletasks()  # Ensure the window updates

    def close(self):
        """Close the loader window
        """
        self.root.destroy()