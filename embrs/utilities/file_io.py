"""Module used to create GUIs and handle backend processing for user file input and output.
"""

from tkinter import BOTH, filedialog
from tkinter import ttk
import tkinter.simpledialog as sd
import tkinter as tk
from time import sleep
import sys
import os
import importlib
import inspect
import json
from typing import Callable, Tuple
import numpy as np
import time
from datetime import datetime, date, time
from tkcalendar import DateEntry

from embrs.utilities.fire_util import FuelConstants
from embrs.base_classes.control_base import ControlClass

from embrs.utilities.data_classes import MapParams

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
        frame.pack(fill=BOTH, padx=5, pady=5)
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

    def create_spinbox_with_two_labels(self, frame: tk.Frame, left_label:str, max_val:float,
                                       var: any, right_label:str):
        """Creates a frame containing a spinbox with one or two labels

        :param frame: Root frame where the new frame should be located 
        :type frame: tk.Frame
        :param left_label: Label to be displayed to the left of the spinbox
        :type left_label: str
        :param max_val: Max allowable value in the spinbox
        :type max_val: float
        :param var: Variable where the result of the spinbox entry should be stored
        :type var: any
        :param right_label: Label to be displayed to the right of the spinbox
        :type right_label: str
        """
        new_frame = self.create_frame(frame)
        tk.Label(new_frame, text=left_label).grid(row=0, column=0)
        tk.Spinbox(new_frame, from_=1, to=max_val, textvariable=var,
                   width=5).grid(row=0, column=1, sticky='w')

        if right_label is not None:
            tk.Label(new_frame, text=right_label).grid(row=0, column=2)

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

class MapGenFileSelector(FileSelectBase):
    """Class used to prompt user for map generation files
    """
    def __init__(self):
        """Constructor method, populates tk window with all necessary elements and initializes
        necessary variables
        """

        super().__init__("Select Input Data Files and Map Destination Folder")

        # Define variables
        self.uniform_fuel = tk.BooleanVar()
        self.uniform_fuel.trace_add("write", self.uniform_options_toggled)
        self.fuel_selection = tk.StringVar()
        self.fuel_selection.set("Short grass")
        self.fuel_selection.trace_add("write", self.fuel_selection_changed)
        self.fuel_selection_val = 1
        self.fuel_map_filename = tk.StringVar()
        self.uniform_elev = tk.BooleanVar()
        self.uniform_elev.trace_add("write", self.uniform_options_toggled)
        self.elev_map_filename = tk.StringVar()
        self.aspect_map_filename = tk.StringVar()
        self.slope_map_filename = tk.StringVar()
        self.output_map_folder = tk.StringVar()
        self.import_roads = tk.BooleanVar()
        self.import_roads.set(False)
        self.sim_width = tk.IntVar()
        self.sim_width.set(1000)
        self.sim_height = tk.IntVar()
        self.sim_height.set(1000)

        frame = self.create_frame(self.root)

        # Create field to select save destination
        self.create_folder_selector(frame, "Save map to:   ", self.output_map_folder)

        # Create frame for fuel map selection
        _, _, self.fuel_button, self.fuel_frame = self.create_file_selector(frame, "Fuel Map:         ",
                                                  self.fuel_map_filename,
                                                  [("Tagged Image File Format","*.tif"),
                                                  ("Tagged Image File Format","*.tiff")])

        # Create field for uniform fuel option
        tk.Checkbutton(self.fuel_frame, text='Uniform Fuel',
                       variable=self.uniform_fuel).grid(row=0, column=3)

        uniform_fuel_frame = tk.Frame(frame)
        uniform_fuel_frame.pack(padx=10,pady=5)
        tk.Label(uniform_fuel_frame, text="Uniform Fuel Type:",
                 anchor="center").grid(row=0, column=0)

        tk.OptionMenu(uniform_fuel_frame, self.fuel_selection,
                      *FuelConstants.fuel_names.values()).grid(row=0, column=1)

        # Create frame for topography map selection
        _, _, self.elev_button, self.elev_frame = self.create_file_selector(frame,
                                                  "Elevation Map: ", self.elev_map_filename,
                    [("Tagged Image File Format","*.tif"), ("Tagged Image File Format","*.tiff")])

        # Create field for uniform elev option
        tk.Checkbutton(self.elev_frame, text="Uniform Elevation",
                       variable=self.uniform_elev).grid(row=0, column=3)

        # Create frame for aspect map selection
        _, _, self.asp_button, self.asp_frame = self.create_file_selector(frame,
                                                "Aspect Map: ", self.aspect_map_filename,
            [("Tagged Image File Format","*.tif"), ("Tagged Image File Format","*.tiff")])


        # Create frame for slope map selection
        _, _, self.slope_button, self.slope_frame = self.create_file_selector(frame,
                                                "Slope Map: ", self.slope_map_filename,
            [("Tagged Image File Format","*.tif"), ("Tagged Image File Format","*.tiff")])


        # Create frame for importing roads
        import_road_frame = tk.Frame(frame)
        import_road_frame.pack(padx=10,pady=5)
        self.import_roads_button = tk.Checkbutton(import_road_frame,
                                   text='Import Roads from OpenStreetMap',
                                   variable = self.import_roads, anchor="center")

        self.import_roads_button.grid(row=0, column=0)

        # Create frame for specifying sim size
        sim_size_frame = tk.Frame(frame)
        sim_size_frame.pack(padx=10,pady=5)
        tk.Label(sim_size_frame, text="width (m)", anchor="center").grid(row=0, column=0)
        self.width_entry = tk.Entry(sim_size_frame, textvariable=self.sim_width, width=20,
                                                                         state='disabled')

        self.width_entry.grid(row=0, column=1)
        tk.Label(sim_size_frame, text="height (m)", anchor="center").grid(row=0, column=2)
        self.height_entry = tk.Entry(sim_size_frame, textvariable=self.sim_height, width=20,
                                     state='disabled')

        self.height_entry.grid(row=0, column=3)

        # Create a submit button
        self.submit_button = tk.Button(frame, text="Submit", command=self.submit, state='disabled')
        self.submit_button.pack(pady=10)

    def uniform_options_toggled(self, *args):
        """Callback function that handles uniform fuel and uniform elevation options,
        turns on/off functionality based on these selections
        """
        if self.uniform_fuel.get() and self.uniform_elev.get():
            self.fuel_button.configure(state='disabled')
            self.elev_button.configure(state='disabled')
            self.asp_button.configure(state='disabled')
            self.slope_button.configure(state='disabled')
            self.height_entry.config(state='normal')
            self.width_entry.config(state='normal')
            self.import_roads_button.config(state='disabled')
            self.import_roads.set(False)

        elif self.uniform_fuel.get() and not self.uniform_elev.get():
            self.fuel_button.configure(state='disabled')
            self.elev_button.configure(state='active')
            self.elev_button.configure(state='active')
            self.asp_button.configure(state='active')
            self.height_entry.config(state='disabled')
            self.width_entry.config(state='disabled')
            self.import_roads_button.config(state='active')

        elif not self.uniform_fuel.get() and self.uniform_elev.get():
            self.fuel_button.configure(state='active')
            self.elev_button.configure(state='disabled')
            self.slope_button.configure(state='disabled')
            self.asp_button.configure(state='disabled')
            self.height_entry.config(state='disabled')
            self.width_entry.config(state='disabled')
            self.import_roads_button.config(state='active')

        else:
            self.fuel_button.configure(state='active')
            self.elev_button.configure(state='active')
            self.slope_button.configure(state='active')
            self.asp_button.configure(state='active')
            self.width_entry.config(state='disabled')
            self.height_entry.config(state='disabled')
            self.import_roads_button.config(state='active')

        self.validate_fields()

    def fuel_selection_changed(self, *args):
        """Callback function to handle the uniform fuel type value being changed
        """
        self.fuel_selection_val = FuelConstants.fuel_type_reverse_lookup[self.fuel_selection.get()]


    def validate_fields(self):
        """Function used to validate the inputs, primarily responsible for activating/disabling
        the submit button based on if all necessary input has been provided.
        """
        # Check that all fields are filled before enabling submit button
        if all([(self.fuel_map_filename.get() or self.uniform_fuel.get()),
                ((self.elev_map_filename.get() and self.aspect_map_filename.get() and self.slope_map_filename.get()) or self.uniform_elev.get()),
                 self.output_map_folder.get()]):

            self.submit_button.config(state='normal')

        else:
            self.submit_button.config(state='disabled')

    def submit(self):
        """Callback when the submit button is pressed. Stores all the relevant data in the result
        variable so it can be retrieved
        """
        if self.import_roads.get():
            # check that metadata.xml exists
            elev_path = os.path.dirname(self.elev_map_filename.get())
            fuel_path = os.path.dirname(self.fuel_map_filename.get())
            asp_path  = os.path.dirname(self.aspect_map_filename.get())
            slope_path = os.path.dirname(self.slope_map_filename.get())

            if os.path.exists(elev_path + "/metadata.xml"):
                metadata_path = elev_path + "/metadata.xml"

            elif os.path.exists(fuel_path + "/metadata.xml"):
                metadata_path = fuel_path + "/metadata.xml"

            else:
                self.import_roads.set(False)
                metadata_path = ""
                window = tk.Tk()
                window.withdraw()
                tk.messagebox.showwarning("Error",
                "Cannot locate 'metadata.xml' file, roads will not be imported.")

                window.destroy()
        else:
            metadata_path = ""

        self.result = MapParams()

        self.result.output_folder = self.output_map_folder.get()
        self.result.metadata_path = metadata_path
        self.result.import_roads = self.import_roads.get()

        if self.uniform_fuel.get():
            self.result.uniform_fuel = True
            self.result.fuel_type = self.fuel_selection_val
            self.result.fuel_data.tiff_filepath = ""
        
        else:
            self.result.uniform_fuel = False
            self.result.fuel_data.tiff_filepath = self.fuel_map_filename.get()

        if self.uniform_elev.get():
            self.result.uniform_elev = True
            self.result.elev_data.tiff_filepath = ""
            self.result.asp_data.tiff_filepath = ""
            self.result.slp_data.tiff_filepath = ""
        
        else:
            self.result.uniform_elev = False
            self.result.elev_data.tiff_filepath = self.elev_map_filename.get()
            self.result.asp_data.tiff_filepath = self.aspect_map_filename.get()
            self.result.slp_data.tiff_filepath = self.slope_map_filename.get()

        if self.uniform_elev.get() and self.uniform_fuel.get():
            self.result.width_m = self.sim_width.get()
            self.result.height_m = self.sim_width.get()

        self.root.withdraw()
        self.root.quit()

class HistoricFireSelector(MapGenFileSelector):
    """Class used to prompt user for historic fire data along with map generation files
    """
    def __init__(self):
        """Constructor method, populates tk window with necessary elements and initializes variables"""
        super().__init__()

        # Define new variables
        self.ignition_date = tk.StringVar()
        self.scenario_start_date = tk.StringVar()
        self.ignition_lat = tk.DoubleVar()
        self.ignition_lon = tk.DoubleVar()

        frame = self.create_frame(self.root)

        # Create fields for Ignition Date and Scenario Start Date
        tk.Label(frame, text="Ignition Date:").pack(pady=5)
        self.ignition_date_entry = DateEntry(frame, textvariable=self.ignition_date, date_pattern="y-mm-dd")
        self.ignition_date_entry.pack(pady=5)

        tk.Label(frame, text="Scenario Start Date:").pack(pady=5)
        self.scenario_start_date_entry = DateEntry(frame, textvariable=self.scenario_start_date, date_pattern="y-mm-dd")
        self.scenario_start_date_entry.pack(pady=5)

        # Create fields for Ignition Location (Lat, Lon)
        ignition_location_frame = tk.Frame(frame)
        ignition_location_frame.pack(padx=10, pady=5)

        tk.Label(ignition_location_frame, text="Ignition Latitude:").grid(row=0, column=0)
        self.lat_entry = tk.Entry(ignition_location_frame, textvariable=self.ignition_lat)
        self.lat_entry.grid(row=0, column=1)

        tk.Label(ignition_location_frame, text="Ignition Longitude:").grid(row=0, column=2)
        self.lon_entry = tk.Entry(ignition_location_frame, textvariable=self.ignition_lon)
        self.lon_entry.grid(row=0, column=3)

        # Modify the submit button callback to include validation for dates and location
        self.submit_button.config(command=self.submit)

    def submit(self):
        """Override the submit method to include validation for historic fire data"""
        # Validate the ignition date and scenario start date
        try:
            ignition_date = datetime.strptime(self.ignition_date.get(), "%Y-%m-%d")
            scenario_start_date = datetime.strptime(self.scenario_start_date.get(), "%Y-%m-%d")
            if scenario_start_date <= ignition_date:
                raise ValueError("Scenario Start Date must be after Ignition Date.")
        except ValueError as e:
            tk.messagebox.showerror("Date Error", str(e))
            return

        # Validate the ignition location (latitude and longitude)
        if not (-90 <= self.ignition_lat.get() <= 90):
            tk.messagebox.showerror("Input Error", "Latitude must be between -90 and 90.")
            return
        if not (-180 <= self.ignition_lon.get() <= 180):
            tk.messagebox.showerror("Input Error", "Longitude must be between -180 and 180.")
            return

        # Call the original submit method to handle the map generation part
        super().submit()

        # Extend the result dictionary with the additional data
        self.result["Ignition Date"] = self.ignition_date.get()
        self.result["Scenario Start Date"] = self.scenario_start_date.get()
        self.result["Ignition Latitude"] = self.ignition_lat.get()
        self.result["Ignition Longitude"] = self.ignition_lon.get()

        # Close the window after successful submission
        self.root.quit() 




class SimFolderSelector(FileSelectBase):
    # TODO: need to deal with duration based on the weather file currently input
    """Class used to prompt user for inputs to set up a sim
    """
    def __init__(self, submit_callback:Callable):
        """Constructor method, populates tk window with all necessary elements and initializes
        necessary variables
        """
        super().__init__("EMBRS")

        self.submit_callback = submit_callback

        # Define variables
        self.map_folder = tk.StringVar()
        self.log_folder = tk.StringVar()
        self.weather = tk.StringVar()
        self.time_step = tk.IntVar()
        self.cell_size = tk.IntVar()
        self.sim_time = tk.DoubleVar()
        self.time_unit = tk.StringVar()
        self.num_runs = tk.IntVar()
        self.user_path = tk.StringVar()
        self.user_class_name = tk.StringVar()
        self.viz_on = tk.BooleanVar()
        self.no_log = tk.BooleanVar()
        self.open_meteo = tk.BooleanVar()
        self.weather_file = tk.BooleanVar()
        self.mesh_resolution = tk.IntVar()
        self.start_date = tk.StringVar()
        self.end_date = tk.StringVar() 
        self.start_hour = tk.IntVar()
        self.start_min = tk.IntVar()
        self.end_hour = tk.IntVar()
        self.end_min = tk.IntVar()
        self.start_ampm = tk.StringVar()
        self.end_ampm = tk.StringVar()

        # Define some useful variables
        self.prev_t_unit = "hours"
        self.max_duration = np.inf
        self.class_options = []
        self.user_module = None

        # Set some initial values
        self.time_step.set(5)
        self.cell_size.set(10)
        self.sim_time.set(1.0)
        self.num_runs.set(1)
        self.viz_on.set(False)
        self.time_unit.set("hours")
        self.no_log.set(False)
        self.open_meteo.set(True)
        self.weather_file.set(False)
        self.mesh_resolution.set(250)
        self.start_hour.set(12)
        self.start_min.set(0)
        self.end_hour.set(12)
        self.end_min.set(0)
        self.start_ampm.set("AM")
        self.end_ampm.set("AM")

        frame = self.create_frame(self.root)

        # Create frame for map folder selection
        self.create_folder_selector(frame, "Map folder:    ", self.map_folder)

        # Create frame for log folder selection
        _, self.log_entry, self.log_button, self.log_frame = self.create_folder_selector(frame, "Log folder:     ", self.log_folder)
        tk.Checkbutton(self.log_frame, text='Disable logging',
                variable=self.no_log).grid(row=0, column=3)


        weather_option_frame = self.create_frame(frame)
        tk.Label(weather_option_frame, text="Weather Option:").grid(row=0, column=0)
        tk.Checkbutton(weather_option_frame, text='Import OpenMeteo Weather',
                       variable=self.open_meteo).grid(row=0, column=1)
        
        tk.Checkbutton(weather_option_frame, text='Import Weather .json File',
                       variable=self.weather_file).grid(row=0, column=2)

        self.open_meteo_frame = self.create_frame(frame)

        tk.Label(self.open_meteo_frame, text="Start Date:").grid(row=0, column=0)
        self.start_cal = DateEntry(self.open_meteo_frame, date_pattern="y-mm-dd", background='white')
        self.start_cal.grid(row=0, column =1)
        self.start_cal.bind("<<DateEntrySelected>>", lambda e: self.update_datetime())
        
        # Start time selection
        tk.Label(self.open_meteo_frame, text="Start Time:").grid(row=0, column=2)
        tk.Spinbox(self.open_meteo_frame, from_=1, to=12, width=5, format="%2.0f", textvariable=self.start_hour).grid(row=0, column=3)
        tk.Spinbox(self.open_meteo_frame, from_=0, to=59, width=5, format="%02.0f", textvariable=self.start_min).grid(row=0, column=4, padx=5)
        ttk.Combobox(self.open_meteo_frame, values=["AM", "PM"], width=5, state='readonly', textvariable=self.start_ampm).grid(row=0, column=5, padx=5)
    
        # End time selection
        tk.Label(self.open_meteo_frame, text="End Time:").grid(row=1, column=2)
        tk.Spinbox(self.open_meteo_frame, from_=1, to=12, width=5, format="%2.0f", textvariable=self.end_hour).grid(row=1, column=3)
        tk.Spinbox(self.open_meteo_frame, from_=0, to=59, width=5, format="%02.0f", textvariable=self.end_min).grid(row=1, column=4, padx=5)
        ttk.Combobox(self.open_meteo_frame, values=["AM", "PM"], width=5, state='readonly', textvariable=self.end_ampm).grid(row=1, column=5, padx=5)

        tk.Label(self.open_meteo_frame, text="End Date:").grid(row=1, column=0)
        self.end_cal = DateEntry(self.open_meteo_frame, date_pattern="y-mm-dd", background='white')
        self.end_cal.grid(row=1, column =1)
        self.end_cal.bind("<<DateEntrySelected>>", lambda e: self.update_datetime())

        # Create frame for wind file selection
        _, self.weather_entry, self.weather_button, self.weather_file_frame = self.create_file_selector(frame, "Weather file: ", self.weather, [("JavaScript Object Notation","*.JSON")])

        self.weather.set("")
        self.weather_button.configure(state='disabled')
        self.weather_entry.configure(state='disabled')

        self.create_spinbox_with_two_labels(frame, "Wind Mesh Resolution:       ", 250, self.mesh_resolution, "meters")

        # Create frame for time step selection
        self.create_spinbox_with_two_labels(frame, "Time step:     ", 100, self.time_step, "seconds")

        # Create frame for cell size selection
        self.create_spinbox_with_two_labels(frame, "Cell size:       ", 100, self.cell_size, "meters")

        # Create frame for sim time selection
        sim_time_frame = self.create_frame(frame)
        tk.Label(sim_time_frame, text="Duration:       ").grid(row=2, column=0)

        self.duration_spinbox = tk.Spinbox(sim_time_frame, from_=1, to=self.max_duration,
                                           textvariable=self.sim_time, width=5, format='%.2f', increment = 0.5)

        self.duration_spinbox.grid(row=2, column=1, sticky='w')
        tk.OptionMenu(sim_time_frame, self.time_unit,
                      "seconds", "minutes", "hours").grid(row=2, column=2)

        # Create frame for num runs selection
        self.create_spinbox_with_two_labels(frame, "Iterations:      ", np.inf,

                                            self.num_runs, None)

        # Create frame for user module selection
        self.create_file_selector(frame, "User module:", self.user_path, [("python file","*.py")])

        # Create frame for user class selection
        class_name_frame = tk.Frame(frame)
        class_name_frame.pack(padx=10,pady=5)
        tk.Label(class_name_frame, text = "User class name:").grid(row=0, column=0)
        self.class_opt_menu = tk.OptionMenu(class_name_frame, self.user_class_name,
                                            self.class_options)
        self.class_opt_menu.grid(row=0, column=1)

        # Create frame for viz selection
        viz_frame = tk.Frame(frame)
        viz_frame.pack(padx=10, pady=5)
        tk.Checkbutton(viz_frame, text='Visualize in Real-time',
                       variable = self.viz_on).grid(row=0, column=0)

        # Create a submit button
        self.submit_button = tk.Button(frame, text="Submit", command=self.submit, state='disabled')
        self.submit_button.pack(pady=10)

        # Define variable callbacks
        self.map_folder.trace_add("write", self.map_folder_changed)
        self.log_folder.trace_add("write", self.log_folder_changed)
        self.sim_time.trace_add("write", self.duration_changed)
        self.time_unit.trace_add("write", self.time_unit_changed)
        self.user_path.trace_add("write", self.user_path_changed)
        self.user_class_name.trace_add("write", self.class_changed)
        self.no_log.trace_add("write", self.write_logs_toggled)
        self.open_meteo.trace_add("write", self.open_meteo_toggled)
        self.weather_file.trace_add("write", self.weather_file_toggled)

        self.start_hour.trace_add("write", self.update_datetime)
        self.start_min.trace_add("write", self.update_datetime)
        self.start_ampm.trace_add("write", self.update_datetime)
        self.end_hour.trace_add("write", self.update_datetime)
        self.end_min.trace_add("write", self.update_datetime)
        self.end_ampm.trace_add("write", self.update_datetime)


        self.update_datetime()


    def update_datetime(self, *args):
        """Update datetime values whenever any input field changes"""
        start_hr = self.convert_to_24_hr_time(self.start_hour.get(), self.start_ampm.get())
        start_time = time(start_hr, self.start_min.get())
        self.start_datetime = datetime.combine(self.start_cal.get_date(), start_time)

        end_hr = self.convert_to_24_hr_time(self.end_hour.get(), self.end_ampm.get())
        end_time = time(end_hr, self.end_min.get())
        self.end_datetime = datetime.combine(self.end_cal.get_date(), end_time)

    def duration_changed(self, *args):
        """Callback function that handles the sim duration changing, prevents values greater than 
        the max being input
        """
        try:
            self.sim_time.get()
        except tk.TclError:
            return

        if self.sim_time.get() > self.max_duration:
            self.sim_time.set(self.max_duration)

    def map_folder_changed(self, *args):
        """Callback function that handles the map folder selection being changed, sets the max
        duration allowed based on the length of the wind forecast in the map
        """
        # open json
        folderpath = self.map_folder.get()
        foldername = os.path.basename(folderpath)
        map_filepath = folderpath + "/" + foldername + ".json"

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

        if os.path.exists(folderpath + '/init_fire_state.pkl') or os.path.exists(folderpath + '/run_0'):
            window = tk.Tk()
            window.withdraw()
            msg = """Warning: Selected folder already contains log files, these will be overwritten
                     if you keep this selection"""
            tk.messagebox.showwarning("Warning", msg)
            window.destroy()

    def time_unit_changed(self, *args):
        """Callback function that handles the time unit selection being changed, adjusts the 
        displayed duration based on the unit
        """
        time_units = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}

        # Get the ratio between the new and old time unit
        ratio = time_units[self.prev_t_unit] / time_units[self.time_unit.get()]

        self.max_duration *= ratio
        self.duration_spinbox.configure(to=self.max_duration)

        # Update sim_time
        current_sim_time = float(self.sim_time.get())
        new_sim_time = current_sim_time * ratio
        self.sim_time.set(new_sim_time)

        # Update prev_t_unit
        self.prev_t_unit = self.time_unit.get()

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

    def open_meteo_toggled(self, *args):
        """Callback for the OpenMeteo toggle button. Ensures only OpenMeteo is selected."""
        if self.open_meteo.get():
            # Turn off weather file option
            self.weather_file.set(False)
            # Disable weather file widgets
            self.weather.set("")
            self.weather_button.configure(state='disabled')
            self.weather_entry.configure(state='disabled')
            # Enable OpenMeteo widgets
            for widget in self.open_meteo_frame.winfo_children():
                widget.configure(state='normal')
        else:
            self.weather_file.set(True)
            for widget in self.open_meteo_frame.winfo_children():
                widget.configure(state='disabled')

            self.weather_button.configure(state='normal')
            self.weather_entry.configure(state='normal')

        self.validate_fields()

    def weather_file_toggled(self, *args):
        """Callback for the Weather File toggle button. Ensures only Weather File is selected."""
        if self.weather_file.get():
            # Turn off OpenMeteo option
            self.open_meteo.set(False)
            # Disable OpenMeteo widgets
            for widget in self.open_meteo_frame.winfo_children():
                widget.configure(state='disabled')
            # Enable weather file widgets
            self.weather_button.configure(state='normal')
            self.weather_entry.configure(state='normal')
        else:
            # Prevent both options from being off
            self.open_meteo.set(True)
            self.weather_file.set(False)
            # Disable weather file widgets
            self.weather.set("")
            self.weather_button.configure(state='disabled')
            self.weather_entry.configure(state='disabled')
            # Enable OpenMeteo widgets
            for widget in self.open_meteo_frame.winfo_children():
                widget.configure(state='normal')

        self.validate_fields()

    def write_logs_toggled(self, *args):
        if self.no_log.get():
            self.log_button.configure(state='disabled')
            self.log_entry.configure(state='disabled')
            self.log_folder.set("")
        
        else:
            self.log_button.configure(state='normal')
            self.log_entry.configure(state='normal')

        self.validate_fields()

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
        if self.map_folder.get() and  (self.no_log.get() or self.log_folder.get()) and (self.open_meteo.get() or self.weather.get()):
            self.submit_button.config(state='normal')
        else:
            self.submit_button.config(state='disabled')

    def submit(self):
        """Callback when the submit button is pressed. Stores all the relevant data in the result
        variable so it can be retrieved
        """
        sim_time_raw = self.sim_time.get()

        if self.time_unit.get() == "seconds":
            sim_time = sim_time_raw

        elif self.time_unit.get() == "minutes":
            sim_time = sim_time_raw * 60

        elif self.time_unit.get() == "hours":
            sim_time = sim_time_raw * 3600

        if not (self.start_datetime < self.end_datetime < datetime.now()):
            tk.messagebox.showwarning("Invalid Date Selection", 
                                "Start date and time must be before the end date and time, and the end date must be in the past.")

        else:
            self.result = {
                "input": self.map_folder.get(),
                "log": self.log_folder.get(),
                "weather_type": "OpenMeteo" if self.open_meteo.get() else "Weather File",
                "weather_file": self.weather.get(),
                "mesh_resolution": self.mesh_resolution.get(),
                "start_datetime": self.start_datetime,
                "end_datetime": self.end_datetime,
                "t_step": self.time_step.get(),
                "cell_size": self.cell_size.get(),
                "sim_time": sim_time,
                "viz_on": self.viz_on.get(),
                "num_runs": self.num_runs.get(),
                "user_path": self.user_path.get(),
                "class_name": self.user_class_name.get(),
                "write_log_files": not self.no_log.get()
            }

            self.submit_callback(self.result)

class VizFolderSelector(FileSelectBase):
    """Class used to prompt user for log files to be visualized

    :param submit_callback: Function that should be called when the submit button is pressed
    :type submit_callback: Callable
    """
    def __init__(self, submit_callback: Callable):
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
        self.has_agents = False
        self.run_folders = ["No runs available, select a folder"]

        self.init_location = False

        # Save submit callback function
        self.submit_callback = submit_callback

        frame = self.create_frame(self.root)

        # # Create a frame to select log file
        self.create_folder_selector(frame, "Log folder:    ", self.viz_folder)

        run_selection_frame = tk.Frame(frame)
        run_selection_frame.pack(padx=10,pady=5)
        tk.Label(run_selection_frame, text="Run to Visualize:",
                 anchor="center").grid(row=0, column=0)

        self.run_options = tk.OptionMenu(run_selection_frame, self.run_folder,
                      *self.run_folders)

        self.run_options.grid(row=0, column=1)

        # Create a frame to select sim time per frame
        self.create_spinbox_with_two_labels(frame, "Update Frequency:", np.inf,
                                            self.viz_freq, "seconds")

        # Create a frame to select scale bar size
        self.create_spinbox_with_two_labels(frame, "Scale bar size:        ", 100, self.scale_km, "km")

        # Create a frame to choose legend display
        legend_frame = tk.Frame(frame)
        legend_frame.pack(padx=10, pady=5)
        tk.Checkbutton(legend_frame, text="Display fuel legend",
                       variable=self.legend).grid(row=0,column=0)

        self.submit_button = tk.Button(frame, text='Submit', command = self.submit,
                                       state='disabled')
        self.submit_button.pack(pady=10)

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

        if os.path.exists(os.path.join(run_folderpath, 'agents.msgpack')):
            self.has_agents = True
            self.agent_file = os.path.join(run_folderpath, 'agents.msgpack')

        else:
            self.has_agents = False

        if os.path.exists(os.path.join(run_folderpath, 'log.msgpack')):
            self.viz_file = os.path.join(run_folderpath, 'log.msgpack')

        else:
            window = tk.Tk()
            window.withdraw()
            tk.messagebox.showwarning("Error",
            "Selected folder does not have a valid log file!")

            window.destroy()

            self.run_folder.set("")

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

        if not os.path.exists(os.path.join(folderpath, 'init_fire_state.pkl')):
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
        self.result = {
            "file": self.viz_file,
            "freq": self.viz_freq.get(),
            "scale_km": self.scale_km.get(),
            "legend": self.legend.get(),
            "init_location": self.init_location,
            "has_agents": self.has_agents
        }

        if self.has_agents:
            self.result["agent_file"] = self.agent_file

        self.submit_callback(self.result)

    def validate_fields(self, *args):
        """Function used to validate the inputs, primarily responsible for activating/disabling
        the submit button based on if all necessary input has been provided.
        """
        if self.viz_folder.get() and self.viz_freq.get() > 0:
            self.submit_button.config(state='normal')
        else:
            self.submit_button.config(state='disabled')

class WindForecastGen(FileSelectBase):
    """Class used to allow users to generate wind forecasts

    :param submit_callback: Function that should be called when the submit button is pressed
    :type submit_callback: Callable
    """
    def __init__(self, submit_callback: Callable):
        """Constructor method, populates tk window with all necessary elements and initializes
        necessary variables
        """
        super().__init__("Wind Forecast Generator")

        self.submit_callback = submit_callback

        # define variables
        self.file_name = tk.StringVar()
        self.folderpath = tk.StringVar()
        self.duration = tk.DoubleVar()
        self.duration.set(1.0)
        self.time_step = tk.DoubleVar()
        self.curr_speed_val = tk.DoubleVar()
        self.curr_direction_val = tk.DoubleVar()

        self.wind_forecast = []

        self.time_step_last_changed = 0
        self.duration_last_changed = 0

        frame = self.create_frame(self.root)

        filename_frame = tk.Frame(frame)
        filename_frame.pack(padx=10,pady=5)
        tk.Label(filename_frame, text="Filename: ", anchor="center").grid(row=0, column=0)
        self.filename_entry = tk.Entry(filename_frame, textvariable=self.file_name, width = 25)
        self.filename_entry.grid(row=0, column=1)
        tk.Label(filename_frame, text=".json ", anchor="center").grid(row=0, column=2)

        self.create_folder_selector(frame, "Save forecast to: ", self.folderpath)

        entry_frame = tk.Frame(frame)
        entry_frame.pack(padx=10,pady=5)
        tk.Label(entry_frame, text="Wind Speed: ", anchor="center").grid(row=0, column=0)
        self.speed_entry = tk.Entry(entry_frame, textvariable=self.curr_speed_val, width = 10)
        self.speed_entry.grid(row=0, column=1)
        tk.Label(entry_frame, text="(m/s)", anchor="center").grid(row=0, column=2)

        time_step_frame = tk.Frame(frame)
        time_step_frame.pack(padx=20,pady=5)
        tk.Label(time_step_frame, text="Time Step: ", anchor="center").grid(row=0, column=0)
        self.t_step_entry = tk.Spinbox(time_step_frame, from_=1, to=1000, increment=1, textvariable=self.time_step, width=10)
        self.t_step_entry.grid(row=0, column=1)
        tk.Label(time_step_frame, text="mins", anchor="center").grid(row=0, column=2)

        # spacer
        tk.Label(time_step_frame, text="").grid(row=0, column=3, padx=20)

        tk.Label(time_step_frame, text="Duration: ", anchor="center").grid(row=0, column=4)
        self.duration_entry = tk.Spinbox(time_step_frame, from_=0.0166666, to=1000, increment=0.5, textvariable=self.duration, width=10)
        self.duration_entry.grid(row=0, column=5)
        tk.Label(time_step_frame, text="hours", anchor="center").grid(row=0, column=6)


        # spacer
        tk.Label(entry_frame, text="").grid(row=0, column=3, padx=20)

        tk.Label(entry_frame, text="Wind Direction: ", anchor="center").grid(row=0, column=4)
        self.dir_entry = tk.Entry(entry_frame, textvariable=self.curr_direction_val, width = 10)
        self.dir_entry.grid(row=0, column=5)
        tk.Label(entry_frame, text="degrees", anchor="center").grid(row=0, column=6)

        # Buttons to add, delete, and edit entries
        btn_add = tk.Button(entry_frame, text="Add", command=self.add_entry)
        btn_add.grid(row=0,column=7, padx=10)

        # Listbox to display entries
        self.listbox = tk.Listbox(frame, width=40, height=10)
        self.listbox.pack(pady=10)

        edit_del_frame = tk.Frame(frame)
        edit_del_frame.pack(padx=10,pady=5)

        btn_edit = tk.Button(edit_del_frame, text="Overwrite", command=self.overwrite_entry)
        btn_edit.grid(row=0, column=0)
        btn_delete = tk.Button(edit_del_frame, text="Delete", command=self.delete_entry)
        btn_delete.grid(row=0, column=1)

        # Create a submit button
        self.submit_button = tk.Button(frame, text="Save to File", command=self.submit, state='disabled')
        self.submit_button.pack(pady=10)

        self.time_step.trace_add('write', self.time_step_changed)
        self.duration.trace_add('write', self.duration_changed)

    def validate_fields(self):
        """Function used to validate the inputs, primarily responsible for activating/disabling
        the submit button based on if all necessary input has been provided.
        """
        folder_exists = os.path.exists(self.folderpath.get())
        filename_entered = self.file_name.get()
        entries_exist = len(self.wind_forecast)

        if entries_exist and self.time_step.get() > 0 and self.duration.get() > 0 and folder_exists and filename_entered:
            self.submit_button.config(state='normal')

    def add_entry(self):
        """Callback function for 'add' button that adds the current input for direction and speed
        to the working wind forecast.
        """

        entry = (self.curr_speed_val.get(), self.curr_direction_val.get())
        if entry:  # only add non-empty entries
            self.wind_forecast.append(entry)
            hours = int(np.floor(len(self.listbox.get(0, tk.END)) * self.time_step.get() / 60))
            mins = int(len(self.listbox.get(0, tk.END)) * self.time_step.get() % 60)
            formatted_entry = f"{hours} h {mins} m: {entry[0]} m/s, {entry[1]} deg"
            self.listbox.insert(tk.END, formatted_entry)

        self.duration.set(len(self.wind_forecast) * self.time_step.get() / 60)

        self.validate_fields()

    def delete_entry(self):
        """Callback function for 'delete' button that removes the selected entry from the working
        wind forecast.
        """
        try:
            # get the index of the selected entry
            index = self.listbox.curselection()[0]
            self.listbox.delete(index)

            del self.wind_forecast[index]

            self.duration.set(len(self.wind_forecast) * self.time_step.get() / 60)

        except IndexError:  # if no entry is selected
            pass

        self.validate_fields()

    def overwrite_entry(self):
        """Callback function for 'overwrite' button that overwrites the selected entry in the
        working wind forecast with the current input for speed and direction.
        """
        try:
            # get the index of the selected entry
            index = self.listbox.curselection()[0]

            updated_entry = (self.curr_speed_val.get(), self.curr_direction_val.get())

            self.wind_forecast[index] = updated_entry

            mins = index * self.time_step.get()

            hours = int(np.floor(mins / 60))
            mins = int(mins % 60)

            formatted_entry = f"{hours} h {mins} m: {updated_entry[0]} m/s, {updated_entry[1]} deg"

            self.listbox.delete(index)
            self.listbox.insert(index, formatted_entry)

        except IndexError:  # if no entry is selected
            pass

        self.validate_fields()

    def time_step_changed(self, *args):
        """Callback function that handles when the time step of the forecast has been changed.
        Ensures the entered time step is valid and updates the duration based on the new time 
        step and the current number of entries in the forecast.
        """
        try:
            current_time_step = self.time_step.get()
        except tk.TclError:
            current_time_step = 0  # default value if Spinbox is empty

        if current_time_step < 0:
            self.time_step.set(1)
            return

        self.listbox.delete(0, tk.END)
        for idx, (speed, direction) in enumerate(self.wind_forecast):
            mins = int((idx * current_time_step) % 60)
            hours = int(np.floor(idx * current_time_step) / 60)
            formatted_entry = f"{hours} h {mins} m: {speed} m/s, {direction} deg"
            self.listbox.insert(tk.END, formatted_entry)

        self.duration.set(len(self.wind_forecast) * current_time_step / 60)  # Convert to hours

        self.validate_fields()

    def duration_changed(self, *args):
        """Callback function that handles when the duration of the forecast has been changed.
        Ensures the entered duration is valid and updates the time step based on the new
        duration and the current number of entries in the forecast.
        """
        try:
            curr_duration = self.duration.get()
        except tk.TclError:
            curr_duration = 0  # default value if Spinbox is empty

        # If duration was the last changed variable, update time_step
        if self.duration_last_changed > self.time_step_last_changed:
            if len(self.wind_forecast) > 0:
                new_time_step = curr_duration * 60 / len(self.wind_forecast)  # Convert duration to minutes
                self.time_step.set(new_time_step)
        else:
            # Update duration based on new time_step and number of entries
            self.duration.set(len(self.wind_forecast) * self.time_step.get() / 60)  # Convert to hours

        # Update the last change time for duration
        self.duration_last_changed = time.time()

        self.validate_fields()

    def submit(self):
        """Callback when the submit button is pressed. Stores all the relevant data in the result
        variable so it can be retrieved
        """
        data = {
            "save_location": os.path.join(self.folderpath.get(), self.file_name.get() + ".json"),
            "time_step": self.time_step.get(),
            "forecast": self.wind_forecast
        }

        self.submit_callback(data)

        self.root.withdraw()
        self.root.quit()

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
