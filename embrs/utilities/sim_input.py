"""Simulation input handling for wildfire modeling.

This module defines classes used to store and manage simulation input parameters, 
including terrain data, fuel properties, wind conditions, and fire break locations.

Classes:
    - SimInput: Stores all input parameters required for initializing a fire simulation.
    - DataMap: Represents a structured data map with spatial resolution.
    - WindDataMap: Extends `DataMap` to include wind-specific time step data.

"""

import numpy as np

class SimInput():
    """A container for all input parameters required to initialize a fire simulation.

    This class holds the environmental and simulation settings such as terrain 
    properties, wind data, fire breaks, and time step parameters.

    Attributes:
        fuel (Optional[DataMap]): A `DataMap` representing fuel types across the simulation grid.
        elevation (Optional[DataMap]): A `DataMap` storing terrain elevation data.
        aspect (Optional[DataMap]): A `DataMap` representing slope aspect (direction of steepest slope).
        slope (Optional[DataMap]): A `DataMap` representing terrain slope steepness.
        wind (Optional[WindDataMap]): A `WindDataMap` storing wind conditions and time-dependent data.
        roads (list): A list of road geometries within sim region
        fire_breaks (list): A list of fire break geometries that limit fire propagation.
        time_step (Optional[float]): The simulation time step in seconds.
        cell_size (Optional[float]): The size of each simulation cell in meters.
        duration_s (Optional[float]): The total simulation duration in seconds.
        initial_ignition (Optional[list]): A list of initial ignition locations.
        size (Optional[tuple]): The total size of the simulation domain.
        shape (Optional[tuple]): The shape (rows, columns) of the simulation grid.
        burnt_cells (list): A list of cells that are pre-designated as burned at initialization.
        display_freq_s (int): The frequency (in seconds) at which visualization updates occur.
        viz_on (bool): Flag to enable or disable visualization.

    """
    def __init__(self):
        """Initializes the `SimInput` object with default values.

        This constructor sets up placeholders for terrain, fuel, and wind data, as well 
        as simulation parameters such as time step, duration, and visualization settings.
        """
        self.fuel = None
        self.elevation = None
        self.aspect = None
        self.slope = None
        self.wind = None
        self.roads = []
        self.fire_breaks = []
        self.time_step = None
        self.cell_size = None
        self.north_angle = None
        self.duration_s = None
        self.initial_ignition = None
        self.size = None
        self.shape = None
        self.burnt_cells = []
        self.display_freq_s = 300
        self.viz_on = False

class DataMap():
    """Represents a structured spatial data map with resolution metadata.

    This class is used to store and process raster-like simulation data, 
    such as fuel models, elevation maps, slope maps, and aspect maps.

    Attributes:
        map (np.ndarray): A NumPy array representing the spatial data grid.
        res (float): The spatial resolution of the data (e.g., meters per grid cell).
    """
    def __init__(self, map: np.ndarray, resolution: float):
        """Initializes a `DataMap` object.

        Args:
            map (np.ndarray): A 2D NumPy array containing the mapped data values.
            resolution (float): The spatial resolution of the map (meters per grid cell).
        """
        self.map = map
        self.res = resolution
        
class WindDataMap(DataMap):
    """Represents wind data in a structured spatial format with a time-dependent component.

    This class extends `DataMap` to include a time step parameter, 
    allowing for dynamic wind updates during the simulation.

    Attributes:
        time_step (float): The time step interval (in seconds) at which 
                           wind data is updated.
    """
    def __init__(self, map: np.ndarray, resolution: float, time_step: float):
        """Initializes a `WindDataMap` object.

        Args:
            map (np.ndarray): A 2D NumPy array representing wind speed or direction values.
            resolution (float): The spatial resolution of the wind data (meters per grid cell).
            time_step (float): The time step interval (seconds) for wind data updates.
        """
        super().__init__(map, resolution)
        self.time_step = time_step