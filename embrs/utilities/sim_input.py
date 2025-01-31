import numpy as np

class SimInput():
    def __init__(self):
        """_summary_
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
        self.duration_s = None
        self.initial_ignition = None
        self.size = None
        self.shape = None
        self.burnt_cells = []
        self.display_freq_s = 300
        self.viz_on = False

class DataMap():
    def __init__(self, map: np.ndarray, resolution: float):
        """_summary_

        Args:
            map (np.ndarray): _description_
            resolution (float): _description_
        """
        self.map = map
        self.res = resolution
        
class WindDataMap(DataMap):
    def __init__(self, map: np.ndarray, resolution: float, time_step: float):
        """_summary_

        Args:
            map (np.ndarray): _description_
            resolution (float): _description_
            time_step (float): _description_
        """

        super().__init__(map, resolution)
        self.time_step = time_step