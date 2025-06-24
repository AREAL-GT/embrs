"""Fuel model definitions for fire spread simulation.

This module defines fuel models used in fire behavior modeling, including the 
Anderson 13 Fire Behavior Fuel Models (FBFMs). It provides structured data for 
different fuel types, including their physical properties and combustion characteristics.

Classes:
    - Fuel: Base class representing a generic fuel type with physical properties.
    - Anderson13: Subclass representing the 13 standard Anderson fuel models.

References:
    - Anderson, H. E. (1982). Aids to Determining Fuel Models for Estimating Fire Behavior.
      USDA Forest Service General Technical Report INT-122.

"""
import numpy as np
import os
import json
from embrs.utilities.unit_conversions import *

class Fuel:
    """Represents a generic fuel type with physical and combustion properties.

    This class serves as a base for defining fuel models, including their 
    loading parameters, fuel moisture, and combustion characteristics.

    Args:
        name (str): Name of the fuel model.
        model_num (int): Fuel model number (e.g., Anderson 13 fuel model ID).
        fuel_load_params (dict): Fuel load parameters for different time-lag classes 
                                 (e.g., 1-hour, 10-hour fuels).
        sav_ratio (int): Surface-area-to-volume ratio (cm²/cm³) of the dominant fuel.
        fuel_depth (float): Fuel bed depth in meters.
        m_x (float): Moisture content of extinction (fraction).
        rel_packing_ratio (float): Packing ratio relative to optimal.
        rho_b (float): Bulk density of the fuel bed (kg/m³).
        burnable (bool): Indicates if the fuel type is burnable.

    Attributes:
        heat_content (float): Heat content of the fuel (BTU/lb).
        fuel_moisture (float): Initial fuel moisture content (default = 0.01).
        net_fuel_load (float): Computed net fuel load (lb/ft²).

    Methods:
        - set_net_fuel_load(): Computes the net fuel load based on the fuel properties.
        - set_fuel_moisture(moisture): Placeholder for updating fuel moisture dynamically.
    """
    def __init__(self, name: str, model_num: int, burnable: bool, dynamic: bool, f_i: np.ndarray, f_ij: np.ndarray, g_ij: np.ndarray, w_0: np.ndarray,
                 s: np.ndarray, s_total: int, dead_mx: float, fuel_depth: float,
                 rho_b: float, rel_packing_ratio: float):
        """Initializes a generic fuel model with its physical and combustion properties.

        This constructor defines the primary attributes of a fuel model, including 
        its loading parameters, packing ratio, and moisture content. 

        Args:
            name (str): The name of the fuel model (e.g., "Short Grass").
            model_num (int): The model number (e.g., Anderson 13 fuel model ID).
            fuel_load_params (dict): A dictionary of fuel load parameters for different 
                                    fuel classes (e.g., `{"1-h": (fuel_load, sav_ratio)}`).
            sav_ratio (int): The surface-area-to-volume ratio of fine fuels (cm²/cm³).
            fuel_depth (float): The depth of the fuel bed (meters).
            m_x (float): The moisture content of extinction (fraction, 0 to 1).
            rel_packing_ratio (float): The relative packing ratio of the fuel.
            rho_b (float): The bulk density of the fuel bed (kg/m³).
            burnable (bool): Whether this fuel model is burnable (`True`) or not (`False`).

        Attributes:
            heat_content (float): Heat content of the fuel (BTU/lb), default = `8000`.
            fuel_moisture (float): Initial fuel moisture content, default = `0.01`.
            net_fuel_load (float): Computed net fuel load (lb/ft²), set to `0` if non-burnable.

        Behavior:
            - Computes `net_fuel_load` automatically for burnable fuels using `set_net_fuel_load()`.
            - Defines standard physical parameters such as mineral content (`s_T = 0.0555`).
        """
        
        self.name = name
        self.model_num = model_num
        self.burnable = burnable
        self.dynamic = dynamic
        self.rel_indices = []

        if self.burnable:
            self.s_T = 0.055
            self.s_e = 0.010
            self.rho_p = 32

            self.f_i = f_i
            self.f_ij = f_ij
            self.g_ij = g_ij

            self.f_dead_arr = self.f_ij[0, 0:4]
            self.f_live_arr = self.f_ij[1, 4:]

            self.g_dead_arr = self.g_ij[0, 0:4]
            self.g_live_arr = self.g_ij[1, 4:]

            self.w_0 = TPA_to_Lbsft2(w_0) # convert to lbs/ft^2 
            w_n = self.w_0 * (1 - self.s_T)
            self.set_fuel_loading(w_n)

            self.w_n_dead_nominal = self.w_n_dead

            self.s = s

            self.sav_ratio = s_total

            self.W = self.calc_W()

            self.fuel_depth_ft = fuel_depth
            self.dead_mx = dead_mx

            self.heat_content = 8000 # btu/lb

            self.rel_packing_ratio = rel_packing_ratio
            self.rho_b = rho_b

            for i in range(6):
                if self.w_0[i] > 0:
                    self.rel_indices.append(i)

            self.rel_indices = np.array(self.rel_indices)
            self.num_classes = len(self.rel_indices)

    def set_fuel_loading(self, w_n):
        self.w_n = w_n
        self.w_n_dead = np.dot(self.g_dead_arr, self.w_n[0:4])
        self.w_n_live = np.dot(self.g_live_arr, self.w_n[4:])

    def calc_W(self):

        w = self.w_0
        s = self.s

        num = 0

        for i in range(3):
            num += w[i] * np.exp(-138/s[i])

        den = 0
        for i in range(4, 6):
            if s[i] != 0:
                den += w[i] * np.exp(-500/s[i])

        if den == 0:
            W = np.inf # Live moisture does not apply here
        
        else:
            W = num/den

        return W

class Anderson13(Fuel):
    _fuel_models = None # class-level cache

    @classmethod
    def load_fuel_models(cls):
        if cls._fuel_models is None:
            json_path = os.path.join(os.path.dirname(__file__), "Anderson13.json")
            with open(json_path, "r") as f:
                cls._fuel_models = json.load(f)

    def __init__(self, model_number: int, live_h_mf: float = 0):
        self.load_fuel_models()

        model_number = int(model_number)

        model_id = str(model_number)
        if model_id not in self._fuel_models["names"]:
            raise ValueError(f"{model_number} is not a valid Anderson 13 model number")
        
        burnable = model_number <= 13
        name = self._fuel_models["names"][model_id]
        dynamic = False

        if not burnable:
            f_i = None
            f_ij = None
            g_ij = None
            w_0 = None
            s = None
            s_total = None
            mx_dead = None
            fuel_bed_depth = None
            rho_b = None
            rel_packing_ratio = None

        else:
            f_i = np.array(self._fuel_models["f_i"][model_id])
            f_ij = np.array(self._fuel_models["f_ij"][model_id])
            g_ij = f_ij
            w_0 = np.array(self._fuel_models["w_0"][model_id])
            s = np.array(self._fuel_models["s"][model_id])
            s_total = self._fuel_models["s_total"][model_id]
            mx_dead = self._fuel_models["mx_dead"][model_id]
            fuel_bed_depth = self._fuel_models["fuel_bed_depth"][model_id]
            rho_b = self._fuel_models["rho_b"][model_id]
            rel_packing_ratio = self._fuel_models["rel_packing_ratio"][model_id]

        super().__init__(name, model_number, burnable, dynamic, f_i, f_ij, g_ij, w_0, s, s_total, mx_dead, fuel_bed_depth, rho_b, rel_packing_ratio)

class ScottBurgan40(Fuel):
    _fuel_models = None # class-level cache

    @classmethod
    def load_fuel_models(cls):
        if cls._fuel_models is None:
            json_path = os.path.join(os.path.dirname(__file__), "ScottBurgan40.json")
            with open(json_path, "r") as f:
                cls._fuel_models = json.load(f)

    def __init__(self, model_number: int, live_h_mf: float = 0):
        self.load_fuel_models()

        model_number = int(model_number)

        model_id = str(model_number)
        if model_id not in self._fuel_models["names"]:
            raise ValueError(f"{model_number} is not a valid ScottBurgan 40 model number")
        
        burnable = model_number >= 101
        name = self._fuel_models["names"][model_id]

        if not burnable: 
            f_i = None
            f_ij = None
            g_ij = None
            w_0 = None
            s = None
            s_total = None
            mx_dead = None
            fuel_bed_depth = None
            rho_b = None
            rel_packing_ratio = None
            dynamic = False

        else:
            dynamic = self._fuel_models["dynamic"][model_id]
            if not dynamic:
                f_ij = np.array(self._fuel_models["f_ij"][model_id])
                g_ij = f_ij
                f_i = np.array(self._fuel_models["f_i"][model_id])
                w_0 = np.array(self._fuel_models["w_0"][model_id])
            else:
                T = self.calc_curing_level(live_h_mf)
                f_ij_by_curing = self._fuel_models["f_ij"][model_id]
                f_ij = self.get_dynamic_weights(f_ij_by_curing, T)
                f_i_by_curing = self._fuel_models["f_i"][model_id]
                f_i = self.get_dynamic_weights(f_i_by_curing, T)
                g_ij_by_curing = self._fuel_models["g_ij"][model_id]
                g_ij = self.get_dynamic_weights(g_ij_by_curing, T)
                w_0 = np.array(self._fuel_models["w_0"][model_id])

                dead_herb_new = T * w_0[4]
                live_h_new = w_0[4] - dead_herb_new

                w_0[3] = dead_herb_new
                w_0[4] = live_h_new 

            s = np.array(self._fuel_models["s"][model_id])
            s_total = self._fuel_models["s_total"][model_id]
            mx_dead = self._fuel_models["mx_dead"][model_id]
            fuel_bed_depth = self._fuel_models["fuel_bed_depth"][model_id]
            rho_b = self._fuel_models["rho_b"][model_id]
            rel_packing_ratio = self._fuel_models["rel_packing_ratio"][model_id]

        super().__init__(name, model_number, burnable, dynamic, f_i, f_ij, g_ij, w_0, s, s_total, mx_dead, fuel_bed_depth, rho_b, rel_packing_ratio)

    def calc_curing_level(self, live_h_mf: float):
        T = -1.11 * live_h_mf + 1.33
        T = min(max(T, 0), 1)
        return T
    
    def get_dynamic_weights(self, weights_by_curing: dict, T: float) -> np.ndarray:
        curing = T * 100

        levels = sorted(map(int, weights_by_curing.keys()))

        if int(curing) in levels:
            key = str(int(curing))
            return np.array(weights_by_curing[key])

        lower = max(l for l in levels if l <= curing)
        upper = min(l for l in levels if l >= curing)

        f_lower = np.array(weights_by_curing[str(lower)])
        f_upper = np.array(weights_by_curing[str(upper)])
        alpha = (curing - lower) / (upper - lower)

        return (1 - alpha) * f_lower + alpha * f_upper

class FuelConstants:
    """_summary_
    """
    burnout_thresh = 0.01

    # Dictionary of fuel number to name
    fuel_names = {1: "Short grass", 2: "Timber grass", 3: "Tall grass", 4: "Chaparral",
                5: "Brush", 6: "Hardwood slash", 7: "Southern rough", 8: "Closed timber litter",
                9: "Hardwood litter", 10: "Timber litter", 11: "Light logging slash",
                12: "Medium logging slash", 13: "Heavy logging slash", 91: 'Urban', 92: 'Snow/ice',
                93: 'Agriculture', 98: 'Water', 99: 'Barren',  101: "GR1", 102: "GR2", 103: "GR3",
                104: "GR4", 105: "GR5", 106: "GR6", 107: "GR7", 108: "GR8", 109: "GR9", 121: "GS1",
                122: "GS2", 123: "GS3", 124: "GS4", 141: "SH1", 142: "SH2", 143: "SH3", 144: "SH4",
                145: "SH5", 146: "SH6", 147: "SH7", 148: "SH8", 149: "SH9", 161: "TU1", 162: "TU2",
                163: "TU3", 164: "TU4", 165: "TU5", 181: "TL1", 182: "TL2", 183: "TL3", 184: "TL4", 
                185: "TL5", 186: "TL6", 187: "TL7", 188: "TL8", 189: "TL9", 201: "SB1", 202: "SB2",
                203: "SB3", 204: "SB4"
    }

    fuel_type_reverse_lookup = {
                "Short grass": 1, "Timber grass": 2, "Tall grass": 3, "Chaparral": 4,
                "Brush": 5, "Hardwood slash": 6, "Southern rough": 7, "Closed timber litter": 8,
                "Hardwood litter": 9, "Timber litter": 10, "Light logging slash": 11,
                "Medium logging slash": 12, "Heavy logging slash": 13, "Urban": 91,
                "Snow/ice": 92, "Agriculture": 93, "Water": 98, "Barren": 99,
                "GR1": 101, "GR2": 102, "GR3": 103, "GR4": 104, "GR5": 105,
                "GR6": 106, "GR7": 107, "GR8": 108, "GR9": 109,
                "GS1": 121, "GS2": 122, "GS3": 123, "GS4": 124,
                "SH1": 141, "SH2": 142, "SH3": 143, "SH4": 144, "SH5": 145,
                "SH6": 146, "SH7": 147, "SH8": 148, "SH9": 149,
                "TU1": 161, "TU2": 162, "TU3": 163, "TU4": 164, "TU5": 165,
                "TL1": 181, "TL2": 182, "TL3": 183, "TL4": 184, "TL5": 185,
                "TL6": 186, "TL7": 187, "TL8": 188, "TL9": 189,
                "SB1": 201, "SB2": 202, "SB3": 203, "SB4": 204
    }

    # Color mapping for each fuel type
    fuel_color_mapping = {
                1: 'xkcd:pale green', 2:'xkcd:lime', 3: 'xkcd:bright green',
                4: 'xkcd:teal', 5: 'xkcd:bluish green', 6: 'xkcd:greenish teal',
                7: 'xkcd:light blue green', 8: 'xkcd:pale olive' , 9: 'xkcd:olive',
                10: 'xkcd:light forest green', 11: 'xkcd:bright olive',
                12: 'xkcd:tree green', 13: 'xkcd:avocado green', 91: '#830089',
                92: '#9ea1ef' , 93: "#e872ff", 98: '#0000ff',
                99: 'xkcd:black', -100: 'xkcd:red', 
                101: "#ffebbe",             # GR1
                102: "#ffd273",             # GR2
                103: "#ffeb8b",             # GR3
                104: "#ffff73",             # GR4
                105: "#f4dd28",             # GR5
                106: "#e6e640",             # GR6
                107: "#ccc573",             # GR7
                108: "#8b854d",             # GR8
                109: "xkcd:drab green",     # GR9

                121: "#ffaa00",             # GS1
                122: "#ffa67e",             # GS2
                123: "#ff6300",             # GS3
                124: "#cc6600",             # GS4

                141: "#d7c19d",             # SH1
                142: "#d7af9d",             # SH2
                143: "#cc8966",             # SH3
                144: "#895a44",             # SH4
                145: "#ccaa66",             # SH5
                146: "#ed7044",             # SH6
                147: "#cc7c39",             # SH7
                148: "#a83700",             # SH8
                149: "#731a00",             # SH9

                161: "#e8ffbe",             # TU1
                162: "#aaff00",             # TU2
                163: "#b3d79d",             # TU3
                164: "#70a800",             # TU4
                165: "#257300",             # TU5

                181: "#beffe8",             # TL1
                182: "#00ffc5",             # TL2
                183: "#bed2ff",             # TL3
                184: "#7a68ed",             # TL4
                185: "#bee8ff",             # TL5
                186: "#00c5ff",             # TL6
                187: "#0084a8",             # TL7
                188: "#005be6",             # TL8
                189: "#4d6e90",             # TL9

                201: "#e8beff",             # SB1
                202: "#c500ff",             # SB2
                203: "#ffbee8",             # SB3
                204: "#ff7e7e",             # SB4
    }
