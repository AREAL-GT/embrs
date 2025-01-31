class Fuel:
    def __init__(self, name: str, model_num: int, fuel_load_params: dict, sav_ratio: int, fuel_depth: float,
                 m_x: float, rel_packing_ratio: float, rho_b: float, burnable: bool):
        """_summary_

        Args:
            name (str): _description_
            model_num (int): _description_
            fuel_load_params (dict): _description_
            sav_ratio (int): _description_
            fuel_depth (float): _description_
            m_x (float): _description_
            rel_packing_ratio (float): _description_
            rho_b (float): _description_
            burnable (bool): _description_
        """

        self.name = name
        self.model_num = model_num
        self.fuel_load_params = fuel_load_params
        self.sav_ratio = sav_ratio
        self.fuel_depth = fuel_depth
        self.m_x = m_x
        self.heat_content = 8000 # btu/lb

        self.rel_packing_ratio = rel_packing_ratio
        self.rho_b = rho_b

        self.s_T = 0.0555 # Total mineral content
        self.partical_density = 32 # lb/ft^3
        
        self.fuel_moisture = 0.01 # TODO: make this function of weather

        self.burnable = burnable

        self.net_fuel_load = 0
        if burnable:
            self.net_fuel_load = self.set_net_fuel_load()

    def set_net_fuel_load(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """

        fuel_classes = ["1-h", "10-h", "100-h", "Live H", "Live W"]

        denom = 0

        for fuel_class in fuel_classes:
            fuel_load, sav_ratio = self.fuel_load_params[fuel_class]
            class_value = (sav_ratio * fuel_load)/self.partical_density
            denom += class_value
        
        net_fuel_load = 0

        for fuel_class in fuel_classes:
            fuel_load, sav_ratio = self.fuel_load_params[fuel_class]

            class_term = (sav_ratio * fuel_load)/self.partical_density
            class_term /= denom
            class_term *= fuel_load * (1 - self.s_T)

            net_fuel_load += class_term

        net_fuel_load *= 0.0459137 # convert to lbs/ft^2

        return net_fuel_load

    def set_fuel_moisture(self, moisture):
        # TODO: this can be set as a function of relative humidity and temperature
        return

    def __str__(self):
        return (f"Fuel Model: {self.name}\n"
                f"Fuel Load: {self.net_fuel_load}\n"
                f"SAV Ratio: {self.sav_ratio}\n"
                f"Fuel Depth: {self.fuel_depth}\n"
                f"Dead Fuel Extinction Moisture: {self.m_x}\n"
                f"Heat Content: {self.heat_content}")


class Anderson13(Fuel):
    def __init__(self, model_number: int):
        """_summary_

        Args:
            model_number (int): _description_

        Raises:
            ValueError: _description_
        """
        
        # TODO: convert fuel load to lb/ft^2 (multiply by 0.0459137)
        fuel_models = {
            1: {"name": "Short Grass",          "fuel_load_params": {"1-h": (0.74, 3500), "10-h": (0.00, 109), "100-h": (0.00, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 3500, "fuel_depth": 1.0,  "m_x": 0.12,  "rho_b": 0.03,  "rel_packing_ratio": 0.25},
            2: {"name": "Timber Grass",         "fuel_load_params": {"1-h": (2.00, 3000), "10-h": (1.00, 109), "100-h": (0.50, 30), "Live H": (0.50, 1500), "Live W": (0.00, 0.00)}, "sav_ratio": 2784, "fuel_depth": 1.0,  "m_x": 0.15,  "rho_b": 0.18,  "rel_packing_ratio": 1.14},
            3: {"name": "Tall Grass",           "fuel_load_params": {"1-h": (3.00, 1500), "10-h": (0.00, 109), "100-h": (0.00, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 1500, "fuel_depth": 2.5,  "m_x": 0.25,  "rho_b": 0.06,  "rel_packing_ratio": 0.21},
            4: {"name": "Chaparral",            "fuel_load_params": {"1-h": (5.00, 2000), "10-h": (4.00, 109), "100-h": (2.00, 30), "Live H": (0.00, 0.00), "Live W": (5.00, 1500)}, "sav_ratio": 1739, "fuel_depth": 6.0,  "m_x": 0.20,  "rho_b": 0.12,  "rel_packing_ratio": 0.52},
            5: {"name": "Brush",                "fuel_load_params": {"1-h": (1.00, 2000), "10-h": (0.50, 109), "100-h": (0.00, 30), "Live H": (0.00, 0.00), "Live W": (2.00, 1500)}, "sav_ratio": 1683, "fuel_depth": 2.0,  "m_x": 0.20,  "rho_b": 0.08,  "rel_packing_ratio": 0.33},
            6: {"name": "Dormant Brush",        "fuel_load_params": {"1-h": (1.50, 1750), "10-h": (2.50, 109), "100-h": (2.00, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 1564, "fuel_depth": 2.5,  "m_x": 0.25,  "rho_b": 0.11,  "rel_packing_ratio": 0.43},
            7: {"name": "Southern Rough",       "fuel_load_params": {"1-h": (1.10, 1750), "10-h": (1.90, 109), "100-h": (1.50, 30), "Live H": (0.00, 0.00), "Live W": (0.37, 1500)}, "sav_ratio": 1552, "fuel_depth": 2.5,  "m_x": 0.40,  "rho_b": 0.09,  "rel_packing_ratio": 0.34},
            8: {"name": "Short Needle Litter",  "fuel_load_params": {"1-h": (1.50, 2000), "10-h": (1.00, 109), "100-h": (2.50, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 1889, "fuel_depth": 0.2,  "m_x": 0.30,  "rho_b": 1.15,  "rel_packing_ratio": 5.17},
            9: {"name": "Hardwood Litter",      "fuel_load_params": {"1-h": (2.90, 1500), "10-h": (0.41, 109), "100-h": (0.15, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 2484, "fuel_depth": 0.2,  "m_x": 0.25,  "rho_b": 0.80,  "rel_packing_ratio": 4.50},
            10:{"name": "Timber Litter",        "fuel_load_params": {"1-h": (3.00, 2000), "10-h": (2.00, 109), "100-h": (5.00, 30), "Live H": (0.00, 0.00), "Live W": (2.00, 1500)}, "sav_ratio": 1764, "fuel_depth": 1.0,  "m_x": 0.25,  "rho_b": 0.55,  "rel_packing_ratio": 2.35},
            11:{"name": "Light Logging Slash",  "fuel_load_params": {"1-h": (1.50, 1500), "10-h": (4.50, 109), "100-h": (5.50, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 1182, "fuel_depth": 1.0,  "m_x": 0.15,  "rho_b": 0.53,  "rel_packing_ratio": 1.62},
            12:{"name": "Medium Logging Slash", "fuel_load_params": {"1-h": (4.00, 1500), "10-h": (14.0, 109), "100-h": (16.5, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 1145, "fuel_depth": 2.3,  "m_x": 0.20,  "rho_b": 0.69,  "rel_packing_ratio": 2.06},
            13:{"name": "Heavy Logging Slash",  "fuel_load_params": {"1-h": (7.00, 1500), "10-h": (23.0, 109), "100-h": (28.0, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 1159, "fuel_depth": 3.0,  "m_x": 0.25,  "rho_b": 0.89,  "rel_packing_ratio": 2.68},
            91:{"name": "Urban",                "fuel_load_params": {"1-h": (0.00, 9999), "10-h": (0.00, 109), "100-h": (0.00, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 9999, "fuel_depth": 9999, "m_x": 9999, "rho_b": 9999, "rel_packing_ratio": 9999},
            92:{"name": "Snow/Ice",             "fuel_load_params": {"1-h": (0.00, 9999), "10-h": (0.00, 109), "100-h": (0.00, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 9999, "fuel_depth": 9999, "m_x": 9999, "rho_b": 9999, "rel_packing_ratio": 9999},
            93:{"name": "Agriculture",          "fuel_load_params": {"1-h": (0.00, 9999), "10-h": (0.00, 109), "100-h": (0.00, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 9999, "fuel_depth": 9999, "m_x": 9999, "rho_b": 9999, "rel_packing_ratio": 9999},
            98:{"name": "Water",                "fuel_load_params": {"1-h": (0.00, 9999), "10-h": (0.00, 109), "100-h": (0.00, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 9999, "fuel_depth": 9999, "m_x": 9999, "rho_b": 9999, "rel_packing_ratio": 9999},
            99:{"name": "Barren",               "fuel_load_params": {"1-h": (0.00, 9999), "10-h": (0.00, 109), "100-h": (0.00, 30), "Live H": (0.00, 0.00), "Live W": (0.00, 0.00)}, "sav_ratio": 9999, "fuel_depth": 9999, "m_x": 9999, "rho_b": 9999, "rel_packing_ratio": 9999},
        }

        if model_number not in fuel_models:
            raise ValueError(f"{model_number} is not a valid Anderson 13 model number.")

        # Get model parameters based on number input
        model = fuel_models[model_number]

        # Set burnable variable
        burnable = model_number <= 13

        super().__init__(model["name"], model_number, model["fuel_load_params"], model["sav_ratio"], model["fuel_depth"], model["m_x"], model["rel_packing_ratio"], model["rho_b"], burnable)


# class ScottBurgan40(Fuel):
#     def __init__(self, model_number):
        
#         fuel_models = {
#             101: {"name": "GR1", "fuel_load": [0.10, 0.00, 0.00, 0.30, 0.00], "sav_ratio": [2200, 2000, 9999], "fuel_depth": 0.4, "m_x": 15, "heat_content": 8000},
#             102: {"name": "GR2", "fuel_load": [0.10, 0.00, 0.00, 1.00, 0.00], "sav_ratio": [2000, 1800, 9999], "fuel_depth": 1.0, "m_x": 15, "heat_content": 8000},
#             103: {"name": "GR3", "fuel_load": [0.10, 0.40, 0.00, 1.50, 0.00], "sav_ratio": [1500, 1300, 9999], "fuel_depth": 2.0, "m_x": 30, "heat_content": 8000},
#             104: {"name": "GR4", "fuel_load": [0.25, 0.00, 0.00, 1.90, 0.00], "sav_ratio": [2000, 1800, 9999], "fuel_depth": 2.0, "m_x": 15, "heat_content": 8000},
#             105: {"name": "GR5", "fuel_load": [0.40, 0.00, 0.00, 2.50, 0.00], "sav_ratio": [1800, 1600, 9999], "fuel_depth": 1.5, "m_x": 40, "heat_content": 8000},
#             106: {"name": "GR6", "fuel_load": [0.10, 0.00, 0.00, 3.40, 0.00], "sav_ratio": [2200, 2000, 9999], "fuel_depth": 1.5, "m_x": 40, "heat_content": 9000},
#             107: {"name": "GR7", "fuel_load": [1.00, 0.00, 0.00, 5.40, 0.00], "sav_ratio": [2000, 1800, 9999], "fuel_depth": 3.0, "m_x": 15, "heat_content": 8000},
#             108: {"name": "GR8", "fuel_load": [0.50, 1.00, 0.00, 7.30, 0.00], "sav_ratio": [1500, 1300, 9999], "fuel_depth": 4.0, "m_x": 30, "heat_content": 8000},
#             109: {"name": "GR9", "fuel_load": [1.00, 1.00, 0.00, 9.00, 0.00], "sav_ratio": [1800, 1600, 9999], "fuel_depth": 1.5, "m_x": 40, "heat_content": 8000},
#             121: {"name": "GS1", "fuel_load": [0.20, 0.00, 0.00, 0.50, 0.65], "sav_ratio": [2000, 1800, 1800], "fuel_depth": 0.9, "m_x": 15, "heat_content": 8000},
#             122: {"name": "GS2", "fuel_load": [0.50, 0.50, 0.00, 0.60, 1.00], "sav_ratio": [2000, 1800, 1800], "fuel_depth": 1.5, "m_x": 15, "heat_content": 8000},
#             123: {"name": "GS3", "fuel_load": [0.30, 0.25, 0.00, 1.45, 1.25], "sav_ratio": [1800, 1600, 1600], "fuel_depth": 1.8, "m_x": 40, "heat_content": 8000},
#             124: {"name": "GS4", "fuel_load": [1.90, 0.30, 0.10, 3.40, 7.10], "sav_ratio": [1800, 1600, 1600], "fuel_depth": 2.1, "m_x": 40, "heat_content": 8000},
#             141: {"name": "SH1", "fuel_load": [0.25, 0.25, 0.00, 0.15, 1.30], "sav_ratio": [2000, 1800, 1600], "fuel_depth": 1.0, "m_x": 15, "heat_content": 8000},
#             142:
#             143:
#             144:
#             145:
#             146:
#             147:
#             148:
#             149:
#             161:
#             162:
#             163:
#             164:
#             165:
#             181:
#             182:
#             183:
#             184:
#             185:
#             186:
#             187:
#             188:
#             189:
#             201:
#             202:
#             203:
#             204





#         }


