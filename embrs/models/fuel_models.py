"""Fuel model definitions for fire spread simulation.

Define fuel model classes used in Rothermel fire behavior calculations. Each
fuel model encapsulates physical properties (loading, surface-area-to-volume
ratios, moisture of extinction, fuel depth) and precomputes derived constants
needed by the Rothermel equations.

Supports Anderson 13 and Scott-Burgan 40 fuel model classification systems.

Classes:
    - Fuel: Base class representing a generic fuel model with physical properties.
    - Anderson13: Anderson 13 standard fire behavior fuel models.
    - ScottBurgan40: Scott and Burgan 40 fuel models with dynamic herbaceous transfer.
    - FuelConstants: Lookup tables mapping fuel model numbers to names and colors.

References:
    Anderson, H. E. (1982). Aids to Determining Fuel Models for Estimating
    Fire Behavior. USDA Forest Service General Technical Report INT-122.

    Scott, J. H. & Burgan, R. E. (2005). Standard Fire Behavior Fuel Models.
    USDA Forest Service General Technical Report RMRS-GTR-153.
"""
import numpy as np
import os
import json
from embrs.utilities.unit_conversions import *

class Fuel:
    """Base fuel model for Rothermel fire spread calculations.

    Encapsulate the physical properties of a fuel type and precompute
    derived constants used in the Rothermel (1972) equations. Non-burnable
    fuel types (e.g., water, urban) store only name and model number.

    All internal units follow the Rothermel convention:
    loading in lb/ft², surface-area-to-volume ratio in 1/ft, fuel depth in
    ft, heat content in BTU/lb.

    Attributes:
        name (str): Human-readable fuel model name.
        model_num (int): Numeric fuel model identifier.
        burnable (bool): Whether this fuel can sustain fire.
        dynamic (bool): Whether herbaceous fuel transfer is applied.
        load (np.ndarray): Fuel loading per class (tons/acre), shape (6,).
            Order: [1h, 10h, 100h, dead herb, live herb, live woody].
        s (np.ndarray): Surface-area-to-volume ratio per class (1/ft),
            shape (6,).
        sav_ratio (int): Characteristic SAV ratio (1/ft).
        dead_mx (float): Dead fuel moisture of extinction (fraction).
        fuel_depth_ft (float): Fuel bed depth (feet).
        heat_content (float): Heat content (BTU/lb), default 8000.
        rho_p (float): Particle density (lb/ft³), default 32.
    """

    def __init__(self, name: str, model_num: int, burnable: bool, dynamic: bool, w_0: np.ndarray,
                 s: np.ndarray, s_total: int, dead_mx: float, fuel_depth: float):
        """Initialize a fuel model.

        Args:
            name (str): Human-readable fuel model name.
            model_num (int): Numeric identifier for the fuel model.
            burnable (bool): Whether this fuel can sustain fire.
            dynamic (bool): Whether herbaceous transfer applies.
            w_0 (np.ndarray): Fuel loading per class (tons/acre), shape (6,).
                None for non-burnable models.
            s (np.ndarray): SAV ratio per class (1/ft), shape (6,).
                None for non-burnable models.
            s_total (int): Characteristic SAV ratio (1/ft).
            dead_mx (float): Dead fuel moisture of extinction (fraction).
            fuel_depth (float): Fuel bed depth (feet).
        """
        self.name = name
        self.model_num = model_num
        self.burnable = burnable
        self.dynamic = dynamic
        self.rel_indices = []

        if self.burnable:
            # Standard constants
            self.s_T = 0.055
            self.s_e = 0.010
            self.rho_p = 32
            self.heat_content = 8000 # btu/lb (used for live and dead)

            # Load data defining the fuel type
            self.load = w_0
            self.s = s
            self.sav_ratio = s_total
            self.fuel_depth_ft = fuel_depth
            self.dead_mx = dead_mx
            
            # Compute weighting factors
            self.compute_f_and_g_weights()

            # Compute f live and f dead
            self.f_dead_arr = self.f_ij[0, 0:4]
            self.f_live_arr = self.f_ij[1, 4:]

            # Compute g live and g dead
            self.g_dead_arr = self.g_ij[0, 0:4]
            self.g_live_arr = self.g_ij[1, 4:]

            # Compute the net fuel loading
            self.w_0 = TPA_to_Lbsft2(self.load)
            w_n = self.w_0 * (1 - self.s_T)
            self.set_fuel_loading(w_n)

            # Store nominal net dead loading
            self.w_n_dead_nominal = self.w_n_dead

            # Compute helpful constants for rothermel equations
            self.beta = np.sum(self.w_0) / 32 / self.fuel_depth_ft
            self.beta_op = 3.348 / (self.sav_ratio ** 0.8189)
            self.rat = self.beta / self.beta_op
            self.A = 133 * self.sav_ratio ** (-0.7913)
            self.gammax = (self.sav_ratio ** 1.5) / (495 + 0.0594 * self.sav_ratio ** 1.5)
            self.gamma = self.gammax * (self.rat ** self.A) * np.exp(self.A*(1-self.rat))
            self.rho_b = np.sum(self.w_0) / self.fuel_depth_ft
            self.flux_ratio = self.calc_flux_ratio()
            self.E, self.B, self.C = self.calc_E_B_C()
            self.W = self.calc_W(w_0)

            # Mark indices that are relevant for this fuel model
            for i in range(6):
                if self.w_0[i] > 0:
                    self.rel_indices.append(i)

            self.rel_indices = np.array(self.rel_indices)
            self.num_classes = len(self.rel_indices)

    def calc_flux_ratio(self) -> float:
        """Compute propagating flux ratio for the Rothermel equation.

        Returns:
            float: Propagating flux ratio (dimensionless).
        """
        packing_ratio = self.rho_b / self.rho_p
        flux_ratio = (192 + 0.2595*self.sav_ratio)**(-1) * np.exp((0.792 + 0.681*np.sqrt(self.sav_ratio))*(packing_ratio + 0.1))

        return flux_ratio

    def calc_E_B_C(self) -> tuple:
        """Compute wind factor coefficients E, B, and C.

        These coefficients parameterize the wind factor equation in the
        Rothermel model as a function of the characteristic SAV ratio.

        Returns:
            Tuple[float, float, float]: ``(E, B, C)`` wind factor
                coefficients (dimensionless).
        """

        sav_ratio = self.sav_ratio

        E = 0.715 * np.exp(-3.59e-4 * sav_ratio)
        B = 0.02526 * sav_ratio ** 0.54
        C = 7.47 * np.exp(-0.133 * sav_ratio**0.55)

        return E, B, C

    def compute_f_and_g_weights(self):
        """Compute fuel class weighting factors f_ij, g_ij, and category fractions f_i.

        Derive weighting arrays from fuel loading and SAV ratios. ``f_ij``
        gives fractional area weights within dead/live categories. ``g_ij``
        gives SAV-bin-based moisture weighting factors. ``f_i`` gives the
        dead vs. live category fractions.

        Side Effects:
            Sets ``self.f_ij`` (2×6), ``self.g_ij`` (2×6), and
            ``self.f_i`` (2,) arrays.
        """
        f_ij = np.zeros((2, 6))
        g_ij = np.zeros((2, 6))
        f_i = np.zeros(2)
        
        # ── Class grouping ─────────────────────────────────────────────
        dead_indices = [0, 1, 2, 3]  # 1h, 10h, 100h, dead herb
        live_indices = [4, 5]       # live herb, live woody
        
        # ── Compute a[i] = load[i] * SAV[i] / 32 ──────────────────────
        a_dead = np.array([
            self.load[i] * self.s[i] / 32.0 for i in dead_indices
        ])
        a_live = np.array([
            self.load[i] * self.s[i] / 32.0 for i in live_indices
        ])

        a_sum_dead = np.sum(a_dead)
        a_sum_live = np.sum(a_live)

        # ── f_ij (fractional weights) ─────────────────────────────────
        for idx in dead_indices:
            f_ij[0, idx] = a_dead[dead_indices.index(idx)] / a_sum_dead if a_sum_dead > 0 else 0.0
        for idx in live_indices:
            f_ij[1, idx] = a_live[live_indices.index(idx)] / a_sum_live if a_sum_live > 0 else 0.0

        # ── g_ij (SAV bin-based moisture weighting) ───────────────────
        def sav_bin(sav):
            if sav >= 1200.0: return 0
            elif sav >= 192.0: return 1
            elif sav >= 96.0: return 2
            elif sav >= 48.0: return 3
            elif sav >= 16.0: return 4
            else: return -1

        # Dead gx bin sums
        gx_dead = np.zeros(5)
        for i in dead_indices:
            bin_idx = sav_bin(self.s[i])
            if bin_idx >= 0:
                gx_dead[bin_idx] += f_ij[0, i]

        # Live gx bin sums
        gx_live = np.zeros(5)
        for i in live_indices:
            bin_idx = sav_bin(self.s[i])
            if bin_idx >= 0:
                gx_live[bin_idx] += f_ij[1, i]

        for i in dead_indices:
            bin_idx = sav_bin(self.s[i])
            g_ij[0, i] = gx_dead[bin_idx] if bin_idx >= 0 else 0.0

        for i in live_indices:
            bin_idx = sav_bin(self.s[i])
            g_ij[1, i] = gx_live[bin_idx] if bin_idx >= 0 else 0.0

        f_i[0] = a_sum_dead / (a_sum_dead + a_sum_live)
        f_i[1] = 1.0 - f_i[0]

        self.f_ij = f_ij
        self.g_ij = g_ij
        self.f_i = f_i

    def set_fuel_loading(self, w_n: np.ndarray):
        """Set net fuel loading and recompute weighted dead/live net loadings.

        Args:
            w_n (np.ndarray): Net fuel loading per class (lb/ft²), shape (6,).

        Side Effects:
            Updates ``self.w_n``, ``self.w_n_dead``, and ``self.w_n_live``.
        """
        self.w_n = w_n
        self.w_n_dead = np.dot(self.g_dead_arr, self.w_n[0:4])
        self.w_n_live = np.dot(self.g_live_arr, self.w_n[4:])

    def calc_W(self, w_0_tpa: np.ndarray) -> float:
        """Compute dead-to-live fuel loading ratio W.

        W is used to determine live fuel moisture of extinction. Returns
        ``np.inf`` when there is no live fuel loading (denominator is zero).

        Args:
            w_0_tpa (np.ndarray): Fuel loading per class (tons/acre),
                shape (6,).

        Returns:
            float: Dead-to-live loading ratio (dimensionless), or ``np.inf``
                if no live fuel is present.
        """
        w = w_0_tpa
        s = self.s

        num = 0

        for i in range(4):
            if s[i] != 0:
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
    """Anderson 13 standard fire behavior fuel models.

    Load fuel properties from the bundled ``Anderson13.json`` data file.
    Model numbers 1-13 are burnable; higher numbers (91, 92, 93, 98, 99)
    represent non-burnable types.

    The JSON data is cached at the class level and loaded only once.
    """

    _fuel_models = None  # class-level cache

    @classmethod
    def load_fuel_models(cls):
        """Load Anderson 13 fuel model data from the bundled JSON file.

        Data is cached at the class level after the first call.
        """
        if cls._fuel_models is None:
            json_path = os.path.join(os.path.dirname(__file__), "Anderson13.json")
            with open(json_path, "r") as f:
                cls._fuel_models = json.load(f)

    def __init__(self, model_number: int, live_h_mf: float = 0):
        """Initialize an Anderson 13 fuel model by model number.

        Args:
            model_number (int): Anderson fuel model number (1-13 for
                burnable, 91/92/93/98/99 for non-burnable).
            live_h_mf (float): Live herbaceous fuel moisture (fraction).
                Unused for Anderson 13 (not dynamic). Defaults to 0.

        Raises:
            ValueError: If ``model_number`` is not a valid Anderson 13 model.
        """
        self.load_fuel_models()

        model_number = int(model_number)

        model_id = str(model_number)
        if model_id not in self._fuel_models["names"]:
            raise ValueError(f"{model_number} is not a valid Anderson 13 model number")
        
        burnable = model_number <= 13
        name = self._fuel_models["names"][model_id]
        dynamic = False

        if not burnable:
            w_0 = None
            s = None
            s_total = None
            mx_dead = None
            fuel_bed_depth = None

        else:
            w_0 = np.array(self._fuel_models["w_0"][model_id])
            s = np.array(self._fuel_models["s"][model_id])
            s_total = self._fuel_models["s_total"][model_id]
            mx_dead = self._fuel_models["mx_dead"][model_id]
            fuel_bed_depth = self._fuel_models["fuel_bed_depth"][model_id]

        super().__init__(name, model_number, burnable, dynamic, w_0, s, s_total, mx_dead, fuel_bed_depth)

class ScottBurgan40(Fuel):
    """Scott-Burgan 40 fire behavior fuel models.

    Load fuel properties from the bundled ``ScottBurgan40.json`` data file.
    Model numbers >= 101 are burnable. Dynamic models transfer herbaceous
    fuel loading between live and dead categories based on a curing level
    computed from live herbaceous moisture content.

    The JSON data is cached at the class level and loaded only once.
    """
    _fuel_models = None  # class-level cache

    @classmethod
    def load_fuel_models(cls):
        """Load Scott-Burgan 40 fuel model data from the bundled JSON file.

        Data is cached at the class level after the first call.
        """
        if cls._fuel_models is None:
            json_path = os.path.join(os.path.dirname(__file__), "ScottBurgan40.json")
            with open(json_path, "r") as f:
                cls._fuel_models = json.load(f)

    def __init__(self, model_number: int, live_h_mf: float = 0):
        """Initialize a Scott-Burgan 40 fuel model by model number.

        For dynamic models, the herbaceous fuel loading is transferred
        between live and dead categories based on the curing level derived
        from ``live_h_mf``.

        Args:
            model_number (int): Scott-Burgan fuel model number
                (e.g., 101 for GR1, 201 for SB1).
            live_h_mf (float): Live herbaceous fuel moisture (fraction).
                Used to compute curing level for dynamic models. Defaults
                to 0.

        Raises:
            ValueError: If ``model_number`` is not a valid Scott-Burgan 40
                model.
        """
        self.load_fuel_models()

        model_number = int(model_number)

        model_id = str(model_number)
        if model_id not in self._fuel_models["names"]:
            raise ValueError(f"{model_number} is not a valid ScottBurgan 40 model number")
        
        burnable = model_number >= 101
        name = self._fuel_models["names"][model_id]

        if not burnable: 
            w_0 = None
            s = None
            s_total = None
            mx_dead = None
            fuel_bed_depth = None
            dynamic = False

        else:
            dynamic = self._fuel_models["dynamic"][model_id]
            if not dynamic:
                w_0 = np.array(self._fuel_models["w_0"][model_id])
            else:
                T = self.calc_curing_level(live_h_mf)
                w_0 = np.array(self._fuel_models["w_0"][model_id])

                dead_herb_new = T * w_0[4]
                live_h_new = w_0[4] - dead_herb_new

                w_0[3] = dead_herb_new
                w_0[4] = live_h_new 

            s = np.array(self._fuel_models["s"][model_id])
            s_total = self._fuel_models["s_total"][model_id]
            mx_dead = self._fuel_models["mx_dead"][model_id]
            fuel_bed_depth = self._fuel_models["fuel_bed_depth"][model_id]    

        super().__init__(name, model_number, burnable, dynamic, w_0, s, s_total, mx_dead, fuel_bed_depth)

    def calc_curing_level(self, live_h_mf: float) -> float:
        """Compute herbaceous curing level from live herbaceous moisture.

        The curing level T determines the fraction of live herbaceous
        loading transferred to the dead herbaceous class. Clamped to [0, 1].

        Args:
            live_h_mf (float): Live herbaceous fuel moisture (fraction).

        Returns:
            float: Curing level in [0, 1] where 1 = fully cured (all
                herbaceous loading treated as dead).
        """
        T = -1.11 * live_h_mf + 1.33
        T = min(max(T, 0), 1)
        return T

class FuelConstants:
    """Lookup tables mapping fuel model numbers to names and display colors.

    Attributes:
        fuel_names (dict): Maps fuel model number (int) to human-readable name.
        fuel_type_reverse_lookup (dict): Maps fuel model name to number.
        fuel_color_mapping (dict): Maps fuel model number to hex color string
            for visualization.
    """
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
                1: '#ffff00', 2:'#00e0e0', 3: '#ffb900',
                4: '#801010', 5: '#985a1b', 6: '#80302f',
                7: '#a26415', 8: '#6699cb' , 9: '#008484',
                10: '#82c160', 11: '#ff84ff',
                12: '#bf57bf', 13: '#7f2b7f', 91: '#cdcdcd',
                92: '#999999' , 93: "#666666", 98: '#0600ff',
                99: '#000000', -100: 'xkcd:red', 
                101: "#ffff98",             # GR1
                102: "#ffff40",             # GR2
                103: "#ffff5f",             # GR3
                104: "#ffe500",             # GR4
                105: "#ffff20",             # GR5
                106: "#ffcc01",             # GR6
                107: "#fca401",             # GR7
                108: "#fa8b00",             # GR8
                109: "#f96600",     # GR9

                121: "#99994f",             # GS1
                122: "#77771a",             # GS2
                123: "#888833",             # GS3
                124: "#666600",             # GS4

                141: "#c38402",             # SH1
                142: "#c38402",             # SH2
                143: "#b87900",             # SH3
                144: "#8d5022",             # SH4
                145: "#8d5022",             # SH5
                146: "#793c30",             # SH6
                147: "#804040",             # SH7
                148: "#802020",             # SH8
                149: "#660000",             # SH9

                161: "#d9fea0",             # TU1
                162: "#addf80",             # TU2
                163: "#2b8420",             # TU3
                164: "#56a240",             # TU4
                165: "#006600",             # TU5

                181: "#aad5ff",             # TL1
                182: "#02ffff",             # TL2
                183: "#88b7e5",             # TL3
                184: "#447bb2",             # TL4
                185: "#235d99",             # TL5
                186: "#01c1c1",             # TL6
                187: "#004080",             # TL7
                188: "#00a3a3",             # TL8
                189: "#006565",             # TL9

                201: "#df6ddf",             # SB1
                202: "#9f419f",             # SB2
                203: "#5f145f",             # SB3
                204: "#3f003f",             # SB4
    }
