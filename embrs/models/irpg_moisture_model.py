"""IRPG fine dead fuel moisture estimation model.

Implement the Incident Response Pocket Guide (IRPG) method for estimating
fine dead fuel moisture (FDFM) from temperature, relative humidity, and
site condition correction factors (shading, aspect, slope, elevation,
time of day, and month).

Classes:
    - FuelMoisturePriors: Prior probability distributions for site conditions.
    - IRPGMoistureModel: FDFM estimation combining RFM lookup and stochastic
      correction factors.

References:
    National Wildfire Coordinating Group. (2014). Incident Response Pocket
    Guide (IRPG). PMS 461, NFES 1077.
"""
import bisect
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class FuelMoisturePriors:
    """Probability distributions for stochastic sampling of site conditions.

    Attributes:
        p_shaded: Probability that the site is shaded (0 to 1)
        aspect_probs: Probabilities for aspects [N, E, S, W], must sum to 1
        slope_probs: Probabilities for slope bins [0-30%, 31%+], must sum to 1
        elev_probs: Probabilities for elevation relation [Below, Level, Above], must sum to 1
    """
    p_shaded: float = 0.5
    aspect_probs: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])  # [N, E, S, W]
    slope_probs: List[float] = field(default_factory=lambda: [0.5, 0.5])  # [0-30, 31+]
    elev_probs: List[float] = field(default_factory=lambda: [1/3, 1/3, 1/3])  # [B, L, A]


# =============================================================================
# RFM Table Constants (Table A)
# =============================================================================

# Bin edges for RH (21 bins: 0-4, 5-9, 10-14, ..., 95-99, 100)
# Using bisect_right, value x maps to bin i where EDGES[i] <= x < EDGES[i+1]
RH_EDGES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 101]

# Bin edges for Temperature in °F (6 bins: 10-29, 30-49, 50-69, 70-89, 90-109, 109+)
T_EDGES = [10, 30, 50, 70, 90, 110, 10**9]

# RFM_TABLE[rh_idx][t_idx] - 21 RH bins (rows) × 6 temperature bins (columns)
# Returns Reference Fuel Moisture percentage
# Transcribed from Table A, transposed so RH is row index and T is column index
RFM_TABLE = [
    [1, 1, 1, 1, 1, 1],       # RH 0-4
    [2, 2, 2, 1, 1, 1],       # RH 5-9
    [2, 2, 2, 2, 2, 2],       # RH 10-14
    [3, 3, 3, 2, 2, 2],       # RH 15-19
    [4, 4, 4, 3, 3, 3],       # RH 20-24
    [5, 5, 5, 4, 4, 4],       # RH 25-29
    [5, 5, 5, 5, 4, 4],       # RH 30-34
    [6, 6, 6, 5, 5, 5],       # RH 36-39
    [7, 7, 6, 6, 6, 6],       # RH 40-44
    [8, 7, 7, 7, 7, 7],       # RH 45-49
    [8, 7, 7, 7, 7, 7],       # RH 50-54
    [8, 8, 8, 8, 8, 8],       # RH 55-59
    [9, 9, 8, 8, 8, 8],       # RH 60-64
    [9, 9, 9, 8, 8, 8],       # RH 65-69
    [10, 10, 9, 9, 9, 9],     # RH 70-74
    [11, 10, 10, 10, 10, 10], # RH 75-79
    [12, 11, 11, 10, 10, 10], # RH 80-84
    [12, 12, 12, 11, 11, 11], # RH 85-89
    [13, 13, 12, 12, 12, 12], # RH 90-94
    [13, 13, 12, 12, 12, 12], # RH 95-99
    [14, 13, 13, 13, 13, 12]  # RH 100
]

# Validate RFM table dimensions at import time
assert len(RFM_TABLE) == len(RH_EDGES) - 1, "RFM_TABLE row count must match RH bins"
assert len(RFM_TABLE[0]) == len(T_EDGES) - 1, "RFM_TABLE column count must match T bins"


# =============================================================================
# Delta (Correction) Table Constants (Tables B, C, D)
# =============================================================================

ASPECTS = ("N", "E", "S", "W")
SLOPE_BINS = ("0_30", "31_plus")
ELEV_REL = ("B", "L", "A")  # Below, Level, Above (relative to wx station)
TIME_BINS = ("0800>", "1000>", "1200>", "1400>", "1600>", "1800>")

# Month to table mapping based on IRPG tables
# Table B: May, June, July (summer - less correction needed)
# Table C: Feb, Mar, Apr, Aug, Sep, Oct (transition seasons)
# Table D: Nov, Dec, Jan (winter - more correction needed)
MONTH_TO_TABLE = {
    1: "D",   # January
    2: "C",   # February
    3: "C",   # March
    4: "C",   # April
    5: "B",   # May
    6: "B",   # June
    7: "B",   # July
    8: "C",   # August
    9: "C",   # September
    10: "C",  # October
    11: "D",  # November
    12: "D"   # December
}

# Row order for exposed delta tables: (aspect, slope) combinations - 8 rows
DELTA_EXPOSED_ROW_ORDER = [
    ("N", "0_30"), ("N", "31_plus"),
    ("E", "0_30"), ("E", "31_plus"),
    ("S", "0_30"), ("S", "31_plus"),
    ("W", "0_30"), ("W", "31_plus"),
]

# Column order for delta tables: (time_bin, elevation) combinations - 18 cols
DELTA_COL_ORDER = [
    ("0800>", "B"), ("0800>", "L"), ("0800>", "A"),
    ("1000>", "B"), ("1000>", "L"), ("1000>", "A"),
    ("1200>", "B"), ("1200>", "L"), ("1200>", "A"),
    ("1400>", "B"), ("1400>", "L"), ("1400>", "A"),
    ("1600>", "B"), ("1600>", "L"), ("1600>", "A"),
    ("1800>", "B"), ("1800>", "L"), ("1800>", "A")
]

# Shaded tables have 4 rows (aspect only, no slope) × 18 cols (time×elev)
DELTA_SHADED_ROW_ORDER = ASPECTS  # N, E, S, W

# Index maps for fast lookup
DELTA_EXPOSED_ROW_MAP = {key: i for i, key in enumerate(DELTA_EXPOSED_ROW_ORDER)}
DELTA_COL_MAP = {key: j for j, key in enumerate(DELTA_COL_ORDER)}
DELTA_SHADED_ROW_MAP = {key: i for i, key in enumerate(DELTA_SHADED_ROW_ORDER)}

# Delta tables - correction factors to add to RFM
# Exposed: 8 rows (aspect×slope) × 18 cols (time×elev)
# Shaded: 4 rows (aspect) × 18 cols (time×elev)
#
# Elevation key:
#   B = Area of concern 1000'-2000' below wx site location
#   L = Area of concern within +/- 1000' of wx site location
#   A = Area of concern 1000'-2000' above wx site location

DELTA_TABLES = {
    "B": {  # May, June, July
        "exposed": [
            # Row format: [0800 B,L,A | 1000 B,L,A | 1200 B,L,A | 1400 B,L,A | 1600 B,L,A | 1800 B,L,A]
            [2, 3, 4, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 3, 4],  # N 0-30%
            [3, 4, 4, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 3, 4, 4],  # N 31%+
            [2, 2, 3, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 3, 4],  # E 0-30%
            [1, 2, 2, 0, 0, 1, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6],  # E 31%+
            [2, 3, 3, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 3, 3],  # S 0-30%
            [2, 3, 3, 1, 1, 2, 0, 1, 1, 0, 1, 1, 1, 1, 2, 2, 3, 3],  # S 31%+
            [2, 3, 4, 1, 1, 2, 0, 0, 1, 0, 0, 1, 0, 1, 1, 2, 3, 3],  # W 0-30%
            [4, 5, 6, 2, 3, 4, 1, 1, 2, 0, 0, 1, 0, 0, 1, 1, 2, 2],  # W 31%+
        ],
        "shaded": [
            # Row format: [0800 B,L,A | 1000 B,L,A | 1200 B,L,A | 1400 B,L,A | 1600 B,L,A | 1800 B,L,A]
            [4, 5, 5, 3, 4, 5, 3, 3, 4, 3, 3, 4, 3, 4, 5, 4, 5, 5],  # N 0%+
            [4, 4, 5, 3, 4, 5, 3, 3, 4, 3, 4, 4, 3, 4, 5, 4, 5, 6],  # E 0%+
            [4, 4, 5, 3, 4, 5, 3, 3, 4, 3, 3, 4, 3, 4, 5, 4, 5, 5],  # S 0%+
            [4, 5, 6, 3, 4, 5, 3, 3, 4, 3, 3, 4, 3, 4, 5, 4, 4, 5],  # W 0%+
        ],
    },
    "C": {  # February, March, April, August, September, October
        "exposed": [
            [3, 4, 5, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 5],  # N 0-30%
            [3, 4, 5, 3, 3, 4, 2, 3, 4, 2, 3, 4, 3, 3, 4, 3, 4, 5],  # N 31%+
            [3, 4, 5, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 5],  # E 0-30%
            [3, 3, 4, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 5, 4, 5, 6],  # E 31%+
            [3, 4, 5, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 5],  # S 0-30%
            [3, 4, 5, 1, 2, 2, 0, 1, 1, 0, 1, 1, 1, 1, 1, 3, 4, 5],  # S 31%+
            [3, 4, 5, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 5],  # W 0-30%
            [4, 5, 6, 3, 4, 5, 1, 2, 3, 1, 1, 1, 1, 1, 1, 3, 3, 4],  # W 31%+
        ],
        "shaded": [
            [4, 5, 6, 4, 5, 5, 3, 4, 5, 3, 4, 5, 4, 5, 5, 4, 5, 6],  # N 0%+
            [4, 5, 6, 3, 4, 5, 3, 4, 5, 3, 4, 5, 4, 5, 6, 4, 5, 6],  # E 0%+
            [4, 5, 6, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 4, 5, 6],  # S 0%+
            [4, 5, 6, 4, 5, 6, 3, 4, 5, 3, 4, 5, 3, 4, 5, 4, 5, 6],  # W 0%+
        ],
    },
    "D": {  # November, December, January
        "exposed": [
            [4, 5, 6, 3, 4, 5, 2, 3, 4, 2, 3, 4, 3, 4, 5, 4, 5, 6],  # N 0-30%
            [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6],  # N 31%+
            [4, 5, 6, 3, 4, 4, 2, 3, 3, 2, 3, 3, 3, 3, 4, 4, 5, 6],  # E 0-30%
            [4, 5, 6, 2, 3, 4, 2, 2, 3, 3, 4, 4, 4, 5, 6, 4, 5, 6],  # E 31%+
            [4, 5, 6, 3, 4, 5, 2, 3, 3, 2, 2, 3, 3, 3, 4, 4, 5, 6],  # S 0-30%
            [4, 5, 6, 2, 3, 3, 1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 5, 6],  # S 31%+
            [4, 5, 6, 3, 4, 5, 2, 3, 3, 2, 3, 3, 3, 3, 4, 4, 5, 6],  # W 0-30%
            [4, 5, 6, 4, 5, 6, 3, 4, 4, 2, 2, 3, 2, 3, 4, 4, 5, 6],  # W 31%+
        ],
        "shaded": [
            [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6],  # N 0%+
            [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6],  # E 0%+
            [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6],  # S 0%+
            [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6],  # W 0%+
        ],
    },
}

# Validate delta table dimensions at import time
for table_id in DELTA_TABLES:
    exposed = DELTA_TABLES[table_id]["exposed"]
    shaded = DELTA_TABLES[table_id]["shaded"]
    assert len(exposed) == 8, f"Table {table_id} exposed should have 8 rows"
    assert all(len(row) == 18 for row in exposed), f"Table {table_id} exposed rows should have 18 cols"
    assert len(shaded) == 4, f"Table {table_id} shaded should have 4 rows"
    assert all(len(row) == 18 for row in shaded), f"Table {table_id} shaded rows should have 18 cols"


class IRPGMoistureModel:
    """IRPG-based fine dead fuel moisture estimation model.

    Combines Reference Fuel Moisture (RFM) lookup from temperature and relative
    humidity with stochastic correction factors (Δ) based on site conditions
    (shading, aspect, slope, elevation, time of day, and month).

    Example:
        >>> priors = FuelMoisturePriors(p_shaded=0.3)
        >>> model = IRPGMoistureModel(priors)
        >>> rng = np.random.default_rng(42)
        >>> fdfm = model.sample_fdfm(T=75.0, RH=35.0, month=7, local_time_hr=14, rng=rng)
    """

    def __init__(self, priors: FuelMoisturePriors):
        """Initialize the model with prior distributions.

        Args:
            priors: FuelMoisturePriors instance with probability distributions
                   for site condition sampling.

        Raises:
            ValueError: If priors have invalid lengths or don't sum to ~1.0
        """
        self._validate_priors(priors)
        self.priors = priors

    def _validate_priors(self, priors: FuelMoisturePriors) -> None:
        """Validate that priors have correct structure and sum to ~1.0."""
        tol = 1e-6

        if not 0.0 <= priors.p_shaded <= 1.0:
            raise ValueError(f"p_shaded must be in [0, 1], got {priors.p_shaded}")

        if len(priors.aspect_probs) != 4:
            raise ValueError(f"aspect_probs must have 4 elements, got {len(priors.aspect_probs)}")
        if abs(sum(priors.aspect_probs) - 1.0) > tol:
            raise ValueError(f"aspect_probs must sum to 1.0, got {sum(priors.aspect_probs)}")

        if len(priors.slope_probs) != 2:
            raise ValueError(f"slope_probs must have 2 elements, got {len(priors.slope_probs)}")
        if abs(sum(priors.slope_probs) - 1.0) > tol:
            raise ValueError(f"slope_probs must sum to 1.0, got {sum(priors.slope_probs)}")

        if len(priors.elev_probs) != 3:
            raise ValueError(f"elev_probs must have 3 elements, got {len(priors.elev_probs)}")
        if abs(sum(priors.elev_probs) - 1.0) > tol:
            raise ValueError(f"elev_probs must sum to 1.0, got {sum(priors.elev_probs)}")

    def rfm(self, T: float, RH: float) -> float:
        """Look up Reference Fuel Moisture from temperature and relative humidity.

        Args:
            T: Temperature in degrees Fahrenheit
            RH: Relative humidity as percentage (0-100)

        Returns:
            Reference Fuel Moisture as a float (typically 1-14%)
        """
        # Clamp inputs to valid ranges
        rh = min(100.0, max(0.0, RH))
        t = max(10.0, T)  # Table starts at 10°F

        # Find bin indices using bisect
        rh_idx = bisect.bisect_right(RH_EDGES, rh) - 1
        t_idx = bisect.bisect_right(T_EDGES, t) - 1

        # Clamp indices to valid range (handles edge cases)
        rh_idx = max(0, min(rh_idx, len(RFM_TABLE) - 1))
        t_idx = max(0, min(t_idx, len(RFM_TABLE[0]) - 1))

        return float(RFM_TABLE[rh_idx][t_idx])

    @staticmethod
    def _month_to_table_id(month: int) -> str:
        """Map month (1-12) to IRPG correction table ID (B, C, or D).

        Table B: May, June, July (summer)
        Table C: Feb, Mar, Apr, Aug, Sep, Oct (transition seasons)
        Table D: Nov, Dec, Jan (winter)

        Args:
            month: Month as integer 1-12

        Returns:
            Table ID string: "B", "C", or "D"
        """
        return MONTH_TO_TABLE[month]

    @staticmethod
    def _time_hr_to_time_bin(local_time_hr: int) -> str:
        """Map hour of day to IRPG time bin.

        Time bins represent "greater than or equal to" thresholds:
        - 0800>: 8:00 AM to 9:59 AM (hours 8-9)
        - 1000>: 10:00 AM to 11:59 AM (hours 10-11)
        - 1200>: 12:00 PM to 1:59 PM (hours 12-13)
        - 1400>: 2:00 PM to 3:59 PM (hours 14-15)
        - 1600>: 4:00 PM to 5:59 PM (hours 16-17)
        - 1800>: 6:00 PM onwards or before 8:00 AM (hours 18+ or <8)

        Args:
            local_time_hr: Hour of day in 24-hour format (0-23)

        Returns:
            Time bin string
        """
        if local_time_hr < 8:
            return "1800>"  # Before 8 AM uses evening values
        if local_time_hr < 10:
            return "0800>"
        if local_time_hr < 12:
            return "1000>"
        if local_time_hr < 14:
            return "1200>"
        if local_time_hr < 16:
            return "1400>"
        if local_time_hr < 18:
            return "1600>"
        return "1800>"

    def _delta_lookup_exposed(self, table_id: str, aspect: str, slope_bin: str,
                              time_bin: str, elev_rel: str) -> int:
        """Look up delta correction value for exposed site.

        Args:
            table_id: "B", "C", or "D"
            aspect: "N", "E", "S", or "W"
            slope_bin: "0_30" or "31_plus"
            time_bin: One of TIME_BINS
            elev_rel: "B", "L", or "A"

        Returns:
            Integer correction value from table
        """
        row_idx = DELTA_EXPOSED_ROW_MAP[(aspect, slope_bin)]
        col_idx = DELTA_COL_MAP[(time_bin, elev_rel)]
        return DELTA_TABLES[table_id]["exposed"][row_idx][col_idx]

    def _delta_lookup_shaded(self, table_id: str, aspect: str,
                             time_bin: str, elev_rel: str) -> int:
        """Look up delta correction value for shaded site.

        Shaded sites don't use slope differentiation.

        Args:
            table_id: "B", "C", or "D"
            aspect: "N", "E", "S", or "W"
            time_bin: One of TIME_BINS
            elev_rel: "B", "L", or "A"

        Returns:
            Integer correction value from table
        """
        row_idx = DELTA_SHADED_ROW_MAP[aspect]
        col_idx = DELTA_COL_MAP[(time_bin, elev_rel)]
        return DELTA_TABLES[table_id]["shaded"][row_idx][col_idx]

    def sample_delta(self, month: int, local_time_hr: int,
                     rng: np.random.Generator) -> float:
        """Sample a correction factor (Δ) based on site conditions.

        Stochastically samples shading, aspect, slope, and elevation
        according to the configured priors, then looks up the appropriate
        correction value from IRPG tables.

        Args:
            month: Month as integer 1-12
            local_time_hr: Hour of day in 24-hour format (0-23)
            rng: NumPy random Generator for reproducible sampling

        Returns:
            Correction factor as a float (typically 0-6)
        """
        table_id = self._month_to_table_id(month)
        time_bin = self._time_hr_to_time_bin(local_time_hr)

        # Sample site conditions
        is_shaded = rng.random() < self.priors.p_shaded
        aspect_idx = rng.choice(4, p=self.priors.aspect_probs)
        aspect = ASPECTS[aspect_idx]
        elev_idx = rng.choice(3, p=self.priors.elev_probs)
        elev_rel = ELEV_REL[elev_idx]

        if is_shaded:
            delta = self._delta_lookup_shaded(table_id, aspect, time_bin, elev_rel)
        else:
            slope_idx = rng.choice(2, p=self.priors.slope_probs)
            slope_bin = SLOPE_BINS[slope_idx]
            delta = self._delta_lookup_exposed(table_id, aspect, slope_bin, time_bin, elev_rel)

        return float(delta)

    def sample_fdfm(self, T: float, RH: float, month: int, local_time_hr: int,
                    rng: np.random.Generator) -> float:
        """Sample Fine Dead Fuel Moisture (FDFM) percentage.

        Combines Reference Fuel Moisture lookup with stochastically sampled
        correction factors based on site conditions.

        Args:
            T: Temperature in degrees Fahrenheit
            RH: Relative humidity as percentage (0-100)
            month: Month as integer 1-12
            local_time_hr: Hour of day in 24-hour format (0-23)
            rng: NumPy random Generator for reproducible sampling

        Returns:
            Fine dead fuel moisture as percentage (float)
        """
        rfm = self.rfm(T, RH)
        delta = self.sample_delta(month, local_time_hr, rng)
        return rfm + delta
