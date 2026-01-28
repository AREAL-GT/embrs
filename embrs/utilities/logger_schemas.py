"""Data schemas for logged simulation entries.

This module defines dataclass schemas for various log entry types used by
the Logger to record simulation state over time.

Classes:
    - CellLogEntry: Schema for cell state log entries.
    - AgentLogEntry: Schema for agent position log entries.
    - ActionsEntry: Schema for suppression action log entries.
    - PredictionEntry: Schema for fire prediction log entries.

.. autoclass:: CellLogEntry
    :members:

.. autoclass:: AgentLogEntry
    :members:

.. autoclass:: ActionsEntry
    :members:

.. autoclass:: PredictionEntry
    :members:
"""

from dataclasses import dataclass, asdict
from typing import Literal, Union, Tuple, List, Dict
import json


@dataclass
class CellLogEntry:
    """Log entry for cell state at a simulation timestamp.

    Attributes:
        timestamp (int): Simulation time in seconds.
        id (int): Unique cell identifier.
        x (float): Cell center x position in meters.
        y (float): Cell center y position in meters.
        fuel (int): Fuel model type ID.
        state (int): Cell state (0=BURNT, 1=FUEL, 2=FIRE).
        crown_state (int): Crown fire status (0=NONE, 1=PASSIVE, 2=ACTIVE).
        w_n_dead (float): Net dead fuel loading.
        w_n_dead_start (float): Initial dead fuel loading.
        w_n_live (float): Net live fuel loading.
        dfm_1hr (float): 1-hour dead fuel moisture content (fraction).
        dfm_10hr (float): 10-hour dead fuel moisture content (fraction).
        dfm_100hr (float): 100-hour dead fuel moisture content (fraction).
        ros (float): Rate of spread in m/s.
        I_ss (float): Steady-state fireline intensity. TODO:verify units.
        wind_speed (float): Wind speed in m/s.
        wind_dir (float): Wind direction in degrees.
        retardant (bool): Whether retardant is applied to this cell.
        arrival_time (float): Time when fire arrived at this cell in seconds.
    """

    timestamp: int
    id: int
    x: float
    y: float
    fuel: int
    state: int
    crown_state: int
    w_n_dead: float
    w_n_dead_start: float
    w_n_live: float
    dfm_1hr: float
    dfm_10hr: float
    dfm_100hr: float
    ros: float
    I_ss: float
    wind_speed: float
    wind_dir: float
    retardant: bool
    arrival_time: float

    def to_dict(self) -> dict:
        """Convert entry to dictionary for serialization."""
        return asdict(self)


@dataclass
class AgentLogEntry:
    """Log entry for agent position at a simulation timestamp.

    Attributes:
        timestamp (int): Simulation time in seconds.
        id (int): Unique agent identifier.
        label (str): Agent display label.
        x (float): Agent x position in meters.
        y (float): Agent y position in meters.
        marker (str): Matplotlib marker style.
        color (str): Matplotlib color string.
    """

    timestamp: int
    id: int
    label: str
    x: float
    y: float
    marker: str
    color: str

    def to_dict(self) -> dict:
        """Convert entry to dictionary for serialization."""
        return asdict(self)


@dataclass
class ActionsEntry:
    """Log entry for a suppression action at a simulation timestamp.

    Attributes:
        timestamp (int): Simulation time in seconds.
        action_type (str): Type of action ('long_term_retardant',
            'short_term_suppressant', or 'active_fireline').
        x_coords (List[float]): X coordinates of action geometry in meters.
        y_coords (List[float]): Y coordinates of action geometry in meters.
        width (float): Width of action (for firelines) in meters.
        effectiveness (List[float]): Effectiveness values for the action.
    """

    timestamp: int
    action_type: Literal['long_term_retardant', 'short_term_suppressant', 'active_fireline']
    x_coords: List[float] = None
    y_coords: List[float] = None
    width: float = None
    effectiveness: List[float] = None

    def to_dict(self) -> dict:
        """Convert entry to dictionary for serialization."""
        return asdict(self)


@dataclass
class PredictionEntry:
    """Log entry for a fire spread prediction.

    Attributes:
        timestamp (int): Simulation time when prediction was made in seconds.
        prediction (dict): Prediction data mapping time to cell positions.
    """

    timestamp: int
    prediction: Dict[int, Tuple[int, int]]

    def to_dict(self) -> dict:
        """Convert entry to dictionary with JSON-serializable prediction."""
        serializable_pred = {str(k): list(v) for k, v in self.prediction.items()}
        return {
            "timestamp": self.timestamp,
            "prediction": json.dumps(serializable_pred)
        }