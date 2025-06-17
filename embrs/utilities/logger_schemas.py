from dataclasses import dataclass, asdict
from typing import Literal, Union, Tuple
import json

@dataclass
class CellLogEntry:
    timestamp: int
    id: int
    x: float
    y: float
    fuel: int
    state: int
    crown_state: int
    w_n_dead: float
    w_n_live: float
    dfm_1hr: float
    dfm_10hr: float
    dfm_100hr: float
    ros: float
    I_ss: float
    wind_speed: float
    wind_dir: float
    retardant: bool

    def to_dict(self):
        return asdict(self)

@dataclass
class AgentLogEntry:
    timestamp: int
    id: int
    label: str
    x: float
    y: float
    marker: str
    color: str

    def to_dict(self):
        return asdict(self)

@dataclass
class ActionsEntry:
    timestamp: int
    action_type: Literal['long_term_retardant', 'short_term_suppressant', 'active_fireline']
    x_coords: list[float] = None
    y_coords: list[float] = None

    # Parameters speicific to action types
    width: float = None
    effectiveness: list[float] = None

    def to_dict(self):
        return asdict(self)
    
@dataclass
class PredictionEntry:
    timestamp: int
    prediction: dict  # dict[int, Tuple[int, int]]

    def to_dict(self):
        # Convert keys to strings and values to lists for JSON serialization
        serializable_pred = {str(k): list(v) for k, v in self.prediction.items()}
        return {
            "timestamp": self.timestamp,
            "prediction": json.dumps(serializable_pred)
        }

