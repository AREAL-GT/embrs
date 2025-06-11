from dataclasses import dataclass, asdict

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