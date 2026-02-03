"""EMBRS - Engineering Model for Burning and Real-time Suppression."""

from embrs.fire_simulator.fire import FireSim
from embrs.tools.fire_predictor import FirePredictor
from embrs.base_classes.control_base import ControlClass
from embrs.base_classes.agent_base import AgentBase
from embrs.exceptions import (
    EMBRSError,
    ConfigurationError,
    SimulationError,
    ValidationError,
)

__version__ = "0.2.0"

__all__ = [
    "FireSim",
    "FirePredictor",
    "ControlClass",
    "AgentBase",
    "EMBRSError",
    "ConfigurationError",
    "SimulationError",
    "ValidationError",
]
