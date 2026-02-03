"""Abstract base classes for the EMBRS fire simulation framework.

This package provides base classes that define interfaces for extensibility:
control algorithms, agents, fire simulation, and visualization.

Classes:
    - ControlClass: Abstract base for user-defined fire suppression strategies.
    - AgentBase: Base class for agents displayed in simulation.
    - BaseFireSim: Core fire simulation logic shared by FireSim and FirePredictor.
    - BaseVisualizer: Base visualization functionality for simulation display.
    - GridManager: Manages the hexagonal cell grid for fire simulation.
    - WeatherManager: Manages weather data and forecasts for fire simulation.
    - ControlActionHandler: Handles fire suppression control actions.

.. autoclass:: embrs.base_classes.control_base.ControlClass
    :members:

.. autoclass:: embrs.base_classes.agent_base.AgentBase
    :members:

.. autoclass:: embrs.base_classes.base_fire.BaseFireSim
    :members:

.. autoclass:: embrs.base_classes.base_visualizer.BaseVisualizer
    :members:

.. autoclass:: embrs.base_classes.grid_manager.GridManager
    :members:

.. autoclass:: embrs.base_classes.weather_manager.WeatherManager
    :members:

.. autoclass:: embrs.base_classes.control_handler.ControlActionHandler
    :members:
"""

from embrs.base_classes.base_fire import BaseFireSim
from embrs.base_classes.control_base import ControlClass
from embrs.base_classes.agent_base import AgentBase
from embrs.base_classes.grid_manager import GridManager
from embrs.base_classes.weather_manager import WeatherManager
from embrs.base_classes.control_handler import ControlActionHandler

__all__ = [
    "BaseFireSim",
    "ControlClass",
    "AgentBase",
    "GridManager",
    "WeatherManager",
    "ControlActionHandler",
]
