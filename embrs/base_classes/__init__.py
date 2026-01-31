"""Abstract base classes for the EMBRS fire simulation framework.

This package provides base classes that define interfaces for extensibility:
control algorithms, agents, fire simulation, and visualization.

Classes:
    - ControlClass: Abstract base for user-defined fire suppression strategies.
    - AgentBase: Base class for agents displayed in simulation.
    - BaseFireSim: Core fire simulation logic shared by FireSim and FirePredictor.
    - BaseVisualizer: Base visualization functionality for simulation display.

.. autoclass:: embrs.base_classes.control_base.ControlClass
    :members:

.. autoclass:: embrs.base_classes.agent_base.AgentBase
    :members:

.. autoclass:: embrs.base_classes.base_fire.BaseFireSim
    :members:

.. autoclass:: embrs.base_classes.base_visualizer.BaseVisualizer
    :members:
"""
