"""Base class for agents displayed in fire simulation.

Agents represent entities (vehicles, personnel, etc.) that can be registered
with the simulation and displayed in visualizations.

Classes:
    - AgentBase: Base class for simulation agents.

.. autoclass:: AgentBase
    :members:
"""

from embrs.utilities.logger_schemas import AgentLogEntry

class AgentBase:
    """Base class for agents in user code.

    Agent objects must be an instance of this class to be registered with the
    simulation and displayed in visualizations.

    Attributes:
        id: Unique identifier of the agent.
        x (float): X position in meters within the simulation.
        y (float): Y position in meters within the simulation.
        label (str): Label displayed with the agent, or None for no label.
        marker (str): Matplotlib marker style for display.
        color (str): Matplotlib color for display.
    """

    def __init__(self, id, x: float, y: float, label: str = None, marker: str = '*',
                 color: str = 'magenta'):
        """Initialize an agent with position and display properties.

        Args:
            id: Unique identifier of the agent.
            x (float): X position in meters within the simulation.
            y (float): Y position in meters within the simulation.
            label (str, optional): Label displayed with the agent. Defaults to None.
            marker (str, optional): Matplotlib marker style. Defaults to '*'.
            color (str, optional): Matplotlib color. Defaults to 'magenta'.
        """
        self.id = id
        self.x = x
        self.y = y
        self.label = label
        self.marker = marker
        self.color = color

    def to_log_entry(self, timestamp) -> AgentLogEntry:
        """Convert agent state to a log entry for recording.

        Args:
            timestamp: Current simulation timestamp.

        Returns:
            AgentLogEntry: Log entry containing the agent's current state.
        """
        entry = AgentLogEntry(
            timestamp=timestamp,
            id=self.id,
            label=self.label,
            x=self.x,
            y=self.y,
            marker=self.marker,
            color=self.color
        )

        return entry