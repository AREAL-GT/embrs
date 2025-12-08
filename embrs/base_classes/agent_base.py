"""Module containing base class for agents

.. autoclass:: AgentBase
    :members:

"""

from embrs.utilities.logger_schemas import AgentLogEntry

class AgentBase:
    """Base class for agents in user code. 
    
    Agent objects must be an instance of this class in order to be registered with the sim and
    displayed in visualizations.

    :param id: unique identifier of the agent
    :type id: any
    :param x_m: x position in meters of the agent within the sim.
    :type x_m: float
    :param y_m: y position in meters of the agent within the sim.
    :type y_m: float
    :param label: label added to the agent when it is displayed, if `None` no label will be
                  added, defaults to `None`.
    :type label: str, optional
    :param marker: marker used to represent the agent when displayed, any
                   :py:attr:`matplotlib.markers` can be used, defaults to '*'.
    :type marker: str, optional
    :param color: color used to represent the agent when displayed, any 
                  :py:attr:`matplotlib.colors` can be used, defaults to 'magenta'.
    :type color: str, optional
    """

    def __init__(self, id: any, x: float, y: float, label:str=None, marker:str='*',
                color:str='magenta'):
        """Constructor method that sets the basic parameters for an agent
        """
        self.id = id
        self.x = x
        self.y = y
        self.label = label
        self.marker = marker
        self.color = color

    def to_log_entry(self, timestamp) -> dict:
        """_summary_

        Returns:
            dict: _description_
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