"""Abstract control class for user-defined fire suppression strategies.

Users extend ControlClass to implement custom control logic that interacts
with the fire simulation at each time step.

Classes:
    - ControlClass: Abstract base for fire suppression algorithms.

.. autoclass:: ControlClass
    :members:
"""

from abc import ABC, abstractmethod
from embrs.fire_simulator.fire import FireSim

class ControlClass(ABC):
    """Abstract base class for user-defined fire suppression control code.

    Subclasses must implement the process_state method, which is called
    after each simulation iteration to apply suppression actions.
    """

    @abstractmethod
    def process_state(self, fire: FireSim) -> None:
        """Process the current simulation state and apply control actions.

        Called after each simulation iteration. Implement this method to
        access fire state and apply suppression actions such as retardant
        drops, water drops, or fireline construction.

        Args:
            fire (FireSim): The current FireSim instance. Access fire state
                via fire.burning_cells, fire.get_frontier(), fire.curr_time_s, etc.
        """
