"""Class of templated actions that can be performed on a :class:`~base_classes.base_fire.BaseFireSim` object.

Not required to be used to complete actions, but provide a nice compact way to define action
sequences.

.. autoclass:: Action
    :members:


.. autoclass:: SetFuelMoisture
    :members:

.. autoclass:: SetFuelContent
    :members:

.. autoclass:: SetIgnition
    :members:

"""

from embrs.base_classes.base_fire import BaseFireSim

class Action:
    """Base class which all other actions implement.

    :param time: time at which the action takes place. Note this is just for the user's use,
                 the actions will not be scheduled to be performed at this time.
    :type time: float
    :param x: x location in meters where the action takes place.
    :type x: float
    :param y: y location in meters where the action takes place.
    :type y: float
    """
    def __init__(self, time: float, x: float, y: float):
        self.time = time
        self.loc = (x, y)

    def __lt__(self, other):
        if self.time != other.time:
            return self.time < other.time

        elif self.loc[0] != self.loc[0]:
            return self.loc[0] < self.loc[1]

        elif self.loc[1] != self.loc[1]:
            return self.loc[1] < self.loc[1]

        else:
            return True
        
class SetFuelMoisture(Action):
    """Class defining the action of setting the fuel moisture at a location.

    :param time: time at which the action takes place. Note this is just for the user's use,
                 the actions will not be scheduled to be performed at this time.
    :type time: float
    :param x: x location in meters where the action takes place.
    :type x: float
    :param y: y location in meters where the action takes place.
    :type y: float
    :param moisture: Moisture value to set the fuel to at the location and time specified.
    :type moisture: float
    """
    def __init__(self, time: float, x: float, y: float, moisture: float):
        super().__init__(time, x, y)
        self.moisture = moisture

    def perform(self, fire: BaseFireSim):
        """Function that carries out the SetFuelMoisture action defined by the object, it should
        noted that the action will take place at whatever sim time the fire instance is on when
        this function is called, NOT the time parameter of this object.

        :param fire: Fire instance to perform the action on.
        :type fire: BaseFireSim
        """
        cell = fire.get_cell_from_xy(self.loc[0], self.loc[1], oob_ok=True)
        if cell is not None:
            fire.set_fuel_moisture_at_xy(self.loc[0], self.loc[1], self.moisture)
        
class SetFuelContent(Action):
    """Class defining the action of setting the fuel content at a location.

    :param time: time at which the action takes place. Note this is just for the user's use,
                 the actions will not be scheduled to be performed at this time.
    :type time: float
    :param x: x location in meters where the action takes place.
    :type x: float
    :param y: y location in meters where the action takes place.
    :type y: float
    :param fuel_content: Fuel content value to set the fuel to at the location and time specified.
    :type fuel_content: float
    """
    def __init__(self, time: float, x: float, y: float, fuel_content: float):
        super().__init__(time, x, y)
        self.content = fuel_content

    def perform(self, fire: BaseFireSim):
        """Function that carries out the SetFuelContent action defined by the object, it should
        noted that the action will take place at whatever sim time the fire instance is on when
        this function is called, NOT the time parameter of this object.


        :param fire: Fire instance to perform the action on.
        :type fire: BaseFireSim
        """
        cell = fire.get_cell_from_xy(self.loc[0], self.loc[1], oob_ok=True)
        if cell is not None:
            fire.set_fuel_content_at_cell(cell, self.content)

class SetIgnition(Action):
    """Class defining the action of starting an ignition at a location.

    :param time: time at which the action takes place. Note this is just for the user's use,
                 the actions will not be scheduled to be performed at this time.
    :type time: float
    :param x: x location in meters where the action takes place.
    :type x: float
    :param y: y location in meters where the action takes place.
    :type y: float
    """
    def __init__(self, time: float, x: float, y: float):
        super().__init__(time, x, y)

    def perform(self, fire: BaseFireSim):
        """Function that carries out the SetIgnition action defined by the object, it should
        noted that the action will take place at whatever sim time the fire instance is on when
        this function is called, NOT the time parameter of this object.

        :param fire: Fire instance to perform the action on.
        :type fire: BaseFireSim
        """

        cell = fire.get_cell_from_xy(self.loc[0], self.loc[1], oob_ok=True)
        if cell is not None:
            fire.set_wild_fire_at_xy(self.loc[0], self.loc[1])
