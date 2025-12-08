from embrs.base_classes.base_fire import BaseFireSim

from shapely.geometry import LineString

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
            fire.set_ignition_at_cell(cell)


class DropRetardant(Action):
    def __init__(self, time, x, y, duration_hr: float, effectiveness: float):
        super().__init__(time, x, y)

        self.duration_hr = duration_hr
        self.effectiveness = effectiveness


    def perform(self, fire: BaseFireSim):

        cell = fire.get_cell_from_xy(self.loc[0], self.loc[1], oob_ok=True)
        if cell is not None:
            fire.add_retardant_at_cell(cell)


class DropWaterAsRain(Action):
    def __init__(self, time, x, y, water_depth_cm: float = 0.0):
        super().__init__(time, x, y)

        self.water_depth_cm = water_depth_cm

    def perform(self, fire: BaseFireSim):

        cell = fire.get_cell_from_xy(self.loc[0], self.loc[1], oob_ok=True)
        if cell is not None:
            fire.water_drop_at_cell_as_rain(cell, self.water_depth_cm)

class DropWaterAsMoistureInc(Action):
    def __init__(self, time, x, y, moisture_inc: float):
        super().__init__(time, x, y)

        self.moisture_inc = moisture_inc

    def perform(self, fire: BaseFireSim):
        cell = fire.get_cell_from_xy(self.loc[0], self.loc[1], oob_ok=True)
        if cell is not None:
            fire.water_drop_at_cell_as_moisture_bump(cell, self.moisture_inc)

class ConstructFireline(Action):
    def __init__(self, time, x, y, line: LineString, width_m: float, rate_m_s: float):
        super().__init__(time, x, y)

        self.line = line
        self.width_m = width_m
        self.rate_m_s = rate_m_s

    def perform(self, fire: BaseFireSim):
        fire.construct_fireline(self.line, self.width_m, self.rate_m_s)