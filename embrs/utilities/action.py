"""Suppression action classes for fire control operations.

This module defines action classes that represent discrete fire suppression
operations such as setting ignitions, dropping retardant, water drops, and
constructing firelines. Actions are meant to be instantiated and then
performed on a fire simulation instance.

Classes:
    - Action: Base class for all actions.
    - SetIgnition: Set an ignition at a specified location.
    - DropRetardant: Drop fire retardant at a location.
    - DropWaterAsRain: Simulate water drop as rainfall.
    - DropWaterAsMoistureInc: Increase fuel moisture at a location.
    - ConstructFireline: Construct a fireline along a path.

.. autoclass:: Action
    :members:

.. autoclass:: SetIgnition
    :members:

.. autoclass:: DropRetardant
    :members:

.. autoclass:: DropWaterAsRain
    :members:

.. autoclass:: DropWaterAsMoistureInc
    :members:

.. autoclass:: ConstructFireline
    :members:
"""

from embrs.base_classes.base_fire import BaseFireSim

from shapely.geometry import LineString


class Action:
    """Base class for all fire suppression actions.

    Actions are sortable by time and location for scheduling purposes.
    Note that the time attribute is for user reference only; actions execute
    at whatever simulation time they are called, not at their stored time.

    Attributes:
        time (float): Reference time for the action in seconds.
        loc (tuple): Location (x, y) of the action in meters.
    """

    def __init__(self, time: float, x: float, y: float):
        """Initialize an action.

        Args:
            time (float): Reference time for the action in seconds.
            x (float): X location of the action in meters.
            y (float): Y location of the action in meters.
        """
        self.time = time
        self.loc = (x, y)

    def __lt__(self, other):
        """Compare actions for sorting by time, then location.

        Args:
            other (Action): Another action to compare against.

        Returns:
            bool: True if this action should come before the other.
        """
        if self.time != other.time:
            return self.time < other.time

        elif self.loc[0] != self.loc[0]:
            return self.loc[0] < self.loc[1]

        elif self.loc[1] != self.loc[1]:
            return self.loc[1] < self.loc[1]

        else:
            return True


class SetIgnition(Action):
    """Action to set an ignition at a specified location.

    When performed, ignites the cell containing the specified (x, y) location.

    Attributes:
        time (float): Reference time for the action in seconds.
        loc (tuple): Location (x, y) of the ignition in meters.
    """

    def __init__(self, time: float, x: float, y: float):
        """Initialize a set ignition action.

        Args:
            time (float): Reference time for the action in seconds.
            x (float): X location of the ignition in meters.
            y (float): Y location of the ignition in meters.
        """
        super().__init__(time, x, y)

    def perform(self, fire: BaseFireSim):
        """Execute the ignition action on the fire simulation.

        The ignition occurs at the current simulation time, not the stored
        time attribute.

        Args:
            fire (BaseFireSim): Fire simulation instance to modify.
        """

        cell = fire.get_cell_from_xy(self.loc[0], self.loc[1], oob_ok=True)
        if cell is not None:
            fire.set_ignition_at_cell(cell)


class DropRetardant(Action):
    """Action to drop fire retardant at a location.

    Applies long-term fire retardant to the cell at the specified location.

    Attributes:
        time (float): Reference time for the action in seconds.
        loc (tuple): Location (x, y) of the drop in meters.
        duration_hr (float): Duration of retardant effectiveness in hours.
        effectiveness (float): Effectiveness factor of the retardant.
            TODO:verify range and units for effectiveness parameter.
    """

    def __init__(self, time, x, y, duration_hr: float, effectiveness: float):
        """Initialize a drop retardant action.

        Args:
            time (float): Reference time for the action in seconds.
            x (float): X location of the drop in meters.
            y (float): Y location of the drop in meters.
            duration_hr (float): Duration of retardant effectiveness in hours.
            effectiveness (float): Effectiveness factor of the retardant.
        """
        super().__init__(time, x, y)

        self.duration_hr = duration_hr
        self.effectiveness = effectiveness

    def perform(self, fire: BaseFireSim):
        """Execute the retardant drop on the fire simulation.

        Args:
            fire (BaseFireSim): Fire simulation instance to modify.
        """

        cell = fire.get_cell_from_xy(self.loc[0], self.loc[1], oob_ok=True)
        if cell is not None:
            fire.add_retardant_at_cell(cell)


class DropWaterAsRain(Action):
    """Action to simulate water drop as rainfall at a location.

    Models water application as equivalent rainfall depth to affect
    fuel moisture calculations.

    Attributes:
        time (float): Reference time for the action in seconds.
        loc (tuple): Location (x, y) of the drop in meters.
        water_depth_cm (float): Equivalent rainfall depth in centimeters.
    """

    def __init__(self, time, x, y, water_depth_cm: float = 0.0):
        """Initialize a water drop (as rain) action.

        Args:
            time (float): Reference time for the action in seconds.
            x (float): X location of the drop in meters.
            y (float): Y location of the drop in meters.
            water_depth_cm (float): Equivalent rainfall depth in centimeters.
                Defaults to 0.0.
        """
        super().__init__(time, x, y)

        self.water_depth_cm = water_depth_cm

    def perform(self, fire: BaseFireSim):
        """Execute the water drop on the fire simulation.

        Args:
            fire (BaseFireSim): Fire simulation instance to modify.
        """

        cell = fire.get_cell_from_xy(self.loc[0], self.loc[1], oob_ok=True)
        if cell is not None:
            fire.water_drop_at_cell_as_rain(cell, self.water_depth_cm)


class DropWaterAsMoistureInc(Action):
    """Action to increase fuel moisture at a location.

    Directly increases the fuel moisture content of the cell at the
    specified location.

    Attributes:
        time (float): Reference time for the action in seconds.
        loc (tuple): Location (x, y) of the action in meters.
        moisture_inc (float): Moisture content increase as a fraction.
            TODO:verify if this is absolute increase or multiplier.
    """

    def __init__(self, time, x, y, moisture_inc: float):
        """Initialize a moisture increase action.

        Args:
            time (float): Reference time for the action in seconds.
            x (float): X location of the action in meters.
            y (float): Y location of the action in meters.
            moisture_inc (float): Moisture content increase as a fraction.
        """
        super().__init__(time, x, y)

        self.moisture_inc = moisture_inc

    def perform(self, fire: BaseFireSim):
        """Execute the moisture increase on the fire simulation.

        Args:
            fire (BaseFireSim): Fire simulation instance to modify.
        """
        cell = fire.get_cell_from_xy(self.loc[0], self.loc[1], oob_ok=True)
        if cell is not None:
            fire.water_drop_at_cell_as_moisture_bump(cell, self.moisture_inc)


class ConstructFireline(Action):
    """Action to construct a fireline along a path.

    Constructs a fireline (fuel break) along the specified line geometry.
    If construction_rate is None, the fireline is applied instantly.
    Otherwise, the fireline is constructed progressively at the specified
    rate over simulation time steps.

    Attributes:
        time (float): Reference time for the action in seconds.
        loc (tuple): Starting location (x, y) of the fireline in meters.
        line (LineString): Shapely LineString defining the fireline path.
        width_m (float): Width of the fireline in meters.
        construction_rate (float): Construction rate in meters per second.
            If None, the fireline is applied instantly.
    """

    def __init__(self, time, x, y, line: LineString, width_m: float,
                 construction_rate: float = None):
        """Initialize a construct fireline action.

        Args:
            time (float): Reference time for the action in seconds.
            x (float): X location of the action origin in meters.
            y (float): Y location of the action origin in meters.
            line (LineString): Shapely LineString defining the fireline path.
            width_m (float): Width of the fireline in meters.
            construction_rate (float): Construction rate in meters per second.
                If None, the fireline is applied instantly. Defaults to None.
        """
        super().__init__(time, x, y)

        self.line = line
        self.width_m = width_m
        self.construction_rate = construction_rate

    def perform(self, fire: BaseFireSim):
        """Execute the fireline construction on the fire simulation.

        If construction_rate is None, the entire fireline is applied instantly.
        Otherwise, an active fireline is created that progresses over time.

        Args:
            fire (BaseFireSim): Fire simulation instance to modify.
        """
        fire.construct_fireline(self.line, self.width_m, self.construction_rate)