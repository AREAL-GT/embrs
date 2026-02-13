# Base Classes

The `embrs.base_classes` package provides the abstract interfaces and manager classes that form the backbone of the EMBRS simulation framework. Users extending EMBRS will primarily interact with [`ControlClass`](#embrs.base_classes.control_base.ControlClass) (to implement suppression strategies) and [`AgentBase`](#embrs.base_classes.agent_base.AgentBase) (to define agents displayed in visualizations). The remaining classes—[`BaseFireSim`](#embrs.base_classes.base_fire.BaseFireSim), [`GridManager`](#embrs.base_classes.grid_manager.GridManager), [`WeatherManager`](#embrs.base_classes.weather_manager.WeatherManager), [`ControlActionHandler`](#embrs.base_classes.control_handler.ControlActionHandler), and [`BaseVisualizer`](#embrs.base_classes.base_visualizer.BaseVisualizer)—are internal infrastructure that advanced users may need to understand when debugging or extending the simulation engine.

::: embrs.base_classes.control_base

::: embrs.base_classes.agent_base

::: embrs.base_classes.base_fire

::: embrs.base_classes.grid_manager

::: embrs.base_classes.weather_manager

::: embrs.base_classes.control_handler

::: embrs.base_classes.base_visualizer
