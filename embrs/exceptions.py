"""Custom exceptions for the EMBRS simulation framework.

This module defines a hierarchy of exceptions used throughout EMBRS
to provide clear, specific error messages and enable targeted exception
handling by users of the library.

Exception Hierarchy:
    EMBRSError (base)
    ├── ConfigurationError - Invalid configuration files or parameters
    ├── SimulationError - Errors during simulation execution
    ├── ValidationError - Input validation failures
    ├── WeatherDataError - Weather data issues
    ├── ExternalServiceError - External service failures (WindNinja, OpenMeteo)
    └── GridError - Grid operations failures

Example:
    >>> from embrs.exceptions import ConfigurationError
    >>> raise ConfigurationError("Missing required parameter 'cell_size'")
"""

from typing import Optional


class EMBRSError(Exception):
    """Base exception for all EMBRS-related errors.

    All custom exceptions in EMBRS inherit from this class, allowing
    users to catch all EMBRS errors with a single except clause if desired.

    Example:
        >>> try:
        ...     fire.iterate()
        ... except EMBRSError as e:
        ...     print(f"EMBRS error occurred: {e}")
    """

    pass


class ConfigurationError(EMBRSError):
    """Raised when configuration file or parameters are invalid.

    This exception is raised when:
    - Required parameters are missing from config files
    - Parameter values are out of valid ranges
    - Config file format is incorrect
    - Incompatible parameter combinations are specified

    Attributes:
        message (str): Explanation of the configuration error.
        config_path (str): Path to the configuration file, if applicable.
        parameter (str): Name of the problematic parameter, if applicable.

    Example:
        >>> raise ConfigurationError(
        ...     "Start datetime must be before end datetime",
        ...     config_path="/path/to/config.cfg"
        ... )
    """

    def __init__(self, message: str, config_path: Optional[str] = None, parameter: Optional[str] = None):
        self.config_path = config_path
        self.parameter = parameter

        # Build detailed message
        parts = []
        if config_path:
            parts.append(f"in {config_path}")
        if parameter:
            parts.append(f"parameter '{parameter}'")

        if parts:
            full_message = f"{message} ({', '.join(parts)})"
        else:
            full_message = message

        super().__init__(full_message)


class SimulationError(EMBRSError):
    """Raised when an error occurs during simulation execution.

    This exception is raised when:
    - The simulation enters an invalid state
    - A required resource becomes unavailable
    - An unrecoverable error occurs during iteration

    Attributes:
        message (str): Explanation of the simulation error.
        sim_time_s (float): Simulation time when error occurred, if available.

    Example:
        >>> raise SimulationError(
        ...     "Weather forecast exhausted before simulation end",
        ...     sim_time_s=3600.0
        ... )
    """

    def __init__(self, message: str, sim_time_s: Optional[float] = None):
        self.sim_time_s = sim_time_s

        if sim_time_s is not None:
            full_message = f"{message} (at sim time {sim_time_s:.1f}s)"
        else:
            full_message = message

        super().__init__(full_message)


class ValidationError(EMBRSError):
    """Raised when input validation fails.

    This exception is raised when:
    - Function arguments are invalid
    - Data structures have invalid contents
    - Coordinate values are out of bounds

    Attributes:
        message (str): Explanation of the validation failure.
        field (str): Name of the field that failed validation, if applicable.
        value: The invalid value, if applicable.

    Example:
        >>> raise ValidationError(
        ...     "Fuel moisture must be between 0 and 3",
        ...     field="dead_fuel_moisture",
        ...     value=-0.5
        ... )
    """

    def __init__(self, message: str, field: Optional[str] = None, value=None):
        self.field = field
        self.value = value

        parts = []
        if field:
            parts.append(f"field '{field}'")
        if value is not None:
            parts.append(f"value={value!r}")

        if parts:
            full_message = f"{message} ({', '.join(parts)})"
        else:
            full_message = message

        super().__init__(full_message)


class WeatherDataError(EMBRSError):
    """Raised when weather data is invalid or unavailable.

    This exception is raised when:
    - Weather file format is incorrect
    - Weather data is missing required fields
    - Weather stream is exhausted
    - Weather values are physically impossible

    Attributes:
        message (str): Explanation of the weather data error.
        source (str): Weather data source (file path or 'OpenMeteo').

    Example:
        >>> raise WeatherDataError(
        ...     "Weather stream ended before simulation completion",
        ...     source="OpenMeteo"
        ... )
    """

    def __init__(self, message: str, source: Optional[str] = None):
        self.source = source

        if source:
            full_message = f"{message} (source: {source})"
        else:
            full_message = message

        super().__init__(full_message)


class ExternalServiceError(EMBRSError):
    """Raised when an external service (WindNinja, OpenMeteo) fails.

    This exception is raised when:
    - WindNinja subprocess fails or times out
    - OpenMeteo API request fails
    - Network connectivity issues occur
    - External service returns invalid data

    Attributes:
        message (str): Explanation of the service failure.
        service (str): Name of the external service.
        original_error (Exception): The underlying exception, if any.

    Example:
        >>> raise ExternalServiceError(
        ...     "WindNinja process timed out after 300 seconds",
        ...     service="WindNinja"
        ... )
    """

    def __init__(self, message: str, service: Optional[str] = None, original_error: Optional[Exception] = None):
        self.service = service
        self.original_error = original_error

        parts = []
        if service:
            parts.append(f"service: {service}")
        if original_error:
            parts.append(f"caused by: {type(original_error).__name__}: {original_error}")

        if parts:
            full_message = f"{message} ({', '.join(parts)})"
        else:
            full_message = message

        super().__init__(full_message)


class GridError(EMBRSError):
    """Raised when grid operations fail.

    This exception is raised when:
    - Grid coordinates are out of bounds
    - Cell lookup fails
    - Grid initialization fails
    - Neighbor calculations encounter invalid state

    Attributes:
        message (str): Explanation of the grid error.
        row (int): Row index involved, if applicable.
        col (int): Column index involved, if applicable.

    Example:
        >>> raise GridError(
        ...     "Cell coordinates outside grid bounds",
        ...     row=150,
        ...     col=200
        ... )
    """

    def __init__(self, message: str, row: Optional[int] = None, col: Optional[int] = None):
        self.row = row
        self.col = col

        parts = []
        if row is not None:
            parts.append(f"row={row}")
        if col is not None:
            parts.append(f"col={col}")

        if parts:
            full_message = f"{message} ({', '.join(parts)})"
        else:
            full_message = message

        super().__init__(full_message)


class FuelModelError(EMBRSError):
    """Raised when fuel model operations fail.

    This exception is raised when:
    - Invalid fuel model ID is specified
    - Fuel model data is missing or corrupt
    - Fuel model calculations fail

    Attributes:
        message (str): Explanation of the fuel model error.
        fuel_model_id (int): The fuel model ID involved, if applicable.

    Example:
        >>> raise FuelModelError(
        ...     "Unknown fuel model ID",
        ...     fuel_model_id=999
        ... )
    """

    def __init__(self, message: str, fuel_model_id: Optional[int] = None):
        self.fuel_model_id = fuel_model_id

        if fuel_model_id is not None:
            full_message = f"{message} (fuel model ID: {fuel_model_id})"
        else:
            full_message = message

        super().__init__(full_message)


class PredictionError(EMBRSError):
    """Raised when fire prediction operations fail.

    This exception is raised when:
    - Ensemble execution fails
    - Prediction results are invalid
    - State synchronization fails

    Attributes:
        message (str): Explanation of the prediction error.
        ensemble_member (int): The ensemble member that failed, if applicable.

    Example:
        >>> raise PredictionError(
        ...     "Ensemble member failed during parallel execution",
        ...     ensemble_member=5
        ... )
    """

    def __init__(self, message: str, ensemble_member: Optional[int] = None):
        self.ensemble_member = ensemble_member

        if ensemble_member is not None:
            full_message = f"{message} (ensemble member: {ensemble_member})"
        else:
            full_message = message

        super().__init__(full_message)
