# Exceptions

EMBRS defines a hierarchy of custom exceptions rooted at `EMBRSError`. Catching `EMBRSError` in your control code lets you handle all EMBRS-specific failures in one place, while the more specific subclasses (`ConfigurationError`, `GridError`, `WeatherDataError`, etc.) allow targeted recovery when you know which operation might fail. Most users will encounter these exceptions when loading configuration files, looking up cells by coordinate, or running ensemble predictions.

::: embrs.exceptions
