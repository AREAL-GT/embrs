# Fuel Models

This module defines the fuel model classes used throughout EMBRS. Each fuel model encapsulates the physical properties (loading, surface-area-to-volume ratios, moisture of extinction, fuel bed depth) required by the Rothermel spread equations. EMBRS supports both the Anderson 13 and Scott-Burgan 40 classification systems, loaded from bundled JSON data files. The `FuelConstants` class provides lookup tables for mapping fuel model numbers to human-readable names and visualization colors.

::: embrs.models.fuel_models
