"""Shared utilities for the EMBRS fire simulation framework.

This package provides common utilities used across the simulation, including
data structures, logging, file I/O, unit conversions, and visualization helpers.

Modules:
    - action: Suppression action classes for fire control.
    - fire_util: Core constants and hexagonal grid math utilities.
    - logger: Simulation logging with Parquet output.
    - logger_schemas: Data schemas for logged entries.
    - parquet_writer: Parquet file writing utilities.
    - unit_conversions: Imperial-metric unit conversion functions.
    - file_io: GUI file selectors and configuration readers.
    - data_classes: Dataclasses for simulation parameters and state.
    - map_drawer: Interactive map drawing tools for ignitions and fire-breaks.
    - ensemble_video: Video generation for ensemble predictions.
"""
