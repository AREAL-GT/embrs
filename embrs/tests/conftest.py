"""Shared pytest fixtures for EMBRS test suite.

This module provides reusable fixtures for testing EMBRS components,
including mock weather data, cell data, and grid utilities.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ============================================================================
# Weather Fixtures
# ============================================================================

@pytest.fixture
def sample_weather_entry():
    """Provide a standard weather entry for testing.

    Returns:
        WeatherEntry: A typical summer afternoon weather condition.
    """
    from embrs.utilities.data_classes import WeatherEntry
    return WeatherEntry(
        wind_speed=4.5,           # m/s (~10 mph)
        wind_dir_deg=180.0,       # From south
        temp=25.0,                # Celsius (~77F)
        rel_humidity=0.30,        # 30%
        cloud_cover=0.0,          # Clear
        rain=0.0,                 # No rain
        dni=800.0,                # W/m^2
        dhi=100.0,                # W/m^2
        ghi=850.0,                # W/m^2
        solar_zenith=30.0,        # degrees
        solar_azimuth=180.0       # degrees
    )


@pytest.fixture
def hot_dry_weather():
    """Provide extreme fire weather conditions.

    Returns:
        WeatherEntry: Hot, dry, windy conditions typical of red flag warnings.
    """
    from embrs.utilities.data_classes import WeatherEntry
    return WeatherEntry(
        wind_speed=13.4,          # m/s (~30 mph)
        wind_dir_deg=270.0,       # From west
        temp=38.0,                # Celsius (~100F)
        rel_humidity=0.10,        # 10%
        cloud_cover=0.0,
        rain=0.0,
        dni=900.0,
        dhi=100.0,
        ghi=950.0,
        solar_zenith=20.0,
        solar_azimuth=180.0
    )


@pytest.fixture
def humid_weather():
    """Provide high humidity weather conditions.

    Returns:
        WeatherEntry: Humid conditions that suppress fire spread.
    """
    from embrs.utilities.data_classes import WeatherEntry
    return WeatherEntry(
        wind_speed=2.0,           # m/s (~4.5 mph)
        wind_dir_deg=90.0,        # From east
        temp=15.0,                # Celsius (~59F)
        rel_humidity=0.85,        # 85%
        cloud_cover=0.8,
        rain=0.0,
        dni=200.0,
        dhi=150.0,
        ghi=300.0,
        solar_zenith=60.0,
        solar_azimuth=90.0
    )


# ============================================================================
# Cell Data Fixtures
# ============================================================================

@pytest.fixture
def sample_cell_data():
    """Provide minimal CellData for testing.

    Returns:
        CellData: A cell with Anderson Fuel Model 1 (short grass).
    """
    from embrs.utilities.data_classes import CellData
    from embrs.models.fuel_models import Anderson13

    return CellData(
        fuel_type=Anderson13(1),  # Short grass
        elevation=100.0,          # meters
        aspect=0.0,               # North-facing
        slope_deg=0.0,            # Flat
        canopy_cover=0.0,         # No canopy
        canopy_height=0.0,
        canopy_base_height=0.0,
        canopy_bulk_density=0.0,
        init_dead_mf=0.08,        # 8% dead fuel moisture
        live_h_mf=0.30,
        live_w_mf=0.30
    )


@pytest.fixture
def timber_cell_data():
    """Provide CellData for timber/forest fuel type.

    Returns:
        CellData: A cell with Anderson Fuel Model 10 (timber litter).
    """
    from embrs.utilities.data_classes import CellData
    from embrs.models.fuel_models import Anderson13

    return CellData(
        fuel_type=Anderson13(10),  # Timber litter
        elevation=500.0,
        aspect=180.0,              # South-facing
        slope_deg=15.0,            # Moderate slope
        canopy_cover=0.6,          # 60% canopy
        canopy_height=20.0,        # meters
        canopy_base_height=3.0,
        canopy_bulk_density=0.1,
        init_dead_mf=0.10,
        live_h_mf=0.50,
        live_w_mf=0.80
    )


@pytest.fixture
def brush_cell_data():
    """Provide CellData for brush/chaparral fuel type.

    Returns:
        CellData: A cell with Anderson Fuel Model 4 (chaparral).
    """
    from embrs.utilities.data_classes import CellData
    from embrs.models.fuel_models import Anderson13

    return CellData(
        fuel_type=Anderson13(4),   # Chaparral
        elevation=300.0,
        aspect=225.0,              # Southwest-facing
        slope_deg=30.0,            # Steep slope
        canopy_cover=0.0,
        canopy_height=0.0,
        canopy_base_height=0.0,
        canopy_bulk_density=0.0,
        init_dead_mf=0.06,         # Dry
        live_h_mf=0.60,
        live_w_mf=0.90
    )


# ============================================================================
# Grid and Math Fixtures
# ============================================================================

@pytest.fixture
def hex_grid_math():
    """Provide HexGridMath instance with standard cell size.

    Returns:
        HexGridMath: Grid math utilities for 30m cells.
    """
    from embrs.utilities.fire_util import HexGridMath
    return HexGridMath(cell_size=30.0)


@pytest.fixture
def small_hex_grid_math():
    """Provide HexGridMath instance with small cell size for detailed tests.

    Returns:
        HexGridMath: Grid math utilities for 10m cells.
    """
    from embrs.utilities.fire_util import HexGridMath
    return HexGridMath(cell_size=10.0)


# ============================================================================
# Fuel Model Fixtures
# ============================================================================

@pytest.fixture
def grass_fuel():
    """Provide Anderson Fuel Model 1 (short grass).

    Returns:
        Anderson13: Fuel model for short grass.
    """
    from embrs.models.fuel_models import Anderson13
    return Anderson13(1)


@pytest.fixture
def brush_fuel():
    """Provide Anderson Fuel Model 4 (chaparral).

    Returns:
        Anderson13: Fuel model for chaparral/brush.
    """
    from embrs.models.fuel_models import Anderson13
    return Anderson13(4)


@pytest.fixture
def timber_fuel():
    """Provide Anderson Fuel Model 10 (timber litter).

    Returns:
        Anderson13: Fuel model for timber litter.
    """
    from embrs.models.fuel_models import Anderson13
    return Anderson13(10)


@pytest.fixture
def scott_burgan_fuel():
    """Provide Scott & Burgan Fuel Model SH2 (low load shrub).

    Returns:
        ScottBurgan40: Fuel model for low load shrub.
    """
    from embrs.models.fuel_models import ScottBurgan40
    return ScottBurgan40(142)


# ============================================================================
# Moisture Fixtures
# ============================================================================

@pytest.fixture
def standard_fuel_moisture():
    """Provide standard fuel moisture array.

    Returns:
        np.ndarray: Moisture values for [1hr, 10hr, 100hr, live_herb, live_woody].
    """
    return np.array([0.06, 0.07, 0.08, 0.60, 0.90])


@pytest.fixture
def dry_fuel_moisture():
    """Provide dry fuel moisture conditions.

    Returns:
        np.ndarray: Low moisture values indicating dry conditions.
    """
    return np.array([0.03, 0.04, 0.05, 0.30, 0.60])


@pytest.fixture
def wet_fuel_moisture():
    """Provide wet fuel moisture conditions.

    Returns:
        np.ndarray: High moisture values indicating wet conditions.
    """
    return np.array([0.15, 0.18, 0.22, 1.20, 1.50])


# ============================================================================
# Random Number Generator Fixtures
# ============================================================================

@pytest.fixture
def seeded_rng():
    """Provide seeded random number generator for reproducible tests.

    Returns:
        np.random.Generator: Seeded RNG with seed 42.
    """
    return np.random.default_rng(42)


@pytest.fixture
def rng_factory():
    """Provide factory for creating seeded random number generators.

    Returns:
        Callable: Function that takes a seed and returns an RNG.
    """
    def _create_rng(seed=42):
        return np.random.default_rng(seed)
    return _create_rng


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_cell():
    """Provide a mock Cell object for testing functions that depend on Cell.

    Returns:
        MagicMock: Mock cell with common attributes set.
    """
    mock = MagicMock()
    mock.row = 5
    mock.col = 5
    mock.x = 150.0
    mock.y = 150.0
    mock.elevation = 100.0
    mock.aspect = 0.0
    mock.slope = 0.0
    mock.fmois = np.array([0.06, 0.07, 0.08, 0.60, 0.90])
    mock.curr_wind = (8.8, 180.0)  # ft/min, degrees
    return mock


@pytest.fixture
def mock_parent_sim():
    """Provide a mock parent simulation object.

    Returns:
        MagicMock: Mock BaseFireSim with common attributes.
    """
    mock = MagicMock()
    mock._cell_size = 30.0
    mock._num_rows = 100
    mock._num_cols = 100
    mock.curr_time_s = 0
    mock._weather_stream = MagicMock()
    return mock
