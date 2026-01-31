"""Wind forecast processing using WindNinja.

This module provides functions for generating and processing wind forecasts using WindNinja. 
It supports different initialization methods, handles WindNinja CLI execution, and structures 
wind data into NumPy arrays for use in wildfire simulations.

Features:
    - **Wind Forecast Generation**:
        - `gen_forecast()`: Runs WindNinja with domain-average, point-based, or weather model initialization.
        - `run_domain_avg_windninja()`: Executes WindNinja using domain-average wind initialization.
        - `run_point_windninja()`: Placeholder for point-based wind initialization.

    - **Wind Data Processing**:
        - `rename_windninja_outputs()`: Renames WindNinja-generated files to a standardized format.
        - `create_forecast_array()`: Loads WindNinja outputs into a structured NumPy array.
        - `convert_to_cartesian()`: Converts wind direction data from meteorological to Cartesian convention.

Dependencies:
    - **External Software**: WindNinja CLI must be installed and accessible at `cli_path`.
    - **Python Libraries**: Requires `subprocess`, `os`, `json`, `numpy`, and `typing`.

Paths:
    - `cli_path`: Path to the WindNinja CLI executable. 
    - `temp_file_path`: Path for storing temporary WindNinja outputs. 

Usage:
    To generate a wind forecast using a domain-average wind:
    ```python
    forecast, time_step = gen_forecast(
        elevation_path="path/to/elevation.asc",
        vegetation_path="path/to/vegetation.asc",
        forecast_seed_path="path/to/wind_seed.json",
        forecast_seed_type="Domain Average Wind"
    )
    ```
"""

import subprocess
import os
import shutil
import numpy as np
from typing import Tuple
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from embrs.utilities.data_classes import MapParams, WindNinjaTask
from embrs.models.weather import WeatherStream
from datetime import timedelta, datetime
import pytz

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Get path to the WindNinja CLI executable from environment variable or use a fallback
fallback_cli_path = os.path.abspath(os.path.join(PACKAGE_DIR, "../../../wind/build/src/cli/WindNinja_cli"))
cli_path = os.getenv("WINDNINJA_CLI_PATH", fallback_cli_path)

# Get path to the temporary files directory from environment variable or use a fallback
fallback_temp_file_path = os.path.abspath(os.path.join(PACKAGE_DIR, "../../../wind/build/src/cli/temp"))
temp_file_path = os.getenv("WINDNINJA_TEMP_PATH", fallback_temp_file_path)

def run_windninja_single(task: WindNinjaTask):
    """Runs WindNinja for a single time step in parallel."""
    output_path = os.path.join(task.temp_file_path, f"{task.index}")
    os.makedirs(output_path, exist_ok=True)

    curr_datetime = task.start_datetime + timedelta(minutes=task.index * task.time_step)

    wind_dir = (task.entry.wind_dir_deg + task.north_angle) % 360

    command = [
        cli_path,
        "--initialization_method", "domainAverageInitialization",
        "--elevation_file", task.elevation_path,
        "--output_path", output_path,
        "--mesh_resolution", str(task.mesh_resolution),
        "--units_mesh_resolution", "m",
        "--time_zone", task.timezone,
        "--uni_air_temp", str(task.entry.temp),
        "--air_temp_units", task.temperature_units,
        "--uni_cloud_cover", str(task.entry.cloud_cover),
        "--cloud_cover_units", "percent",
        "--diurnal_winds", "false",
        "--year", str(curr_datetime.year),
        "--month", str(curr_datetime.month),
        "--day", str(curr_datetime.day),
        "--hour", str(curr_datetime.hour),
        "--minute", str(curr_datetime.minute),
        "--num_threads", "4",
        "--output_wind_height", "6.1",
        "--units_output_wind_height", "m",
        "--output_speed_units", "mps",
        "--input_speed", str(task.entry.wind_speed),
        "--input_speed_units", task.input_speed_units,
        "--input_direction", str(wind_dir),
        "--input_wind_height", str(task.wind_height),
        "--units_input_wind_height", task.wind_height_units,
        "--write_ascii_output", "true",
        "--write_goog_output", "true"
    ]

    try:
        log_file = os.path.join(output_path, "windninja_log.txt")

        with open(log_file, "w") as file:
            subprocess.run(command, check=True, stdout=file, stderr=file)
        
        rename_windninja_outputs(output_path, task.index)

    except subprocess.CalledProcessError as e:
        print(f"Error running WindNinja CLI at step {task.index}: {e}")

def run_windninja(weather: WeatherStream, map: MapParams, custom_temp_dir: str = None, num_workers: int = None) -> Tuple[np.ndarray, float]:
    """Runs WindNinja with domain-average initialization in parallel.

    Args:
        weather: Weather stream with wind data
        map: Map parameters
        custom_temp_dir: Optional custom temporary directory for this run.
                        If provided, uses this instead of the global temp_file_path.
                        This is essential for parallel ensemble predictions to avoid
                        race conditions between workers.
        num_workers: Number of worker processes for parallel WindNinja execution.
                    If None (default), uses min(cpu_count(), num_tasks).
                    Set to 1 for sequential execution (useful when caller is
                    already parallelizing at a higher level).

    Returns:
        Tuple of (forecast array, time_step)
    """

    # Use custom temp dir if provided (for ensemble workers), otherwise use global
    work_temp_path = custom_temp_dir if custom_temp_dir is not None else temp_file_path

    # Extract data from forecast seed
    time_step = weather.time_step
    wind_height = weather.input_wind_ht
    wind_height_units = weather.input_wind_ht_units
    input_speed_units = weather.input_wind_vel_units
    temperature_units = weather.input_temp_units
    start_datetime = weather.params.start_datetime
    mesh_resolution = weather.params.mesh_resolution
    timezone = map.geo_info.timezone
    north_angle = map.geo_info.north_angle_deg

    if not type(start_datetime) == datetime:
        # Convert to naive datetime
        dt_local = datetime.fromisoformat(start_datetime)

        # Localize the datetime with the correct timezone
        tz = pytz.timezone(timezone)
        start_datetime = tz.localize(dt_local)

    # Clear temp folder
    if os.path.exists(work_temp_path):
        for file_name in os.listdir(work_temp_path):
            file_path = os.path.join(work_temp_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                # Use shutil.rmtree to handle arbitrarily nested directories
                shutil.rmtree(file_path)

    # Prepare arguments for parallel execution
    tasks = [
        WindNinjaTask(
            index=i,
            time_step=time_step,
            entry=entry,
            elevation_path=map.cropped_lcp_path,
            timezone=timezone,
            north_angle=north_angle,
            mesh_resolution=mesh_resolution,
            temp_file_path=work_temp_path,  # Use work_temp_path instead of global
            cli_path=cli_path,
            start_datetime=start_datetime,
            wind_height=wind_height,
            wind_height_units=wind_height_units,
            input_speed_units=input_speed_units,
            temperature_units=temperature_units
        )
        for i, entry in enumerate(weather.stream[weather.sim_start_idx:])
    ]
    # Use multiprocessing Pool to parallelize
    num_tasks = len(tasks)
    # Use provided num_workers, or auto-detect based on CPU cores
    if num_workers is None:
        num_workers = min(cpu_count(), num_tasks)  # Limit workers to available CPU cores
    else:
        num_workers = min(num_workers, num_tasks)  # Don't use more workers than tasks

    pool = Pool(processes=num_workers)

    for _ in tqdm(pool.imap_unordered(run_windninja_single, tasks), total = num_tasks, desc="Generating wind with WindNinja: "):
        pass
    
    # Merge data into a forecast
    forecast = create_forecast_array(num_tasks, work_temp_path)

    return forecast

def run_point_windninja():
    raise NotImplementedError("Point initialization not yet implemented")

def rename_windninja_outputs(output_path: str, time_step_index: int):
    """Renames WindNinja output files in a specified directory to a standardized format.

    This function processes WindNinja-generated output files, extracting wind speed, 
    wind direction, and cloud cover data. It renames the files using a structured 
    naming convention based on the provided time step index.

    Args:
        output_path (str): The directory where WindNinja outputs are stored.
        time_step_index (int): The index of the time step to include in the file names.

    Behavior:
        - Identifies files containing `_vel` (wind speed), `_ang` (wind direction), 
          and `_cld` (cloud cover).
        - Renames them to `wind_speed_<time_step>.asc`, `wind_direction_<time_step>.asc`, etc.
        - Ensures all files remain in the same directory.

    Notes:
        - Assumes WindNinja outputs are stored in ASCII format (`.asc`).
        - Cloud cover renaming is included but may not be relevant for all WindNinja runs.
    """
    for file_name in os.listdir(output_path):
        old_path = os.path.join(output_path, file_name)
        if os.path.isfile(old_path):
            # Create a standardized file name
            extension = os.path.splitext(file_name)[1]
            if "_vel" in file_name:
                new_file_name = f"wind_speed_{time_step_index}{extension}"
            elif "_ang" in file_name:
                new_file_name = f"wind_direction_{time_step_index}{extension}"
            elif "_cld" in file_name:
                new_file_name = f"cloud_cover_{time_step_index}{extension}"
            elif extension == ".kmz" or extension == ".kml":
                new_file_name = f"google_earth_{time_step_index}{extension}"
            else:
                continue

            new_path = os.path.join(output_path, new_file_name)
            os.rename(old_path, new_path)

def create_forecast_array(num_files: int, work_temp_path: str = None) -> np.ndarray:
    """Loads WindNinja wind forecast outputs into a structured NumPy array.


    This function reads ASCII files produced by WindNinja, extracts wind speed
    and direction data, and compiles them into a multi-dimensional NumPy array
    for use in fire simulations.

    Args:
        num_files (int): The number of time steps (i.e., number of WindNinja-generated files).
        work_temp_path (str): Optional custom temp directory. If None, uses global temp_file_path.

    Returns:
        np.ndarray: A structured array with shape `(num_files, height, width, 2)`, 
                    where:
                    - `height` and `width` are the dimensions of the wind raster.
                    - The last axis stores wind components:
                        - `[:,:,0]` = Wind speed (m/s).
                        - `[:,:,1]` = Wind direction (converted to Cartesian).

    Behavior:
        - Iterates through each time step’s wind speed and direction files.
        - Loads wind speed (`wind_speed_<i>.asc`) and direction (`wind_direction_<i>.asc`).
        - Converts wind direction data using `convert_to_cartesian()`.
        - Constructs a forecast array with separate wind speed and direction layers.
        - Cleans up temporary WindNinja output directories after processing.

    Notes:
        - Expects WindNinja outputs to follow the standardized naming convention 
          produced by `rename_windninja_outputs()`.
        - Assumes WindNinja output files contain ASCII grid data with a 6-line header.
        - Uses `np.loadtxt()` to efficiently load numerical wind data.
    """
    # Use work_temp_path if provided, otherwise use global
    if work_temp_path is None:
        work_temp_path = temp_file_path

    forecast = None  # Initialize before loop

    for i in range(num_files):
        output_path = os.path.join(work_temp_path, f"{i}")

        speed_file = os.path.join(output_path, f"wind_speed_{i}.asc")
        direction_file = os.path.join(output_path, f"wind_direction_{i}.asc")

        if os.path.exists(speed_file) and os.path.exists(direction_file):
            with open(speed_file, 'r') as file:
                speed_data = np.loadtxt(file, skiprows=6)

                # Initialize forecast array on first valid file
                if forecast is None:
                    forecast = np.zeros((num_files, *speed_data.shape, 2))

                forecast[i, :, :, 0] = speed_data

            with open(direction_file, 'r') as file:
                direction_data = np.loadtxt(file, skiprows=6)
                direction_data = convert_to_cartesian(direction_data)
                forecast[i, :, :, 1] = direction_data

    # Ensure we found at least some files
    if forecast is None:
        raise FileNotFoundError(
            f"No WindNinja output files found in {work_temp_path}. "
            f"Expected files like wind_speed_0.asc and wind_direction_0.asc. "
            f"WindNinja may have failed to generate outputs."
        )

    # Cleanup worker-specific temp directory after loading data
    # Only delete if this is a worker temp dir (contains "worker_" in path)
    if work_temp_path is not None and "worker_" in work_temp_path:
        if os.path.exists(work_temp_path):
            shutil.rmtree(work_temp_path)

    return forecast

def convert_to_cartesian(direction_data: np.ndarray) -> np.ndarray:
    """Converts wind direction data to a Cartesian-compatible format.

    Wind direction in WindNinja outputs is typically measured in meteorological degrees 
    (0° = North-South, 90° = East-West, 180° = South-North, 270° = West-East). This function adjusts the values 
    to align with a Cartesian coordinate system.

    Args:
        direction_data (np.ndarray): A NumPy array containing wind direction values in degrees.

    Returns:
        np.ndarray: The transformed wind direction data, where:

            direction_cartesian = (180 + direction_data) % 360

    Notes:
        - This transformation ensures compatibility with fire spread models that 
          use Cartesian angle conventions.
        - Assumes wind direction values are given in degrees (0-360).
    """
    return (180 + direction_data) % 360