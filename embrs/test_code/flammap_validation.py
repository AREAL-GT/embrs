import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Load the cell log file
df = pd.read_parquet("/Users/rui/Documents/Research/Code/embrs_logs/48hr_line_fire_test/run_0/cell_logs.parquet")

# Round coordinates to avoid floating point issues when checking uniqueness
df["x_rounded"] = df["x"].round(3)
df["y_rounded"] = df["y"].round(3)

# Drop duplicates based on (x, y), keeping the last arrival for each cell
df_unique = df.sort_values("arrival_time").drop_duplicates(subset=["x_rounded", "y_rounded"], keep="last")

# Filter out unburnt cells (those with null or sentinel arrival times like -999)
df_unique = df_unique[
    (df_unique["arrival_time"].notnull()) & 
    (df_unique["arrival_time"] != -999)
]

# Create the scatter plot
fig, ax = plt.subplots(figsize=(10, 8))

sc = ax.scatter(
    df_unique["x"],
    df_unique["y"],
    c=df_unique["arrival_time"],
    cmap="rainbow_r",
    s=10,
    edgecolor="none",
    norm=mcolors.Normalize(
        vmin=df_unique["arrival_time"].min(),
        vmax=df_unique["arrival_time"].max()
    )
)

# Add colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Arrival Time (s)", fontsize=12)

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Fire Arrival Times by Cell Location")
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

import rasterio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Path to FlamMap arrival time raster
flammap_path = "/Users/rui/Documents/Research/Code/embrs/embrs/test_code/48hr_validation_test.tif"

# Load the raster
with rasterio.open(flammap_path) as src:
    arrival = src.read(1)  # Read the first (and usually only) band
    transform = src.transform  # Affine transform to get spatial coordinates
    crs = src.crs  # Coordinate reference system
    nodata = src.nodata  # Value used for unburned or masked-out cells

# Mask out nodata values
arrival = np.where((arrival == nodata) | (arrival < 0), np.nan, arrival)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

cmap = "rainbow_r"
norm = mcolors.Normalize(vmin=np.nanmin(arrival), vmax=np.nanmax(arrival))

cax = ax.imshow(arrival, cmap=cmap, norm=norm, origin="upper")
cbar = plt.colorbar(cax, ax=ax)
cbar.set_label("Arrival Time (minutes or seconds)", fontsize=12)  # Adjust depending on FlamMap config

ax.set_title("FlamMap Fire Arrival Time")
ax.set_xlabel("Raster column")
ax.set_ylabel("Raster row")
plt.tight_layout()
plt.show()
