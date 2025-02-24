import pvlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
import pytz
from tqdm import tqdm
import numpy as np

filename = '/Users/rui/Documents/Research/Code/embrs/embrs/utilities/irradiance_test_data.pkl'

# load pandas dataframe with weather data
weather_data = pd.read_pickle(filename)

# ----- Input Parameters -----



# Geographic location (latitude and longitude in decimal degrees)
latitude =  36.24381166486823
longitude = -112.45742255388956

# Initialize a list to store the total irradiance values
I_values = []

# Calculate the solar position at the given date, time, and location
times = pd.date_range('2025-02-18 00:00', '2025-2-18 23:00', freq='H', tz='America/Phoenix')
solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

aspect = np.load('/Users/rui/Documents/Research/Code/embrs_maps/grand_canyon/aspect.npy')
# aspect = (aspect+ 180) % 360
slope = np.load('/Users/rui/Documents/Research/Code/embrs_maps/grand_canyon/slope.npy')

I_values = []  # list to store 2D arrays for each time step

for idx in tqdm(range(len(times)), desc="Calculating Irradiance"):
    date_time = weather_data["date"].iloc[idx]
    ghi = weather_data["ghi"].iloc[idx]
    dni = weather_data["dni"].iloc[idx]
    dhi = weather_data["dhi"].iloc[idx]

    solar_zenith = solpos['zenith'].iloc[idx]
    solar_azimuth = solpos['azimuth'].iloc[idx]

    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=slope,          # pass entire 2D array
        surface_azimuth=aspect,      # pass entire 2D array
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        model='isotropic'
    )
    I_values.append(total_irradiance['poa_global'])


I_min = min(np.min(frame) for frame in I_values)
I_max = max(np.max(frame) for frame in I_values)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(I_values[0], vmin=I_min, vmax=I_max, cmap='gray', animated=True)
cbar = plt.colorbar(im, ax=ax)
ax.set_xticks([])
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_xticklabels([])

# Create an animated text object (placed inside the axes) with a bounding box.
time_text = ax.text(0.5, 0.90, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=ax.transAxes, ha="center", animated=True)

def update(frame_idx):
    # Update the image data
    im.set_data(I_values[frame_idx])
    # Update the overlaid text with the current timestamp.
    # Format the datetime as desired; here we use year-month-day hour:minute.
    time_str = pd.to_datetime(times[frame_idx]).strftime('%Y-%m-%d %H:%M')
    time_text.set_text(f"Time: {time_str}")
    return im, time_text

ani = animation.FuncAnimation(fig, update, frames=len(I_values),
                              interval=200, blit=True)

plt.title('Irradiance Across Landscape')
plt.show()