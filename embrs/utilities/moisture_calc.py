import pvlib
import numpy as np
import pickle
from embrs.utilities.dead_fuel_moisture import DeadFuelMoisture
from embrs.fire_simulator.cell import Cell

import pandas as pd
from tqdm import tqdm
from datetime import datetime
from embrs.utilities.weather import WeatherStream
from embrs.utilities.data_classes import WeatherParams
from matplotlib.colors import Normalize
import matplotlib as mpl
import matplotlib.pyplot as plt

def site_specific(elev_diff, temp_air, rh_air):
    ## elev_diff is always in feet, temp is in Fahrenheit, humid is %
    
    dewptref = -398.0-7469.0 / (np.log(rh_air/100.0)-7469.0/(temp_air+398.0))

    temp = temp_air + elev_diff/1000.0*5.5 # Stephenson 1988 found summer adiabat at 3.07 F/1000ft
    dewpt = dewptref + elev_diff/1000.0*1.1 # new humidity, new dewpt, and new humidity
    rh = 7469.0*(1.0/(temp+398.0)-1.0/(dewpt+398.0))
    rh = np.exp(rh)*100.0 #convert from ln units
    if (rh > 99.0):
        rh=99.0

    # Convert temp to celius
    temp = temp-32
    temp /= 1.8 

    # Return humidity as a decimal
    rh /= 100.0

    return temp, rh


def calc_solar_radiation(slope, aspect, canopy_cvr, curr_weather):

    # Calculate total irradiance using pvlib
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=slope,
        surface_azimuth=aspect,
        solar_zenith=curr_weather.solar_zenith,
        solar_azimuth=curr_weather.solar_azimuth,
        dni=curr_weather.dni,
        ghi=curr_weather.ghi,
        dhi=curr_weather.dhi,
        model='isotropic'
    )

    # Adjust for canopy transmittance (Only thing not modelled in pvlib)
    canopy_transmittance = 1 - (canopy_cvr/100)
    I = total_irradiance['poa_global'] * canopy_transmittance

    return I

# TODO:
    # - Implement moisture conditioning
    # - Implement the function in the Cell class
    # - Test the function in the Cell class
    # - Integrate with the rest of the model

def single_loc_test():
    map_params_path = "/Users/rjdp3/Documents/Research/embrs_map/map_params.pkl"

    with open(map_params_path, "rb") as f:
        map_params = pickle.load(f)
    
    weather = WeatherParams(
        input_type="OpenMeteo",
        file="",
        mesh_resolution=250,
        start_datetime=datetime(2025, 1, 31, 12, 0, 0),
        end_datetime=datetime(2025, 2, 1, 12, 0, 0)
    )

    geo = map_params.geo_info
    weather_stream = WeatherStream(weather, geo)
    
    ref_elev = weather_stream.ref_elev

    elev = 1000
    aspect = 0
    slope = 5
    canopy_cvr = 25
    cell = Cell(0, 0, 0, 30, elev, aspect, slope, canopy_cvr)

    m_w1_arr = []
    m_w10_arr = []
    m_w100_arr = []
    m_w1000_arr = []

    for curr_weather in weather_stream.stream:
        m_w1, m_w10, m_w100, m_w1000 = update_moisture(cell, curr_weather, ref_elev)
        m_w1_arr.append(m_w1)
        m_w10_arr.append(m_w10)
        m_w100_arr.append(m_w100)
        m_w1000_arr.append(m_w1000)

    plt.plot(m_w1_arr, label="1hr")
    plt.plot(m_w10_arr, label="10hr")
    plt.plot(m_w100_arr, label="100hr")
    plt.plot(m_w1000_arr, label="1000hr")
    plt.legend()
    plt.show()


def full_map_test():

    map_params_path = "/Users/rjdp3/Documents/Research/embrs_map/map_params.pkl"

    with open(map_params_path, "rb") as f:
        map_params = pickle.load(f)
    

    weather = WeatherParams(
        input_type="OpenMeteo",
        file="",
        mesh_resolution=250,
        start_datetime=datetime(2024, 7, 4, 12, 0, 0),
        end_datetime=datetime(2024, 7, 11, 12, 0, 0)
    )

    geo = map_params.geo_info
    weather_stream = WeatherStream(weather, geo)
    
    ref_elev = weather_stream.ref_elev

    elev_map = np.load("/Users/rjdp3/Documents/Research/embrs_map/elev.npy")
    slope_map = np.load("/Users/rjdp3/Documents/Research/embrs_map/slope.npy")
    aspect_map = np.load("/Users/rjdp3/Documents/Research/embrs_map/aspect.npy")
    canopy_cvr_map = np.load("/Users/rjdp3/Documents/Research/embrs_map/canopy_cover.npy")

    # Downscale the maps by a factor of 100 by selecting every 100th pixel along each axis
    elev_map = elev_map[::100, ::100]
    slope_map = slope_map[::100, ::100]
    aspect_map = aspect_map[::100, ::100]
    canopy_cvr_map = canopy_cvr_map[::100, ::100]

    moisture_map1hr = np.zeros((len(weather_stream.stream), elev_map.shape[0], elev_map.shape[1]))
    moisture_map10hr = np.zeros((len(weather_stream.stream), elev_map.shape[0], elev_map.shape[1]))
    moisture_map100hr = np.zeros((len(weather_stream.stream), elev_map.shape[0], elev_map.shape[1]))
    moisture_map1000hr = np.zeros((len(weather_stream.stream), elev_map.shape[0], elev_map.shape[1]))

    for t in tqdm(range(len(weather_stream.stream))):
        curr_weather = weather_stream.stream[t]
        for i in range(elev_map.shape[0]):
            for j in range(elev_map.shape[1]):
                cell = Cell(0, 0, 0, 30, elev_map[i][j], aspect_map[i][j], slope_map[i][j], canopy_cvr_map[i][j])

                m_w1, m_w10, m_w100, m_w1000 = update_moisture(cell, curr_weather, ref_elev)

                moisture_map1hr[t][i][j] = m_w1
                moisture_map10hr[t][i][j] = m_w10
                moisture_map100hr[t][i][j] = m_w100
                moisture_map1000hr[t][i][j] = m_w1000

    np.save("moisture_map1hr.npy", moisture_map1hr)
    np.save("moisture_map10hr.npy", moisture_map10hr)
    np.save("moisture_map100hr.npy", moisture_map100hr)
    np.save("moisture_map1000hr.npy", moisture_map1000hr)

def plot_moisture_map():

    map_params_path = "/Users/rjdp3/Documents/Research/embrs_map/map_params.pkl"

    with open(map_params_path, "rb") as f:
        map_params = pickle.load(f)

    weather = WeatherParams(
        input_type="OpenMeteo",
        file="",
        mesh_resolution=250,
        start_datetime=datetime(2024, 7, 4, 12, 0, 0),
        end_datetime=datetime(2024, 7, 11, 12, 0, 0)
    )

    geo = map_params.geo_info
    weather_stream = WeatherStream(weather, geo)
    
    import matplotlib.gridspec as gridspec
    import matplotlib.animation as animation

    # Load precomputed weather stream data arrays.
    # (These files should be created when running your weather simulation.)
    weather_time = pd.date_range(weather.start_datetime, weather.end_datetime, freq='h')      # 1D array of datetime (or numerical) values
    
    weather_temp, weather_rh, weather_rain = zip(*[
        (entry.temp, entry.rel_humidity, entry.rain)
        for entry in weather_stream.stream
    ])
    weather_temp = list(weather_temp)
    weather_rh = list(weather_rh)
    weather_rain = list(weather_rain)

    # Create a figure with a gridspec layout to combine moisture maps and weather data
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax_weather = fig.add_subplot(gs[1, :])

    # Preload the moisture maps to compute common limits
    m1 = np.load("moisture_map1hr.npy")
    m10 = np.load("moisture_map10hr.npy")
    m100 = np.load("moisture_map100hr.npy")
    m1000 = np.load("moisture_map1000hr.npy")

    global_vmin = min(m1.min(), m10.min(), m100.min(), m1000.min())
    global_vmax = max(m1.max(), m10.max(), m100.max(), m1000.max())
    norm = Normalize(vmin=global_vmin, vmax=global_vmax)

    # Override imshow on all Axes so that they all use the same normalization unless otherwise specified.
    orig_imshow = mpl.axes.Axes.imshow
    def normalized_imshow(self, *args, **kwargs):
        kwargs.setdefault("norm", norm)
        return orig_imshow(self, *args, **kwargs)
    mpl.axes.Axes.imshow = normalized_imshow
    # Display initial moisture maps (loaded from files)
    im1 = ax1.imshow(np.load("moisture_map1hr.npy")[0], interpolation='none')
    ax1.set_title("1hr Moisture")
    fig.colorbar(im1, ax=ax1)

    im10 = ax2.imshow(np.load("moisture_map10hr.npy")[0], interpolation='none')
    ax2.set_title("10hr Moisture")
    fig.colorbar(im10, ax=ax2)

    im100 = ax3.imshow(np.load("moisture_map100hr.npy")[0], interpolation='none')
    ax3.set_title("100hr Moisture")
    fig.colorbar(im100, ax=ax3)

    im1000 = ax4.imshow(np.load("moisture_map1000hr.npy")[0], interpolation='none')
    ax4.set_title("1000hr Moisture")
    fig.colorbar(im1000, ax=ax4)

    # Plot weather stream data as continuous time series
    ax_weather.set_title("Weather Stream Data")
    ax_weather.set_xlabel("Time Step")
    ax_weather.set_ylabel("Value")
    line_temp, = ax_weather.plot(weather_time, weather_temp, label="Temperature")
    line_rh, = ax_weather.plot(weather_time, weather_rh, label="Relative Humidity")
    line_rain, = ax_weather.plot(weather_time, weather_rain, label="Rainfall")
    vline = ax_weather.axvline(x=weather_time[0], color='black', linestyle='--', linewidth=2)
    ax_weather.legend()

    # Load moisture maps animations from file arrays
    moisture_map1hr = np.load("moisture_map1hr.npy")
    moisture_map10hr = np.load("moisture_map10hr.npy")
    moisture_map100hr = np.load("moisture_map100hr.npy")
    moisture_map1000hr = np.load("moisture_map1000hr.npy")

    def animate(i):
        im1.set_data(moisture_map1hr[i])
        im10.set_data(moisture_map10hr[i])
        im100.set_data(moisture_map100hr[i])
        im1000.set_data(moisture_map1000hr[i])
        # Update the vertical line in the weather plot to indicate current time step
        vline.set_xdata([weather_time[i]]) 
        return [im1, im10, im100, im1000, vline]

    ani = animation.FuncAnimation(fig, animate, frames=moisture_map1hr.shape[0],
                                  interval=400, blit=True)
    
    plt.show()

def update_moisture(cell, curr_weather, ref_elev):
    bp0 = 0.0218
    update_interval_hr = 1

    elev_diff = ref_elev - cell.z 
    elev_diff *= 3.2808 # convert to ft

    t_f_celsius, h_f_frac = site_specific(elev_diff, curr_weather.temp, curr_weather.rel_humidity)
    solar_radiation = calc_solar_radiation(cell.slope_deg, cell.aspect, cell.canopy_cover, curr_weather)

    mx = 0.1

    # TODO: these will be class members for Cell, should only initialize them once
    dfm1 = DeadFuelMoisture.createDeadFuelMoisture1()
    dfm10 = DeadFuelMoisture.createDeadFuelMoisture10()
    dfm100 = DeadFuelMoisture.createDeadFuelMoisture100()
    dfm1000 = DeadFuelMoisture.createDeadFuelMoisture1000()

    dfms = [dfm1, dfm10, dfm100, dfm1000]

    for dfm in dfms:
        if not dfm.initialized():
            dfm.initializeEnvironment(
                t_f_celsius, # Intial ambient air temeperature
                h_f_frac, # Initial ambient air rel. humidity (g/g)
                solar_radiation, # Initial solar radiation (W/m^2)
                curr_weather.rain, # Initial cumulative rainfall (cm)
                t_f_celsius, # Initial stick temperature (degrees C)
                h_f_frac, # Intial stick surface relative humidity (g/g)
                mx, # Initial stick fuel moisture fraction (g/g) # TODO: implement how to get this
                bp0) # Initial stick barometric pressure (cal/cm^3)

        dfm.update_internal(
            update_interval_hr, # Elapsed time since the previous observation (hours)
            t_f_celsius, # Current observation's ambient air temperature (degrees C)
            h_f_frac, # Current observation's ambient air relative humidity (g/g)
            solar_radiation, # Current observation's solar radiation (W/m^2)
            curr_weather.rain, # Current observation's total cumulative rainfall (cm)
            bp0) # Current observation's stick barometric pressure (cal/cm^3)
        
        dfm.m_w = dfm.meanWtdMoisture()

    return dfm1.m_w, dfm10.m_w, dfm100.m_w, dfm1000.m_w

if __name__ == "__main__":
    plot_moisture_map()