import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from utils.visualization import plot_with_peaks, zoom_plot


def count_steps(csv_file, return_debug=False):
    """
    Count steps from accelerometer data.
    Input:
        csv_file : str ---> Path to CSV with columns: time, x, y, z
        return_debug : bool, optional ---> If True, return both the 
        step count and a dictionary with extra details (peaks, signal, dataframe) 
        for plotting/debugging.

    Returns:
        int: Number of steps detected (default behavior).
        or (int, dict): Step count plus debug info when return_debug=True.
    """
    
    # load the accelerometer file
    data = pd.read_csv(csv_file)
    
    # combine x,y,z into a single magnitude signal
    data['mag'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    SR = 50
    b, a = butter(3, [0.5/(SR/2), 4.0/(SR/2)], btype='bandpass')

    # VP edited
    magnitude = data['mag'].to_numpy()

    filtered = filtfilt(b, a, magnitude)

    data['mag_filtered'] = filtered #DA added

    # originally was mean, now utilizing median
    data['mag_smooth'] = data['mag_filtered'].rolling(window= 5, center=True).median() #DA edited

    # originally used height, now utilizes prominence
    peaks, props = find_peaks(data['mag_smooth'], prominence = 1.5, distance= 45, width = 8) #DA edited

    step_count = len(peaks)

    if return_debug:
        debug = {
            "peaks": peaks,             # indices of detected peaks
            "props": props,             # properties from find_peaks
            "signal": data['mag_smooth'].to_numpy(),  # the signal we ran peaks on
            "df": data                  # full dataframe with raw + processed columns
        }
        return step_count, debug

    return step_count


if __name__ == "__main__":
    # sample file we swapped out for real data
    steps, dbg = count_steps('data/Class_Data/sensor_data/accelormeter_2026-09-10_19-11-09.csv', return_debug=True)

    print(f"Steps detected: {steps}")
    print(f"First few peak indices: {dbg['peaks'][:30]}")

    # VP edited
    plot_with_peaks(dbg['df'], mag_col='mag', peaks=dbg['peaks'])
    

