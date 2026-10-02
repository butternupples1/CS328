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
    
    #TODO: The below is a basic scaffolding that you can modify/improve on based on your data.
    
    # combine x,y,z into a single magnitude signal
    data['mag'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    SR = 50
    b, a = butter(3, [0.5/(SR/2), 4.0/(SR/2)], btype='bandpass')

    # apply the filter to the magnitude signal

    # Victor added this
    magnitude = data['mag'].to_numpy()

    filtered = filtfilt(b, a, magnitude)

    # TODO: Apply a simple moving average to reduce noise (modify to apply different filter/parameters)
    data['mag_smooth'] = data['mag'].rolling(window=5, center=True).mean()

    # run a simple peak detector; these params are just placeholders -> feel free to change
    peaks, props = find_peaks(data['mag_smooth'], height=3, distance=13)

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
    # try it on the sample file
    steps, dbg = count_steps('data/Class_Data/sensor_data/accelormeter_2026-09-10_19-11-09.csv', return_debug=True)

    # 20 true steps for this sample and the placement is in their hand

    print(f"Steps detected: {steps}")
    print(f"First few peak indices: {dbg['peaks'][:30]}")

    # Victor added these
    # Plot full signal with peak markers
    # Pass the filtered column name created in solution.py
    plot_with_peaks(dbg['df'], mag_col='mag', peaks=dbg['peaks'])
    

