# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu), Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: June 8, 2025
# Description: The script applies lowpass of 4Hz and downsamples the clean (interpolated) pupil data to 50 Hz
# Downsampled via averaging (i.e., taking the mean of every N samples)

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
EXP_TYPE = "recall" # "encoding" or "recall"
SDSCORE = 2
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/3_interpolated/' + EXP_TYPE))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/4_downsampled/' + EXP_TYPE))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

if EXP_TYPE == "encoding":
    runs = ['run_1', 'run_2']
else:
    runs = [None]

SAMPLE_RATE_HZ = int(500) # Sampling frequency in Hz
SAMPLE_RATE_MS = 1/SAMPLE_RATE_HZ * 1000 # 500 Hz in ms (2 ms)

DOWNSAMPLE_RATE_HZ = int(50) # Downsample to 50 Hz
DOWNSAMPLE_RATE_MS = 1/DOWNSAMPLE_RATE_HZ * 1000 # 50 Hz in ms (20 ms)

SUBJ_IDS = range(1001,1046)

# ------------------ Filtering Function ------------------ #
def lowpass_filter(data, sample_rate, cutoff, order=3):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# ------------------- Main ------------------ #
for run in runs:
    # Set current paths
    if EXP_TYPE == "encoding":
        current_dat = os.path.join(DAT_PATH, run)
        current_save = os.path.join(SAVE_PATH, run)
    else:
        current_dat = DAT_PATH
        current_save = SAVE_PATH

    # Ensure output directory exists
    os.makedirs(current_save, exist_ok=True)

    for sub in SUBJ_IDS:
        group_num = (sub - 1000) % 3
        if group_num == 0:
            group_num = 3

        # Load clean pupil data
        input_file = os.path.join(current_dat, f"{sub}_{group_num}_{SDSCORE}SD_interpolated_{EXP_TYPE}.csv")
        if not os.path.exists(input_file):
            print(f"No Input File for Subject {sub}")
            continue
        dat = pd.read_csv(input_file)

        # Apply 4 Hz lowpass filter
        filtered_pupil = lowpass_filter(dat['pupilSize_clean'].values, SAMPLE_RATE_HZ, cutoff=4)

        # Replace original signal with filtered one
        dat['pupilSize_clean'] = filtered_pupil
        
        # Convert time to datetime and set as index
        dat['time_in_ms_corrected'] = pd.to_datetime(dat['time_in_ms_corrected'], unit='ms')
        dat.set_index('time_in_ms_corrected', inplace=True)
        
        # Resample to 50 Hz (20 ms intervals) and aggregate (e.g., take mean of each chunk)
        pupilSize_downsampled = dat['pupilSize_clean'].resample(f"{DOWNSAMPLE_RATE_MS}ms").mean()
        pupilSize_downsampled.reset_index(drop=True, inplace=True)
        
        # Create a new time column
        time_in_ms_downsampled = np.arange(0, len(pupilSize_downsampled) * DOWNSAMPLE_RATE_MS, DOWNSAMPLE_RATE_MS)
        
        # Create new dataframe with downsampled data
        df_downsampled = pd.DataFrame({'time_in_ms': time_in_ms_downsampled, 'pupilSize_clean': pupilSize_downsampled})
        
        # Check if the length is more or less the same across subjects
        print("Subject", sub, "; num samples: ", len(time_in_ms_downsampled))
        
        # Save downsampled data
        output_file = os.path.join(current_save, f"{sub}_{group_num}_{SDSCORE}SD_downsampled_{EXP_TYPE}.csv")
        df_downsampled.to_csv(output_file, index=False)
