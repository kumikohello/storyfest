# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: July 21, 2025
# Description: The script interpolates over blinks (detected by the Eyelink blink detection algorithm)

# Steps:
# 1. Load aligned (and validated) pupil data
# 2. Identify blinks in the data within the range of 0.1-0.5s
# 3. Interpolate over blinks with ±150ms buffer (if possible), fallback to ±1 or edge fill
# 4. Detect additional missing segments (NaNs) ≤ 1000ms not classified as blinks
# 5. Interpolate over missing data
# 6. Calculate % of long missing segments (>1s) remaining post-interpolation

import numpy as np
import pandas as pd
import os

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/UChicago/CASNL/storyfest/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
EXP_TYPE = "encoding" # "encoding" or "recall"

# Standard score cutoffs
SDSCORE = 3 

DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/2_valid_pts/' + EXP_TYPE, str(SDSCORE) + "SD"))
RAW_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/2_csv/' + EXP_TYPE))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/3_interpolated/' + EXP_TYPE))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

if EXP_TYPE == "encoding":
    runs = ['run_1','run_2']
else:
    runs = [None]

SUBJ_IDS = range(1001,1046)
BLINKMIN = 100
BLINKMAX = 500
WINSIZE = 1000 ## cap for ms needed for interpolation
SAMPLE_RATE = int(500) # Sampling frequency/rate(Hz)

# ------------------ Define functions ------------------ # 
def interpolate_nans(start_idx, end_idx, pupilSize, buffer=75):
    """
    Linearly interpolates pupil size data between start and end index (with optional buffer).
    
    Params:
    - start_idx (int): start index of blink (inclusive)
    - end_idx (int): end index of blink (exclusive)
    - pupilSize (1D numpy array): original pupil size time series (with NaNs)
    - buffer (int): number of samples to buffer on both sides of blink
    
    Returns:
    - pupilSize (1D numpy array): modified array with interpolated values    
    """
    n = len(pupilSize)

    # Preferred buffer-based interpolation
    s_minus_buffer = start_idx - buffer
    e_plus_buffer = end_idx + buffer + 1

    # Safer fallback range
    s_minus1 = start_idx - 1   
    e_plus1 = end_idx + 1 + 1 

    # Case 1: buffer interpolation (most preferred)
    if (s_minus_buffer >= 0) and (e_plus_buffer < n):
        left_val = pupilSize[s_minus_buffer]
        right_val = pupilSize[e_plus_buffer]
        if not np.isnan(left_val) and not np.isnan(right_val):
            interp_vals = np.linspace(left_val, right_val, e_plus_buffer - s_minus_buffer)[1:-1]
            return interp_vals, s_minus_buffer + 1, e_plus_buffer - 1  # exclude endpoints

    # Case 2: ±1 interpolation (fallback)
    elif (s_minus1 >= 0) and (e_plus1 < n):
        left_val = pupilSize[s_minus1]
        right_val = pupilSize[e_plus1]
        if not np.isnan(left_val) and not np.isnan(right_val):
            interp_vals = np.linspace(left_val, right_val, e_plus1 - s_minus1)[1:-1]
            return interp_vals, s_minus1 + 1, e_plus1 - 1

    # Case 3: tail fallback — only one side has valid value
    elif s_minus1 >= 0:
        left_val = pupilSize[s_minus1]
        if not np.isnan(left_val):
            return np.full(end_idx - start_idx, left_val), start_idx, end_idx
    elif end_idx < n:
        right_val = pupilSize[end_idx]
        if not np.isnan(right_val):
            return np.full(end_idx - start_idx, right_val), start_idx, end_idx
    
    # Case 4: unable to interpolate, fill with NaN
    return np.full(end_idx - start_idx, np.nan), start_idx, end_idx

def id_nans(arr):
    """
    Identify segments in the data where there are NaNs.
    Each row in the output contains the start and end index
    of a consecutive NaN segment.

    Params:
    - arr (numpy array): pupil size data

    Returns:
    - ranges (numpy array): indices of consecutive NaNs
    """
    is_nan = np.concatenate(([0], np.isnan(arr).astype(int), [0]))
    absdiff = np.abs(np.diff(is_nan))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def detect_missing_data_type(arr, blink_min_len = 50, blink_max_len = 250, max_len = 500):
    """
    Identify if the segments in data can be considered as blinks or general missing data.
    
    Params:
    - arr (numpy array): raw pupil size data (e.g., with NaNs or zeros)
    - blink_min_len (int): minimum length of blink (in samples)
    - blink_max_len (int): maximum length of blink (in samples)
    - max_len (int): maximum length of data to be interpolated (e.g., 1s = 500 at 500Hz)
    
    Returns:
    - blinks (list of tuples): [(start, end), ...] indices of blink-like gaps
    - missing_data (list of tuples): indices of short missing data gaps (non-blinks)

    """
    nan_runs = id_nans(arr)
    blinks = []
    missing_data = []

    for start, end in nan_runs:
        duration = end - start
        if blink_min_len <= duration <= blink_max_len:
            blinks.append((start,end))
        elif duration < blink_min_len or (blink_max_len < duration <= max_len):
            missing_data.append((start, end))
    return blinks, missing_data

# ------------------- Main ------------------ #
for run in runs:
    if EXP_TYPE == "encoding":
        current_dat_path = os.path.join(DAT_PATH, run)
        current_save_path = os.path.join(SAVE_PATH, run)
    else:
        current_dat_path = DAT_PATH
        current_save_path = SAVE_PATH

    os.makedirs(current_save_path, exist_ok=True)

    for sub in SUBJ_IDS:
        group_num = (sub - 1000) % 3
        if group_num == 0:
            group_num = 3
        input_file = os.path.join(current_dat_path, f"{sub}_valid_{run}_{SDSCORE}SD.csv")
        raw_pupil_file = os.path.join(RAW_PATH, str(sub), "samples.csv")
        if not os.path.exists(input_file) or not os.path.exists(raw_pupil_file):
            print(f"No Input File for Participant {sub}")
            continue
        
        raw_df = pd.read_csv(raw_pupil_file)
        dat = pd.read_csv(input_file)
        
        raw_pupil = raw_df['pupil'].replace(0, np.nan).values
        raw_ts = raw_df['timestamp'].values

        blinks, missing_data = detect_missing_data_type(raw_pupil)
        # Apply blink interpolation
        pupil_noblinks = raw_pupil.copy()
        interpolated_mask = np.zeros(len(pupil_noblinks), dtype=bool)
        for s, e in blinks:
            interp_vals, start, end = interpolate_nans(s, e, raw_pupil, buffer=75)
            pupil_noblinks[start:end] = interp_vals
            interpolated_mask[start:end] = True


        # Apply missing data interpolation
        pupil_interp_all = pupil_noblinks.copy()
        for s, e in missing_data:
            interp_vals, start, end = interpolate_nans(s, e, pupil_noblinks, buffer=1)
            if interpolated_mask[start:end].any():
                # print(f"[SKIP] [missing data] Overlap at {s}-{e} mapped to {start}-{end}")
                continue
            pupil_interp_all[start:end] = interp_vals
            interpolated_mask[start:end] = True
            
        # Align interpolated data back to validated timestamps
        raw_df['pupil_interp_all'] = pupil_interp_all
        dat = dat.rename(columns={'time_in_ms': 'timestamp'})

        dat = dat.merge(
            raw_df[['timestamp', 'pupil_interp_all']], 
            on='timestamp', 
            how='left'
        )

        pupilSize_clean = dat['pupil_interp_all'].replace(0, np.nan)

        # Calculate the percentage of data with missing data longer than 1 seconds
        prop_missing_data = pupilSize_clean.isna().sum() / len(pupilSize_clean) * 100
        print("Subject", sub, "in", run, "has", prop_missing_data, "% missing data points")
        
        # Save clean pupil data
        output_file = os.path.join(current_save_path, f"{sub}_{SDSCORE}SD_interpolated.csv")
        dat.to_csv(output_file, index=False)
        