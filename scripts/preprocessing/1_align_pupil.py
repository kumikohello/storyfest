# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: July 18, 2025
# Description:
#   Align pupil size time series to stimulus presentation
#   using pre-extracted timestamps from events.csv
#   Input: samples.csv, timestamps.csv
#   Output: run_1 and run_2 aligned pupil data

import numpy as np
import pandas as pd
import os

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/UChicago/CASNL/storyfest/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
EXP_TYPE = "encoding" 
CSV_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/2_csv'))
TIMESTAMP_PATH = os.path.join(_THISDIR, '../../data/timestamps')  
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/1_aligned', EXP_TYPE))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

SUBJ_IDS = range(1001, 1046)
SAMPLING_RATE = 500  # Hz

# ------------------ Define functions ------------------ # 
def align_run(pupil_df, start_ms, end_ms):
    timestamps = pupil_df['timestamp'].values
    pupil = pupil_df['pupil'].values

    start_idx = np.argmin(np.abs(timestamps - start_ms))
    end_idx = np.argmin(np.abs(timestamps - end_ms))

    aligned_time = timestamps[start_idx:end_idx]
    aligned_pupil = pupil[start_idx:end_idx]

    # Replace invalid 0 pupil values with NaN
    aligned_pupil[aligned_pupil == 0] = np.nan

    time_corrected = aligned_time - aligned_time[0]

    return pd.DataFrame({
        'pupilSize': aligned_pupil,
        'time_in_ms': aligned_time,
        'time_corrected': time_corrected
    })

# ------------------- Main ------------------ #
if EXP_TYPE == 'encoding':
    for sub in SUBJ_IDS:
        try:
            pupil_path = os.path.join(CSV_PATH, EXP_TYPE, str(sub), "samples.csv")
            ts_path = os.path.join(TIMESTAMP_PATH, f"{sub}_storyfest_timestamps.csv")

            pupil_df = pd.read_csv(pupil_path)
            ts_df = pd.read_csv(ts_path)

            # Extract timestamps
            r1_start = ts_df["story1_start"].iloc[0]
            r1_end = ts_df["story1_end"].iloc[0]
            r2_start = ts_df["story2_start"].iloc[0]
            r2_end = ts_df["story2_end"].iloc[0]

            # Align each run
            run1_df = align_run(pupil_df, r1_start, r1_end)
            run2_df = align_run(pupil_df, r2_start, r2_end)

            # Save
            out_dir_1 = os.path.join(SAVE_PATH, "run_1")
            out_dir_2 = os.path.join(SAVE_PATH, "run_2")
            os.makedirs(out_dir_1, exist_ok=True)
            os.makedirs(out_dir_2, exist_ok=True)

            run1_df.to_csv(os.path.join(out_dir_1, f"{sub}_aligned.csv"), index=False)
            run2_df.to_csv(os.path.join(out_dir_2, f"{sub}_aligned.csv"), index=False)

            print(f"Saved aligned runs for {sub}")

        except Exception as e:
            print(f"Skipped {sub}: {e}")