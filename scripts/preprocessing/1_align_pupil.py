# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu), Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: June 8, 2025
# Description: The script aligns pupil data to stimulus presentation and excludes non-encoding data

import numpy as np
import pandas as pd
import os
import mat73 # to load .mat files in MATLAB v7.3

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
EXP_TYPE = "encoding" # "encoding" or "recall"
MAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/2_mat/', EXP_TYPE))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/1_aligned/', EXP_TYPE))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

SUBJ_IDS = range(1001,1046)
SAMPLING_RATE = 500 # Hz
# PUPIL_INFO = Area (Area or Diameter)

# ------------------ Define functions ------------------ # 
def fetch_mat(mat_path, sub_id):
    """
    Grabs .mat file for a given subject and saves each struct as an array.
    
    Samples (1x1 Struct): contains time, posX, posY, pupilSize, etc.
    Events (1x1 Struct): contains Messages (another Struct), Sblink, Eblink, etc.
        Sblink: time of the start of the blink
        Eblink: time of the start and end of the blink, and blink duration
        Detailed description of the variables: http://sr-research.jp/support/EyeLink%201000%20User%20Manual%201.5.0.pdf
    
    """
    mat = mat73.loadmat(os.path.join(mat_path, str(sub_id) + "_" + EXP_TYPE + "_ET.mat"))
    samples = mat['Samples']
    events = mat['Events']
        
    return samples, events

# ------------------- Main ------------------ #
for sub in SUBJ_IDS:
    
    # Load .mat data
    samples, events = fetch_mat(MAT_PATH, sub)

    # Time stamp of samples
    samples_time = samples['time'] # in milliseconds; samples_time[1] - samples_time[0] = 2 ms
    
    # Pupil size during the entire timecourse
    samples_pupilSize = samples['pupilSize']

    # Event messages
    events_messages_info = events['Messages']['info']
    events_messages_time = events['Messages']['time']

    # Helper function to align pupil data
    def align_pupil_data(start_time, end_time):
        # Find the closest available sample index
        pupil_start_idx = np.searchsorted(samples_time, start_time, side="left")
        pupil_end_idx = np.searchsorted(samples_time, end_time, side="right") - 1

        # Ensure indices are within valid bounds
        pupil_start_idx = max(0, min(pupil_start_idx, len(samples_time) - 1))
        pupil_end_idx = max(0, min(pupil_end_idx, len(samples_time) - 1))

        # Handle cases where start or end times are out of range
        if pupil_start_idx == 0 and start_time < samples_time[0]:
            print(f"Warning: Adjusted start_time {start_time} is before the first sample for subject {sub}. Using first available sample.")

        if pupil_end_idx == len(samples_time) - 1 and end_time > samples_time[-1]:
            print(f"Warning: Adjusted end_time {end_time} is after the last sample for subject {sub}. Using last available sample.")
        
        # New array of samples during stimulus presentation
        pupilSize_encoding = samples_pupilSize[pupil_start_idx:pupil_end_idx]
        
        # Corresponding time stamp of the new array
        encoding_time = samples_time[pupil_start_idx:pupil_end_idx]
        encoding_time_corrected = encoding_time - encoding_time[0]

        return pupilSize_encoding, encoding_time, encoding_time_corrected

    if EXP_TYPE == "encoding":

        # Ensure the output directories exist
        run_1_path = os.path.join(SAVE_PATH, "run_1")
        run_2_path = os.path.join(SAVE_PATH, "run_2")

        os.makedirs(run_1_path, exist_ok=True)
        os.makedirs(run_2_path, exist_ok=True)

        # Index and timestamp of STORY_START and STORY_END
        run_1_start_idx = events_messages_info.index('ALL_STORY_START')
        run_1_end_idx = events_messages_info.index('STORY_1_END')
        run_2_start_idx = events_messages_info.index('STORY_2_START')
        run_2_end_idx = events_messages_info.index('STORY_2_END')

        run_1_start_time = events_messages_time[run_1_start_idx]
        run_1_end_time = events_messages_time[run_1_end_idx]
        run_2_start_time = events_messages_time[run_2_start_idx]
        run_2_end_time = events_messages_time[run_2_end_idx]

        group_num = (sub - 1000) % 3
        if group_num == 0:
            group_num = 3

        # Process run_1
        if run_1_start_time is not None and run_1_end_time is not None:
            pupilSize_run_1, encoding_time_run_1, encoding_time_corrected_run_1 = align_pupil_data(run_1_start_time, run_1_end_time)

            if pupilSize_run_1 is not None:
                filename = os.path.join(run_1_path, f"{sub}_{group_num}_aligned_encoding_run_1_ET.csv")
                pd.DataFrame({'pupilSize': pupilSize_run_1, 
                            'time_in_ms': encoding_time_run_1, 
                            'time_in_ms_corrected': encoding_time_corrected_run_1}).to_csv(filename, index=False)
                print(f"Saved run_1 data for subject {sub}.")

        # Process run_2
        if run_2_start_time is not None and run_2_end_time is not None:
            pupilSize_run_2, encoding_time_run_2, encoding_time_corrected_run_2 = align_pupil_data(run_2_start_time, run_2_end_time)

            if pupilSize_run_2 is not None:
                filename = os.path.join(run_2_path, f"{sub}_{group_num}_aligned_encoding_run_2_ET.csv")
                pd.DataFrame({'pupilSize': pupilSize_run_2, 
                            'time_in_ms': encoding_time_run_2, 
                            'time_in_ms_corrected': encoding_time_corrected_run_2}).to_csv(filename, index=False)
                print(f"Saved run_2 data for subject {sub}.")

    else: # if EXP_TYPE == "recall"
        
        # Index and timestamp of STORY_START and STORY_END
        story_start_idx = events_messages_info.index('REC_START')
        story_end_idx = events_messages_info.index('REC_END')

        story_start_time = events_messages_time[story_start_idx]
        story_end_time = events_messages_time[story_end_idx]

        pupilSize_encoding, encoding_time, encoding_time_corrected = align_pupil_data(story_start_time, story_end_time)
        
        print(f"Saving subject {sub} ... ")
        
        group_num = (sub - 1000) % 3
        if group_num == 0:
            group_num = 3

        filename = os.path.join(SAVE_PATH, str(sub) + "_" + str(group_num) + "_aligned_" + EXP_TYPE + "_ET.csv")
        pd.DataFrame({'pupilSize': pupilSize_encoding, 'time_in_ms': encoding_time, 'time_in_ms_corrected': encoding_time_corrected}).to_csv(filename, index=False)

# Encoding: no input file for pts 1033
# Recall: no input file for pts 1008, 1010, 1022, 1034
