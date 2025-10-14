# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: June 29, 2025
# Description: Extract key event timestamps from EyeLink event.csv files for each participant
#              (e.g., story start/end, recording markers) in the "encoding" phase.


import numpy as np
import pandas as pd
import os
import math

# ------------------ Hardcoded parameters ------------------ 
os.chdir('/Users/UChicago/CASNL/storyfest/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
EXP_TYPE = "encoding" # only "encoding"
CSV_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/2_csv'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/timestamps'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

SUBJ_IDS = range(1001, 1046)
SAMPLING_RATE = 500 # Hz

# ------------------ Define functions ------------------ # 
def get_timestamp(events_df, keyword):
    match = events_df[events_df['detail'].str.contains(keyword, na=False)]
    if not match.empty:
        return float(match.iloc[0]['timestamp'])
    return np.nan

def fetch_timestamps(csv_path, sub_id):
    events = pd.read_csv(os.path.join(csv_path, 'events.csv'), names=['event_type', 'timestamp', 'detail'], encoding='utf-8', engine='python')
    
    return pd.DataFrame([{
        'sub_id': sub_id,
        'recording_start': get_timestamp(events, 'ALL_STORY_START'),
        'story1_start': get_timestamp(events, 'START'),
        'story1_end': get_timestamp(events, 'STORY_1_END'),
        'story2_start': get_timestamp(events, 'STORY_2_START'),
        'story2_end': get_timestamp(events, 'STORY_2_END'),
        'recording_end': get_timestamp(events, 'END')
    }])


# ------------------- Main ------------------ #
for sub in SUBJ_IDS:
    csv_path = os.path.join(CSV_PATH, EXP_TYPE, str(sub))
    if os.path.exists(csv_path):
        timestamp_df = fetch_timestamps(csv_path, sub)
        if timestamp_df is not None:
            out_path = os.path.join(SAVE_PATH, f"{sub}_storyfest_timestamps.csv")
            timestamp_df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")
    else:
        print(f"Skipped {sub}: folder not found at {csv_path}")