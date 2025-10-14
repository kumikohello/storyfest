# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: July 6, 2025
# Description: This script downsamples pupil data to stories.

# Steps:
# 1. Load downsampled (50 Hz; sampled every 20 ms) pupil data
# 2. Compute story-level average pupil size

import os
import pandas as pd
import numpy as np

# ---------- Configuration ---------- #
EXP_TYPE = "encoding"
SUBJ_IDS = range(1001, 1046)

# Paths
os.chdir('/Users/UChicago/CASNL/storyfest/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/5_standardized/{EXP_TYPE}'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/6_storylocked/{EXP_TYPE}'))
EVENTS_PATH = os.path.normpath(os.path.join(_THISDIR, '../../experiment/Storyfest_Event_Segmentation.xlsx'))
os.makedirs(SAVE_PATH, exist_ok=True)

runs = ['run_1', 'run_2'] if EXP_TYPE == "encoding" else [None]

GROUP_STORY_ORDER = {
    1: ['Pool Party', 'Sea Ice', 'Natalie Wood', 'Grandfather Clocks', 'Impatient Billionaire', 'Dont Look'],
    2: ['Dont Look', 'Pool Party', 'Grandfather Clocks', 'Impatient Billionaire', 'Natalie Wood', 'Sea Ice'],
    3: ['Sea Ice', 'Dont Look', 'Impatient Billionaire', 'Natalie Wood', 'Grandfather Clocks', 'Pool Party'],
}

STORY_LENGTH_MS = {
    'Pool Party': 374000,
    'Sea Ice': 381000,
    'Natalie Wood': 815000,
    'Impatient Billionaire': 330000,
    'Grandfather Clocks': 535000,
    'Dont Look': 710000
}

STORY_VALENCE = {
    'Pool Party': 'positive',
    'Sea Ice': 'neutral',
    'Natalie Wood': 'negative',
    "Impatient Billionaire": 'positive',
    "Grandfather Clocks": 'neutral',
    "Dont Look": 'negative'
}

# ---------- Processing ---------- #
for sub in SUBJ_IDS:
    group_num = (sub - 1000) % 3
    if group_num == 0:
        group_num = 3
    stories = GROUP_STORY_ORDER[group_num]
    subject_rows = []

    for run in runs:
        current_dat = os.path.join(DAT_PATH, run) if run else DAT_PATH
        pupil_file = os.path.join(current_dat, f"{sub}_{group_num}_standardized.csv")
        
        if not os.path.exists(pupil_file):
            print(f"File not found: {pupil_file}")
            continue

        df = pd.read_csv(pupil_file)
        pupil_array = df['pupilSize']
        time_array = df['time_in_ms']

        run_stories = stories[:3] if run == 'run_1' else stories[3:]
        start = 0
        for story in run_stories:
            duration = STORY_LENGTH_MS[story]
            end = start + duration

            mask = (df['time_in_ms'] >= start) & (df['time_in_ms'] <= end)
            segment = pupil_array[mask]
            valence = STORY_VALENCE[story]

            if len(segment) == 0:
                mean = np.nan
            else:
                seg_mean = segment.mean()
                seg_std = segment.std()
                # filtered = segment[(segment >= seg_mean - 2*seg_std) & (segment <= seg_mean + 2*seg_std)]
                
                # mean = np.nan if len(filtered) == 0 else filtered.mean()
                mean = segment.mean()

            subject_rows.append({
                    "subject": sub,
                    "group": group_num,
                    "story": story,
                    "valence": valence,
                    "mean_pupil_size": mean,  # still included for clarity
                    "z_pupil": mean,          # same as mean since input is z-scored
                    "run": run
                })
            # subject_rows.append({
            #     "story": story,
            #     "z_pupil": mean,
            # })

            start = end + 2000  # add 2s pause

    if subject_rows:
        df_sub = pd.DataFrame(subject_rows)
        df_sub['z_pupil'] = df_sub['z_pupil']
        df_sub.to_csv(os.path.join(SAVE_PATH, f"{sub}_{group_num}_story_aligned.csv"), index=False)
        print(f"Saved story-aligned file for subject {sub}")
