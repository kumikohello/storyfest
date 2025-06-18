# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last edited: June 17, 2025
# Description: This script analyzes the frequency of pupil data using Fourier Transform and spectral entropy.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime
from scipy.signal import welch

# ---------- Configuration ---------- #
EXP_TYPE = "encoding" # "encoding" or "recall"
SAMPLE_HZ = 1 #50
SUBJ_IDS = range(1001,1046) # keep range from 1001

# Paths
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/6_eventlocked/' + EXP_TYPE))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/9_frequency_analysis/' + EXP_TYPE))
#EVENTS_PATH = os.path.normpath(os.path.join(_THISDIR, '../../experiment/Storyfest_Event_Segmentation.xlsx'))
os.makedirs(SAVE_PATH, exist_ok=True)

if EXP_TYPE == "encoding":
    runs = ['run_1', 'run_2']
else:
    runs = [None]

# Story orders per group
GROUP_STORY_ORDER = {
    1: ['Pool Party', 'Sea Ice', 'Natalie Wood', 'Grandfather Clocks', 'Impatient Billionaire', 'Dont Look'],
    2: ['Dont Look', 'Pool Party', 'Grandfather Clocks', 'Impatient Billionaire', 'Natalie Wood', 'Sea Ice'],
    3: ['Sea Ice', 'Dont Look', 'Impatient Billionaire', 'Natalie Wood', 'Grandfather Clocks', 'Pool Party'],
}

# Valence per story (match sheet names!)
STORY_VALENCE = {
    'Pool Party': 'positive',
    'Sea Ice': 'neutral',
    'Natalie Wood': 'negative',
    "Impatient Billionaire": 'positive',
    "Grandfather Clocks": 'neutral',
    "Dont Look": 'negative'
}

# ---------- Processing ---------- #
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

        # Load pupil data
        pupil_file = os.path.join(current_dat, f"{sub}_{group_num}_event_aligned_{EXP_TYPE}.csv")
        if not os.path.exists(pupil_file):
            print(f"Missing pupil file for subject {sub}")
            continue
        pupil_data = pd.read_csv(pupil_file)
        pupil_array = np.array(pupil_data["pupilSize"])

                # Load Excel file with all story sheets
        event_file = EVENTS_PATH

        xl = pd.ExcelFile(event_file)
        story_order = GROUP_STORY_ORDER[group_num]

        event_rows = []

        # Determine which stories are in this run
        if EXP_TYPE == "encoding":
            if run == 'run_1':
                story_order_in_run = story_order[:3] 
            else:
                story_order_in_run = story_order[3:]
        else:
            story_order_in_run = story_order

        # Build list of (story_name, start_time) pairs with 2s gap between stories
        story_start_times = []
        current_time = 0

        for sheet_name in story_order_in_run:
            if sheet_name not in xl.sheet_names or sheet_name not in STORY_VALENCE:
                print(f"Skipping sheet: {sheet_name}")
                continue

            sheet = xl.parse(sheet_name)
            

        event_num = 1
        for sheet_name, timeslot_start_sec in story_start_times:
            valence = STORY_VALENCE[sheet_name]
            timeslot_start_sec = dict(story_start_times)[sheet_name]
            sheet = xl.parse(sheet_name)

            



# pupil_signal: 1D array of pupil diameter over time
fs = 20  # Sampling rate in Hz (change if different)

# Welch method gives you power spectral density
frequencies, power = welch(pupil_file, fs=fs, nperseg=256)

power_norm = power / np.sum(power)  # make it like a probability distribution

# Compute Spectral Entropy
spectral_entropy = -np.sum(power_norm * np.log2(power_norm + 1e-12))  # add small term to avoid log(0)
