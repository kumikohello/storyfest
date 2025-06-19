# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: June 11, 2025
# Description: This script downsamples pupil data to stories.

# Steps:
# 1. Load downsampled (1 Hz; sampled every 1 s) pupil data
# 2. Compute event-level average pupil size

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime

# ---------- Configuration ---------- #
EXP_TYPE = "encoding" # "encoding" or "recall"
SAMPLE_HZ = 1 #50
SUBJ_IDS = range(1001,1046) # keep range from 1001
FILTER_TYPE = "lowpass"  # lowpass or bandpass
LOWCUT_HZ = None#0.01 # Only used if FILTER_TYPE is "bandpass"
HIGHCUT_HZ = 0.2 #0.3 # Used in both "lowpass" and "bandpass"

# Paths
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/5_timelocked/' + EXP_TYPE))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/6_eventlocked/' + EXP_TYPE))
EVENTS_PATH = os.path.normpath(os.path.join(_THISDIR, '../../experiment/Storyfest_Event_Segmentation.xlsx'))
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


def time_str_to_sec(t):
    if isinstance(t, datetime.time):
        return t.hour * 60 + t.minute
    return np.nan


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
        pupil_file = os.path.join(current_dat, f"{sub}_{group_num}_{run}_{FILTER_TYPE}_2SD_downsample_to_sec_{EXP_TYPE}.csv")
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
            end_times = sheet['Segment_end_time'].dropna().apply(time_str_to_sec)
            if end_times.empty:
                print(f"No valid end times for story: {sheet_name}")
                continue
            max_end = end_times.max()

            story_start_times.append((sheet_name, current_time))
            current_time += max_end + 2  # Add 2s gap after each story

        event_num = 1
        for sheet_name, timeslot_start_sec in story_start_times:
            valence = STORY_VALENCE[sheet_name]
            timeslot_start_sec = dict(story_start_times)[sheet_name]
            sheet = xl.parse(sheet_name)

            for _, row in sheet.iterrows():
                start = time_str_to_sec(row['Segment_start_time'])
                end = time_str_to_sec(row['Segment_end_time'])
                if pd.isna(start) or pd.isna(end):
                    continue

                transcript = row['Transcript']
                absolute_start = timeslot_start_sec + start
                absolute_end = timeslot_start_sec + end
                start_idx = int(absolute_start * SAMPLE_HZ)
                end_idx = int(absolute_end * SAMPLE_HZ)
                if start_idx >= len(pupil_array):
                    continue
                end_idx = min(end_idx, len(pupil_array))

                segment = pupil_array[start_idx:end_idx]
                if len(segment) == 0:
                    continue

                # Optional artifact removal
                mean = np.mean(segment)
                std = np.std(segment)
                clean_segment = segment[(segment > mean - 2 * std) & (segment < mean + 2 * std)]

                mean_pupil = np.nan if len(clean_segment) == 0 else np.mean(clean_segment)
                duration = absolute_end - absolute_start

                event_rows.append({
                    "subject": sub,
                    "group": group_num,
                    "story": sheet_name,
                    "event_num": event_num,
                    "valence": valence,
                    "event_start_sec": absolute_start,
                    "event_end_sec": absolute_end,
                    "event_duration_sec": duration,
                    "transcript": transcript,
                    "mean_pupil_size": mean_pupil
                })
                event_num += 1

        df_out = pd.DataFrame(event_rows)

        df_out['z_pupil'] = stats.zscore(df_out['mean_pupil_size'], nan_policy='omit')

        # Save per subject
        out_csv = os.path.join(current_save, f"{sub}_{group_num}_{run}_{FILTER_TYPE}_event_aligned.csv")
        df_out.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        # ---------- Plot ---------- #
        plt.figure(figsize=(12, 4))
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}

        for val in df_out['valence'].unique():
            val_df = df_out[df_out['valence'] == val]
            plt.scatter(val_df['event_start_sec'], val_df['z_pupil'], 
                        label=val, color=colors[val], alpha=0.7)

        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.xlabel('Event Start Time (sec)')
        plt.ylabel('Standardized Pupil Size (z)')
        plt.title(f'Event-Aligned Pupil Data: Subject {sub}')
        plt.legend(title='Valence')
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(current_save, f"{sub}_{group_num}_{run}_{FILTER_TYPE}_event_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot: {plot_path}")

# ---------- Group-Level Average Plots (Encoding Only, Per Run) ---------- #
if EXP_TYPE == "encoding":
    from collections import defaultdict

    for run in runs:
        current_save = os.path.join(SAVE_PATH, run)
        group_event_data = defaultdict(list)

        for sub in SUBJ_IDS:
            group_num = (sub - 1000) % 3
            if group_num == 0:
                group_num = 3
            file_path = os.path.join(current_save, f"{sub}_{group_num}_event_aligned_{run}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df["group_num"] = group_num
                group_event_data[group_num].append(df)

        for group_num, dfs in group_event_data.items():
            print(f"Plotting group {group_num} with {len(dfs)} subjects.")
            group_df = pd.concat(dfs, ignore_index=True)

            # Sort by time for clarity
            group_df = group_df.sort_values("event_start_sec")

            # Create plot
            plt.figure(figsize=(12, 4))
            colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}

            for val in group_df['valence'].unique():
                val_df = group_df[group_df['valence'] == val]
                plt.scatter(val_df['event_start_sec'], val_df['z_pupil'], 
                            label=val, color=colors[val], alpha=0.5)

            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.xlabel('Event Start Time (sec)')
            plt.ylabel('Standardized Pupil Size (z)')
            plt.title(f'Group {group_num} Event-Aligned Pupil Data ({run})')
            plt.legend(title='Valence')
            plt.tight_layout()

            group_plot_path = os.path.join(SAVE_PATH, run, f"{run}_{group_num}_event_plot.png")
            plt.savefig(group_plot_path, dpi=300)
            plt.close()
            print(f"Saved group-level plot: {group_plot_path}")
