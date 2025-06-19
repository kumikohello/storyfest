# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: June 13, 2025
# Description: This script downsamples pupil data to stories.

# Steps:
# 1. Load downsampled (1 Hz; sampled every 1 s) pupil data
# 2. Compute story-level average pupil size

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime
from collections import defaultdict

# ---------- Configuration ---------- #
EXP_TYPE = "encoding"
SUBJ_IDS = range(1001, 1046)
FILTER_TYPE = "lowpass"  # lowpass or bandpass
LOWCUT_HZ = None # Only used if FILTER_TYPE is "bandpass"
HIGHCUT_HZ = 0.2 # Used in both "lowpass" and "bandpass"

# Paths
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/5_timelocked/{EXP_TYPE}'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/6_storylocked/{EXP_TYPE}'))
EVENTS_PATH = os.path.normpath(os.path.join(_THISDIR, '../../experiment/Storyfest_Event_Segmentation.xlsx'))
os.makedirs(SAVE_PATH, exist_ok=True)

runs = ['run_1', 'run_2'] if EXP_TYPE == "encoding" else [None]

GROUP_STORY_ORDER = {
    1: ['Pool Party', 'Sea Ice', 'Natalie Wood', 'Grandfather Clocks', 'Impatient Billionaire', 'Dont Look'],
    2: ['Dont Look', 'Pool Party', 'Grandfather Clocks', 'Impatient Billionaire', 'Natalie Wood', 'Sea Ice'],
    3: ['Sea Ice', 'Dont Look', 'Impatient Billionaire', 'Natalie Wood', 'Grandfather Clocks', 'Pool Party'],
}

STORY_VALENCE = {
    'Pool Party': 'positive',
    'Sea Ice': 'neutral',
    'Natalie Wood': 'negative',
    'Impatient Billionaire': 'positive',
    'Grandfather Clocks': 'neutral',
    'Dont Look': 'negative'
}

def time_str_to_sec(t):
    if isinstance(t, datetime.time):
        return t.hour * 60 + t.minute
    return np.nan

# ---------- Processing ---------- #
all_story_segments = defaultdict(list)

for run in runs:
    current_dat = os.path.join(DAT_PATH, run) if run else DAT_PATH
    current_save = os.path.join(SAVE_PATH, run) if run else SAVE_PATH
    os.makedirs(current_save, exist_ok=True)

    for sub in SUBJ_IDS:
        group_num = (sub - 1000) % 3
        if group_num == 0:
            group_num = 3
        pupil_file = os.path.join(current_dat, f"{sub}_{group_num}_{run}_{FILTER_TYPE}_2SD_downsample_to_sec_{EXP_TYPE}.csv")
        if not os.path.exists(pupil_file):
            print(f"Missing pupil file for subject {sub}")
            continue

        pupil_data = pd.read_csv(pupil_file)
        pupil_array = np.array(pupil_data["pupilSize"])

        xl = pd.ExcelFile(EVENTS_PATH)
        story_order = GROUP_STORY_ORDER[group_num]
        story_order_in_run = story_order[:3] if run == 'run_1' else story_order[3:] if run else story_order

        current_time = 0
        subject_rows = []

        for story in story_order_in_run:
            if story not in xl.sheet_names or story not in STORY_VALENCE:
                continue

            sheet = xl.parse(story)
            end_times = sheet['Segment_end_time'].dropna().apply(time_str_to_sec)
            if end_times.empty:
                continue

            max_end = end_times.max()
            story_start_sec = current_time
            story_end_sec = story_start_sec + max_end

            start_idx = int(story_start_sec)
            end_idx = min(int(story_end_sec), len(pupil_array))
            segment = pupil_array[start_idx:end_idx]
            if len(segment) == 0:
                continue

            mean = np.mean(segment)
            std = np.std(segment)
            clean_segment = segment[(segment > mean - 2 * std) & (segment < mean + 2 * std)]
            mean_pupil = np.nan if len(clean_segment) == 0 else np.mean(clean_segment)

            subject_rows.append({
                "subject": sub,
                "story": story,
                "order": story_order.index(story) + 1,
                "valence": STORY_VALENCE[story],
                "story_start_sec": story_start_sec,
                "story_end_sec": story_end_sec,
                "story_duration_sec": story_end_sec - story_start_sec,
                "mean_pupil_size": mean_pupil
            })

            z_segment = stats.zscore(segment, nan_policy='omit')
            all_story_segments[story].append(z_segment)
            current_time += max_end + 2 # 2 second pause between each story

        if subject_rows:
            df_sub = pd.DataFrame(subject_rows)
            df_sub['z_pupil'] = stats.zscore(df_sub['mean_pupil_size'], nan_policy='omit')
            df_sub.to_csv(os.path.join(current_save, f"{sub}_{group_num}_{run}_{FILTER_TYPE}_story_aligned.csv"), index=False)

            # Plot per subject
            plt.figure(figsize=(10, 5))
            colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            for valence in df_sub['valence'].unique():
                val_df = df_sub[df_sub['valence'] == valence]
                plt.plot(val_df['story_start_sec'], val_df['z_pupil'], 'o-', label=valence, color=colors[valence])
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.xlabel("Story Start Time (sec)")
            plt.ylabel("Standardized Pupil Size (z)")
            plt.title(f"Story-Level Pupil Size: Subject {sub}")
            plt.legend(title="Valence")
            plt.tight_layout()
            plot_path = os.path.join(current_save, f"{sub}_{group_num}_{run}_{FILTER_TYPE}_story_plot.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved: {plot_path}")

# ---------- Group-Level Plotting ---------- #
valence_bins = {'negative': [], 'neutral': [], 'positive': []}
for story, valence in STORY_VALENCE.items():
    if story in all_story_segments:
        valence_bins[valence].append(story)

valence_labels = ['negative', 'neutral', 'positive']
plot_positions = {
    0: [0, 3],
    1: [1, 4],
    2: [2, 5],
}

plt.figure(figsize=(18, 10))
for col_idx, valence in enumerate(valence_labels):
    stories = valence_bins[valence]
    for row_idx, story in enumerate(stories):
        pos = plot_positions[col_idx][row_idx]
        segments = all_story_segments[story]
        max_len = max(len(seg) for seg in segments)
        padded = np.full((len(segments), max_len), np.nan)
        for j, seg in enumerate(segments):
            padded[j, :len(seg)] = seg
        avg_course = np.nanmean(padded, axis=0)

        ax = plt.subplot(2, 3, pos + 1)
        ax.plot(avg_course, color='black', linewidth=2)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(story)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Standardized Pupil Size (z)")
        ax.set_ylim(-1.0, 1.5)
        if row_idx == 0:
            ax.annotate(valence, xy=(0.5, 1.2), xycoords='axes fraction', ha='center', fontsize=14, fontweight='bold')

plt.suptitle(f"Average Pupil Time Courses by Story Sorted by Valence {(FILTER_TYPE, LOWCUT_HZ, HIGHCUT_HZ)}", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plot_name = f"all_stories_avg_time_courses_sorted_by_valence_{(FILTER_TYPE, LOWCUT_HZ, HIGHCUT_HZ)}.png"
plt.savefig(os.path.join(SAVE_PATH, plot_name), dpi=300)
plt.close()
print(f"Saved combined 6-story plot: {plot_name}")
