import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict

# ---------- Configuration ---------- #
EXP_TYPE = "encoding"  # "encoding" or "recall"
SUBJ_IDS = range(1001, 1046)
FILTER_TYPE = "bandpass"  # "lowpass" or "bandpass"
SAMPLE_HZ = 1

# Paths
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/5_timelocked/{EXP_TYPE}'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/9_FFT_story_from_sec/{EXP_TYPE}'))
EVENTS_PATH = os.path.normpath(os.path.join(_THISDIR, '../../experiment/Storyfest_Event_Segmentation.xlsx'))
os.makedirs(SAVE_PATH, exist_ok=True)

# Runs
if EXP_TYPE == "encoding":
    runs = ['run_1', 'run_2']
else:
    runs = [None]

# Story order and valence
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

# Storage for group-level FFT
all_story_power = defaultdict(list)
story_freqs = {}

# ------------------ Define functions ------------------ # 
def time_str_to_sec(t):
    """
    Convert a time string MM:SS:ms or an Excel datetime.time to seconds (float).
    """
    if isinstance(t, datetime.time):
        return t.hour * 60 + t.minute
    return np.nan

def compute_fft(signal):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/SAMPLE_HZ)
    fft_vals = np.fft.rfft(signal)
    power = np.abs(fft_vals)**2 / n
    return freqs, power

# ------------------ Processing ------------------ # 
# Load event segmentation excel
xl = pd.ExcelFile(EVENTS_PATH)

# Process each subject and run
for run in runs:
    # Define directories
    current_dat = os.path.join(DAT_PATH, run) if run else DAT_PATH
    current_save = os.path.join(SAVE_PATH, run) if run else SAVE_PATH
    os.makedirs(current_save, exist_ok=True)

    for sub in SUBJ_IDS:
        # Compute group number
        group_num = (sub - 1000) % 3
        if group_num == 0:
            group_num = 3

        # Load 1Hz pupil data
        file_name = f"{sub}_{group_num}_{run}_{FILTER_TYPE}_2SD_downsample_to_sec_{EXP_TYPE}.csv"
        pupil_file = os.path.join(current_dat, file_name)
        if not os.path.exists(pupil_file):
            print(f"Missing pupil file for subject {sub}: {pupil_file}")
            continue

        df = pd.read_csv(pupil_file)
        pupil_array = df['pupilSize'].values

        # Determine story order for this subject
        story_order = GROUP_STORY_ORDER[group_num]
        stories_in_run = (
            story_order[:3] if run == 'run_1' else story_order[3:]
        )

        # Segment stories using event end times
        current_time = 0
        for story in stories_in_run:
            if story not in xl.sheet_names:
                continue

            sheet = xl.parse(story)
            end_times = sheet['Segment_end_time'].dropna().apply(time_str_to_sec)
            if end_times.empty:
                continue
            max_end = end_times.max()

            start_idx = int(current_time)
            end_idx = min(int(current_time + max_end), len(pupil_array))
            segment = pupil_array[start_idx:end_idx]
            current_time += max_end + 2  # add 2s pause

            if len(segment) < 3:
                continue

            # Compute FFT
            freqs, power = compute_fft(segment)
            # Store for group-level plots
            all_story_power[story].append(power)
            if story not in story_freqs:
                story_freqs[story] = freqs

            # Save individual FFT CSV
            story_clean = story.replace(' ', '_').replace('/', '_')
            out_csv = pd.DataFrame({'frequency': freqs, 'power': power})
            csv_name = f"{sub}_{group_num}_{run}_{story_clean}_fft.csv"
            out_csv.to_csv(os.path.join(current_save, csv_name), index=False)

            # Save individual FFT plot
            plt.figure(figsize=(8, 4))
            plt.plot(freqs, power)
            plt.title(f"FFT - Sub {sub} - {story} (Group {group_num}, {run})")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power")
            plt.grid(True)
            plot_name = f"{sub}_{group_num}_{run}_{story_clean}_fft.png"
            plt.savefig(os.path.join(current_save, plot_name), dpi=300)
            plt.close()

# Group-level FFT plots by valence
valence_bins = defaultdict(list)
for story, val in STORY_VALENCE.items():
    if story in all_story_power:
        valence_bins[val].append(story)

valence_labels = ['negative', 'neutral', 'positive']
plot_positions = {0: [0, 3], 1: [1, 4], 2: [2, 5]}

plt.figure(figsize=(18, 10))
for col_idx, val in enumerate(valence_labels):
    stories = valence_bins.get(val, [])
    for row_idx, story in enumerate(stories):
        powers = np.vstack(all_story_power[story])
        mean_power = np.nanmean(powers, axis=0)
        sem_power = np.nanstd(powers, axis=0, ddof=1) / np.sqrt(powers.shape[0])
        freqs = story_freqs[story]

        ax = plt.subplot(2, 3, plot_positions[col_idx][row_idx] + 1)
        ax.plot(freqs, mean_power, linewidth=2)
        ax.fill_between(freqs, mean_power - sem_power, mean_power + sem_power, alpha=0.3)
        ax.set_title(story)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.set_ylim(0, 80000)
        ax.grid(True)
        if row_idx == 0:
            ax.text(0.5, 1.2, val.capitalize(), transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')
            #ax.annotate(val.capitalize(), xy=(0.5, 1.05), xycoords='axes fraction', ha='center', fontsize=14, fontweight='bold')

plt.suptitle(f"Average Story FFT Power by Valence (Filter={FILTER_TYPE})", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
combined_plot = f"all_stories_avg_fft_by_valence_{FILTER_TYPE}.png"
plt.savefig(os.path.join(SAVE_PATH, combined_plot), dpi=300)
plt.close()
print(f"Saved combined valence-level FFT plot: {combined_plot}")
