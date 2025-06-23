"""
Spectral Centroid Calculation Script
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

# ---------- Configuration ---------- #
EXP_TYPE = "encoding"
FILTER_TYPE = "bandpass"  # "lowpass" or "bandpass"
FFT_DIR = f"/Users/UChicago/CASNL/storyfest/data/pupil/3_processed/9_FFT_story_from_sec/{EXP_TYPE}"
SAVE_SUMMARY = f"/Users/UChicago/CASNL/storyfest/data/pupil/3_processed/11_spectral_centroid/{EXP_TYPE}"
os.makedirs(SAVE_SUMMARY, exist_ok=True)

STORY_VALENCE = {
    'Pool Party': 'positive',
    'Sea Ice': 'neutral',
    'Natalie Wood': 'negative',
    'Impatient Billionaire': 'positive',
    'Grandfather Clocks': 'neutral',
    'Dont Look': 'negative'
}

runs = ['run_1', 'run_2']

# ---------- Process FFT Files ---------- #
summary_records = []

for run in runs:
    fft_run_dir = os.path.join(FFT_DIR, run)
    fft_files = [f for f in os.listdir(fft_run_dir) if f.endswith('.csv')]

    for file in fft_files:
        file_path = os.path.join(fft_run_dir, file)

        # Parse metadata from filename: sub_group_run_story_fft.csv
        base = file.replace("_fft.csv", "")
        parts = base.split("_")
        sub_id = parts[0]
        group = parts[1]
        story = "_".join(parts[4:]).replace("_", " ")

        # Load FFT data
        df = pd.read_csv(file_path)
        freqs = df['frequency'].values
        power = df['power'].values

        # Compute Spectral Centroid
        spectral_centroid = np.sum(freqs * power) / np.sum(power)

        summary_records.append({
            'subject': sub_id,
            'group': group,
            'run': run,
            'story': story,
            'valence': STORY_VALENCE.get(story, 'unknown'),
            'spectral_centroid': spectral_centroid
        })

# ---------- Save Summary ---------- #
summary_df = pd.DataFrame(summary_records)
summary_file = os.path.join(SAVE_SUMMARY, f'spectral_centroid_summary_{EXP_TYPE}_{FILTER_TYPE}.csv')
summary_df.to_csv(summary_file, index=False)
print(f"Saved spectral centroid summary to {summary_file}")

# ---------- Statistical Testing ---------- #
neg = summary_df[summary_df['valence'] == 'negative']['spectral_centroid']
neu = summary_df[summary_df['valence'] == 'neutral']['spectral_centroid']
pos = summary_df[summary_df['valence'] == 'positive']['spectral_centroid']

anova_result = stats.f_oneway(neg, neu, pos)
kruskal_result = stats.kruskal(neg, neu, pos)

# Save stats to CSV
stats_summary = pd.DataFrame({
    'Test': ['ANOVA', 'Kruskal-Wallis'],
    'Statistic': [anova_result.statistic, kruskal_result.statistic],
    'p_value': [anova_result.pvalue, kruskal_result.pvalue]
})

stats_file = os.path.join(SAVE_SUMMARY, f'spectral_centroid_stats_{EXP_TYPE}_{FILTER_TYPE}.csv')
stats_summary.to_csv(stats_file, index=False)
print(f"Saved statistical test results to {stats_file}")

# ---------- Optional: Plot ---------- #
plt.figure(figsize=(10, 6))
for valence in ['negative', 'neutral', 'positive']:
    data = summary_df[summary_df['valence'] == valence]
    plt.scatter(data['story'], data['spectral_centroid'], label=valence, alpha=0.7)

plt.xlabel('Story')
plt.ylabel('Spectral Centroid (Hz)')
plt.title('Spectral Centroid by Story and Valence')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_SUMMARY, f'spectral_centroid_plot_{EXP_TYPE}_{FILTER_TYPE}.png'), dpi=300)
plt.close()
