# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: June 23, 2025
# Description: Calculate Spectral Entropy

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import entropy

# ---------- Configuration ---------- #
EXP_TYPE = "encoding"
FFT_DIR = f"/Users/UChicago/CASNL/storyfest/data/pupil/3_processed/9_FFT_story_from_sec/{EXP_TYPE}"
SAVE_SUMMARY = f"/Users/UChicago/CASNL/storyfest/data/pupil/3_processed/12_spectral_entropy/{EXP_TYPE}"
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
entropy_records = []

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
        power = df['power'].values

        # Normalize the power spectrum to get probability distribution
        power_sum = np.sum(power)
        if power_sum == 0:
            print(f"Zero power sum for file: {file}")
            continue
        power_normalized = power / power_sum

        # Compute spectral entropy
        spectral_entropy = entropy(power_normalized, base=2)

        entropy_records.append({
            'subject': sub_id,
            'group': group,
            'run': run,
            'story': story,
            'valence': STORY_VALENCE.get(story, 'unknown'),
            'spectral_entropy': spectral_entropy
        })

# ---------- Save Summary ---------- #
entropy_df = pd.DataFrame(entropy_records)
summary_file = os.path.join(SAVE_SUMMARY, f'spectral_entropy_summary_{EXP_TYPE}.csv')
entropy_df.to_csv(summary_file, index=False)
print(f"Saved spectral entropy summary to {summary_file}")

# ---------- Statistical Testing ---------- #
from scipy import stats

neg = entropy_df[entropy_df['valence'] == 'negative']['spectral_entropy']
neu = entropy_df[entropy_df['valence'] == 'neutral']['spectral_entropy']
pos = entropy_df[entropy_df['valence'] == 'positive']['spectral_entropy']

anova_result = stats.f_oneway(neg, neu, pos)
kruskal_result = stats.kruskal(neg, neu, pos)

stats_summary = pd.DataFrame({
    'Test': ['ANOVA', 'Kruskal-Wallis'],
    'Statistic': [anova_result.statistic, kruskal_result.statistic],
    'p_value': [anova_result.pvalue, kruskal_result.pvalue]
})

stats_file = os.path.join(SAVE_SUMMARY, f'spectral_entropy_stats_{EXP_TYPE}.csv')
stats_summary.to_csv(stats_file, index=False)
print(f"Saved statistical test results to {stats_file}")

# ---------- Optional: Plot ---------- #
plt.figure(figsize=(10, 6))
for valence, color in zip(['negative', 'neutral', 'positive'], ['blue', 'orange', 'green']):
    data = entropy_df[entropy_df['valence'] == valence]
    plt.scatter(data['story'], data['spectral_entropy'], label=valence, alpha=0.7, color=color)

plt.xlabel('Story')
plt.ylabel('Spectral Entropy')
plt.title('Spectral Entropy by Story and Valence')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(SAVE_SUMMARY, f'spectral_entropy_plot_{EXP_TYPE}.png')
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Saved plot to {plot_path}")
