import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal

# === CONFIG ===
EXP_TYPE = "encoding"  # "encoding" or "recall"
# Paths
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/9_FFT_story_from_sec/{EXP_TYPE}'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/10_peak_freq_plots/{EXP_TYPE}'))
os.makedirs(SAVE_PATH, exist_ok=True)

# BASE_DIR = "/Users/UChicago/CASNL/storyfest/data/pupil/3_processed/9_convert_FFT/encoding"  # directory with run_1 and run_2
# OUTPUT_CSV = "dominant_frequencies_summary.csv"
# PLOT_DIR = "/Users/UChicago/CASNL/storyfest/data/pupil/3_processed/10_dominant_freq_plots"
os.makedirs(SAVE_PATH, exist_ok=True)

# Load all FFT files from both runs
summary_rows = []

for run in ["run_1", "run_2"]:
    FFT_DIR = os.path.join(DAT_PATH, run)
    for fname in os.listdir(FFT_DIR):
        if fname.endswith("_fft.csv") and not fname.startswith("group"):
            path = os.path.join(FFT_DIR, fname)
            df = pd.read_csv(path)

            # Exclude 0 Hz
            freqs = df['frequency'].values
            power = df['power'].values
            valid = freqs > 0

            freqs = freqs[valid]
            power = power[valid]

            peak_idx = np.argmax(power)
            dominant_freq = freqs[peak_idx]

            # Extract metadata from filename
            base = fname.replace("_fft.csv", "")
            parts = base.split("_")
            subject = parts[0]
            group = parts[1]
            story = "_".join(parts[4:]).replace("_", " ")

            summary_rows.append({
                "subject": int(subject),
                "group": int(group),
                "run": run,
                "story": story,
                "dominant_freq_Hz": dominant_freq
            })

# Save summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(by="subject")
csv_name = f"peak_frequencies_summary.csv"
summary_df.to_csv(os.path.join(SAVE_PATH, csv_name), index=False)
print(f"Saved dominant frequencies to {SAVE_PATH}")

# Map story to valence
STORY_VALENCE = {
    'Pool Party': 'positive',
    'Sea Ice': 'neutral',
    'Natalie Wood': 'negative',
    'Impatient Billionaire': 'positive',
    'Grandfather Clocks': 'neutral',
    'Dont Look': 'negative'
}
summary_df["valence"] = summary_df["story"].map(STORY_VALENCE)

# === Plot histograms ===
for val in ['negative', 'neutral', 'positive']:
    val_df = summary_df[summary_df["valence"] == val]
    plt.figure(figsize=(6, 4))
    plt.hist(val_df["dominant_freq_Hz"], bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Dominant Frequencies - {val.capitalize()}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"dominant_freq_hist_{val}.png"))
    plt.close()
    print(f"Saved histogram: dominant_freq_hist_{val}.png")

# === Statistical tests ===
neg = summary_df[summary_df.valence == 'negative']['dominant_freq_Hz']
neut = summary_df[summary_df.valence == 'neutral']['dominant_freq_Hz']
pos = summary_df[summary_df.valence == 'positive']['dominant_freq_Hz']

anova_result = f_oneway(neg, neut, pos)
kruskal_result = kruskal(neg, neut, pos)

print("\nANOVA result:", anova_result)
print("Kruskal-Wallis result:", kruskal_result)

# === Violin plot by valence ===
plt.figure(figsize=(8, 5))
sns.violinplot(data=summary_df, x="valence", y="dominant_freq_Hz", inner="box", palette="pastel")
sns.stripplot(data=summary_df, x="valence", y="dominant_freq_Hz", color='gray', size=4, jitter=True)
plt.title("Dominant Frequencies by Valence")
plt.ylabel("Dominant Frequency (Hz)")
plt.xlabel("Valence")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "violin_dominant_freq_by_valence.png"))
plt.close()
print("Saved violin plot by valence")

# === Violin plot by story ===
plt.figure(figsize=(12, 6))
sns.violinplot(data=summary_df, x="story", y="dominant_freq_Hz", inner="box", palette="pastel")
sns.stripplot(data=summary_df, x="story", y="dominant_freq_Hz", color='gray', size=4, jitter=True)
plt.xticks(rotation=45, ha='right')
plt.title("Dominant Frequencies by Story")
plt.ylabel("Dominant Frequency (Hz)")
plt.xlabel("Story")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "violin_dominant_freq_by_story.png"))
plt.close()
print("Saved violin plot by story")
