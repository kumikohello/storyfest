import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# ---------- Configuration ---------- #
EXP_TYPE = "encoding" # encoding or recall
SUBJ_IDS = range(1001,1046)

# Paths
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
EVENT_INPUT_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/6_eventlocked/{EXP_TYPE}'))
STORY_OUTPUT_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/7_storylocked_from_events/{EXP_TYPE}'))
os.makedirs(STORY_OUTPUT_PATH, exist_ok=True)

if EXP_TYPE == "encoding":
    runs = ['run_1', 'run_2']
else:
    runs = [None]

# Story valence
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
    current_input = os.path.join(EVENT_INPUT_PATH, run) if run else EVENT_INPUT_PATH
    current_output = os.path.join(STORY_OUTPUT_PATH, run) if run else STORY_OUTPUT_PATH
    os.makedirs(current_output, exist_ok=True)

    for sub in SUBJ_IDS:
        group_num = (sub - 1000) % 3 or 3
        # Find file
        files = [f for f in os.listdir(current_input) if f.startswith(str(sub)) and f.endswith('.csv')]
        if len(files) == 0:
            print(f"No event file found for subject {sub} in {run}")
            continue

        event_file = os.path.join(current_input, files[0])
        df_events = pd.read_csv(event_file)

        if df_events.empty:
            print(f"Empty event file for subject {sub}")
            continue

        # Group by story
        df_stories = df_events.groupby("story").agg({
            "mean_pupil_size": "mean",
            "story_duration_sec": "sum",
            "story_start_sec": "min",
            "story_end_sec": "max"
        }).reset_index()

        # Add valence
        df_stories["valence"] = df_stories["story"].map(STORY_VALENCE)
        df_stories["subject"] = sub
        df_stories["story_duration_sec"] = df_stories["story_end_sec"] - df_stories["story_start_sec"]

        # Z-score within subject
        df_stories["z_pupil"] = stats.zscore(df_stories["mean_pupil_size"], nan_policy="omit")

        # Save CSV
        out_csv = os.path.join(current_output, f"{sub}_{group_num}_story_from_events.csv")
        df_stories.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        # Plot
        plt.figure(figsize=(10, 5))
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        for valence in df_stories['valence'].unique():
            val_df = df_stories[df_stories['valence'] == valence]
            plt.plot(val_df['story_start_sec'], val_df['z_pupil'], 'o-', label=valence, color=colors[valence])

        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.xlabel("Story Start Time (sec)")
        plt.ylabel("Standardized Pupil Size (z)")
        plt.title(f"Story-Level Pupil (from events): Subject {sub}")
        plt.legend(title="Valence")
        plt.tight_layout()

        plot_path = os.path.join(current_output, f"{sub}_{group_num}_{run}_story_from_events_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot: {plot_path}")

# ---------- Group-Level Average Plots (Recall or Encoding) ---------- #
if EXP_TYPE == "encoding":
    from collections import defaultdict

    for run in runs:
        current_output = os.path.join(STORY_OUTPUT_PATH, run) if run else STORY_OUTPUT_PATH
        group_story_data = defaultdict(list)

        for sub in SUBJ_IDS:
            group_num = (sub - 1000) % 3 or 3
            file_path = os.path.join(current_output, f"{sub}_{group_num}_story_from_events.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df["group_num"] = group_num
                group_story_data[group_num].append(df)

        for group_num, dfs in group_story_data.items():
            group_df = pd.concat(dfs, ignore_index=True)

            if group_df.empty:
                continue

            # Sort by time for clarity
            group_df = group_df.dropna(subset=["z_pupil", "story_start_sec"])
            group_df = group_df.sort_values("story_start_sec")

            plt.figure(figsize=(12, 5))
            colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}

            for valence in group_df["valence"].unique():
                val_df = group_df[group_df["valence"] == valence]
                plt.plot(val_df["story_start_sec"], val_df["z_pupil"], 'o', label=valence, color=colors[valence], alpha=0.5)

            plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
            plt.xlabel("Story Start Time (sec)")
            plt.ylabel("Standardized Pupil Size (z)")
            title_str = f"Group {group_num} Story-Level Pupil (from Events)"
            if run:
                title_str += f" - {run}"
            plt.title(title_str)
            plt.legend(title="Valence")
            plt.tight_layout()

            fname = f"{run}_{group_num}_event_plot.png" if run else f"group_{group_num}_event_plot.png"
            group_plot_path = os.path.join(current_output, fname)
            plt.savefig(group_plot_path, dpi=300)
            plt.close()
            print(f"Saved group-level plot: {group_plot_path}")

