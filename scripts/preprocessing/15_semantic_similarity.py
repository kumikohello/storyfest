from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
import pandas as pd
import re
import seaborn as sns
from functools import reduce

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/UChicago/CASNL/storyfest/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/recall_transcripts/event_segmented_recall'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/15_semantic_similarity'))
EVENTS_PATH = os.path.normpath(os.path.join(_THISDIR, '../../experiment/Storyfest_Event_Segmentation.xlsx'))
# COURSE_EVENTS_PATH = os.path.normpath(os.path.join(_THISDIR, '../../experiment/eventsegmentation_coarse.xlsx'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

runs = [None]

SUBJ_IDS = range(1001,1046)

STORIES = ['Pool Party', 'Sea Ice', 'Natalie Wood', 'Grandfather Clocks', 'Impatient Billionaire', 'Dont Look']

STORY_VALENCE = {
    'Pool Party': 'positive',
    'Sea Ice': 'neutral',
    'Natalie Wood': 'negative',
    'Impatient Billionaire': 'positive',
    'Grandfather Clocks': 'neutral',
    'Dont Look': 'negative'
}

EVENT_COUNT = {
    'Pool Party': 20,
    'Sea Ice': 23,
    'Natalie Wood': 55,
    'Grandfather Clocks': 29,
    'Impatient Billionaire': 17,
    'Dont Look': 41
}

# COURSE_EVENT_COUNT = {
#     'Pool Party': 7,
#     'Sea Ice': 8,
#     'Natalie Wood': 17,
#     'Grandfather Clocks': 11,
#     'Impatient Billionaire': 7,
#     'Dont Look': 13
# }

# Download model to local
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

print ("module %s loaded" % module_url)

# ------------------ Define functions ------------------ # 
def embed(input):
    if not input or all(not str(t).strip() for t in input):
        print("⚠️ Skipping embedding: No valid text found.")
        return None
    return model(input)

def cosine_similarity(vector1, vector2):

    dot_product = np.dot(vector1, vector2)

    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)

    return dot_product / (magnitude_vector1 * magnitude_vector2)

# Reduce logging output.
logging.set_verbosity(logging.ERROR)

# Store all-subject data here
all_subject_event_similarities = {story: [] for story in STORIES}
all_subject_mean_similarities = []

for subid in SUBJ_IDS:
    # Import data
    subj_file = os.path.join(DAT_PATH, f"Subj_{subid - 1000}.xlsx")
    if not os.path.exists(subj_file):
        print(f"File missing for Subj {subid}")
        continue

    xl = pd.ExcelFile(subj_file)

    # Dictionary to store story-specific similarities
    subj_cos_sim = {}

    for story in STORIES:
        if story not in xl.sheet_names:
            print(f"Story {story} missing for Subj {subid} — inserting zeros")
            n_events = EVENT_COUNT[story]
            subj_cos_sim[story] = [0.0] * n_events
            all_subject_event_similarities[story].append([0.0] * n_events)
            continue

        sheet = xl.parse(story)

        # Make sure both columns exist
        if 'Transcript' not in sheet.columns or 'Subj_Transcript' not in sheet.columns:
            print(f"Missing columns in {story} for Subj {subid}")
            continue

        # Extract and clean original (annotation) transcript
        original_transcript = sheet['Transcript'].dropna().astype(str).tolist()
        recall_transcript = sheet['Subj_Transcript'].astype(str).tolist()

        # Embed original annotations
        annotation_embeddings = embed(original_transcript)

        # Embed recall; remove nans but keep indices
        recall_transcript_nan_indices = pd.isnull(sheet['Subj_Transcript'])
        recall_transcript_no_nans = [x for x in recall_transcript if str(x) != 'nan']
        recall_embeddings = embed(recall_transcript_no_nans)
        if recall_embeddings is None:
            print("recall_embeddings is None")
            continue  # or handle differently

        # Add the nans back to their original position
        recall_embeddings_with_nans = np.full((len(recall_transcript), recall_embeddings.shape[1]), np.nan)
        j = 0
        for i in range(len(recall_transcript)):
            if not recall_transcript_nan_indices.iloc[i]:
                recall_embeddings_with_nans[i] = recall_embeddings[j]
                j += 1
                
        # Cosine similarity between `annotation_embeddings` and `recall_embeddings_with_nans`
        cos_sim_list = []
        
        for i in range(len(annotation_embeddings)):
            if i >= recall_embeddings_with_nans.shape[0] or np.isnan(recall_embeddings_with_nans[i]).any():
                cos_sim_list.append(0)
            else:
                cos_sim = np.dot(annotation_embeddings[i], recall_embeddings_with_nans[i]) / (norm(annotation_embeddings[i]*norm(recall_embeddings_with_nans[i])))
                cos_sim_list.append(cos_sim)
        
        subj_cos_sim[story] = cos_sim_list
        all_subject_event_similarities[story].append(cos_sim_list)

    # ------------------ Save Results ------------------ #         
    # Save cosine similarity
    # Combine all stories into one DataFrame (column = story, rows = event similarities)
    story_event_frames = []

    for story, sim_list in subj_cos_sim.items():
        if story in xl.sheet_names:
            sheet = xl.parse(story)

            if 'event_number' not in sheet.columns:
                print(f"Missing event_number in {story} for Subj {subid}")
                event_nums = pd.Series(np.arange(1, len(sim_list) + 1))
                continue

            event_nums = sheet['event_number'].reset_index(drop=True)
            orig_trans = sheet['Transcript'].reset_index(drop=True).astype(str)
            subj_trans = sheet['Subj_Transcript'].reset_index(drop=True).astype(str)

            # Recalculate sim_list with the updated rule: 0 if subj_trans is missing
            sim_list_fixed = []
            for i in range(len(event_nums)):
                if pd.isna(event_nums[i]):
                    continue  # Skip this row entirely
                elif subj_trans[i].lower() == 'nan' or subj_trans[i].strip() == '':
                    sim_list_fixed.append(0.0)  # Empty recall = 0
                else:
                    vec1 = embed([orig_trans[i]])[0]
                    vec2 = embed([subj_trans[i]])[0]
                    cos_sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
                    sim_list_fixed.append(cos_sim)

            # Keep only non-NaN event_number rows
            clean_event_nums = event_nums[~event_nums.isna()].reset_index(drop=True)
            sim_series = pd.Series(sim_list_fixed)

        else:
            # Story was missing entirely — use synthetic event numbers
            print(f"Generating dummy event numbers for {story} for Subj {subid}")
            clean_event_nums = pd.Series(np.arange(1, EVENT_COUNT[story] + 1))
            sim_series = pd.Series(sim_list) # this is already all zeros

        df_story = pd.DataFrame({
            'event_number': clean_event_nums,
            story: sim_series
        })

        story_event_frames.append(df_story)

    # Merge all story columns on event_number
    from functools import reduce
    event_df = reduce(lambda left, right: pd.merge(left, right, on='event_number', how='outer'), story_event_frames)

    # Sort by event_number and save
    event_df = event_df.sort_values('event_number').reset_index(drop=True)

    # Save
    if not os.path.exists(SAVE_PATH + '/event'):
        os.makedirs(SAVE_PATH + '/event')

    event_df.to_excel(os.path.join(SAVE_PATH + '/event', f"sub_{subid}_cosine_similarity.xlsx"), index=False)
    print("Saved ", f"sub_{subid}_cosine_similarity_eventwise.xlsx")


    # Mean cosine similarity per story
    mean_per_story = {story: np.nanmean(vals) for story, vals in subj_cos_sim.items()}
    all_subject_mean_similarities.append(mean_per_story)
    mean_df = pd.DataFrame([mean_per_story])
    if not os.path.exists(SAVE_PATH + '/mean'):
        os.makedirs(SAVE_PATH + '/mean')
    mean_df.to_excel(os.path.join(SAVE_PATH + '/mean', f"sub_{subid}_cosine_similarity_story_avg.xlsx"))
    print(f"Saved mean for sub {subid}")

# ------------------ Mean Cosine Similarity per Event ------------------ #

# Collect all subject event-level cosine similarity data
subject_eventwise_paths = os.listdir(SAVE_PATH + '/event')
subject_eventwise_paths = [f for f in subject_eventwise_paths if f.endswith('.xlsx')]

subject_event_dfs = []
for fname in subject_eventwise_paths:
    fpath = os.path.join(SAVE_PATH, 'event', fname)
    df = pd.read_excel(fpath, engine='openpyxl')

    subid_match = re.search(r"sub_(\d+)", fname)
    if subid_match:
        subid = subid_match.group(1)
        renamed_cols = {
            col: f"{col}_{subid}" for col in df.columns if col != 'event_number'
        }
        df = df.rename(columns=renamed_cols)
    
    subject_event_dfs.append(df)

# Merge on event_number
merged_df = reduce(lambda left, right: pd.merge(left, right, on='event_number', how='outer'), subject_event_dfs)

# Group columns by story name
# Group by story name before underscore
story_groups = {}
for col in merged_df.columns:
    if col == 'event_number':
        continue
    story_base = col.rsplit('_', 1)[0]
    story_groups.setdefault(story_base, []).append(col)

# Average across subjects for each story
group_avg_df = pd.DataFrame()
group_avg_df['event_number'] = merged_df['event_number']

for story, cols in story_groups.items():
    group_avg_df[story] = merged_df[cols].mean(axis=1, skipna=True)

# Save the averaged eventwise cosine similarity
group_event_path = os.path.join(SAVE_PATH, 'group_mean')
os.makedirs(group_event_path, exist_ok=True)

output_path = os.path.join(group_event_path, 'group_mean_eventwise_cosine_similarity.xlsx')
group_avg_df.to_excel(output_path, index=False)
print(f"Saved group-level eventwise cosine similarity to: {output_path}")

# ------------------ Save Group-Level Data ------------------ #
# Plot average cosine similarity across events per story
plt.figure(figsize=(10, 6))

# Load group-averaged eventwise similarity
group_avg_df = pd.read_excel(SAVE_PATH + '/group_mean/group_mean_eventwise_cosine_similarity.xlsx')

# Then, inside your plotting code:
for story in STORIES:
    avg_course = group_avg_df[story].values
    plt.plot(avg_course, label=story)

plt.xlabel("Event Index")
plt.ylabel("Cosine Similarity")
plt.title("Average Semantic SImilarity Across Subjects")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "average_cosine_similarity_by_event.png"), dpi=300)
plt.close()
print("Saved figures")

# ------------------ Bar Plot: Mean Cosine Similarity per Story ------------------ #
# Calculate group average across all subjects for each story
story_means = group_avg_df.mean(skipna=True)

# Sort stories by valence: negative -> neutral -> positive
valence_order = ['negative', 'neutral', 'positive']
sorted_stories = []
for valence in valence_order:
    stories = [s for s, v in STORY_VALENCE.items() if v == valence]
    sorted_stories.extend(stories)

story_means_sorted = story_means[sorted_stories]

# Set color map
color_map = {'negative': 'red', 'neutral': 'black', 'positive': 'blue'}
bar_colors = [color_map[STORY_VALENCE[story]] for story in story_means_sorted.index]

# Plot
plt.figure(figsize=(10, 5))
plt.bar(story_means_sorted.index, story_means_sorted.values, color=bar_colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean Cosine Similarity")
plt.title("Average Cosine Similarity per Story (across subjects)")
plt.tight_layout()
plt.grid(axis='y')

# Save
plt.savefig(os.path.join(SAVE_PATH, "mean_cosine_similarity_per_story_bar.png"), dpi=300)
plt.close()
print("Saved mean cosine similarity per story")

# ------------------ Valence-Sorted Story Plots (All in One Figure) ------------------ #
valence_bins = {'negative': [], 'neutral': [], 'positive': []}
for story, valence in STORY_VALENCE.items():
    if story in all_subject_event_similarities:
        valence_bins[valence].append(story)

valence_labels = ['negative', 'neutral', 'positive']
plot_positions = {
    0: [0, 3],  # positions for negative
    1: [1, 4],  # positions for neutral
    2: [2, 5],  # positions for positive
}

plt.figure(figsize=(18, 10))

for col_idx, valence in enumerate(valence_labels):
    stories = valence_bins[valence]
    for row_idx, story in enumerate(stories):
        pos = plot_positions[col_idx][row_idx]
        avg_course = group_avg_df[story].values

        ax = plt.subplot(2, 3, pos + 1)
        ax.plot(avg_course, color='black', linewidth=2)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(story)
        ax.set_xlabel("Event Index")
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(0, 0.6)  # Adjust based on expected range
        if row_idx == 0:
            ax.annotate(valence.capitalize(), xy=(0.5, 1.1), xycoords='axes fraction', ha='center',
                        fontsize=14, fontweight='bold')

plt.suptitle("Average Semantic Similarity by Story Sorted by Valence", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plot_name = "all_stories_cosine_similarity_sorted_by_valence.png"
plt.savefig(os.path.join(SAVE_PATH, plot_name), dpi=300)
plt.close()
print(f"Saved combined 6-story plot: {plot_name}")

