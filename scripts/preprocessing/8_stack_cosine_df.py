import pandas as pd
import glob
import os
import re

# --------- YOU MAY EDIT THESE ---------
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()

ENCODING_CSV = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/8_stack_df/encoding/stacked_events_lowpass.csv'))
COSINE_DIR = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/15_semantic_similarity/event/'))
OUTPUT_DIR  = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/8_stack_cosine_df'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'stacked_events_lowpass_with_cosine.csv')
# --------------------------------------

# Load the stacked events file (from encoding)
stacked = pd.read_csv(ENCODING_CSV)

# Gather all cosine similarity files
cosine_files = glob.glob(os.path.join(COSINE_DIR, '*_cosine_similarity.*'))

long_frames = []
for f in cosine_files:
    df = pd.read_excel(f)

    # Try to extract subject ID from filename
    if 'subject_id' not in df.columns:
        m = re.search(r'(\d{4})', os.path.basename(f))
        if m:
            df['subject_id'] = int(m.group(1))
        else:
            raise ValueError(f"Could not find subject ID in file name: {f}")

    # Reshape: one row per subject, story, and event
    long_df = df.melt(
        id_vars=['subject_id', 'event_number'],
        var_name='story',
        value_name='cosine_similarity'
    ).rename(columns={'subject_id': 'subject', 'event_number': 'event_num'})

    long_frames.append(long_df)

# Combine all cosine similarity data
cosine_long = pd.concat(long_frames, ignore_index=True)

# Make sure key columns are same type
for col in ['subject', 'event_num']:
    stacked[col] = stacked[col].astype(int)
    cosine_long[col] = cosine_long[col].astype(int)

# Merge recall cosine similarity into encoding dataframe
merged = stacked.merge(
    cosine_long,
    on=['subject', 'story', 'event_num'],
    how='left'
)

# Save result
merged.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved: {OUTPUT_FILE}")
