# ----------------------------------------
# Author: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: July 2, 2025
# Description: Convert EyeLink .asc files to structured .csv files
#   - Outputs one CSV for continuous samples (timestamp, x, y, pupil)
#   - Outputs one CSV for discrete events (MSG, SFIX, EFIX, SBLINK, etc.)
# ----------------------------------------

import os
import numpy as np
import pandas as pd

# ------------------ Hardcoded parameters ------------------ #
# Set working directory to storyfest root (e.g., your repo path)
os.chdir('/Users/UChicago/CASNL/storyfest/storyfest')
_THISDIR = os.getcwd()

EXP_TYPE = "encoding"  # Choose between "encoding" or "recall"
ASC_PATH = os.path.normpath(os.path.join(_THISDIR, 'data/pupil/1_raw/'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, 'data/pupil/2_csv/', EXP_TYPE))

SUBJ_IDS = range(1001, 1046)
SAMPLING_RATE = 500  # Hz

os.makedirs(SAVE_PATH, exist_ok=True)

# ------------------ Define Function ------------------ #
def safe_float(val):
    try:
        return float(val)
    except ValueError:
        return np.nan

def parse_asc_file(asc_path):
    """
    Parses a .asc file and separates:
    - sample rows: timestamp, x, y, pupil
    - event rows: event_type, timestamp, detail
    Returns:
        samples_df, events_df
    """
    samples = []
    events = []
    valid_event_types = {'MSG', 'SFIX R', 'EFIX R', 'SSACC R', 'ESACC R', 'SBLINK R', 'EBLINK R'}

    with open(asc_path, 'r') as f:
        for line in f:
            parts = line.strip().split()

            # --- Sample line: starts with digit, has at least 4 values
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    timestamp = int(parts[0])
                    x = safe_float(parts[1])
                    y = safe_float(parts[2])
                    pupil = safe_float(parts[3])
                    samples.append([timestamp, x, y, pupil])
                except ValueError:
                    continue  # skip malformed lines

            
            # --- Event line: MSG or two-part event type
            elif len(parts) >= 2:
                if parts[0] == 'MSG':
                    try:
                        timestamp = float(parts[1])
                        detail = ' '.join(parts[2:]) if len(parts) > 2 else ""
                        events.append(['MSG', timestamp, detail])
                    except ValueError:
                        continue
                else:
                    # Combine first two parts (e.g., "EFIX R")
                    event_key = parts[0] + ' ' + parts[1]
                    if event_key in valid_event_types:
                        try:
                            timestamp = float(parts[2])
                            detail = ' '.join(parts[3:]) if len(parts) > 3 else ""
                            events.append([event_key, timestamp, detail])
                        except ValueError:
                            continue

    samples_df = pd.DataFrame(samples, columns=["timestamp", "x", "y", "pupil"])
    events_df = pd.DataFrame(events, columns=["event_type", "timestamp", "detail"])

    return samples_df, events_df

# ------------------ Save Function ------------------ #
def save_csvs(samples_df, events_df, asc_filename, output_dir):
    """
    Save parsed DataFrames into per-participant subfolders as samples.csv and events.csv.
    """
    base = os.path.splitext(os.path.basename(asc_filename))[0]  # e.g., "1001_encoding"
    subj_id = base.split("_")[0]  # e.g., "1001"

    subj_folder = os.path.join(output_dir, subj_id)
    os.makedirs(subj_folder, exist_ok=True)

    samples_out = os.path.join(subj_folder, "samples.csv")
    events_out = os.path.join(subj_folder, "events.csv")

    samples_df.to_csv(samples_out, index=False)
    events_df.to_csv(events_out, index=False)
    print(f"Saved to: {subj_folder}")

# ------------------ Main ------------------ #
if __name__ == "__main__":
    for fname in os.listdir(ASC_PATH):
        if fname.endswith(".asc"):
            asc_file = os.path.join(ASC_PATH, fname)
            samples_df, events_df = parse_asc_file(asc_file)
            save_csvs(samples_df, events_df, fname, SAVE_PATH)