# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: June 30, 2025
# Description: The script extract blinks

import pandas as pd
import os

BASE_PATH = "/Users/UChicago/CASNL/storyfest/storyfest/data/pupil/2_csv/encoding"
SUBJ_IDS = range(1001, 1046)

for sub in SUBJ_IDS:
    subj_path = os.path.join(BASE_PATH, str(sub))
    events_path = os.path.join(subj_path, "events.csv")
    blinks_out_path = os.path.join(subj_path, "blinks.csv")

    if not os.path.exists(events_path):
        print(f"Missing events.csv for subject {sub}")
        continue

    events_df = pd.read_csv(events_path)

    # Filter out blink lines
    sblinks = events_df[events_df["event_type"] == "SBLINK R"]
    eblinks = events_df[events_df["event_type"] == "EBLINK R"]

    # Check for matching pairs
    if len(sblinks) != len(eblinks):
        print(f"Mismatched blinks for subject {sub}: SBLINK={len(sblinks)}, EBLINK={len(eblinks)}")
        continue

    # Extract start and end timestamps
    start_times = sblinks["timestamp"].astype(float).tolist()
    end_times = (
        eblinks['detail']
        .str.extract(r'^(\d+(?:\.\d+)?)')  # match the first number only
        .iloc[:, 0]
        .astype(float)
        .tolist()
    )

    # Build and save blink DataFrame
    blinks_df = pd.DataFrame({"start_time": start_times, "end_time": end_times})
    blinks_df.to_csv(blinks_out_path, index=False)
    print(f"Saved blinks.csv for subject {sub}")