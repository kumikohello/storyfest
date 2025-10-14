# Storyfest

This repository contains scripts and data for preprocessing and analyzing pupil dilation data collected during an auditory narrative (story-listening) experiment. The goal is to investigate cognitive and emotional responses to naturalistic stimuli.

---

## 🧪 Preprocessing Scripts Overview

All scripts below are located in `scripts/preprocessing/` and are designed to be run in sequential order as part of the pupil preprocessing pipeline.

### 1. `0_blinks.py`
Extracts blinks from the asc files.

---

### 2. `0_timestamps.py`
Extracts timestamps from the asc files.

---

### 3. `1_align_pupil.py`
Aligns raw pupil data to TRs using timing information. This step ensures that all data streams are synchronized to a common timeline.

---

### 4. `2_exclude_noisy_subj.py`
Identifies and removes noisy participants based on pupil derivative statistics. Subjects are excluded if a large proportion of samples show abnormally high change between timepoints.

---

### 5. `3_interpolate_blinks.py`
Performs blink interpolation by identifying short-duration signal dropouts and filling them using linear interpolation. Helps to maintain continuity in the pupil signal.

---

### 6. `4_downsample.py`
Applies lowpass filter of 4Hz to prevent aliasing (Aliased noise can corrupt low-frequency pupil signal). Downsamples the interpolated signal to a lower temporal resolution (e.g., from 500 Hz to 50 Hz). This reduces noise and file size while preserving the signal’s temporal structure.

---

### 7. `5_standardize.py`
An extension of the above script, applies either bandpass (0.01-0.2Hz) or lowpass (-0.2Hz) filter, and standardizes.

---

### 8. `6_segment_stories.py`
Aggregates pupil data by entire stories, aligning the full signal to narrative start and end times. Useful for averaging time courses across longer narrative arcs.

---

### 9. `7_segment_events.py`
Segments and downsamples pupil data according to predefined event boundaries (e.g., narrative moments of interest). Useful for event-level analyses of arousal or attention.

---

### 10. `7_isc_pupil.py`
Computes inter-subject correlation (ISC) of pupil dilation across subjects. This measures the degree of synchronization in pupil responses to the same narrative, indicative of shared attention or emotion.

---

### 11. `8_stack_df.py`
Stacks all processed pupil data (e.g., from all participants, all runs) into a single dataframe for downstream analysis and visualization. Helpful for group-level stats.

---

### 12. `15_semantic_similarity.py`
Calculates the semantic similarity.

---

### 13. `16_sentiment_analysis.py`
Calculates the cosine similarity score of the original transcript and participants recall transcript at the 20sec level.

---

### 14. `16_coarse_sentiment_analysis.py`
Calculates the cosine similarity score of the original transcript and participants recall transcript at the event level.

---

### 15. `17_sentiment_centrality`
Calculates the semantic centrality.

---

## ⚠️ Notes

- Output is saved under `data/pupil/3_processed/` in corresponding subfolders.

