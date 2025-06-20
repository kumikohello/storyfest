# Storyfest

This repository contains scripts and data for preprocessing and analyzing pupil dilation data collected during an auditory narrative (story-listening) experiment. The goal is to investigate cognitive and emotional responses to naturalistic stimuli.

---

## 🧪 Preprocessing Scripts Overview

All scripts below are located in `scripts/preprocessing/` and are designed to be run in sequential order as part of the pupil preprocessing pipeline.

### 1. `1_align_pupil.py`
Aligns raw pupil data to TRs using timing information. This step ensures that all data streams are synchronized to a common timeline.

---

### 2. `2_exclude_noisy_pts.py`
Identifies and removes noisy participants based on pupil derivative statistics. Subjects are excluded if a large proportion of samples show abnormally high change between timepoints.

---

### 3. `3_interpolate_blinks.py`
Performs blink interpolation by identifying short-duration signal dropouts and filling them using linear interpolation. Helps to maintain continuity in the pupil signal.

---

### 4. `4_downsample.py`
Downsamples the interpolated signal to a lower temporal resolution (e.g., from 500 Hz to 50 Hz). This reduces noise and file size while preserving the signal’s temporal structure.

---

### 5. `5_downsample_to_sec.py`
An extension of the above script, this one downsamples pupil data to 1 sample per second (1 Hz), explicitly for alignment with second-wise event-level data.

---

### 6. `6_downsample_to_events.py`
Segments and downsamples pupil data according to predefined event boundaries (e.g., narrative moments of interest). Useful for event-level analyses of arousal or attention.

---

### 7. `6_downsample_to_stories.py`
Aggregates pupil data by entire stories, aligning the full signal to narrative start and end times. Useful for averaging time courses across longer narrative arcs.

---

### 8. `7_isc_pupil.py`
Computes inter-subject correlation (ISC) of pupil dilation across subjects. This measures the degree of synchronization in pupil responses to the same narrative, indicative of shared attention or emotion.

---

### 9. `8_stack_df.py`
Stacks all processed pupil data (e.g., from all participants, all runs) into a single dataframe for downstream analysis and visualization. Helpful for group-level stats.

---

### 10. `9_convert_FFT.py`
Performs Fast Fourier Transformation on pupil data. Useful for examining signal complexity or frequency-domain dynamics across events or stories.

---

### 11. `9_frequency_analysis.py`
Performs spectral analysis on pupil data, including Fourier transforms and entropy metrics. Useful for examining signal complexity or frequency-domain dynamics across events or stories.

---

## ⚠️ Notes

- Output is saved under `data/pupil/3_processed/` in corresponding subfolders.

