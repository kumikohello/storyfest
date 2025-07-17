# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: Julyt 15, 2025
# Description: This script calculates one-to-average ISC at the event level

import os
import numpy as np
from numpy.fft import fft, ifft
import pandas as pd
from scipy import stats
from sklearn.utils import check_random_state
from collections import defaultdict

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
EXP_TYPE = "encoding"
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/6_eventlocked', EXP_TYPE))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/7_isc_event', EXP_TYPE))
os.makedirs(SAVE_PATH, exist_ok=True)

FILTER_TYPE = "lowpass"  # "lowpass" or "bandpass"
LOWCUT_HZ = None # Only used if FILTER_TYPE is "bandpass"
HIGHCUT_HZ = 0.2 # Used in both "lowpass" and "bandpass"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    
SUBJ_IDS = range(1001, 1046)

# Number of iterations for permutation test
ITERATIONS = 5000

# ------------------ Define functions ------------------ # 
def phase_randomize(data, random_state=None):
    """Perform phase randomization on time-series signal (from nltools.stats)

    This procedure preserves the power spectrum/autocorrelation,
    but destroys any nonlinear behavior. Based on the algorithm
    described in:

    Theiler, J., Galdrikian, B., Longtin, A., Eubank, S., & Farmer, J. D. (1991).
    Testing for nonlinearity in time series: the method of surrogate data
    (No. LA-UR-91-3343; CONF-9108181-1). Los Alamos National Lab., NM (United States).

    Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. (2018).
    Surrogate data for hypothesis testing of physical systems. Physics Reports, 748, 1-60.

    1. Calculate the Fourier transform ftx of the original signal xn.
    2. Generate a vector of random phases in the range[0, 2π]) with
       length L/2,where L is the length of the time series.
    3. As the Fourier transform is symmetrical, to create the new phase
       randomized vector ftr , multiply the first half of ftx (i.e.the half
       corresponding to the positive frequencies) by exp(iφr) to create the
       first half of ftr.The remainder of ftr is then the horizontally flipped
       complex conjugate of the first half.
    4. Finally, the inverse Fourier transform of ftr gives the FT surrogate.

    Args:

        data: (np.array) data (can be 1d or 2d, time by features)
        random_state: (int, None, or np.random.RandomState) Initial random seed (default: None)

    Returns:

        shifted_data: (np.array) phase randomized data
    """
    random_state = check_random_state(random_state)

    data = np.array(data)
    fft_data = fft(data, axis=0)

    if data.shape[0] % 2 == 0:
        pos_freq = np.arange(1, data.shape[0] // 2)
        neg_freq = np.arange(data.shape[0] - 1, data.shape[0] // 2, -1)
    else:
        pos_freq = np.arange(1, (data.shape[0] - 1) // 2 + 1)
        neg_freq = np.arange(data.shape[0] - 1, (data.shape[0] - 1) // 2, -1)

    if len(data.shape) == 1:
        phase_shifts = random_state.uniform(0, 2 * np.pi, size=(len(pos_freq)))
        fft_data[pos_freq] *= np.exp(1j * phase_shifts)
        fft_data[neg_freq] *= np.exp(-1j * phase_shifts)
    else:
        phase_shifts = random_state.uniform(
            0, 2 * np.pi, size=(len(pos_freq), data.shape[1])
        )
        fft_data[pos_freq, :] *= np.exp(1j * phase_shifts)
        fft_data[neg_freq, :] *= np.exp(-1j * phase_shifts)
        
    return np.real(ifft(fft_data, axis=0))


# ------------------ Main: Event-Level ISC ------------------ #
event_data = defaultdict(list)

# Step 1: Load and group z-scored data by story across subjects
for sub in SUBJ_IDS:
    group_num = (sub - 1000) % 3 or 3
    for run in ['run_1', 'run_2']:
        file_path = os.path.join(DAT_PATH, run, f"{sub}_{group_num}_{run}_{FILTER_TYPE}_event_aligned.csv")
        pupil_file = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/5_timelocked/{EXP_TYPE}/{run}/{sub}_{group_num}_{run}_{FILTER_TYPE}_2SD_downsample_to_sec_{EXP_TYPE}.csv'))
        if not os.path.exists(file_path):
            print(f"Missing pupil file for subject {sub}, run {run}")
            continue
        df = pd.read_csv(file_path)
        full_pupil = pd.read_csv(pupil_file)['pupilSize'].values
        for _, row in df.iterrows():
            story = row['story']
            event_num = int(row['event_num'])
            start = int(row['event_start_sec'])
            end = int(row['event_end_sec'])

            if end > len(full_pupil):
                continue

            segment = full_pupil[start:end]
            if len(segment) < 2:
                continue

            z_segment = stats.zscore(segment, nan_policy='omit')
            if np.isnan(z_segment).all():
                continue

            key = (story, event_num)
            event_data[key].append(z_segment)
            # print(f"Added: {story} - Event {event_num} - Segment length: {len(z_segment)}")

# Step 2: Compute ISC + Permutation
isc_results = []

for (story, ev_num), segments in event_data.items():

    if len(segments) < 2:
        print(f"Skipping event {(story, ev_num)}: not enough subjects.")
        continue

    # Pad to equal length
    max_len = max(len(s) for s in segments)
    data_matrix = np.full((max_len, len(segments)), np.nan)
    for i, s in enumerate(segments):
        data_matrix[:len(s), i] = s

    # One-to-average ISC
    isc_vals = []
    for i in range(data_matrix.shape[1]):
        this = data_matrix[:, i]
        others = np.delete(data_matrix, i, axis=1)
        avg = np.nanmean(others, axis=1)
        mask = ~np.isnan(this) & ~np.isnan(avg)
        if np.sum(mask) > 1:
            r = np.corrcoef(this[mask], avg[mask])[0, 1]
            # r = np.clip(r, -0.999999, 0.999999)      # avoid ±1
            isc_vals.append(r)
    
    # if len(isc_vals) < 2:
    #     continue

    isc_z = np.arctanh(isc_vals)
    true_mean_z = np.nanmean(isc_z)
    true_mean_r = np.tanh(true_mean_z)
    print(f"True ISC (r): {true_mean_r:.3f}")

    # Step 3: Permutation test
    perm_ISC_mean = []
    for _ in range(ITERATIONS):
        perm_vals = []
        for i in range(data_matrix.shape[1]):
            this = phase_randomize(data_matrix[:, i])
            others = np.delete(data_matrix, i, axis=1)
            avg = np.nanmean(others, axis=1)
            mask = ~np.isnan(this) & ~np.isnan(avg)
            if np.sum(mask) > 1:
                r = np.corrcoef(this[mask], avg[mask])[0, 1]
                # r = np.clip(r, -0.999999, 0.999999)
                perm_vals.append(r)
        perm_z = np.arctanh(perm_vals)
        perm_ISC_mean.append(np.tanh(np.nanmean(perm_z)))

    perm_ISC_mean = np.array(perm_ISC_mean)
    p_one = (1 + np.sum(perm_ISC_mean > true_mean_r)) / (1 + ITERATIONS)
    p_opposite = (1 + np.sum(perm_ISC_mean < -true_mean_r)) / (1 + ITERATIONS)
    p_twotail = p_one + p_opposite

    isc_results.append({
        'story': story,
        'event_num': ev_num,
        'isc_r': true_mean_r,
        'p': p_twotail
    })

# Save results
isc_df = pd.DataFrame(isc_results)
out_path = os.path.join(SAVE_PATH, f'event_level_isc_{FILTER_TYPE, LOWCUT_HZ, HIGHCUT_HZ}.csv')
isc_df.to_csv(out_path, index=False)
print(f"\nSaved event-level ISC results: {out_path}")
