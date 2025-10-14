# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu), Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: July 19, 2025
# Description: The script calculated group mean pupil dilation and excludes noisy participants
# Noise is calculated based on the "derivative" (i.e., change in pupil size relative to the preceding sample; 
#                                                aka,  sample N - sample N-1 pupil size)
# Steps:
# 1. Load aligned pupil data
# 2. Create a distribution of the "derivatives"
# 3. Calculate the cutoff from the distribution (e.g., 1 SD, 2 SD, 3 SD)
# 4. Exclude participants if 25% data points are above/below the cutoff
# i.e., if 25% of the data points have a derivative greater/less than the cutoff, exclude the participant

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/UChicago/CASNL/storyfest/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
EXP_TYPE = "encoding" # "encoding" or "recall"
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/1_aligned/' + EXP_TYPE))

# Standard score for identifying cutoffs (SDSCORE = 1, 2, 3, ...)
# The higher the SDSCORE, the more stringent the cutoff for noise and more participants will be included
SDSCORE = 3

SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/2_valid_pts/' + EXP_TYPE, str(SDSCORE) + "SD"))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    
SUBJ_IDS = range(1001,1046)

runs = ['run_1', 'run_2']

# ------------------ Define functions ------------------ #
def calculate_derivative(arr):
    """
    Calculates the sample-to-sample difference in pupil size, while preserving NaNs for better downstream consistency.
    
    Input: 
    - arr (pd.Series or np.array)
    
    Output: 
    - pd.Series of differences with same length

    """

    if isinstance(arr, np.ndarray):
        arr = pd.Series(arr)
    
    diff = arr.diff()  # preserves NaN
    
    return diff

def create_dist_find_cutoff(pupil_diff, z):
    """
    Creates a distribution of the data and finds the cutoff value based on the distribution's standard deviation (or pupil differences).
    
    Inputs:
    - pupil_diff (numpy array) containing the derivative of the pupil size data
    - z (float) specifying the number of standard deviations to consider
    
    Outputs:
    - cutoff (float) specifying the cutoff value for the data

    """
    
    # Calculates median absolute deviation (MAD), which is more robust to outliers
    pupil_diff = pupil_diff[~np.isnan(pupil_diff)]

    median = np.median(pupil_diff)
    mad = np.median(np.abs(pupil_diff - median))
    robust_sd = mad * 1.4826
    robust_cutoff = median + z * (mad * 1.4826) # pseudo SD for a normal distribution

    mean = np.mean(pupil_diff)
    std = np.std(pupil_diff)
    standard_cutoff = mean + z * std

    return {
        "robust_cutoff": robust_cutoff,
        "robust_sd": robust_sd,
        "median": median,
        "mad": mad,
        "standard_cutoff": standard_cutoff,
        "mean": mean,
        "std": std
    }

def calc_prop_noisy(pupil_diff, cutoff):
    """
    Calculates what proportion of the data is considered noisy based on the cutoff value.
    
    Inputs:
    - pupil_diff (numpy array) containing the derivative of the pupil size data
    - cutoff (float) specifying the cutoff value for the data
    
    Outputs:
    - prop_noisy (float) specifying the proportion of noisy data points (in percentage)

    """
    valid_diff = pupil_diff[~np.isnan(pupil_diff)]
    noisy_pts = np.where(np.abs(valid_diff) > cutoff) # Indices of the "noisy" data points
    prop_noisy = len(noisy_pts[0])/len(valid_diff)*100
    
    return prop_noisy

# ------------------- Main ------------------ #
summary_rows = []
for run in runs:
    current_dat_path = os.path.join(DAT_PATH, run)
    current_save_path = os.path.join(SAVE_PATH, run)
    os.makedirs(current_save_path, exist_ok=True)

    pupil_diff_allsub = []

    for sub in SUBJ_IDS:
        group_num = (sub - 1000) % 3
        if group_num == 0:
            group_num = 3
        input_file = os.path.join(current_dat_path, f"{sub}_aligned.csv")
        
        if not os.path.exists(input_file):
            print(f"No Input File for Participant {sub}: {input_file}")
            continue

        dat = pd.read_csv(input_file)
        pupil_raw = dat['pupilSize']
        pupil_diff = calculate_derivative(pupil_raw)
        pupil_diff_allsub.extend(pupil_diff.dropna())

    stats = create_dist_find_cutoff(np.array(pupil_diff_allsub), SDSCORE)
    cutoff = stats["robust_cutoff"]
    print(f'cutoff: {cutoff}')

    excluded = []
    for sub in SUBJ_IDS:
        group_num = (sub - 1000) % 3
        if group_num == 0:
            group_num = 3

        input_file = os.path.join(current_dat_path, f"{sub}_aligned.csv")

        if not os.path.exists(input_file):
            continue

        dat = pd.read_csv(input_file)
        pupil_raw = dat['pupilSize']
        pupil_diff = calculate_derivative(pupil_raw)
        prop_noisy = calc_prop_noisy(pupil_diff, cutoff)

        included = prop_noisy < 25

        summary_rows.append({
            "subject": sub,
            "group": group_num,
            "percent_noisy": round(prop_noisy, 2),
            "included": included,
            "run": run,
            "SDSCORE": SDSCORE,
            "median": stats["median"],
            "mad": stats["mad"],
            "robust_sd": stats["robust_sd"],
            "robust_cutoff": stats["robust_cutoff"],
            "mean": stats["mean"],
            "std": stats["std"],
            "standard_cutoff": stats["standard_cutoff"]
        })

        if not included:
            excluded.append(sub)
            print(f"{sub}: {prop_noisy:.2f}% noisy. EXCLUDED.") 
        else:
            outname = f"{sub}_valid_{run}_{SDSCORE}SD.csv"
            dat.to_csv(os.path.join(current_save_path, outname), index=False)
            print(f"{sub}: {prop_noisy:.2f}% noisy. INCLUDED and saved to {outname}")
        
    plt.hist(pupil_diff_allsub, bins=200)
    plt.title("Distribution of pupil size derivatives")
    plt.xlabel("Derivative value")
    plt.ylabel("Frequency")
    plot_output_path = os.path.join(SAVE_PATH, f"{run}_pupil_size_derivatives.png")
    plt.savefig(plot_output_path, dpi=300)
    print(f"Saved plot")
    plt.close()

        # if prop_noisy >= 25:
        #     excluded.append(sub)
        #     print(f"{sub}: {prop_noisy:.2f}% noisy. EXCLUDED.") 

        # else:
        #     outname = f"{sub}_valid_{run}_{SDSCORE}SD.csv"
        #     dat.to_csv(os.path.join(current_save_path, outname), index=False)
        #     print(f"{sub}: {prop_noisy:.2f}% noisy. INCLUDED and saved to {outname}")
        
    print(f"{run} (n={len(excluded)}): excluded {excluded}")

summary_df = pd.DataFrame(summary_rows)
excel_output_path = os.path.join(SAVE_PATH, f"noise_summary_SDSCORE_{SDSCORE}.xlsx")
summary_df.to_excel(excel_output_path, index=False)
print(f"\nSummary saved to {excel_output_path}")

            # 2D:
            # run_1 (n=11): excluded [1002, 1005, 1007, 1008, 1009, 1011, 1014, 1022, 1028, 1036, 1042]
            # run_2 (n=15): excluded [1002, 1005, 1007, 1008, 1009, 1011, 1013, 1014, 1016, 1022, 1025, 1028, 1036, 1039, 1042]

            # 3D:
            # run_1 (n=6): excluded [1002, 1009, 1011, 1014, 1028, 1042]
            # run_2 (n=8): excluded [1002, 1009, 1011, 1014, 1016, 1022, 1028, 1042]