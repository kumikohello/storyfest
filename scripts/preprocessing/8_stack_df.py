import pandas as pd
import glob
import os

EXP_TYPE = "encoding"
FILTER_TYPE = "lowpass"  # "lowpass" or "bandpass"

os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/6_eventlocked/' + EXP_TYPE))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/8_stack_df/' + EXP_TYPE))
os.makedirs(SAVE_PATH, exist_ok=True)

if EXP_TYPE == "encoding":
    runs = ['run_1', 'run_2']

def concatenate_csv_files(directories, output_file, runs):
    """
    Concatenates CSV files from multiple directories into a single CSV file.

    Args:
        directories (list): A list of directory paths containing CSV files.
        output_file (str): The path to the output CSV file.
    """
    current_save = output_file
    os.makedirs(current_save, exist_ok=True)

    all_files = []

    for run in runs:
        current_dat = os.path.join(directories, run) if run else directories

        #for directory in current_dat:
        csv_files = glob.glob(os.path.join(current_dat, f"*_{FILTER_TYPE}_*.csv"))
        all_files.extend(csv_files)

        all_df = []
        for f in all_files:
            df = pd.read_csv(f)
            all_df.append(df)

    merged_df = pd.concat(all_df, ignore_index=True)
    merged_df.to_csv(os.path.join(output_file, f"stacked_events_{FILTER_TYPE}.csv"), index=False)

concatenate_csv_files(DAT_PATH, SAVE_PATH, runs)

