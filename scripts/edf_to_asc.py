import scipy.io
import os
import pandas as pd
import mat73

# === Step 1: Load your .mat file ===
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
filepath = os.path.join(_THISDIR, '../../data/pupil/2_mat/encoding/1001_encoding_ET.mat')
mat = mat73.loadmat(filepath)

# === Step 2: Extract arrays (adjust names if needed) ===
time = mat['time'].squeeze()
x = mat['x'].squeeze()
y = mat['y'].squeeze()
pupil = mat['pupil'].squeeze()

# === Step 3: Prepare output ASC-style file ===
with open("1001_storyfest_encoding_generated.asc", "w") as f:
    # --- HEADER (mocked for structure) ---
    f.write("START	EDF2ASC CONVERTED FILE\n")
    f.write("MSG\t{}\t!CONVERSION_START\n".format(int(time[0])))
    f.write("MSG\t{}\tALL_STORY_START\n".format(int(time[0])))

    # --- SAMPLES ---
    for t, x_val, y_val, p in zip(time, x, y, pupil):
        # Format: timestamp \t x \t y \t pupil
        f.write(f"{int(t)}\t{x_val:.1f}\t{y_val:.1f}\t{p:.1f}\n")
    
    # --- FOOTER ---
    f.write("MSG\t{}\t!CONVERSION_END\n".format(int(time[-1])))
    f.write("END\n")
