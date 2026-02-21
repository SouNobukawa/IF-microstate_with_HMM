"""01_preprocess.py

EEG preprocessing pipeline for theta- and alpha-band microstate HMM analysis.

For each subject:
  1. Load raw EEG (tab-delimited, 16 channels)
  2. Z-score normalize across time
  3. Band-pass filter in the theta-alpha band (4-13 Hz) using a linear-phase FIR filter
  4. Compute analytic signal via Hilbert transform
  5. Extract instantaneous frequency (IF) from unwrapped phase derivative
  6. Extract instantaneous amplitude (IA) from wrapped phase
  7. Apply median filtering to IF and IA
  8. Spatially demean and z-score across channels
  9. Concatenate IF and IA features [T-1, 32]
 10. Extract a fixed epoch [1001:11000] (MATLAB 1-indexed) per subject

Output: result_Y_O_IF_IA.mat containing
  - c_z_diff      : all subjects stacked [N_all * 10000, 32]
  - c_z_diff_hc   : younger group        [N_hc  * 10000, 32]
  - c_z_diff_ad   : older group          [N_ad  * 10000, 32]

Subject lists are read from low.txt (younger) and high.txt (older).
Data files are expected at ../../DPD/HC_all_h_unwrap/<id>/<id>.txt
relative to this script's directory.
"""

import numpy as np
import pandas as pd
import scipy.signal
import scipy.ndimage
import scipy.io
import scipy.stats
from pathlib import Path

np.random.seed(1)  # Reproducibility (MATLAB: rng(1))

# -------------------------
# Parameters
# -------------------------
Fs = 200                         # Sampling rate (Hz)
Fl = [2, 4, 4, 13, 30]          # High-pass cutoff frequencies (Hz)
Fh = [4, 8, 13, 30, 60]         # Low-pass cutoff frequencies (Hz)
BAND_IDX = 2                     # Theta-alpha band index: 4-13 Hz

# MATLAB front_list / occi_list (1-indexed) converted to 0-indexed
front_list = [0, 1, 2, 3, 10, 11, 12]  # MATLAB: [1 2 3 4 11 12 13]
occi_list  = [8, 9]                      # MATLAB: [9 10]

# Median filter kernel size
# MATLAB medfilt1(X, 20) internally rounds 20 (even) up to 21 (odd)
MEDFILT_SIZE = 21

# Epoch extraction: MATLAB 1001:11000 (1-indexed) → Python [1000:11000] (0-indexed)
BEGIN_EPOCH = 1000
END_EPOCH   = 11000   # exclusive; yields 10000 samples per subject

# Category file names (lists of subject IDs)
CATE_FILE_NAMES = ['low.txt', 'high.txt']

# Root directory of this script
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_ROOT  = SCRIPT_DIR / '../../DPD/HC_all_h_unwrap'

# -------------------------
# Main processing loop
# -------------------------
c_z_diff_hc = None
c_z_diff_ad = None

for cate_idx, cate_file in enumerate(CATE_FILE_NAMES):
    cate_path = SCRIPT_DIR / cate_file
    with open(cate_path) as f:
        cate_list = [line.strip() for line in f if line.strip()]

    group_name = 'Younger' if cate_idx == 0 else 'Older'
    print(f"Category {cate_idx + 1} ({group_name}): {len(cate_list)} subjects")

    for c_i, subj_name in enumerate(cate_list):
        print(f"  [{c_i + 1}/{len(cate_list)}] {subj_name}")

        file_name = DATA_ROOT / subj_name / f'{subj_name}.txt'

        # Load raw EEG: tab-delimited, 1 header row
        # MATLAB: importdata(f, '\t', 1) → .data(:, 2:17)
        # Columns 2:17 in MATLAB (1-indexed) = columns 1:17 in Python (0-indexed)
        df = pd.read_csv(file_name, sep='\t', header=0)
        RawData = df.values[:, 1:17].astype(float)  # [T, 16]
        T = RawData.shape[0]

        # Z-score across time (MATLAB: zscore(RawData) → axis=0)
        ZsRawData = scipy.stats.zscore(RawData, axis=0)

        # --- FIR band-pass filter (theta-alpha: 4-13 Hz) ---
        passband = [Fl[BAND_IDX] / (Fs / 2), Fh[BAND_IDX] / (Fs / 2)]

        # MATLAB: fir1(floor(T/3) - 1, passband)
        #   filter order N = floor(T/3) - 1  →  num taps = floor(T/3)
        # Band-pass FIR requires an odd number of taps; round up if even.
        numtaps = int(np.floor(T / 3))
        if numtaps % 2 == 0:
            numtaps += 1
        b = scipy.signal.firwin(numtaps, passband, pass_zero=False)

        # Zero-phase filtering (MATLAB: filtfilt(fir, 1, ZsRawData))
        filtered_ch = scipy.signal.filtfilt(b, [1.0], ZsRawData, axis=0)

        # --- Analytic signal via Hilbert transform ---
        ylp_ch = scipy.signal.hilbert(filtered_ch, axis=0)  # [T, 16], complex

        # Unwrapped phase (for instantaneous frequency)
        angle_ch = np.unwrap(np.angle(ylp_ch), axis=0)   # [T, 16]

        # Wrapped phase (for instantaneous amplitude proxy)
        # Note: MATLAB variable was named abs_ch but stores angle(), not abs()
        abs_ch = np.angle(ylp_ch)                         # [T, 16]

        # --- Instantaneous frequency (IF) ---
        # Derivative of unwrapped phase, then median filter
        # MATLAB: medfilt1(diff(angle_ch), 20)
        m_if_ch = scipy.ndimage.median_filter(
            np.diff(angle_ch, axis=0), size=(MEDFILT_SIZE, 1)
        )  # [T-1, 16]

        # --- Instantaneous amplitude (IA) proxy ---
        # Wrapped phase, median filtered
        m_ia_ch = scipy.ndimage.median_filter(abs_ch, size=(MEDFILT_SIZE, 1))  # [T, 16]

        # Convert IF from rad/sample to Hz
        chose_angle_ch = (m_if_ch / (2 * np.pi)) * Fs   # [T-1, 16]

        # Spatial demeaning: subtract channel mean at each time step
        # MATLAB: mean(X, 2) → axis=1
        m_angle_ch = np.mean(chose_angle_ch, axis=1, keepdims=True)  # [T-1, 1]
        m_abs_ch   = np.mean(m_ia_ch,        axis=1, keepdims=True)  # [T, 1]

        diff_angle_m = chose_angle_ch - m_angle_ch            # [T-1, 16]
        diff_abs_m   = (m_ia_ch - m_abs_ch)[:-1, :]           # [T-1, 16]
        # diff_abs_m(1:end-1,:) in MATLAB trims last row to match diff_angle_m length

        # Z-score across channels at each time step
        # MATLAB: zscore(X, 0, 2) → axis=1
        z_diff      = scipy.stats.zscore(diff_angle_m, axis=1)   # [T-1, 16]
        z_diff_abs_m = scipy.stats.zscore(diff_abs_m,  axis=1)   # [T-1, 16]

        # Concatenate IF and IA features
        z_diff = np.concatenate([z_diff, z_diff_abs_m], axis=1)  # [T-1, 32]

        # Extract fixed epoch: MATLAB 1001:11000 → Python [1000:11000]
        epoch = z_diff[BEGIN_EPOCH:END_EPOCH, :]  # [10000, 32]

        if cate_idx == 0:
            c_z_diff_hc = epoch if c_z_diff_hc is None else np.vstack([c_z_diff_hc, epoch])
        else:
            c_z_diff_ad = epoch if c_z_diff_ad is None else np.vstack([c_z_diff_ad, epoch])

# Concatenate both groups (young first, then old)
c_z_diff = np.vstack([c_z_diff_hc, c_z_diff_ad])

out_path = SCRIPT_DIR / 'result_Y_O_IF_IA.mat'
print(f"\nSaving {out_path}")
print(f"  c_z_diff     shape: {c_z_diff.shape}")
print(f"  c_z_diff_hc  shape: {c_z_diff_hc.shape}")
print(f"  c_z_diff_ad  shape: {c_z_diff_ad.shape}")

scipy.io.savemat(str(out_path), {
    'c_z_diff':     c_z_diff,
    'c_z_diff_hc':  c_z_diff_hc,
    'c_z_diff_ad':  c_z_diff_ad,
})

print("Done.")
