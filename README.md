# EEG Theta- and Alpha-Band Microstate HMM Analysis for Aging

## Project Overview

This repository contains the analysis pipeline for studying age-related changes in EEG theta- and alpha-band (4–13 Hz) microstates using Hidden Markov Models (HMM). The pipeline extracts instantaneous frequency (IF) and instantaneous amplitude (IA) features from multichannel EEG, trains a Gaussian HMM, and compares microstate dwell times, occupancy, and state-transition probabilities between a younger group and an older group.

## Background

EEG microstates are quasi-stable patterns of scalp electric field topography that recur at the millisecond timescale and are thought to reflect global brain network dynamics. This study focuses on the theta-alpha band (4–13 Hz) and uses an HMM-based approach to characterize how microstate statistics—dwell time, occupancy, and transition probabilities—change with aging.

**Key methodological steps:**
1. FIR band-pass filtering in the theta-alpha band (4–13 Hz)
2. Analytic signal computation via Hilbert transform
3. Instantaneous frequency (IF) from the derivative of the unwrapped phase
4. Instantaneous amplitude (IA) proxy from the wrapped phase
5. Spatial demeaning and z-scoring across channels at each time step
6. Gaussian HMM trained on PCA-reduced IF features
7. Group comparison (Young vs. Old) of dwell time, occupancy, and transition probabilities with FDR correction

## Data Requirements

### Input format

Each subject's EEG data must be a tab-delimited text file with:
- One header row
- Column 1: a label/index column (ignored)
- Columns 2–17: 16-channel EEG time series (floating-point, pre-processed to phase)

Channel order: `Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, Fz, Pz, T5, T6`

Sampling rate: **200 Hz**

Recommended minimum recording length: > 55 seconds (11 000 samples at 200 Hz) per subject, as only samples 1001–11000 (MATLAB 1-indexed) are used.

### Directory structure

```
<project_root>/
├── DPD/
│   └── HC_all_h_unwrap/
│       ├── <subject_id_1>/
│       │   └── <subject_id_1>.txt
│       ├── <subject_id_2>/
│       │   └── <subject_id_2>.txt
│       └── ...
└── Frontiers_aging/
    └── code/              ← this repository
        ├── low.txt        ← list of younger subject IDs (one per line)
        └── high.txt       ← list of older subject IDs (one per line)
```

`low.txt` and `high.txt` list subject IDs one per line, matching the directory names under `HC_all_h_unwrap/`.

## Installation

Install [uv](https://docs.astral.sh/uv/) if not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the virtual environment and install all dependencies:

```bash
cd code/
uv sync
```

## Usage

Run the scripts in order from within the `code/` directory:

```bash
cd code/

# Step 1: Preprocess raw EEG → result_Y_O_IF_IA.mat
uv run python 01_preprocess.py

# Step 2: Train Gaussian HMM → hmm_model.pkl
uv run python 02_hmm_train.py

# Step 3: Analyse dwell time and occupancy → SVG figures
uv run python 03_analyze_dwell_occ.py

# Step 4: Analyse state-transition probabilities → SVG figures
uv run python 04_analyze_transition.py
```

Each script resolves file paths relative to its own location, so it can also be run from any working directory using an absolute path:

```bash
uv run python /path/to/code/01_preprocess.py
```

## Pipeline Overview

```
low.txt / high.txt
        │
        ▼
01_preprocess.py
  ├── Load raw EEG (16 ch, tab-delimited)
  ├── Z-score across time
  ├── FIR band-pass filter (theta-alpha: 4–13 Hz)
  ├── Hilbert transform → analytic signal
  ├── Instantaneous frequency (IF): d/dt unwrapped phase
  ├── Instantaneous amplitude (IA): wrapped phase
  ├── Median filter (kernel=21)
  ├── Spatial demeaning + z-score across channels
  ├── Epoch extraction [1001:11000]
  └── Save → result_Y_O_IF_IA.mat
                │
                ▼
        02_hmm_train.py
          ├── Load result_Y_O_IF_IA.mat
          ├── PCA on IF features (16 → 16 components)
          ├── Gaussian HMM (n_components=5, full covariance)
          └── Save → hmm_model.pkl
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
03_analyze_dwell_occ.py    04_analyze_transition.py
  ├── Per-subject dwell        ├── Per-subject transition
  │   times and occupancy      │   probability matrices
  ├── log1p + normality        ├── log1p + normality
  │   → t-test / MWU           │   → t-test / MWU
  ├── FDR correction           ├── FDR correction
  └── Save SVG figures         └── Save SVG figures
```

## Output Files

| Script | Output file | Description |
|--------|-------------|-------------|
| `01_preprocess.py` | `result_Y_O_IF_IA.mat` | Preprocessed features for all subjects |
| `02_hmm_train.py`  | `hmm_model.pkl`         | Trained HMM + PCA transformer |
| `03_analyze_dwell_occ.py` | `dwell_time_violin.svg` | Violin plot: mean dwell time (seconds) per state |
| `03_analyze_dwell_occ.py` | `occupancy_violin.svg` | Violin plot: occupancy per state |
| `04_analyze_transition.py` | `transition_probability.svg` | Signed significance map (sign × -log10 q) |
| `04_analyze_transition.py` | `transition_probability_young.svg` | Mean transition matrix, Young group |
| `04_analyze_transition.py` | `transition_probability_old.svg` | Mean transition matrix, Old group |

### Variables in `result_Y_O_IF_IA.mat`

| Variable | Shape | Description |
|----------|-------|-------------|
| `c_z_diff`    | `[N_all × 10000, 32]` | All subjects (younger then older) |
| `c_z_diff_hc` | `[N_hc  × 10000, 32]` | Younger group only |
| `c_z_diff_ad` | `[N_ad  × 10000, 32]` | Older group only |

Columns 0–15: z-scored IF features; columns 16–31: z-scored IA features.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sampling rate | 200 Hz | EEG acquisition rate |
| Channels | 16 | Standard 10-20 system subset |
| Theta-alpha band | 4–13 Hz | Band-pass filter cutoffs |
| FIR order | `floor(T/3)` taps | Linear-phase FIR (odd taps enforced) |
| Median filter kernel | 21 | Corresponds to MATLAB `medfilt1(X, 20)` |
| Epoch length | 10 000 samples (50 s) | Samples 1001–11000 per subject |
| HMM states | 5 | Number of hidden states |
| HMM iterations | 100 000 | Maximum EM iterations |
| HMM seed | 2 | `np.random.seed(2)` in `02_hmm_train.py` |
| Preprocessing seed | 1 | `np.random.seed(1)` in `01_preprocess.py` |
| Significance level | 0.05 | FDR-corrected (Benjamini-Hochberg) |

## Citation

If you use this code, please cite:

> under review
