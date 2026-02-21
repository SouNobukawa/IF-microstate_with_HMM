"""04_analyze_transition.py

Analyse per-subject microstate state-transition probabilities for the
young (HC) and old (AD) groups using the trained HMM.

Loads:
  result_Y_O_IF_IA.mat  -- preprocessed EEG features (01_preprocess.py)
  hmm_model.pkl         -- trained HMM + PCA (02_hmm_train.py)

Outputs:
  transition_probability.svg        -- signed significance map (sign × -log10(q))
  transition_probability_young.svg  -- mean transition matrix, Young group
  transition_probability_old.svg    -- mean transition matrix, Old group

Statistical testing (per transition i→j):
  - log1p transform applied before testing
  - Normality (D'Agostino-Pearson); equal-variance (Levene)
  - Welch t-test if both groups normal, else Mann-Whitney U
  - FDR correction (Benjamini-Hochberg) across all n_states² transitions
"""

import pickle
from pathlib import Path
from itertools import groupby

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import ttest_ind, levene, normaltest, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ===============================
# Settings
# ===============================
APPLY_LOG_TRANSFORM    = True    # use log1p before testing
PRINT_DIAGNOSTICS      = True    # print test name and p-values
EPS                    = 1e-300  # floor for -log10(q) to avoid log10(0)
USE_LOG_SCALE_FOR_MEAN = True    # log10 scale for mean heatmaps
LOG_EPS                = 1e-12   # clip before log10

SAMPLES_PER_SUBJECT = 10000

SCRIPT_DIR = Path(__file__).parent.resolve()

# ===============================
# Statistical helpers
# ===============================
def _finite(a):
    a = np.asarray(a)
    return a[np.isfinite(a)]


def _maybe_log1p(a):
    if not APPLY_LOG_TRANSFORM:
        return a
    a = np.asarray(a)
    a = np.clip(a, 0, None)
    return np.log1p(a)


def two_sample_test(data_o, data_y, label="", print_diag=PRINT_DIAGNOSTICS):
    """Decide between t-test and Mann-Whitney U based on normality."""
    a = _finite(np.asarray(data_o))
    b = _finite(np.asarray(data_y))

    if a.size < 2 or b.size < 2:
        if print_diag:
            print(f"[{label}] Skipped: insufficient samples (n_old={a.size}, n_young={b.size})")
        return np.nan, np.nan, "insufficient", a, b

    a_t = _maybe_log1p(a)
    b_t = _maybe_log1p(b)

    norm_a_ok = a_t.size >= 8 and normaltest(a_t).pvalue > 0.05
    norm_b_ok = b_t.size >= 8 and normaltest(b_t).pvalue > 0.05

    if norm_a_ok and norm_b_ok:
        p_var     = levene(a_t, b_t).pvalue
        equal_var = p_var > 0.05
        stat, p   = ttest_ind(a_t, b_t, equal_var=equal_var)
        method    = f"t-test ({'equal var' if equal_var else 'Welch'})"
        if print_diag:
            print(f"[{label}] {method}, t={stat:.3g}, p={p:.3g}")
    else:
        stat, p = mannwhitneyu(a_t, b_t, alternative='two-sided')
        method  = "Mann-Whitney U"
        if print_diag:
            print(f"[{label}] {method}, U={stat:.3g}, p={p:.3g}")

    return stat, p, method, a_t, b_t


def cohens_d_from_samples(a, b):
    """
    Cohen's d (Old - Young) using pooled SD.
    a and b are already transformed (log1p if APPLY_LOG_TRANSFORM=True).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    sp = np.sqrt(((a.size - 1) * s1 + (b.size - 1) * s2) / (a.size + b.size - 2))
    if sp == 0 or not np.isfinite(sp):
        return np.nan
    return (np.mean(a) - np.mean(b)) / sp  # Old - Young


def mean_sd(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return np.mean(a), np.std(a, ddof=1)


# ===============================
# Load data
# ===============================
data    = loadmat(str(SCRIPT_DIR / 'result_Y_O_IF_IA.mat'))
X_young = data['c_z_diff_hc']
X_old   = data['c_z_diff_ad']

# Derive subject counts dynamically
n_subjects_y = X_young.shape[0] // SAMPLES_PER_SUBJECT
n_subjects_o = X_old.shape[0]   // SAMPLES_PER_SUBJECT

with open(SCRIPT_DIR / 'hmm_model.pkl', 'rb') as f:
    loaded_model, loaded_pca_IF = pickle.load(f)

n_states = loaded_model.n_components

# ===============================
# Per-subject transition probabilities, dwell times, and occupancy
# ===============================
def compute_subject_stats(X_group, n_subjects):
    transition_probs = []
    dwell_times_all  = []
    occupancy_all    = []

    for i in range(n_subjects):
        X_subj = X_group[i * SAMPLES_PER_SUBJECT : (i + 1) * SAMPLES_PER_SUBJECT]
        X_pca  = loaded_pca_IF.transform(X_subj[:, :16])
        states = loaded_model.predict(X_pca)

        # Transition counts → probabilities
        trans_mat = np.zeros((n_states, n_states))
        for prev_s, next_s in zip(states[:-1], states[1:]):
            trans_mat[prev_s, next_s] += 1
        with np.errstate(invalid='ignore', divide='ignore'):
            trans_prob = trans_mat / trans_mat.sum(axis=1, keepdims=True)
        transition_probs.append(trans_prob)

        # Dwell times
        dwell_times = [[] for _ in range(n_states)]
        for state, group in groupby(states):
            dwell_times[state].append(len(list(group)))
        mean_dwell = np.full(n_states, np.nan)
        for s in range(n_states):
            if len(dwell_times[s]) > 0:
                mean_dwell[s] = np.mean(dwell_times[s])
        dwell_times_all.append(mean_dwell)

        occupancy = np.bincount(states, minlength=n_states) / len(states)
        occupancy_all.append(occupancy)

    return np.array(transition_probs), np.array(dwell_times_all), np.array(occupancy_all)


tp_array_y, _, _ = compute_subject_stats(X_young, n_subjects_y)
tp_array_o, _, _ = compute_subject_stats(X_old,   n_subjects_o)

# ===============================
# Transition significance: signed -log10(q)
# ===============================
raw_stats  = []
raw_methods = []
p_vals     = []
dirs       = []
d_vals     = []

for i in range(n_states):
    for j in range(n_states):
        data_y = tp_array_y[:, i, j]
        data_o = tp_array_o[:, i, j]

        stat, p, method, a_t, b_t = two_sample_test(data_o, data_y, label=f"Transition {i}->{j}")
        raw_stats.append(stat)
        raw_methods.append(method)
        p_vals.append(p)

        # Cohen's d on the same scale as testing (log1p if enabled)
        d = cohens_d_from_samples(a_t, b_t)  # Old - Young
        d_vals.append(d)
        dirs.append(np.sign(d) if np.isfinite(d) else np.nan)

rejected, q_vals, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
q_arr = np.array(q_vals, dtype=float)
q_arr[q_arr <= 0] = EPS
dirs_arr     = np.array(dirs, dtype=float)
signed_scores = dirs_arr * (-np.log10(q_arr))

score_matrix  = signed_scores.reshape(n_states, n_states)
method_matrix = np.array(raw_methods, dtype=object).reshape(n_states, n_states)
sig_mask      = rejected.reshape(n_states, n_states)

# ===============================
# Plot 1: Signed significance map
# ===============================
plt.figure(figsize=(9, 7))
ax = sns.heatmap(score_matrix, vmin=-3.5, vmax=3.5, cmap='RdBu_r', center=0)
for i in range(n_states):
    for j in range(n_states):
        if sig_mask[i, j]:
            marker = '*' if 't-test' in method_matrix[i, j] else '\u2020'
            ax.text(j + 0.5, i + 0.5, marker,
                    color='black', ha='center', va='center',
                    fontsize=16, fontweight='bold')
plt.title("Signed significance map (sign \u00d7 -log10(q))\n* t-test, \u2020 MWU")
plt.xlabel("To State")
plt.ylabel("From State")
plt.tight_layout()
plt.savefig(str(SCRIPT_DIR / 'transition_probability.svg'), format='svg', dpi=300)
plt.close()

# ===============================
# Plot 2 & 3: Mean transition probability heatmaps
# ===============================
mean_tp_y = np.nanmean(tp_array_y, axis=0)
mean_tp_o = np.nanmean(tp_array_o, axis=0)

if USE_LOG_SCALE_FOR_MEAN:
    mean_tp_y_plot = np.log10(np.clip(mean_tp_y, LOG_EPS, 1.0))
    mean_tp_o_plot = np.log10(np.clip(mean_tp_o, LOG_EPS, 1.0))
    vmin  = np.nanmin([mean_tp_y_plot, mean_tp_o_plot])
    vmax  = np.nanmax([mean_tp_y_plot, mean_tp_o_plot])
    label = "log10(Probability)"
else:
    mean_tp_y_plot = mean_tp_y
    mean_tp_o_plot = mean_tp_o
    vmin, vmax = 0.0, 1.0
    label = "Probability"

mask_y = ~np.isfinite(mean_tp_y_plot)
mask_o = ~np.isfinite(mean_tp_o_plot)

# Young group
plt.figure(figsize=(8, 6))
ax = sns.heatmap(mean_tp_y_plot, vmin=vmin, vmax=vmax, cmap='viridis',
                 mask=mask_y, square=True)
plt.xlabel("To State")
plt.ylabel("From State")
ax.collections[0].colorbar.set_label(label, rotation=270, labelpad=15)
plt.tight_layout()
plt.savefig(str(SCRIPT_DIR / 'transition_probability_young.svg'), format='svg', dpi=300)
plt.close()

# Old group
ax = sns.heatmap(mean_tp_o_plot, vmin=vmin, vmax=vmax, cmap='viridis',
                 mask=mask_o, square=True)
plt.xlabel("To State")
plt.ylabel("From State")
ax.collections[0].colorbar.set_label(label, rotation=270, labelpad=15)
plt.tight_layout()
plt.savefig(str(SCRIPT_DIR / 'transition_probability_old.svg'), format='svg', dpi=300)
plt.close()

# ===============================
# Summary: Cohen's d for significant transitions
# ===============================
d_arr     = np.array(d_vals).reshape(n_states, n_states)
q_arr_mat = q_arr.reshape(n_states, n_states)
rej_arr   = rejected.reshape(n_states, n_states)

caption_lines = []
for i in range(n_states):
    for j in range(n_states):
        if rej_arr[i, j] and np.isfinite(d_arr[i, j]):
            caption_lines.append(f"{i}\u2192{j}: d = {d_arr[i, j]:.2f}")

print("\n[Cohen's d for significant transitions]")
print("; ".join(caption_lines))

# ===============================
# Summary table for significant transitions
# ===============================
rows = []
for i in range(n_states):
    for j in range(n_states):
        if rej_arr[i, j]:
            y_vals = tp_array_y[:, i, j]
            o_vals = tp_array_o[:, i, j]
            y_mean, y_sd = mean_sd(y_vals)
            o_mean, o_sd = mean_sd(o_vals)
            rows.append({
                "From":                      i + 1,  # 1-indexed for paper
                "To":                        j + 1,
                "Young_mean":               y_mean,
                "Young_SD":                 y_sd,
                "Old_mean":                 o_mean,
                "Old_SD":                   o_sd,
                "p":                         p_vals[i * n_states + j],
                "q(FDR)":                    q_arr_mat[i, j],
                "Cohen_d(Old-Young,log1p)":  d_arr[i, j],
                "Test":                      method_matrix[i, j],
            })

summary_tp = pd.DataFrame(rows)
print("\n=== Significant transition probabilities (q < 0.05) ===")
print(summary_tp.to_string(index=False))
