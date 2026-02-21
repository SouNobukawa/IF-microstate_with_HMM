"""03_analyze_dwell_occ.py

Analyse per-subject microstate dwell times and occupancy for the
young (HC) and old (AD) groups using the trained HMM.

Loads:
  result_Y_O_IF_IA.mat  -- preprocessed EEG features (01_preprocess.py)
  hmm_model.pkl         -- trained HMM + PCA (02_hmm_train.py)

Outputs:
  dwell_time_violin.svg  -- violin plot of mean dwell time per state
  occupancy_violin.svg   -- violin plot of occupancy per state

Statistical testing (per state):
  - log1p transform applied before testing
  - Normality (D'Agostino-Pearson); equal-variance (Levene)
  - Welch t-test if both groups are approximately normal, else Mann-Whitney U
  - FDR correction (Benjamini-Hochberg) across states
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
APPLY_LOG1P            = True   # apply log1p before statistical tests
PRINT_DIAGNOSTICS      = True   # print test name and p-values
MARKER_SIZE            = 18     # significance marker font size
TOP_MARGIN_RATIO_DWELL = 0.12   # extra headroom fraction for dwell plot
TOP_MARGIN_ABS_OCC     = 0.03   # extra headroom (absolute) for occupancy plot

SAMPLES_PER_SUBJECT = 10000

SCRIPT_DIR = Path(__file__).parent.resolve()

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
# Per-subject dwell times and occupancy
# ===============================
def compute_dwell_occupancy(X_group, n_subjects, n_states):
    dwell_all     = []
    occupancy_all = []
    for i in range(n_subjects):
        X_subj   = X_group[i * SAMPLES_PER_SUBJECT : (i + 1) * SAMPLES_PER_SUBJECT]
        X_pca    = loaded_pca_IF.transform(X_subj[:, :16])
        states   = loaded_model.predict(X_pca)

        dwell_times = [[] for _ in range(n_states)]
        for state, group in groupby(states):
            dwell_times[state].append(len(list(group)))

        mean_dwell = [np.mean(x) if len(x) > 0 else np.nan for x in dwell_times]
        dwell_all.append(mean_dwell)

        occupancy = np.bincount(states, minlength=n_states) / len(states)
        occupancy_all.append(occupancy)

    return np.array(dwell_all), np.array(occupancy_all)


dwell_y, occupancy_y = compute_dwell_occupancy(X_young, n_subjects_y, n_states)
dwell_o, occupancy_o = compute_dwell_occupancy(X_old,   n_subjects_o, n_states)

# ===============================
# Statistical helpers
# ===============================
def _transform(a):
    a = np.asarray(a)
    if APPLY_LOG1P:
        a = np.clip(a, 0, None)
        return np.log1p(a)
    return a


def two_sample_test(data_o, data_y, label=""):
    """log1p → normality check → Welch t-test or Mann-Whitney U."""
    a = _transform(data_o)
    b = _transform(data_y)

    if np.isfinite(a).sum() < 2 or np.isfinite(b).sum() < 2:
        if PRINT_DIAGNOSTICS:
            print(f"[{label}] Skipped: insufficient samples")
        return np.nan, "insufficient", np.nan

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    norm_a = (a.size >= 8) and (normaltest(a).pvalue > 0.05)
    norm_b = (b.size >= 8) and (normaltest(b).pvalue > 0.05)

    if norm_a and norm_b:
        p_var     = levene(a, b).pvalue
        equal_var = p_var > 0.05
        stat, p   = ttest_ind(a, b, equal_var=equal_var)
        method    = f"t-test ({'equal var' if equal_var else 'Welch'})"
    else:
        stat, p = mannwhitneyu(a, b, alternative='two-sided')
        method  = "Mann-Whitney U"

    if PRINT_DIAGNOSTICS:
        print(f"[{label}] {method}, stat={stat:.4g}, p={p:.4g}")
    return p, method, stat


def cohens_d(a, b, transform=True):
    """Cohen's d (b - a) with pooled SD."""
    a = np.asarray(a)
    b = np.asarray(b)
    if transform:
        a = _transform(a)
        b = _transform(b)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    sp = np.sqrt(((a.size - 1) * s1 + (b.size - 1) * s2) / (a.size + b.size - 2))
    if sp == 0:
        return np.nan
    return (np.mean(b) - np.mean(a)) / sp  # Old - Young when called as (young, old)


def hedges_g_from_d(d, n1, n2):
    """Small-sample bias correction (Hedges' g) from Cohen's d."""
    if not np.isfinite(d):
        return np.nan
    df = n1 + n2 - 2
    if df <= 0:
        return np.nan
    J = 1 - (3 / (4 * df - 1))
    return J * d


def mean_sd(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return np.mean(a), np.std(a, ddof=1)


# ===============================
# Dwell time analysis
# ===============================
p_vals, state_labels, methods_dwell, d_vals_dwell, g_vals_dwell = [], [], [], [], []

for s in range(n_states):
    p, method, _ = two_sample_test(dwell_o[:, s], dwell_y[:, s], label=f"Dwell State {s}")
    p_vals.append(p)
    state_labels.append(f"State {s}")
    methods_dwell.append(method)

    d = cohens_d(dwell_y[:, s], dwell_o[:, s], transform=True)
    g = hedges_g_from_d(d, np.isfinite(dwell_y[:, s]).sum(), np.isfinite(dwell_o[:, s]).sum())
    d_vals_dwell.append(d)
    g_vals_dwell.append(g)

rejected, q_vals, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')

sampling_rate = 200
dwell_y_sec   = dwell_y / sampling_rate
dwell_o_sec   = dwell_o / sampling_rate

df_sec = pd.DataFrame(
    [{'Group': 'Young', 'Subject': i, 'State': f'State {s}', 'DwellTime': dwell_y_sec[i, s]}
     for i in range(dwell_y_sec.shape[0]) for s in range(n_states)]
    + [{'Group': 'Old', 'Subject': i, 'State': f'State {s}', 'DwellTime': dwell_o_sec[i, s]}
       for i in range(dwell_o_sec.shape[0]) for s in range(n_states)]
)

plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='State', y='DwellTime', hue='Group', data=df_sec,
                    split=True, inner='quartile')
plt.ylabel('Mean Dwell Time (s)')
plt.legend(title='Group')

overall_max = df_sec['DwellTime'].max()
ax.set_ylim(0, overall_max * (1 + TOP_MARGIN_RATIO_DWELL))

xticks = ax.get_xticks()
for i, state in enumerate(state_labels):
    if rejected[i]:
        marker = '*' if 't-test' in methods_dwell[i] else '\u2020'
        y_top  = overall_max * (0.6 + TOP_MARGIN_RATIO_DWELL * 0.7)
        ax.text(xticks[i] + 0.1, y_top, marker, ha='center', va='center',
                fontsize=MARKER_SIZE, fontweight='bold', color='red')

plt.legend(title='Group', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.grid(True, axis='x', linestyle=':')

out_dwell = SCRIPT_DIR / 'dwell_time_violin.svg'
plt.savefig(str(out_dwell), format='svg', bbox_inches='tight')
plt.show()

# ===============================
# Occupancy analysis
# ===============================
p_vals_occ, state_labels_occ, methods_occ, d_vals_occ, g_vals_occ = [], [], [], [], []

for s in range(n_states):
    p, method, _ = two_sample_test(occupancy_o[:, s], occupancy_y[:, s], label=f"Occupancy State {s}")
    p_vals_occ.append(p)
    state_labels_occ.append(f"State {s}")
    methods_occ.append(method)

    d = cohens_d(occupancy_y[:, s], occupancy_o[:, s], transform=True)
    g = hedges_g_from_d(d, np.isfinite(occupancy_y[:, s]).sum(), np.isfinite(occupancy_o[:, s]).sum())
    d_vals_occ.append(d)
    g_vals_occ.append(g)

rejected_occ, q_vals_occ, _, _ = multipletests(p_vals_occ, alpha=0.05, method='fdr_bh')

df_occ = pd.DataFrame(
    [{'Group': 'Young', 'Subject': i, 'State': f'State {s}', 'Occupancy': occupancy_y[i, s]}
     for i in range(occupancy_y.shape[0]) for s in range(n_states)]
    + [{'Group': 'Old', 'Subject': i, 'State': f'State {s}', 'Occupancy': occupancy_o[i, s]}
       for i in range(occupancy_o.shape[0]) for s in range(n_states)]
)

plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='State', y='Occupancy', hue='Group', data=df_occ,
                    split=True, inner='quartile')
plt.ylabel('Occupancy')
plt.legend(title='Group')
ax.set_ylim(0, 1.0 + TOP_MARGIN_ABS_OCC)

xticks = ax.get_xticks()
for i, state in enumerate(state_labels_occ):
    if rejected_occ[i]:
        marker = '*' if 't-test' in methods_occ[i] else '\u2020'
        y_top  = 0.6 + TOP_MARGIN_ABS_OCC * 0.7
        ax.text(xticks[i] + 0.1, y_top, marker, ha='center', va='center',
                fontsize=MARKER_SIZE, fontweight='bold', color='red')

plt.legend(title='Group', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.grid(True, axis='x', linestyle=':')

out_occ = SCRIPT_DIR / 'occupancy_violin.svg'
plt.savefig(str(out_occ), format='svg', bbox_inches='tight')
plt.show()

# ===============================
# Summary tables
# ===============================
summary_dwell = pd.DataFrame({
    "State":                       state_labels,
    "Test":                        methods_dwell,
    "p":                           p_vals,
    "q(FDR)":                      q_vals,
    "Young_mean":                  [mean_sd(dwell_y_sec[:, s])[0] for s in range(n_states)],
    "Young_SD":                    [mean_sd(dwell_y_sec[:, s])[1] for s in range(n_states)],
    "Old_mean":                    [mean_sd(dwell_o_sec[:, s])[0] for s in range(n_states)],
    "Old_SD":                      [mean_sd(dwell_o_sec[:, s])[1] for s in range(n_states)],
    "Cohen_d(Old-Young,log1p)":    d_vals_dwell,
})

summary_occ = pd.DataFrame({
    "State":                       state_labels_occ,
    "Test":                        methods_occ,
    "p":                           p_vals_occ,
    "q(FDR)":                      q_vals_occ,
    "Young_mean":                  [mean_sd(occupancy_y[:, s])[0] for s in range(n_states)],
    "Young_SD":                    [mean_sd(occupancy_y[:, s])[1] for s in range(n_states)],
    "Old_mean":                    [mean_sd(occupancy_o[:, s])[0] for s in range(n_states)],
    "Old_SD":                      [mean_sd(occupancy_o[:, s])[1] for s in range(n_states)],
    "Cohen_d(Old-Young,log1p)":    d_vals_occ,
})

print("\n=== Dwell time summary ===")
print(summary_dwell.to_string(index=False))
print("\n=== Occupancy summary ===")
print(summary_occ.to_string(index=False))
