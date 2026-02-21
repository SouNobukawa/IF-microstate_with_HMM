"""05_plot_topomap.py

Plot IF-microstate topomaps for each HMM state and group,
and compute group comparisons for transition probabilities,
dwell times, and occupancy.

Loads:
  result_Y_O_IF_IA.mat  -- preprocessed EEG features (01_preprocess.py)
  hmm_model.pkl         -- trained HMM + PCA (02_hmm_train.py)

Outputs:
  topomap_IF_GroupYounger_State{n}.svg  -- IF topomap, younger group
  topomap_IF_GroupOlder_State{n}.svg    -- IF topomap, older group
"""

import pickle
from pathlib import Path
from itertools import groupby

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy.io import loadmat
from scipy.stats import ttest_ind, levene
from statsmodels.stats.multitest import multipletests

plt.rcParams["svg.fonttype"] = "none"  # keep text as text in SVG

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
# MNE channel setup
# ===============================
ch_order = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1',  'O2',  'F7', 'F8', 'Fz', 'Pz', 'T5', 'T6']

montage = mne.channels.make_standard_montage('standard_1020')
if_info = mne.create_info(ch_names=ch_order, sfreq=100, ch_types='eeg')
if_info.set_montage(montage)

# ===============================
# State assignment for each group
# ===============================
X_y_IF_pca = loaded_pca_IF.transform(X_young[:, :16])
states_y    = loaded_model.predict(X_y_IF_pca)

X_o_IF_pca = loaded_pca_IF.transform(X_old[:, :16])
states_o    = loaded_model.predict(X_o_IF_pca)

# ===============================
# Mean IF feature vector per state and group
# ===============================
mean_vectors_y = []
mean_vectors_o = []

for s in range(n_states):
    idx_s_y = np.where(states_y == s)[0]
    mean_y = np.mean(X_young[idx_s_y], axis=0) if len(idx_s_y) > 0 else np.full(X_young.shape[1], np.nan)
    mean_vectors_y.append(mean_y)

    idx_s_o = np.where(states_o == s)[0]
    mean_o = np.mean(X_old[idx_s_o], axis=0) if len(idx_s_o) > 0 else np.full(X_old.shape[1], np.nan)
    mean_vectors_o.append(mean_o)

    print(f"State {s}: younger mean shape {mean_y.shape}, older mean shape {mean_o.shape}")

# ===============================
# Topomap plots
# ===============================
for s in range(n_states):
    print(f"Plotting State {s}")

    # Younger group
    mean_y = mean_vectors_y[s]
    if not np.isnan(mean_y).all():
        if_data   = mean_y[:16]
        evoked_y  = mne.EvokedArray(if_data[:, np.newaxis], if_info)
        fig = evoked_y.plot_topomap(
            times=0, scalings=1,
            time_format=f"Younger - State {s + 1} - IF",
            show=False, vlim=(-1, 1)
        )
        fig.savefig(str(SCRIPT_DIR / f'topomap_IF_GroupYounger_State{s + 1}.svg'),
                    format='svg', bbox_inches='tight')
        plt.close(fig)

    # Older group
    mean_o = mean_vectors_o[s]
    if not np.isnan(mean_o).all():
        if_data   = mean_o[:16]
        evoked_o  = mne.EvokedArray(if_data[:, np.newaxis], if_info)
        fig = evoked_o.plot_topomap(
            times=0, scalings=1,
            time_format=f"Older - State {s + 1} - IF",
            show=False, vlim=(-1, 1)
        )
        fig.savefig(str(SCRIPT_DIR / f'topomap_IF_GroupOlder_State{s + 1}.svg'),
                    format='svg', bbox_inches='tight')
        plt.close(fig)

# ===============================
# Per-subject transition probabilities
# ===============================
def compute_transitions(X_group, n_subjects):
    transition_probs = []
    for i in range(n_subjects):
        X_subj     = X_group[i * SAMPLES_PER_SUBJECT : (i + 1) * SAMPLES_PER_SUBJECT]
        X_pca      = loaded_pca_IF.transform(X_subj[:, :16])
        states     = loaded_model.predict(X_pca)

        trans_mat = np.zeros((n_states, n_states))
        for prev_s, next_s in zip(states[:-1], states[1:]):
            trans_mat[prev_s, next_s] += 1
        with np.errstate(invalid='ignore', divide='ignore'):
            trans_prob = trans_mat / trans_mat.sum(axis=1, keepdims=True)
        transition_probs.append(trans_prob)
        print(f"  Subject {i + 1} transition probabilities:\n{trans_prob}\n")
    return np.array(transition_probs)


print(f"Computing transitions for younger group ({n_subjects_y} subjects)")
tp_array_y = compute_transitions(X_young, n_subjects_y)

print(f"Computing transitions for older group ({n_subjects_o} subjects)")
tp_array_o = compute_transitions(X_old, n_subjects_o)

# ===============================
# Transition probability: Levene + t-test with FDR correction
# ===============================
t_vals = []
p_vals = []

for i in range(n_states):
    for j in range(n_states):
        data_y = tp_array_y[:, i, j]
        data_o = tp_array_o[:, i, j]

        stat_f, p_f  = levene(data_y, data_o)
        equal_var    = p_f > 0.05
        stat_t, p_t  = ttest_ind(data_o, data_y, equal_var=equal_var)

        p_vals.append(p_t)
        t_vals.append(stat_t)
        print(f"Transition {i}->{j}: Levene p={p_f:.4f}, t-test p={p_t:.4f}, t={stat_t:.2f}")

rejected, q_vals, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')

t_matrix        = np.array(t_vals).reshape(n_states, n_states)
significant_mask = rejected.reshape(n_states, n_states)

plt.figure(figsize=(8, 6))
ax = sns.heatmap(t_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
for i in range(n_states):
    for j in range(n_states):
        if significant_mask[i, j]:
            ax.text(j + 0.5, i + 0.5, '*', color='black',
                    ha='center', va='center', fontsize=20, fontweight='bold')
plt.title("T-values heatmap (significant cells marked with *)")
plt.xlabel("To State")
plt.ylabel("From State")
plt.show()

# ===============================
# Per-subject dwell times and occupancy
# ===============================
dwell_times_all_y = []
occupancy_all_y   = []
dwell_times_all_o = []
occupancy_all_o   = []

for i in range(n_subjects_y):
    X_subj = X_young[i * SAMPLES_PER_SUBJECT : (i + 1) * SAMPLES_PER_SUBJECT]
    X_pca  = loaded_pca_IF.transform(X_subj[:, :16])
    states = loaded_model.predict(X_pca)

    dwell_times = [[] for _ in range(n_states)]
    for state, group in groupby(states):
        dwell_times[state].append(len(list(group)))
    mean_dwell = np.full(n_states, np.nan)
    for s in range(n_states):
        if len(dwell_times[s]) > 0:
            mean_dwell[s] = np.mean(dwell_times[s])
    dwell_times_all_y.append(mean_dwell)

    occupancy = np.bincount(states, minlength=n_states) / len(states)
    occupancy_all_y.append(occupancy)
    print(f"Younger subject {i + 1}: dwell={mean_dwell}, occupancy={occupancy}")

for i in range(n_subjects_o):
    X_subj = X_old[i * SAMPLES_PER_SUBJECT : (i + 1) * SAMPLES_PER_SUBJECT]
    X_pca  = loaded_pca_IF.transform(X_subj[:, :16])
    states = loaded_model.predict(X_pca)

    dwell_times = [[] for _ in range(n_states)]
    for state, group in groupby(states):
        dwell_times[state].append(len(list(group)))
    mean_dwell = np.full(n_states, np.nan)
    for s in range(n_states):
        if len(dwell_times[s]) > 0:
            mean_dwell[s] = np.mean(dwell_times[s])
    dwell_times_all_o.append(mean_dwell)

    occupancy = np.bincount(states, minlength=n_states) / len(states)
    occupancy_all_o.append(occupancy)
    print(f"Older subject {i + 1}: dwell={mean_dwell}, occupancy={occupancy}")

dwell_y     = np.array(dwell_times_all_y)
dwell_o     = np.array(dwell_times_all_o)
occupancy_y = np.array(occupancy_all_y)
occupancy_o = np.array(occupancy_all_o)

# ===============================
# Dwell time: Levene + t-test with FDR correction
# ===============================
p_vals_dwell  = []
state_labels  = []

print("\nDwell time t-test + BH correction:")
for s in range(n_states):
    stat_f, p_f = levene(dwell_y[:, s], dwell_o[:, s])
    equal_var   = p_f > 0.05
    stat_t, p_t = ttest_ind(dwell_o[:, s], dwell_y[:, s], equal_var=equal_var)
    print(f"  State {s}: p={p_t:.4f}")
    p_vals_dwell.append(p_t)
    state_labels.append(f"State {s}")

rejected_dwell, _, _, _ = multipletests(p_vals_dwell, alpha=0.05, method='fdr_bh')
significant_states = [state_labels[i] for i, rej in enumerate(rejected_dwell) if rej]
print(f"  Significant (q<0.05): {significant_states}")

sampling_rate = 200
dwell_y_sec   = dwell_y / sampling_rate
dwell_o_sec   = dwell_o / sampling_rate

df_dwell = pd.DataFrame(
    [{'Group': 'Younger', 'Subject': i, 'State': f'State {s}', 'DwellTime': dwell_y_sec[i, s]}
     for i in range(dwell_y_sec.shape[0]) for s in range(n_states)]
    + [{'Group': 'Older', 'Subject': i, 'State': f'State {s}', 'DwellTime': dwell_o_sec[i, s]}
       for i in range(dwell_o_sec.shape[0]) for s in range(n_states)]
)

plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='State', y='DwellTime', hue='Group', data=df_dwell,
                    split=True, inner='quartile')
plt.title('Dwell Time Distribution per State (seconds)')
plt.ylabel('Mean Dwell Time (seconds)')
plt.legend(title='Group')

xticks = ax.get_xticks()
for i, state in enumerate(state_labels):
    if state in significant_states:
        y_max = df_dwell[df_dwell['State'] == state]['DwellTime'].max()
        ax.text(xticks[i], y_max + 0.05, '*', ha='center', va='bottom',
                color='red', fontsize=20)
plt.show()

# ===============================
# Occupancy: Levene + t-test with FDR correction
# ===============================
p_vals_occ      = []
state_labels_occ = []

print("\nOccupancy t-test + BH correction:")
for s in range(n_states):
    stat_f, p_f = levene(occupancy_y[:, s], occupancy_o[:, s])
    equal_var   = p_f > 0.05
    stat_t, p_t = ttest_ind(occupancy_o[:, s], occupancy_y[:, s], equal_var=equal_var)
    print(f"  State {s}: p={p_t:.4f}")
    p_vals_occ.append(p_t)
    state_labels_occ.append(f"State {s}")

rejected_occ, _, _, _ = multipletests(p_vals_occ, alpha=0.05, method='fdr_bh')
significant_states_occ = [state_labels_occ[i] for i, rej in enumerate(rejected_occ) if rej]
print(f"  Significant (q<0.05): {significant_states_occ}")

df_occ = pd.DataFrame(
    [{'Group': 'Younger', 'Subject': i, 'State': f'State {s}', 'Occupancy': occupancy_y[i, s]}
     for i in range(occupancy_y.shape[0]) for s in range(n_states)]
    + [{'Group': 'Older', 'Subject': i, 'State': f'State {s}', 'Occupancy': occupancy_o[i, s]}
       for i in range(occupancy_o.shape[0]) for s in range(n_states)]
)

plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='State', y='Occupancy', hue='Group', data=df_occ,
                    split=True, inner='quartile')
plt.title('Occupancy per State')
plt.ylabel('Occupancy')
plt.legend(title='Group')

xticks = ax.get_xticks()
for i, state in enumerate(state_labels_occ):
    if state in significant_states_occ:
        y_max = df_occ[df_occ['State'] == state]['Occupancy'].max()
        ax.text(xticks[i], y_max + 0.02, '*', ha='center', va='bottom',
                color='red', fontsize=20)
plt.show()
