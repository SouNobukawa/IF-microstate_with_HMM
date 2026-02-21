"""02_hmm_train.py

Train a Gaussian HMM on the preprocessed EEG microstate features.

Loads result_Y_O_IF_IA.mat (produced by 01_preprocess.py), applies PCA
to the instantaneous-frequency (IF) channels, fits a Gaussian HMM, and
saves the model together with the PCA transformer to hmm_model.pkl.

Output:
  hmm_model.pkl  -- tuple (GaussianHMM, PCA) serialised with pickle
"""

import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from scipy.io import loadmat
import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

data = loadmat(str(SCRIPT_DIR / 'result_Y_O_IF_IA.mat'))
X = data['c_z_diff']

# Split into IF (instantaneous frequency) and IA (instantaneous amplitude)
X_IF = X[:, :16]
X_IA = X[:, 16:]

# Model hyperparameters
n_features   = 32
n_components = 5   # number of HMM hidden states

N_if = 16
N_ia = 16

# PCA for IF features
pca_IF     = PCA(n_components=N_if)
X_IF_pca   = pca_IF.fit_transform(X_IF)

# PCA for IA features (computed but not used for HMM training below)
pca_IA     = PCA(n_components=N_ia)
X_IA_pca   = pca_IA.fit_transform(X_IA)

# Use only IF-PCA features for HMM training
X_train = X_IF_pca

# Train Gaussian HMM
np.random.seed(2)
model = hmm.GaussianHMM(
    n_components=n_components,
    covariance_type='full',
    n_iter=100000,
    verbose=True,
)
model.fit(X_train)

# State sequence on training data
hidden_states = model.predict(X_train)

# Print model summary
print("State transition matrix:")
print(model.transmat_)

print("\nInitial state probabilities:")
print(model.startprob_)

print("\nMean vector shape per state:", model.means_.shape)

print(model.score(X_train))

# Save model and PCA transformer
out_path = SCRIPT_DIR / 'hmm_model.pkl'
with open(out_path, 'wb') as f:
    pickle.dump((model, pca_IF), f)

print(f"\nModel saved to {out_path}")
