import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append('src')
from mappings import MAPPINGS
from models import KernelClassifier

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])

# Define analysis parameters
beta = 1
kernel = 'cosine'
ktype = 'similarity'

subs = [str(s).zfill(2) for s in range(1, 61)]

# Initialize model!
model = KernelClassifier(au_cfg=MAPPINGS['Cordaro2018IPC'], param_names=PARAM_NAMES,
                         kernel=kernel, ktype=ktype, binarize_X=False, normalization='softmax', beta=beta)

# Technically, we're not "fitting" anything, but this will set up the mapping matrix (self.Z_)
model.fit(None, None)
Z_orig = model.Z_.copy()

# Initialize scores (one score per subject and per emotion)
scores = np.zeros((len(subs), len(EMOTIONS)))
preds = []

# Compute model performance per subject!
for i, sub in enumerate(subs):
    data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
    data = data.query("emotion != 'other'")
    data = data.loc[data.index != 'empty', :]

    X, y = data.iloc[:, :-2], data.iloc[:, -2]

    # Predict data + compute performance (AUROC)
    scores[i, :] = roc_auc_score(pd.get_dummies(y), model.predict_proba(X), average=None)

    model.Z_.loc[:, 'AU04'] = 0
    scores[i, :] = roc_auc_score(pd.get_dummies(y), model.predict_proba(X), average=None)# - scores[i, :]
    model.Z_ = Z_orig.copy()

# Store scores and raw predictions
scores = pd.DataFrame(scores, columns=EMOTIONS, index=subs).reset_index()
scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
scores = scores.rename({'index': 'sub'}, axis=1)
print(scores.groupby('emotion').mean())