import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from mappings import MAPPINGS
from models import KernelClassifier

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# One-hot encode target label
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(np.arange(6)[:, np.newaxis])

# Define analysis parameters
subs = [str(s).zfill(2) for s in range(1, 61)]
scores_all = []

kernel = 'cosine'
beta = 1

for method in ['predict', 'predict_proba']:
    # Loop across mappings (Darwin, Ekman, etc.)
    for mapp_name, mapp in MAPPINGS.items():
        # ktype = kernel type (infer from kernel name)

        # Initialize model!
        model = KernelClassifier(au_cfg=mapp, param_names=PARAM_NAMES, kernel=kernel, ktype='similarity',
                                binarize_X=False, normalization='softmax', beta=beta)
        
        # Initialize scores (one score per subject and per emotion)
        scores = np.zeros((len(subs), len(EMOTIONS)))
        
        # Compute model performance per subject!
        for i, sub in enumerate(subs):
            data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
            data = data.query("emotion != 'other'")
            X, y = data.iloc[:, :-2], data.iloc[:, -2]

            # Technically, we're not "fitting" anything, but this will set up the mapping matrix (self.Z_)
            model.fit(X, y)

            y_pred = getattr(model, method)(X)
            if method == 'predict':
                scores[i, :] = roc_auc_score(pd.get_dummies(y).values, pd.get_dummies(y_pred).values, average=None)
            else:
                scores[i, :] = roc_auc_score(pd.get_dummies(y), y_pred, average=None)

        # Store scores and raw predictions
        scores = pd.DataFrame(scores, columns=EMOTIONS, index=subs).reset_index()
        scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
        scores = scores.rename({'index': 'sub'}, axis=1)
        scores['mapping'] = mapp_name
        scores['kernel'] = kernel
        scores['beta'] = beta
        scores['method'] = method
        scores_all.append(scores)

# Save scores and predictions
scores = pd.concat(scores_all, axis=0)
print(scores.groupby(['method', 'mapping']).mean())
scores.to_csv('results/prob_vs_discrete_scores.tsv', sep='\t')