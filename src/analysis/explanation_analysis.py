import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from mappings import MAPPINGS
from models import KernelClassifier

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])
ohe = OneHotEncoder(sparse=False)
ohe.fit(EMOTIONS[:, np.newaxis])

subs = [str(s).zfill(2) for s in range(1, 61)]
scores_all = []    

def _parallel_analysis(au, MAPPINGS, EMOTIONS, PARAM_NAMES):
    scores_all = []
    for mapp_name, mapp in MAPPINGS.items():
    
        model = KernelClassifier(au_cfg=mapp, param_names=PARAM_NAMES, kernel='cosine',
                                    ktype='similarity', binarize_X=False, normalization='softmax', beta=1)
        if mapp_name == 'JS':
            these_subs = subs[1::2]
        else:
            these_subs = subs
    
        model.fit(None, None)
        Z_orig = model.Z_.copy()

        for emo in EMOTIONS:

            # Initialize scores (one score per subject and per emotion)
            scores = np.zeros((len(these_subs), len(EMOTIONS)))
            scores[:] = np.nan

            # Compute model performance per subject!
            for i, sub in enumerate(these_subs):
                data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
                data = data.query("emotion != 'other'")
                data = data.loc[data.index != 'empty', :]
            
                if mapp_name == 'JS':
                    data = data.query("data_split == 'test'")
                
                X, y = data.iloc[:, :33], data.loc[:, 'emotion']

                # Predict data + compute performance (AUROC)
                y_pred = pd.DataFrame(model.predict_proba(X), index=X.index, columns=EMOTIONS)
                y_ohe = ohe.transform(y.to_numpy()[:, np.newaxis])
                idx = y_ohe.sum(axis=0) != 0
                scores[i, idx] = roc_auc_score(y_ohe[:, idx], y_pred.to_numpy()[:, idx], average=None)
                model.Z_.loc[emo, au] = 0
                y_pred = pd.DataFrame(model.predict_proba(X), index=X.index, columns=EMOTIONS)
                new = roc_auc_score(y_ohe[:, idx], y_pred.to_numpy()[:, idx], average=None)
                scores[i, idx] = (new - scores[i, idx]) #/ scores[i, idx]) * 100
                model.Z_ = Z_orig.copy()

            # Store scores and raw predictions
            scores = pd.DataFrame(scores, columns=EMOTIONS, index=these_subs).reset_index()
            scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
            scores = scores.rename({'index': 'sub'}, axis=1)
            scores['mapping'] = mapp_name
            scores['kernel'] = 'cosine'
            scores['beta'] = 1
            scores['ablated_au'] = au
            scores['ablated_from'] = emo
            scores_all.append(scores)
    
    # Save scores and predictions
    scores = pd.concat(scores_all, axis=0)
    return scores

from joblib import Parallel, delayed

scores = Parallel(n_jobs=17)(delayed(_parallel_analysis)(
    au, MAPPINGS, EMOTIONS, PARAM_NAMES)
    for au in tqdm(PARAM_NAMES)
)
scores = pd.concat(scores, axis=0)
scores.to_csv('results/scores_ablation.tsv', sep='\t')

