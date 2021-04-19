import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import pairwise_kernels
sys.path.append('src')
from models import KernelClassifier

# Define parameter names (AUs) and target label (EMOTIONS)
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])

# Define analysis parameters
beta = 1
kernel = 'cosine'
ktype = 'similarity'

subs = [str(s).zfill(2) for s in range(1, 61)]

mapp = pd.read_csv('data/HierarchicalClustering.tsv', sep='\t', index_col=0)
scores_ = []
# Compute model performance per subject!
for i, sub in enumerate(tqdm(subs)):
    data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
    data = data.query("emotion != 'other'")
    data = data.loc[data.index != 'empty', :]

    X, y = data.iloc[:, :-2], data.iloc[:, -2]

    for ii, emo in enumerate(EMOTIONS):
        X_ = X.loc[y == emo, :].to_numpy()
        mapp_ = mapp.query("emotion == @emo")
        
        scores = np.zeros(mapp_['hierarchy_idx'].unique().size)
        au_str = []
        for idx in mapp_['hierarchy_idx'].unique():
            mapp__ = mapp_.query("hierarchy_idx == @idx").iloc[:, :-3].to_numpy()
            scores[idx] = pairwise_kernels(X_, mapp__, metric='cosine').mean()
            s = ' + '.join(mapp_.columns[:-3][mapp__.squeeze().astype(bool)].to_list())
            au_str.append(s)
        
        scores = pd.DataFrame(scores, columns=['similarity'])
        scores['level'] = range(scores.shape[0])
        scores['emotion'] = emo
        scores['sub'] = sub
        scores['AUs'] = au_str
        scores_.append(scores)

scores = pd.concat(scores_, axis=0)
scores.to_csv('results/scores_hierarchical_analysis.tsv', index=False, sep='\t')    
