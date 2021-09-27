import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from mappings import MAPPINGS
from models import KernelClassifier

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])
ohe = OneHotEncoder(sparse=False)
ohe.fit(EMOTIONS[:, np.newaxis])

df_abl = pd.read_csv('results/scores_ablation.tsv', sep='\t', index_col=0)
df_abl = df_abl.drop(['sub', 'beta', 'kernel'], axis=1)

subs = [str(s).zfill(2) for s in range(1, 61)]
scores_all = []    


cm_df = []
for mapp_name, mapp in MAPPINGS.items():
    print(mapp_name)

    model = KernelClassifier(au_cfg=mapp, param_names=PARAM_NAMES, kernel='cosine',
                                ktype='similarity', binarize_X=False, normalization='softmax', beta=1)
    if mapp_name == 'JS':
        these_subs = subs[1::2]
    else:
        these_subs = subs

    model.fit(None, None)
    Z_orig = model.Z_.copy()
    
    # Initialize scores (one score per subject and per emotion)
    scores = np.zeros((len(these_subs), len(EMOTIONS)))
    scores[:] = np.nan

    cm_old, cm_new = np.zeros((6, 6)), np.zeros((6, 6))

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
        cm_old += confusion_matrix(y_ohe.argmax(axis=1), y_pred.to_numpy().argmax(axis=1))

        # APPEND TO CONFIG
        for emo in EMOTIONS:
            df_abl_emo = df_abl.query("ablated_from == @emo & emotion == @emo & score != 0")
            aus = df_abl_emo.groupby('ablated_au').mean().query("score < 0").index.tolist()
            for au in aus:
                delta = df_abl_emo.query("ablated_au == @au & mapping == @mapp_name")['score'].to_numpy()
                #if not len(delta):
                #    continue

                #from scipy.stats import ttest_1samp
                #t, p = ttest_1samp(delta, popmean=0, alternative='less')
                
                #if p < 0.05:
                model.Z_.loc[emo, au] = 1

            aus = df_abl_emo.groupby('ablated_au').mean().query("score > 0").index.tolist()
            for au in aus:
                #delta = df_abl_emo.query("ablated_au == @au & mapping == @mapp_name")['score'].to_numpy()
                #if not len(delta):
                #    continue

                #t, p = ttest_1samp(delta, popmean=0, alternative='greater')                
                #if p < 0.05:
                model.Z_.loc[emo, au] = 0

        y_pred = pd.DataFrame(model.predict_proba(X), index=X.index, columns=EMOTIONS)
        new = roc_auc_score(y_ohe[:, idx], y_pred.to_numpy()[:, idx], average=None)
        scores[i, idx] = (new - scores[i, idx])# / scores[i, idx]) * 100

        cm_new += confusion_matrix(y_ohe.argmax(axis=1), y_pred.to_numpy().argmax(axis=1))
        model.Z_ = Z_orig.copy()

    cm_old = pd.DataFrame(cm_old, index=EMOTIONS, columns=EMOTIONS)
    cm_old['mapping'] = mapp_name
    cm_old['type'] = 'orig'
    cm_new = pd.DataFrame(cm_new, index=EMOTIONS, columns=EMOTIONS)
    cm_new['mapping'] = mapp_name
    cm_new['type'] = 'opt'
    cm_df.append(pd.concat((cm_old, cm_new), axis=0))

    # Store scores and raw predictions
    scores = pd.DataFrame(scores, columns=EMOTIONS, index=these_subs).reset_index()
    scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
    scores = scores.rename({'index': 'sub'}, axis=1)
    scores['mapping'] = mapp_name
    scores_all.append(scores)


scores = pd.concat(scores_all, axis=0)
scores.to_csv('results/scores_optimal.tsv', sep='\t')

cm = pd.concat(cm_df, axis=0)
cm.to_csv('results/cm_optimal.tsv', sep='\t')
