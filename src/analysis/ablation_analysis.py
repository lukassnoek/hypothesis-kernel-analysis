import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
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
scores_all = []

# Loop across mappings (Darwin, Ekman, etc.)
MAPPINGS['JS-between'] = 'JS-between'
    
for au in tqdm(PARAM_NAMES):

    for mapp_name, mapp in MAPPINGS.items():
        if mapp_name == 'JS-between':
            mapp = pd.read_csv('data/JackSchyns.tsv', sep='\t', index_col=0)
            av_mapp = mapp.query("sub == 'average_even' & trial_split == 'odd'").drop(['sub', 'trial_split'], axis=1).set_index('emotion')
            model = KernelClassifier(au_cfg=None, param_names=av_mapp.columns.tolist(),
                                    kernel=kernel, ktype=ktype, binarize_X=False,
                                    normalization='softmax', beta=beta)
            model.Z_ = av_mapp
            model.cls_idx_ = range(6)
            these_subs = subs[1::2]
        else:
            #continue
            model = KernelClassifier(au_cfg=mapp, param_names=PARAM_NAMES, kernel=kernel, ktype=ktype,
                                    binarize_X=False, normalization='softmax', beta=beta)
            these_subs = subs
        
        model.fit(None, None)
        Z_orig = model.Z_.copy()

        for emo in EMOTIONS:

            # Initialize scores (one score per subject and per emotion)
            scores = np.zeros((len(these_subs), len(EMOTIONS)))

            # Compute model performance per subject!
            for i, sub in enumerate(these_subs):
                data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
                data = data.query("emotion != 'other'")
                data = data.loc[data.index != 'empty', :]
            
                if mapp_name == 'JS-between':
                    data = data.iloc[1::2, :]
                
                X, y = data.iloc[:, :-2], data.iloc[:, -2]

                # Predict data + compute performance (AUROC)
                y_pred = pd.DataFrame(model.predict_proba(X), index=X.index, columns=EMOTIONS)
                scores[i, :] = roc_auc_score(pd.get_dummies(y), y_pred, average=None)
                model.Z_.loc[emo, au] = 0
                y_pred = pd.DataFrame(model.predict_proba(X), index=X.index, columns=EMOTIONS)
                new = roc_auc_score(pd.get_dummies(y), y_pred, average=None)
                scores[i, :] = (new - scores[i, :]) / scores[i, :] * 100
                model.Z_ = Z_orig.copy()

            # Store scores and raw predictions
            scores = pd.DataFrame(scores, columns=EMOTIONS, index=these_subs).reset_index()
            scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
            scores = scores.rename({'index': 'sub'}, axis=1)
            scores['mapping'] = mapp_name
            scores['kernel'] = kernel
            scores['beta'] = beta
            scores['ablated_au'] = au
            scores['ablated_from'] = emo
            scores_all.append(scores)

# Save scores and predictions
scores = pd.concat(scores_all, axis=0)
scores.to_csv('results/scores_ablation.tsv', sep='\t')
