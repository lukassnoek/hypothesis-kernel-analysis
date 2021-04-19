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
scores_all, preds_all = [], []

# Loop across mappings (Darwin, Ekman, etc.)
for mapp_name, mapp in tqdm(MAPPINGS.items()):

    # Initialize model!
    model = KernelClassifier(au_cfg=mapp, param_names=PARAM_NAMES, kernel=kernel, ktype=ktype,
                             binarize_X=False, normalization='softmax', beta=beta)
    

    # Technically, we're not "fitting" anything, but this will set up the mapping matrix (self.Z_)
    model.fit(None, None)
    model.Z_.to_csv(f'data/{mapp_name}.tsv', sep='\t')

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
        y_pred = pd.DataFrame(model.predict_proba(X), index=X.index, columns=EMOTIONS)
        scores[i, :] = roc_auc_score(pd.get_dummies(y), y_pred, average=None)

        # Save results
        y_pred['sub'] = sub
        y_pred['intensity'] = data['intensity']
        y_pred['y_true'] = data['emotion']
        preds.append(y_pred)

    # Store scores and raw predictions
    scores = pd.DataFrame(scores, columns=EMOTIONS, index=subs).reset_index()
    scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
    scores = scores.rename({'index': 'sub'}, axis=1)
    scores['mapping'] = mapp_name
    scores['kernel'] = kernel
    scores['beta'] = beta
    scores_all.append(scores)

    preds = pd.concat(preds, axis=0)
    preds['mapping'] = mapp_name    
    preds_all.append(preds)

# Save scores and predictions
scores = pd.concat(scores_all, axis=0)
scores.to_csv('results/scores.tsv', sep='\t')
print(scores.groupby(['emotion', 'mapping']).mean())

# Save predictions (takes a while). Not really necessary, but maybe useful for 
# follow-up analyses
preds = pd.concat(preds_all)
preds.to_csv('results/predictions.tsv', sep='\t')