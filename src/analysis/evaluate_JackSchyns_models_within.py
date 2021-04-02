import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append('src')
from models import KernelClassifier

# Define parameter names (AUs) and target label (EMOTIONS)
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])

# Define analysis parameters
beta = 1
kernel = 'cosine'
ktype = 'similarity'

subs = [str(s).zfill(2) for s in range(1, 61)]
mapp = pd.read_csv('data/JackSchyns.tsv', sep='\t')

scores = np.zeros((len(subs), len(EMOTIONS)))
preds = []
# Compute model performance per subject!
for i, sub in enumerate(subs):
    data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
    data = data.iloc[1::2, :]
    data = data.query("emotion != 'other'")
    data = data.loc[data.index != 'empty', :]

    X_odd, y_odd = data.iloc[:, :-2], data.iloc[:, -2]
    
    this_mapp = mapp.query("sub == @sub & trial_split == 'even'").drop(['sub', 'trial_split'], axis=1).set_index('emotion')
    model = KernelClassifier(au_cfg=None, param_names=this_mapp.columns.tolist(),
                            kernel=kernel, ktype=ktype, binarize_X=False,
                            normalization='softmax', beta=beta)
    model.Z_ = this_mapp
    model.cls_idx_ = range(6)
    model.fit(None, None)

    # Predict data + compute performance (AUROC)
    y_pred = pd.DataFrame(model.predict_proba(X_odd), index=X_odd.index, columns=EMOTIONS)
    scores[i, :] = roc_auc_score(pd.get_dummies(y_odd), y_pred, average=None)
    
    y_pred['sub'] = sub
    y_pred['intensity'] = data['intensity']
    y_pred['y_true'] = data['emotion']
    preds.append(y_pred)

# Store scores and raw predictions
scores = pd.DataFrame(scores, columns=EMOTIONS, index=subs).reset_index()
scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
scores = scores.rename({'index': 'sub'}, axis=1)
scores['mapping'] = 'JS-within'
scores['kernel'] = kernel
scores['beta'] = beta

preds = pd.concat(preds, axis=0)
preds['mapping'] = 'JS-within' 

# Save scores and predictions
scores.to_csv('results/JS-within_scores.tsv', sep='\t')

# Save predictions (takes a while). Not really necessary, but maybe useful for 
# follow-up analyses
preds.to_csv('results/JS-within_predictions.tsv', sep='\t')