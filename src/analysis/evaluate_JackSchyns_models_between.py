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

mapp = pd.read_csv('data/JackSchyns.tsv', sep='\t', index_col=0)
av_mapp = mapp.query("sub == 'average_even' & trial_split == 'even'").drop(['sub', 'trial_split'], axis=1)
for emo in av_mapp['emotion']:
    mapp = av_mapp.query("emotion == @emo").iloc[0]
    mapp = mapp.index[mapp == 1].tolist()
    print(f"{emo}: {mapp}")
    
av_mapp = av_mapp.set_index('emotion')
model = KernelClassifier(au_cfg=None, param_names=av_mapp.columns.tolist(),
                         kernel=kernel, ktype=ktype, binarize_X=False,
                         normalization='softmax', beta=beta)
model.Z_ = av_mapp
model.cls_idx_ = range(6)

scores = np.zeros((len(subs[1::2]), len(EMOTIONS)))
preds = []
# Compute model performance per subject!
for i, sub in enumerate(subs[1::2]):
    data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
    data = data.query("emotion != 'other'")
    data = data.loc[data.index != 'empty', :]

    data = data.iloc[1::2, :]
    X, y = data.iloc[:, :-2], data.iloc[:, -2]
    
    model.fit(None, None)
    # Predict data + compute performance (AUROC)
    y_pred = pd.DataFrame(model.predict_proba(X), index=X.index, columns=EMOTIONS)
    scores[i, :] = roc_auc_score(pd.get_dummies(y), y_pred, average=None)
    
    y_pred['sub'] = sub
    y_pred['intensity'] = data['intensity']
    y_pred['y_true'] = data['emotion']
    preds.append(y_pred)

# Store scores and raw predictions
scores = pd.DataFrame(scores, columns=EMOTIONS, index=subs[1::2]).reset_index()
scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
scores = scores.rename({'index': 'sub'}, axis=1)
scores['mapping'] = 'JS-between'
scores['kernel'] = kernel
scores['beta'] = beta

preds = pd.concat(preds, axis=0)
preds['mapping'] = 'JS-between' 

# Save scores and predictions
scores.to_csv('results/JS-between_scores.tsv', sep='\t')

# Save predictions (takes a while). Not really necessary, but maybe useful for 
# follow-up analyses
preds.to_csv('results/JS-between_predictions.tsv', sep='\t')

