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
ohe.fit(np.array(EMOTIONS)[:, np.newaxis])

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
        max_sim = model.sim_.max(axis=1)

        # Save results
        y_pred['sub'] = sub
        y_pred['max_sim'] = max_sim
        y_pred['y_true'] = data['emotion']
        preds.append(y_pred)

    preds = pd.concat(preds, axis=0)
    preds['mapping'] = mapp_name    
    preds_all.append(preds)

preds = pd.concat(preds_all)
preds.to_csv('results/predictions_with_ambiguity.tsv', sep='\t')
mean_max_sim = preds['max_sim'].reset_index().groupby('index').mean()

preds.loc[mean_max_sim.index, 'max_sim'] = mean_max_sim['max_sim']

# Compute quantiles based on this mean intensity: define 6 values, get 5 quantiles
percentiles = preds['max_sim'].quantile([0, 0.25, 0.5, 0.75, 1])

# Initialize results dataframe
scores_int = pd.DataFrame(columns=['sub', 'emotion', 'mapping', 'max_sim', 'score'])
i = 0

# Loop over trials based on intensity levels
for lvl in tqdm([1, 2, 3, 4]):
    # Get current set of trials based on `intensity`
    minn, maxx = percentiles.iloc[lvl-1], percentiles.iloc[lvl]
    preds_amb = preds.query("@minn <= max_sim & max_sim <= @maxx")
    
    # Loop across subjects
    for sub in subs:
        for mapp_name, _ in MAPPINGS.items():
            tmp_preds = preds_amb.query("sub == @sub & mapping == @mapp_name")
            y_true = ohe.transform(tmp_preds['y_true'].to_numpy()[:, np.newaxis])
            try:
                score = roc_auc_score(y_true, tmp_preds.iloc[:, :6], average=None)
            except ValueError:
                score = [np.nan] * 6

            for ii, s in enumerate(score):
                scores_int.loc[i, 'sub'] = sub
                scores_int.loc[i, 'emotion'] = EMOTIONS[ii]
                scores_int.loc[i, 'mapping'] = mapp_name
                scores_int.loc[i, 'max_sim'] = lvl
                scores_int.loc[i, 'score'] = s
                i += 1

scores_int['score'] = scores_int['score'].astype(float)
scores_int = scores_int.sort_values(['mapping', 'sub', 'emotion', 'max_sim'])
scores_int.to_csv('results/score_per_typicality_quantile.tsv', sep='\t')
