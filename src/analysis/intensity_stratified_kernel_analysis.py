import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append('src')
from mappings import MAPPINGS
from models import KernelClassifier


PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])

# NOTE: kernel_analysis.py should be run first
preds = pd.read_csv('results/predictions.tsv', sep='\t', index_col=0)
subs = [str(s).zfill(2) for s in range(1, 61)]

# Compute average intensity across repeated observations
mean_int = preds['intensity'].reset_index().groupby('index').mean()
preds.loc[mean_int.index, 'intensity'] = mean_int['intensity']

# Compute quantiles based on this mean intensity: define 5 values, get 4 quantiles
percentiles = preds['intensity'].quantile([0., .25, .5, .75, 1.])

# Initialize results dataframe
scores_int = pd.DataFrame(columns=['sub', 'emotion', 'mapping', 'intensity', 'score'])
i = 0

# Loop over trials based on intensity levels
for intensity in tqdm([1, 2, 3, 4]):
    # Get current set of trials based on `intensity`
    minn, maxx = percentiles.iloc[intensity-1], percentiles.iloc[intensity]
    preds_int = preds.query("@minn <= intensity & intensity <= @maxx")
    
    # Loop across subjects
    for sub in subs:
        for mapp_name, _ in MAPPINGS.items():
            tmp_preds = preds_int.query("sub == @sub & mapping == @mapp_name")
            y_true = pd.get_dummies(tmp_preds['y_true'])
            score = roc_auc_score(y_true, tmp_preds.iloc[:, :6], average=None)
            for ii, s in enumerate(score):
                scores_int.loc[i, 'sub'] = sub
                scores_int.loc[i, 'emotion'] = EMOTIONS[ii]
                scores_int.loc[i, 'mapping'] = mapp_name
                scores_int.loc[i, 'intensity'] = intensity
                scores_int.loc[i, 'score'] = s
                i += 1

scores_int['score'] = scores_int['score'].astype(float)
scores_int = scores_int.sort_values(['mapping', 'sub', 'emotion', 'intensity'])
scores_int.to_csv('results/score_per_intensity_quantile.tsv', sep='\t')