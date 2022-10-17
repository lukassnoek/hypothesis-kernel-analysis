import sys
import pandas as pd
import numpy as np
import os.path as op
from tqdm import tqdm
from glob import glob
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from models import KernelClassifier
from mappings_conversational import MAPPINGS

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()

scores_all = []

files = sorted(glob('data/ratings/conversational/*/*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
mega_df = mega_df.query("sub_split == 'train' & trial_split == 'train'")
mega_df = mega_df.query("state != 'other'")  # remove non-emo trials
#mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs
ohe = OneHotEncoder(sparse=False)
    
for mapp_name, mapp in MAPPINGS.items():

    # Initialize model!
    model = KernelClassifier(au_cfg=mapp, param_names=PARAM_NAMES, kernel='cosine', ktype='similarity',
                             binarize_X=False, normalization='softmax', beta=1)
    model.fit(None, None)
    model.Z_ = model.Z_.astype(int)
    model.Z_.to_csv(f'data/{mapp_name}_conversational.tsv', sep='\t')
    
    defined_states = sorted(list(set(model.Z_.index.tolist())))
    ohe.fit(np.array(defined_states)[:, None])

    for sub_id in tqdm(mega_df['sub'].unique(), desc=mapp_name):
        df_l1 = mega_df.query("sub == @sub_id & state in @defined_states")

        # Initialize with NaNs in case of no trials for a
        # given emotion category
        scores = np.zeros(len(defined_states))
        scores[:] = np.nan
    
        X, y = df_l1.iloc[:, :33], df_l1.loc[:, 'state']
        y_ohe = ohe.transform(y.to_numpy()[:, None])
        y_pred = model.predict_proba(X)
        idx = y_ohe.sum(axis=0) != 0

        scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)
        scores = pd.DataFrame(scores, columns=['score'])
        scores['state'] = defined_states
        scores['sub'] = sub_id
        ethn = df_l1['sub_ethnicity'].unique()[0]
        scores['sub_ethnicity'] = ethn
        scores['mapping'] =  mapp_name
        scores_all.append(scores)

# Save scores and predictions
scores = pd.concat(scores_all, axis=0)
scores.to_csv('results/scores_conversational.tsv', sep='\t')
print(scores.groupby(['mapping', 'state']).mean())