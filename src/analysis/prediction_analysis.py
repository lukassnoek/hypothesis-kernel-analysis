import sys
import pandas as pd
import numpy as np
import os.path as op
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from models import KernelClassifier

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])
ohe = OneHotEncoder(sparse=False)
ohe.fit(EMOTIONS[:, np.newaxis])

# Define analysis parameters
beta = 1
kernel = 'cosine'
ktype = 'similarity'

scores_all = []

# Loop across mappings (Darwin, Ekman, etc.)
mappings = ['Cordaro2018IPC', 'Cordaro2018ref', 'Darwin', 'Ekman', 'Keltner2019', 'Matsumoto2008',
#            'JackSchyns_ethn-WC_sub-train_trial-train',
#            'JackSchyns_ethn-EA_sub-train_trial-train',
#            'JackSchyns_ethn-all_sub-train_trial-train',
            ]
for mapp_name in mappings:

    # Initialize model!
    model = KernelClassifier(au_cfg=None, param_names=None, kernel=kernel, ktype=ktype,
                             binarize_X=False, normalization='softmax', beta=beta)

    # Note that there is no "fitting" of the model! The mappings themselves
    # can be interpreted as already-fitted models
    model.add_Z(pd.read_csv(f'data/{mapp_name}.tsv', sep='\t', index_col=0))

    sub_files = sorted(glob(f'data/ratings/*/*.tsv'))
    sub_files = [f for i, f in enumerate(sub_files) if i % 3 != 0] 
    for f in tqdm(sub_files, desc=mapp_name):
        df = pd.read_csv(f, sep='\t', index_col=0)
        df = df.query("emotion != 'other'")  # remove non-emo trials
        df = df.loc[df.index != 'empty', :]  # remove trials w/o AUs

        #for trial_split in ['train', 'test', 'all']:
        #    if trial_split != 'all':
        #        data = df.query("trial_split == @trial_split")
        #    else:
        #        data = df
        data = df.query("trial_split == 'train'")
        for face_gender in [0, 1, 'all']:
            if face_gender != 'all':
                this_data = data.query("face_gender == @face_gender")
            else:
                this_data = data

            # Initialize with NaNs in case of no trials for a
            # given emotion category
            scores = np.zeros(len(EMOTIONS))
            scores[:] = np.nan
        
            X, y = this_data.iloc[:, :33], this_data.loc[:, 'emotion']
            y_ohe = ohe.transform(y.to_numpy()[:, None])
            y_pred = model.predict_proba(X)
            idx = y_ohe.sum(axis=0) != 0
            scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)
            scores = pd.DataFrame(scores, columns=['score'])
            scores['emotion'] = EMOTIONS
            scores['sub'] = op.basename(f).split('_')[0].split('-')[1]
            ethn = df['sub_ethnicity'].unique()[0]
            scores['sub_ethnicity'] = {0: 'WC', 1: 'EA'}[ethn]
            scores['sub_split'] = df['sub_split'].unique()[0]
            scores['trial_split'] = 'train'#trial_split
            scores['face_gender'] = {0: 'F', 1: 'M', 'all': 'all'}[face_gender]
            scores['mapping'] =  mapp_name
            scores_all.append(scores)

# Save scores and predictions
scores = pd.concat(scores_all, axis=0)
scores.to_csv('results/scores.tsv', sep='\t')
print(scores.query("face_gender == 'all'").groupby(['mapping', 'sub_ethnicity']).mean())