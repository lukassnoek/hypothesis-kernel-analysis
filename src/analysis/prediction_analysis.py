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

files = sorted(glob('data/ratings/*/*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
mega_df = mega_df.query("sub_split == 'train' & trial_split == 'train'")
mega_df = mega_df.query("emotion != 'other'")  # remove non-emo trials
mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs

for mapp_name in mappings:

    # Initialize model!
    model = KernelClassifier(au_cfg=None, param_names=None, kernel=kernel, ktype=ktype,
                             binarize_X=False, normalization='softmax', beta=beta)

    # Note that there is no "fitting" of the model! The mappings themselves
    # can be interpreted as already-fitted models
    model.add_Z(pd.read_csv(f'data/{mapp_name}.tsv', sep='\t', index_col=0))

    for sub_id in tqdm(mega_df['sub'].unique(), desc=mapp_name):
        df_l1 = mega_df.query("sub == @sub_id")

        for face_gender in ['M', 'F', 'all']:
            if face_gender != 'all':
                df_l2 = df_l1.query("face_gender == @face_gender")
            else:
                df_l2 = df_l1

            # Initialize with NaNs in case of no trials for a
            # given emotion category
            scores = np.zeros(len(EMOTIONS))
            scores[:] = np.nan
        
            X, y = df_l2.iloc[:, :33], df_l2.loc[:, 'emotion']
            y_ohe = ohe.transform(y.to_numpy()[:, None])
            y_pred = model.predict_proba(X)
            idx = y_ohe.sum(axis=0) != 0
            
            scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)
            scores = pd.DataFrame(scores, columns=['score'])
            scores['emotion'] = EMOTIONS
            scores['sub'] = sub_id
            ethn = df_l2['sub_ethnicity'].unique()[0]
            scores['sub_ethnicity'] = ethn
            scores['sub_split'] = df_l2['sub_split'].unique()[0]
            scores['trial_split'] = 'train'
            scores['face_gender'] = face_gender
            scores['mapping'] =  mapp_name
            scores_all.append(scores)

# Save scores and predictions
scores = pd.concat(scores_all, axis=0)
scores.to_csv('results/scores.tsv', sep='\t')
print(scores.query("face_gender == 'all'").groupby(['mapping', 'emotion', 'sub_ethnicity']).mean())