import os
import os.path as op
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from models import KernelClassifier

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])
ohe = OneHotEncoder(sparse=False)
ohe.fit(EMOTIONS[:, np.newaxis])

df_abl = pd.read_csv('results/scores_ablation.tsv', sep='\t', index_col=0)
df_abl = df_abl.query("face_gender == 'all'")

mappings = ['Cordaro2018IPC', 'Cordaro2018ref', 'Darwin', 'Ekman', 'Keltner2019', 'Matsumoto2008',
            'JackSchyns_ethn-all_CV',
            ]

files = sorted(glob('data/ratings/*/*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
mega_df = mega_df.query("sub_split == 'test' & trial_split == 'test'")
mega_df = mega_df.query("emotion != 'other'")  # remove non-emo trials
mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs

scores_all = []    
for mapp_name in tqdm(mappings):

    model = KernelClassifier(au_cfg=None, param_names=None, kernel='cosine', ktype='similarity',
                            binarize_X=False, normalization='softmax', beta=1)

    # Save Z_orig for later!
    Z_orig = pd.read_csv(f'data/{mapp_name}.tsv', sep='\t', index_col=0)
                
    for ethn in ['all', 'WC', 'EA']:

        if ethn != 'all':
            df_abl_l1 = df_abl.query(f"sub_ethnicity == '{ethn}'")
        else:
            df_abl_l1 = df_abl
        
        #Z_opt = pd.read_csv('data/JackSchyns_ethn-all_CV.tsv', sep='\t', index_col=0)
        Z_opt = Z_orig.copy()
        for emo in EMOTIONS:    
            df_abl_l2 = df_abl_l1.query("ablated_from == @emo & emotion == @emo & score != 0")
            aus = df_abl_l2.groupby('ablated_au').mean().query("score < 0").index.tolist()
            
            for au in aus:
                Z_opt.loc[emo, au] = 1

            # Which AUs decrease prediction? 
            aus = df_abl_l2.groupby('ablated_au').mean().query("score > 0").index.tolist()
            for au in aus:
                Z_opt.loc[emo, au] = 0

        sub_ids = mega_df['sub'].unique().tolist()
        for sub_id in sub_ids:
            df = mega_df.query("sub == @sub_id")

            scores = np.zeros(len(EMOTIONS))
            scores[:] = np.nan
        
            X, y = df.iloc[:, :33], df.loc[:, 'emotion']
            y_ohe = ohe.transform(y.to_numpy()[:, None])
            
            # Original prediction
            model.add_Z(Z_orig)
            y_pred = model.predict_proba(X)
            idx = y_ohe.sum(axis=0) != 0
            scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)
            
            model.add_Z(Z_opt)
            y_pred = model.predict_proba(X)
            new = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)
            scores[idx] = new - scores[idx]

            scores = pd.DataFrame(scores, columns=['score'])
            scores['emotion'] = EMOTIONS
            scores['sub'] = sub_id
            sub_ethn = df['sub_ethnicity'].unique()[0]
            scores['sub_ethnicity'] = sub_ethn
            scores['sub_split'] = df['sub_split'].unique()[0]
            scores['trial_split'] = 'train'
            scores['mapping'] =  mapp_name
            scores['model_ethnicity'] = ethn
            scores_all.append(scores)

scores = pd.concat(scores_all, axis=0)
print(scores.groupby(['model_ethnicity', 'sub_ethnicity', 'emotion']).mean())
scores.to_csv('results/scores_optimal.tsv', sep='\t')
