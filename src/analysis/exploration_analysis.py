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

mappings = ['Cordaro2018IPC', 'Cordaro2018ref', 'Darwin', 'Ekman', 'Keltner2019', 'Matsumoto2008',
            'JackSchyns_ethn-WC_sub-train_trial-train',
            'JackSchyns_ethn-EA_sub-train_trial-train',
            'JackSchyns_ethn-all_sub-train_trial-train',
            ]

scores_all = []    
for mapp_name in tqdm(mappings):

    model = KernelClassifier(au_cfg=None, param_names=None, kernel='cosine', ktype='similarity',
                            binarize_X=False, normalization='softmax', beta=1)

    # Save Z_orig for later!
    Z_orig = pd.read_csv(f'data/{mapp_name}.tsv', sep='\t', index_col=0)

    for emo in EMOTIONS:

        for sub_split in ['train', 'test', 'all']:
            # We only report results on the test set (sub & trial)!
            # See figures.ipynb

            if sub_split != 'all':
                this_df = df_abl.query("sub_split == @sub_split")
            else:
                this_df = df_abl
            
            for ethn in [0, 1, 'all']:
                this_df = this_df.query("ablated_from == @emo & emotion == @emo & score != 0")
                if ethn != 'all':
                    this_df = this_df.query(f"sub_ethnicity == {ethn}")
                
                Z_opt = Z_orig.copy()

                # Which AUs improve prediction? 
                aus = this_df.groupby('ablated_au').mean().query("score < 0").index.tolist()
                for au in aus:
                    Z_opt.loc[emo, au] = 1

                # Which AUs decrease prediction? 
                aus = this_df.groupby('ablated_au').mean().query("score > 0").index.tolist()
                for au in aus:
                    Z_opt.loc[emo, au] = 0

                sub_files = sorted(glob(f'data/ratings/*/*.tsv'))

                for f in sub_files:
                    df = pd.read_csv(f, sep='\t', index_col=0)
                    df = df.query("emotion != 'other'")  # remove non-emo trials
                    df = df.loc[df.index != 'empty', :]  # remove trials w/o AUs
                
                    for trial_split in ['train', 'test', 'all']:
                        if trial_split != 'all':
                            data = df.query("trial_split == @trial_split")
                        else:
                            data = df

                        X, y = data.iloc[:, :33], data.loc[:, 'emotion']

                        # Original prediction
                        model.add_Z(Z_orig)

                        # Predict data + compute performance (AUROC)
                        y_pred = model.predict_proba(X)
                        y_ohe = ohe.transform(y.to_numpy()[:, np.newaxis])
                        idx = y_ohe.sum(axis=0) != 0
                        
                        scores = np.zeros(len(EMOTIONS))
                        scores[:] = np.nan
                        scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)

                        model.add_Z(Z_opt)
                        y_pred = model.predict_proba(X)
                        new = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)
                        scores[idx] = new - scores[idx]

                        # Store scores and raw predictions
                        scores = pd.DataFrame(scores, columns=['score'])
                        scores['emotion'] = EMOTIONS
                        scores['sub'] = op.basename(f).split('_')[0].split('-')[1]
                        scores['sub_ethnicity'] = df['sub_ethnicity'].unique()[0]
                        scores['sub_split'] = sub_split
                        scores['trial_split'] = trial_split
                        scores['mapping'] =  mapp_name
                        scores['model_ethnicity'] =  ethn
                        
                    scores_all.append(scores)

scores = pd.concat(scores_all, axis=0)
print(scores.groupby(['sub_ethnicity', 'model_ethnicity']).mean())
scores.to_csv('results/scores_optimal.tsv', sep='\t')
