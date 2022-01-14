import sys
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import roc_auc_score
from noiseceiling import compute_nc_classification
from noiseceiling.bootstrap import run_bootstraps_nc

emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'] 
all_nc = []

# Compute the noise ceiling separately per ethnicity (and pooled)
# as well as per sub-specific and trial-specific train and test set
for ethn in ['*', 'WC', 'EA']:
    for sub_split in ['all', 'train', 'test']:
        files = sorted(glob(f'data/ratings/{ethn}/sub*.tsv'))
        
        if sub_split == 'train':
            files = files[::2]
        elif sub_split == 'test':
            files = files[1::2]
        else:
            pass
    
        ratings = []
        for sub, f in enumerate(files):
            df = pd.read_csv(f, sep='\t', index_col=0)
            df = df.query("emotion != 'other'")
            df = df.loc[df.index != 'empty', :]
            ratings.append(df)

        ratings = pd.concat(ratings, axis=0)

        for trial_split in ['all', 'train', 'test']:
            if trial_split != 'all':
                these_ratings = ratings.query("trial_split == @trial_split")
            else:
                these_ratings = ratings

            kwargs = dict(
                use_repeats_only=True,
                soft=True, per_class=True,
                use_index=False,
                score_func=roc_auc_score,
                progress_bar=True
            )

            idx = [col for col in these_ratings.columns if 'AU' in col]
            nc = compute_nc_classification(
                these_ratings.loc[:, idx], these_ratings['emotion'], **kwargs
            )
            
            nc_b = run_bootstraps_nc(these_ratings.loc[:, idx], these_ratings['emotion'], kwargs=kwargs, n_bootstraps=100)
            nc = pd.DataFrame(np.c_[nc.to_numpy().squeeze(), nc_b.std(axis=0).to_numpy()],
                            columns=['noise_ceiling', 'sd'])
            nc['emotion'] = emotions
            nc['ethn'] = 'all' if ethn == '*' else ethn
            nc['sub_split'] = sub_split
            nc['trial_split'] = trial_split
            all_nc.append(nc)

nc = pd.concat(all_nc, axis=0)
nc.to_csv('results/noise_ceiling.tsv', sep='\t', index=True)