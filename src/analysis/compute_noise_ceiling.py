import sys
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import roc_auc_score
from noiseceiling import compute_nc_classification
from noiseceiling.bootstrap import run_bootstraps_nc


# TODO: convert sub_ethnicity to integer for speed!
kwargs = dict(
    use_repeats_only=True,
    soft=True, per_class=True,
    use_index=False,
    score_func=roc_auc_score,
    progress_bar=True
)

files = sorted(glob('data/ratings/*/*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
mega_df = mega_df.query("emotion != 'other'")  # remove non-emo trials
mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs
#mega_df['sub_ethnicity'] = mega_df['sub_ethnicity'].replace({'WC': 0, 'EA': 1})

emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'] 
emo2int = {s: i for i, s in enumerate(emotions)}

all_nc = []
# Compute the noise ceiling separately per ethnicity (and pooled)
# as well as per sub-specific and trial-specific train and test set
for ethn in ['all', 'WC', 'EA']:
    if ethn != 'all':
        df_l1 = mega_df.query("sub_ethnicity == @ethn")
    else:
        df_l1 = mega_df 

    for sub_split, trial_split in [('all', 'all'), ('train', 'train'), ('test', 'test')]:
        
        if sub_split != 'all':
            df_l2 = df_l1.query("sub_split == @sub_split & trial_split == @trial_split")
        else:
            df_l2 = df_l1
    
        idx = [col for col in df_l2.columns if 'AU' in col]
        X, y = df_l2.loc[:, idx], df_l2['emotion']
        X = X.astype(np.float16).round(1)
        nc = compute_nc_classification(X, y, **kwargs)
        
        nc_b = run_bootstraps_nc(X, y, kwargs=kwargs, n_bootstraps=100)
        nc = pd.DataFrame(np.c_[nc.to_numpy().squeeze(), nc_b.std(axis=0).to_numpy()],
                        columns=['noise_ceiling', 'sd'])
        nc['emotion'] = emotions
        nc['sub_ethnicity'] = ethn
        nc['sub_split'] = sub_split
        nc['trial_split'] = trial_split
        all_nc.append(nc)

nc = pd.concat(all_nc, axis=0)
nc.to_csv('results/noise_ceiling.tsv', sep='\t', index=True)