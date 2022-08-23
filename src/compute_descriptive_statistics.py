import numpy as np
import pandas as pd
from glob import glob
from noiseceiling.utils import _find_repeats

files = sorted(glob('data/ratings/*/*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
#mega_df = mega_df.query("sub_split == 'train' & trial_split == 'train'")
mega_df = mega_df.query("emotion != 'other'")  # remove non-emo trials
mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs

#rep_idx, _ = _find_repeats(mega_df.iloc[:, :33], progress_bar=True)
#unique, counts = np.unique(rep_idx, return_counts=True)
#print(np.mean(counts))

n_trials = np.zeros((60, 2))
for i, ethn in enumerate(['WC', 'EA']):
    df = mega_df.query("sub_ethnicity == @ethn")
    rep_idx, _ = _find_repeats(df.iloc[:, :33], progress_bar=True)
    unique, counts = np.unique(rep_idx, return_counts=True)
    print(np.mean(counts))

    for ii, sub in enumerate(df['sub'].unique()):
        df_ = df.query("sub == @sub")
        n_trials[ii, i] = df_.shape[0]
        
print(n_trials.mean(axis=0))
print(n_trials.mean())
