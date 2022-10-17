import numpy as np
import pandas as pd
from glob import glob
from noiseceiling.utils import _find_repeats

for type_ in ['emotion', 'conversational']:

    files = sorted(glob(f'data/ratings/{type_}/*/*.tsv'))
    mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
    
    if type_ == 'emotion':
        mega_df = mega_df.query("emotion != 'other'")  # remove non-emo trials
    else:
        mega_df = mega_df.query("state != 'other'")  # remove non-emo trials

    mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs
    print(f"Total number of trials ({type_}): {mega_df.shape[0]}")

    rep_idx, _ = _find_repeats(mega_df.iloc[:, :33], progress_bar=True)
    unique, counts = np.unique(rep_idx, return_counts=True)
    print('Repetitions all: ', len(unique), np.mean(counts))

    we_df = mega_df.query("sub_ethnicity == 'WC'")
    print(f"Total number of trials ({type_}), WE: {we_df.shape[0]}")

    rep_idx, _ = _find_repeats(we_df.iloc[:, :33], progress_bar=True)
    unique, counts = np.unique(rep_idx, return_counts=True)
    print('Repetitions WE: ', len(unique), np.mean(counts))

    ea_df = mega_df.query("sub_ethnicity == 'EA'")
    print(f"Total number of trials ({type_}), EA: {ea_df.shape[0]}")

    rep_idx, _ = _find_repeats(ea_df.iloc[:, :33], progress_bar=True)
    unique, counts = np.unique(rep_idx, return_counts=True)
    print('Repetitions EA: ', len(unique), np.mean(counts))
