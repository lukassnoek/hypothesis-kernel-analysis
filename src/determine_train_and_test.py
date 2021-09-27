import numpy as np
import pandas as pd
from glob import glob
from noiseceiling.utils import _find_repeats


files = sorted(glob('data/ratings/sub-*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0).assign(sub=i+1)
                     for i, f in enumerate(files)], axis=0)
if 'data_split' in mega_df.columns:
    mega_df = mega_df.drop('data_split', axis=1)

fids = mega_df.loc[:, 'face_id'].to_numpy()

# Get unique face IDs (fids)
uniq_fids = np.unique(fids)
uniq_fids = uniq_fids[uniq_fids > 7]

# Try to find a face ID split which contains the fewest AU duplicates
# across train and test
fewest_duplicates = np.inf
for i in range(100):
    # Draw train IDs for 0-7 (participant 45-60)
    train_ids45 = np.random.choice(range(7), size=4, replace=False)
    
    # Draw train IDs for >7
    train_ids15 = np.random.choice(uniq_fids, size=uniq_fids.size // 2, replace=False)
    train_ids = np.concatenate((train_ids45, train_ids15))
    
    # Split original DF in train and test and remove duplicates WITHIN train and test
    train_df = mega_df.loc[mega_df['face_id'].isin(train_ids), :].iloc[:, :33]
    train_df_uniq = train_df.drop_duplicates()
    test_df = mega_df.loc[~mega_df['face_id'].isin(train_ids), :].iloc[:, :33]
    test_df_uniq = test_df.drop_duplicates()

    # Check duplicates across train and test
    mrg = pd.concat((train_df_uniq, test_df_uniq))
    dups = mrg.duplicated()
    n_dups = test_df.loc[dups.index[dups], :].shape[0]

    # Keep track of fewest duplicates
    if n_dups < fewest_duplicates:
        best_train_ids = train_ids.copy()
        to_drop = dups.index[dups]
        fewest_duplicates = n_dups
        print(f"Fewest duplicates: {fewest_duplicates}")

mega_df.loc[:, 'data_split'] = ['train' if fid in best_train_ids else 'test' for fid in mega_df['face_id']]
train_df = mega_df.query("data_split == 'train'")
test_df = mega_df.query("data_split == 'test'")

# Remove all duplicates from test
print(f"Test data before removing duplicates: {test_df.shape}")
test_df = test_df.drop(to_drop, axis=0)
print(f"Test data after removing duplicates: {test_df.shape}")

mega_df = pd.concat((train_df, test_df), axis=0)

for sub in range(1, 61):
    this_df = mega_df.query("sub == @sub").copy().drop('sub', axis=1)
    
    # Because I'm paranoid, let's check again whether there are no duplicates across train and test
    train_df = this_df.query("data_split == 'train'").iloc[:, :33].drop_duplicates()
    test_df = this_df.query("data_split == 'test'").iloc[:, :33].drop_duplicates()
    assert(pd.concat((train_df, test_df), axis=0).duplicated().sum() == 0)
    f_out = f'data/ratings/sub-{str(sub).zfill(2)}_ratings.tsv'
    this_df.to_csv(f_out, sep='\t')