import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm 


for type_ in ['emotion', 'conversational']:
    for ethn in ['WC', 'EA']:
        print(f"Processing ethnicity {ethn}")
        files = sorted(glob(f'data/ratings/{type_}/{ethn}/sub-*.tsv'))
        mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0).assign(
                                sub_split='test' if i % 3 == 0 else 'train'
                            ) for i, f in enumerate(files)], axis=0)

        if 'trial_split' in mega_df.columns:
            mega_df = mega_df.drop('trial_split', axis=1)

        fids = mega_df.loc[:, 'face_id'].to_numpy()

        # Get unique face IDs (fids)
        uniq_fids = np.unique(fids)
        if ethn == 'WC':
            uniq_fids = uniq_fids[uniq_fids > 7]

        # Try to find a face ID split which contains the fewest AU duplicates
        # across train and test
        fewest_duplicates = np.inf
        for i in tqdm(range(1000)):
            
            if ethn == 'WC' and type_ == 'emotion':
                # Draw 4 train IDs for 0-7 (participant 45-60)
                train_ids45 = np.random.choice(range(8), size=4, replace=False)
                
                # Draw train IDs for >7
                n_train_id = int(uniq_fids.size * 0.75)
                train_ids15 = np.random.choice(uniq_fids, size=n_train_id, replace=False)
                train_ids = np.concatenate((train_ids45, train_ids15))
            else:
                train_ids = np.random.choice(range(4), size=2, replace=False)

            # Split original DF in train and test and remove duplicates WITHIN train and test
            train_df = mega_df.loc[mega_df['face_id'].isin(train_ids), :].iloc[:, :33]
            train_df_uniq = train_df.drop_duplicates()
            test_df = mega_df.loc[~mega_df['face_id'].isin(train_ids), :].iloc[:, :33]
            test_df_uniq = test_df.drop_duplicates()

            # Check duplicates across train and test: we want to have
            # original faces (ID+AUs) in the train and test set!
            mrg = pd.concat((train_df_uniq, test_df_uniq))
            dups = mrg.duplicated()
            n_dups = test_df.loc[dups.index[dups], :].shape[0]

            # Keep track of fewest duplicates
            if n_dups < fewest_duplicates:
                best_train_ids = train_ids.copy()
                to_drop = dups.index[dups]
                fewest_duplicates = n_dups
                print(f"Fewest duplicates: {fewest_duplicates}")
                if n_dups == 0:
                    break

        mega_df.loc[:, 'trial_split'] = ['train' if fid in best_train_ids else 'test' for fid in mega_df['face_id']]
        train_df = mega_df.query("trial_split == 'train'")
        test_df = mega_df.query("trial_split == 'test'")

        # Remove all duplicates from test
        # (Technically, you can also remove them from the train set, but
        #  that is not strictly necessary)
        print(f"Test data before removing duplicates: {test_df.shape}")
        test_df = test_df.drop(to_drop, axis=0)
        print(f"Test data after removing duplicates: {test_df.shape}")

        # Remerge train and test into 'mega' dataframe
        mega_df = pd.concat((train_df, test_df), axis=0)

        for sub in range(1, 61):  # split by sub
            sub_id = f'{str(sub).zfill(2)}{ethn}'
            this_df = mega_df.query("sub == @sub_id").copy()
            # Because I'm paranoid, let's check again whether there are no duplicates across train and test
            train_df = this_df.query("trial_split == 'train'").iloc[:, :33].drop_duplicates()
            test_df = this_df.query("trial_split == 'test'").iloc[:, :33].drop_duplicates()
            f_out = f'data/ratings/{type_}/{ethn}/sub-{sub_id}_ratings.tsv'
            this_df.to_csv(f_out, sep='\t')