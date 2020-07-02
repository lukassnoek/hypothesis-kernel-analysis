import sys
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
sys.path.append('src')
from noise_ceiling import compute_noise_ceiling


# Load data from all subs
ratings = []
N = 60
nc_df = pd.DataFrame(
    index=range(N * 6),
    columns=['participant_id', 'emotion', 'intensity', 'nr_trials', 'noise_ceiling', 'sd']
)
emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

i = 0
for sub, f in enumerate(tqdm(sorted(glob('data/ratings/sub*.tsv')[:N]))):
    df = pd.read_csv(f, sep='\t', index_col=0)
    # Remove "other" trials (perhaps too optimistic?)
    df = df.query("emotion != 'other'")
    this_sub = f'sub-{str(sub+1).zfill(2)}'
    # Remove "empty" trials (no AUs)
    df = df.loc[df.index != 'empty', :]
    mean_int = df['intensity'].reset_index().groupby('index').mean()
    df.loc[mean_int.index, 'intensity'] = mean_int['intensity']

    percentiles = df['intensity'].quantile([0, .2, .4, .6, .8, 1.])
    for intensity in [0, 1, 2, 3, 4, 5]:
        if intensity == 0:
            tmp_df = df.copy()
        else:
            minn, maxx = percentiles.iloc[intensity-1], percentiles.iloc[intensity]
            tmp_df = df.query("@minn <= intensity & intensity <= @maxx")    
        
        nc, nt = compute_noise_ceiling(
            tmp_df['emotion'],
            scoring=roc_auc_score,
            soft=True,
            progbar=False,
            doubles_only=True,
            K=6,
            return_number=True
        )
        for emo, val in zip(emotions, nc):
            nc_df.loc[i, 'participant_id'] = this_sub
            nc_df.loc[i, 'emotion'] = emo
            nc_df.loc[i, 'intensity'] = intensity
            nc_df.loc[i, 'nr_trials'] = nt
            nc_df.loc[i, 'noise_ceiling'] = val
            i += 1

    ratings.append(df)

nc_df['noise_ceiling'] = nc_df['noise_ceiling'].astype(float)

ratings = pd.concat(ratings, axis=0)
mean_intensity = ratings['intensity'].reset_index().groupby('index').mean()
ratings.loc[mean_intensity.index, 'intensity'] = mean_intensity['intensity']

i = nc_df.shape[0]
percentiles = ratings['intensity'].quantile([0, .2, .4, .6, .8, 1.])
for intensity in [0, 1, 2, 3, 4, 5]:
    if intensity == 0:
       tmp_ratings = ratings.copy()
    else:
        minn, maxx = percentiles.iloc[intensity-1], percentiles.iloc[intensity]
        tmp_ratings = ratings.query("@minn <= intensity & intensity <= @maxx")    
    
    nc, nt = compute_noise_ceiling(
        tmp_ratings['emotion'],
        scoring=roc_auc_score,
        soft=True,
        progbar=False,
        doubles_only=True,
        K=6,
        return_number=True
    )
    bootstrap_nc = np.zeros((10, 6))
    for r in tqdm(range(10)):
        bootstrap_nc[r, :] = compute_noise_ceiling(
            tmp_df['emotion'],
            scoring=roc_auc_score,
            soft=True,
            progbar=False,
            doubles_only=True,
            K=6,
            bootstrap=True
        )

    sd = np.nanstd(bootstrap_nc, axis=0)
    for emo, val, s in zip(emotions, nc, sd):
        nc_df.loc[i, 'participant_id'] = 'between_subjects'
        nc_df.loc[i, 'emotion'] = emo
        nc_df.loc[i, 'intensity'] = intensity
        nc_df.loc[i, 'nr_trials'] = nt
        nc_df.loc[i, 'noise_ceiling'] = val
        nc_df.loc[i, 'sd'] = s
        i += 1

nc_df.to_csv('data/noise_ceilings.tsv', sep='\t', index=True)
