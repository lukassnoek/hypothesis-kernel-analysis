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
emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

i = 0
for sub, f in enumerate(tqdm(sorted(glob('data/ratings/sub*.tsv')[1::2]))):
    df = pd.read_csv(f, sep='\t', index_col=0)
    # Remove "other" trials (perhaps too optimistic?)
    df = df.query("emotion != 'other'")
    df = df.loc[df.index != 'empty', :]
    ratings.append(df)

ratings = pd.concat(ratings, axis=0)
mean_intensity = ratings['intensity'].reset_index().groupby('index').mean()
ratings.loc[mean_intensity.index, 'intensity'] = mean_intensity['intensity']
percentiles = ratings['intensity'].quantile([0, .2, .4, .6, .8, 1.])

nc_df = pd.DataFrame(columns=[
    ['participant_id', 'emotion', 'intensity', 'noise_ceiling', 'sd']
])
i = 0
for intensity in tqdm([0, 1, 2, 3, 4, 5]):
    if intensity == 0:
       tmp_ratings = ratings.copy()
    else:
        minn, maxx = percentiles.iloc[intensity-1], percentiles.iloc[intensity]
        tmp_ratings = ratings.query("@minn <= intensity & intensity <= @maxx")    

    nc = compute_noise_ceiling(
        tmp_ratings['emotion'],
        only_repeats=True
    )

    nc_b = compute_noise_ceiling(
        tmp_ratings['emotion'],
        only_repeats=True,
        n_bootstraps=20
    )

    sd = np.nanstd(nc_b, axis=0)
    print(sd)
    for emo, val, s in zip(emotions, nc, sd):
        nc_df.loc[i, 'participant_id'] = 'between_subjects'
        nc_df.loc[i, 'emotion'] = emo
        nc_df.loc[i, 'intensity'] = intensity
        #nc_df.loc[i, 'nr_trials'] = nt
        nc_df.loc[i, 'noise_ceiling'] = val
        nc_df.loc[i, 'sd'] = s
        i += 1

nc_df.to_csv('results/noise_ceilings_half_subjects.tsv', sep='\t', index=True)
