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
emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

i = 0
for sub, f in enumerate(tqdm(sorted(glob('data/ratings/sub*.tsv')[:N]))):
    df = pd.read_csv(f, sep='\t', index_col=0)
    # Remove "other" trials (perhaps too optimistic?)
    df = df.query("emotion != 'other'")
    df = df.loc[df.index != 'empty', :]
    ratings.append(df)

ratings = pd.concat(ratings, axis=0)
nc = compute_noise_ceiling(
    ratings['emotion'],
    only_repeats=True
)

nc = pd.DataFrame(nc[:, np.newaxis], index=['between_subjects'] * 6, columns=['noise_ceiling'])
nc['emotion'] = emotions
nc_b = compute_noise_ceiling(
    ratings['emotion'],
    only_repeats=True,
    n_bootstraps=20
)
sd = np.nanstd(nc_b, axis=0)
nc['sd'] = sd
nc.to_csv('results/noise_ceiling.tsv', sep='\t', index=True)

### INTENSITY ANALYSIS
mean_intensity = ratings['intensity'].reset_index().groupby('index').mean()
ratings.loc[mean_intensity.index, 'intensity'] = mean_intensity['intensity']
percentiles = ratings['intensity'].quantile([0, .25, 0.5, 0.75, 1.])
nc_df = pd.DataFrame(columns=[
    ['participant_id', 'emotion', 'intensity', 'noise_ceiling', 'sd']
])
dfs = []
i = 0
for intensity in [1, 2, 3, 4]:
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
    df = pd.DataFrame(np.c_[nc])
    print(sd)
    print(nc)   
    for emo, val, s in zip(emotions, nc, sd):
        print(nc_df)
    
        nc_df.loc[i, 'participant_id'] = 'between_subjects'
        nc_df.loc[i, 'emotion'] = emo
        nc_df.loc[i, 'intensity'] = intensity
        nc_df.loc[i, 'noise_ceiling'] = val
        nc_df.loc[i, 'sd'] = s
        i += 1

nc_df.to_csv('results/noise_ceiling_per_intensity_level.tsv', sep='\t', index=True)