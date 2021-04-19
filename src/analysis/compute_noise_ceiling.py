import sys
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
sys.path.append('src')
from noiseceiling import compute_nc_classification
from noiseceiling.bootstrap import run_bootstraps_nc

# Load data from all subs
ratings = []
emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

for sub, f in enumerate(sorted(glob('data/ratings/sub*.tsv'))):
    df = pd.read_csv(f, sep='\t', index_col=0)
    df = df.query("emotion != 'other'")
    df = df.loc[df.index != 'empty', :]
    ratings.append(df)

ratings = pd.concat(ratings, axis=0)
kwargs = dict(
    use_repeats_only=True,
    soft=True, per_class=True,
    use_index=False,
    score_func=roc_auc_score,
    progress_bar=True
)

# nc = compute_nc_classification(
#     ratings.iloc[:, :33], ratings['emotion'], **kwargs
# )
# nc_b = run_bootstraps_nc(ratings.iloc[:, :33], ratings['emotion'], kwargs=kwargs)
# nc = pd.DataFrame(np.c_[nc.to_numpy().squeeze(), nc_b.std(axis=0).to_numpy()],
#                   columns=['noise_ceiling', 'sd'])
# nc['emotion'] = emotions
# nc.to_csv('results/noise_ceiling.tsv', sep='\t', index=True)

# ### ON HALF OF THE SUBJECTS
# ratings = []
# for sub, f in enumerate(sorted(glob('data/ratings/sub*.tsv')[1::2])):
#     df = pd.read_csv(f, sep='\t', index_col=0)
#     df = df.query("emotion != 'other'")
#     df = df.loc[df.index != 'empty', :]
#     ratings.append(df)

# ratings = pd.concat(ratings, axis=0)
# nc = compute_nc_classification(
#     ratings.iloc[:, :33], ratings['emotion'], **kwargs
# )
# nc_b = run_bootstraps_nc(ratings.iloc[:, :33], ratings['emotion'], kwargs=kwargs)
# nc = pd.DataFrame(np.c_[nc.to_numpy().squeeze(), nc_b.std(axis=0).to_numpy()],
#                   columns=['noise_ceiling', 'sd'])
# nc['emotion'] = emotions
# nc.to_csv('results/noise_ceiling_half.tsv', sep='\t', index=True)

### PER INTENSITY LEVEL
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
    nc = compute_nc_classification(
        tmp_ratings.iloc[:, :-2], tmp_ratings['emotion'], **kwargs
    )
    nc_b = run_bootstraps_nc(tmp_ratings.iloc[:, :-2], tmp_ratings['emotion'], kwargs=kwargs)
    nc = np.c_[nc.to_numpy().squeeze(), nc_b.std(axis=0).to_numpy()]
    nc = pd.DataFrame(nc, columns=['noise_ceiling', 'sd'])
    nc['emotion'] = emotions
    nc['intensity_level'] = intensity
    dfs.append(nc)

nc_df = pd.concat(dfs, axis=0)
nc_df.to_csv('results/noise_ceiling_intensity_stratified.tsv', sep='\t', index=True)