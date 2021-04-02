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
kwargs = dict(
    use_repeats_only=True,
    soft=True, per_class=True,
    use_index=False,
    score_func=roc_auc_score
)
nc = compute_nc_classification(
    ratings.iloc[:, :34], ratings['emotion'], **kwargs
)
nc_b = run_bootstraps_nc(ratings.iloc[:, :34], ratings['emotion'], kwargs=kwargs)
nc = pd.DataFrame(np.c_[nc.to_numpy().squeeze(), nc_b.std(axis=0).to_numpy()],
                  columns=['noise_ceiling', 'sd'])
nc['emotion'] = emotions
nc.to_csv('results/noise_ceiling2.tsv', sep='\t', index=True)
