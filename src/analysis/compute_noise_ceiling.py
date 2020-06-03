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
ncs = np.zeros((N, 6))
for i, f in enumerate(tqdm(sorted(glob('data/ratings/sub*.tsv')[:N]))):
    df = pd.read_csv(f, sep='\t', index_col=0)
    # Remove "other" trials (perhaps too optimistic?)
    df = df.query("emotion != 'other'")
    
    # Remove "empty" trials (no AUs)
    df = df.loc[df.index != 'empty', :]
    ncs[i, :] = compute_noise_ceiling(
        df['emotion'],
        scoring=roc_auc_score,
        soft=True,
        progbar=False,
        doubles_only=True
    )
    ratings.append(df)

emo_labels = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
idx = [f"sub-{str(i+1).zfill(2)}" for i in range(N)]
ncs = pd.DataFrame(ncs, columns=emo_labels, index=idx)

ratings = pd.concat(ratings, axis=0)
nc = compute_noise_ceiling(
    ratings['emotion'],
    scoring=roc_auc_score,
    soft=True,
    progbar=True,
    doubles_only=True
)[np.newaxis, :]
ncs = ncs.append(pd.DataFrame(nc, columns=emo_labels, index=['between_subjects']))
ncs.to_csv('data/noise_ceilings.tsv', sep='\t', index=True)
