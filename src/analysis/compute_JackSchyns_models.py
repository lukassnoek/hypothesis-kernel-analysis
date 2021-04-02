import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
from sklearn.preprocessing import OneHotEncoder


def vectorized_corr(X, y):
    """ Assuming X is a 3D array and y a 1D column vector. """
    Xbar = X.mean(axis=2)         # 60 x 42
    ybar = y.mean(axis=1)         # 60 x 6
    
    corrs = np.zeros((X.shape[0], X.shape[1], y.shape[2]))  # 60 x 42 x 6
    for emo_i in range(y.shape[-1]):
        yn = y[:, :, emo_i] - ybar[:, emo_i, np.newaxis]
        Xn = X - Xbar[..., np.newaxis]
        
        for au_i in range(X.shape[1]):
            cov_ = np.sum(Xn[:, au_i, :] * yn, axis=1)
            var_ = np.sqrt(np.sum(Xn[:, au_i, :] ** 2, axis=1) * np.sum(yn ** 2, axis=1))
            corrs[:, au_i, emo_i] = cov_ / var_

    corrs[np.isnan(corrs)] = 0
    return corrs


# Load in data and split in AUs and emo ratings
files = sorted(glob('data/ratings/sub*.tsv'))
data = [pd.read_csv(f, sep='\t', index_col=0) for f in files]
au_data = np.stack([d.iloc[:, :-2].values for d in data])
au_data = np.moveaxis(au_data, 2, 1)
emo_ratings = np.stack([d.iloc[:, -2] for d in data])

# One-hot encode emotions
ohe = OneHotEncoder(sparse=False)
ohe.fit(emo_ratings[0, :, np.newaxis])
emo_ratings = np.stack([ohe.transform(emo_ratings[i, :, np.newaxis]) for i in range(60)])
# au_data: 60 x 34 x 2400
# emo_ratings: 60 x 2400 x 7

# Binarize AU activations
au_data = (au_data > 0).astype(int)

# Remove "other"
idx = np.ones(7, dtype=bool)
cats = ohe.categories_[0]
idx[cats == 'other'] = False
cats = cats[idx]

# Compute all correlations
dfs = []
for t_split in ['odd', 'even', 'all']:
    if t_split == 'even':
        au, emo = au_data[:, :, ::2], emo_ratings[:, ::2, :]
    elif t_split == 'odd':
        au, emo = au_data[:, :, 1::2], emo_ratings[:, 1::2, :] 
    else:
        au, emo = au_data, emo_ratings
    
    corrs = vectorized_corr(au_data, emo_ratings)  # 60 x 42 x 6
    corrs = corrs[:, :, idx] 

    t_corrs = corrs * np.sqrt(au.shape[2] - 2) / np.sqrt(1 - corrs**2)
    t_corrs[t_corrs < 0] = 0
    p_corrs = stats.t.sf(np.abs(t_corrs), au.shape[2] - 1) * 2 
    hyp = (p_corrs < 0.05 / 36).astype(int)
    for i in range(60):
        df = pd.DataFrame(hyp[i, :, :].T, columns=data[0].columns[:-2])
        df['sub'] = str(i + 1).zfill(2)
        df['trial_split'] = t_split
        df['emotion'] = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        dfs.append(df)

    for s_split in ['even', 'odd', 'all']:
        if s_split == 'even':
            corrs_av = corrs[::2, :, :].mean(axis=0)
        elif s_split == 'odd':
            corrs_av = corrs[1::2, :, :].mean(axis=0)
        else:
            corrs_av = corrs.mean(axis=0)

        t_corrs = corrs_av * np.sqrt(au.shape[2] - 2) / np.sqrt(1 - corrs_av**2)
        t_corrs[t_corrs < 0] = 0
        p_corrs = stats.t.sf(np.abs(t_corrs), au.shape[2] - 1) * 2 
        hyp = (p_corrs < 0.05 / 36).astype(int)
        df = pd.DataFrame(hyp.T, columns=data[0].columns[:-2])
        df['sub'] = 'average_' + s_split
        df['trial_split'] = t_split
        df['emotion'] = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        dfs.append(df)       

df = pd.concat(dfs, axis=0)
df.to_csv('data/JackSchyns.tsv', sep='\t', index=False)