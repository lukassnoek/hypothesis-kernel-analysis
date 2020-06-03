import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from glob import glob
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


# Corrs
files = sorted(glob('data/ratings/sub*.tsv'))
data = [pd.read_csv(f, sep='\t', index_col=0) for f in files]
au_data = np.stack([d.iloc[:, :-2].values for d in data])
au_data = np.moveaxis(au_data, 2, 1)
emo_ratings = np.stack([d.iloc[:, -2] for d in data])

ohe = OneHotEncoder(sparse=False)
ohe.fit(emo_ratings[0, :, np.newaxis])
emo_ratings = np.stack([ohe.transform(emo_ratings[i, :, np.newaxis]) for i in range(60)])
corrs = vectorized_corr(au_data, emo_ratings)
idx = np.ones(7, dtype=bool)
cats = ohe.categories_[0]
idx[cats == 'other'] = False
corrs = corrs[:, :, idx] 
cats = cats[idx]

plt.imshow(corrs.mean(axis=0), aspect='auto')
plt.colorbar()
ax = plt.gca()
ax.set_xticks(range(6))
ax.set_xticklabels(cats)
ax.set_yticks(range(au_data.shape[1]))
ax.set_yticklabels(data[0].columns, fontdict=dict(fontsize=8))
plt.savefig('average_corr.png', dpi=200)
