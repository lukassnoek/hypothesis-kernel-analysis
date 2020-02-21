import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm


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

mat = loadmat('data/AU_data_for_Lukas.mat')
au_names = [n[0] for n in mat['AUnames'][0]]
emo_names = [e[0] for e in mat['expnames'][0]]

au_data = mat['data_AUamp']
au_data[np.isnan(au_data)] = 0
au_data_onoff = mat['data_AUon']

# Check whether the on/off data is indeed amplitude > 0
au_data_bin = (au_data > 0).astype(int)
np.testing.assert_array_equal(au_data_bin, au_data_onoff)

# Make sure dimensions match
au_data = np.moveaxis(au_data, -1, 0)              # 60 x 42 x 2400
au_model = np.moveaxis(mat['models_AUon'], -1, 1)  # 60 x 42 x 6
emo_rating = np.moveaxis(mat['data_cat'], -1, 0)   # 60 x 2400 x 7
intensity = mat['data_rat'].T                      # 60 x 2400

# Corrs
corrs = vectorized_corr((au_data > 0).astype(int), emo_rating[:, :, :-1])
plt.imshow(corrs.mean(axis=0), aspect='auto')
plt.colorbar()
ax = plt.gca()
ax.set_xticklabels([''] + emo_names[:-1])
ax.set_yticks(range(len(au_names)))
ax.set_yticklabels(au_names, fontdict=dict(fontsize=8))
plt.savefig('average_corr.png', dpi=200)

for i in tqdm(range(au_data.shape[0])):
    idx = []
    for ii in range(au_data.shape[2]):
        au_on = np.where(au_data[i, :, ii] > 0)[0]
        this_idx= '_'.join(
            [f'{au_names[iii]}-{int(100 * au_data[i, iii, ii])}'
             for iii in au_on]
        )
        
        if not this_idx:
            this_idx = 'empty'

        idx.append(this_idx)

    df = pd.DataFrame(au_data[i, :, :].T, columns=au_names, index=idx)
    df = df.drop(['AU7', 'AU12L', 'AU12R'], axis=1)
    df['emotion'] = [emo_names[idx] for idx in emo_rating[i, :, :].argmax(axis=1)]
    df['intensity'] = intensity[i, :]

    f_out = f'data/ratings/sub-{str(i+1).zfill(2)}_ratings.tsv'
    df.to_csv(f_out, sep='\t')