import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
from scipy.stats import pearsonr

from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from noiseceiling.utils import _find_repeats


def compute_corrs(X, Y):

    corrs = np.zeros((X.shape[1], Y.shape[1]))  # 33 x 7
    for au_i in range(X.shape[1]):

        for emo_i in range(Y.shape[1]):
            corrs[au_i, emo_i] = pearsonr(X[:, au_i], Y[:, emo_i])[0]

    corrs[np.isnan(corrs)] = 0
    return corrs


# Load in data and split in AUs and emo ratings
files = sorted(glob('data/ratings/sub*.tsv'))

# One-hot encode emotions
ohe = OneHotEncoder(sparse=False)
ohe.fit(np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'other'])[:, None])

# Remove "other" (but not from data)
idx = np.ones(7, dtype=bool)
cats = ohe.categories_[0]
idx[cats == 'other'] = False
cats = cats[idx]

# Compute all correlations
dfs = []
corrs = np.zeros((len(files), 33, 7))
N = []
for i, f in enumerate(files):

    df_raw = pd.read_csv(f, sep='\t', index_col=0)#.query("data_split == 'train'")
    X, Y = df_raw.iloc[:, :33].to_numpy(), df_raw.loc[:, 'emotion'].to_numpy()
    Y = ohe.transform(Y[:, None])
    corrs[i, :,:] = compute_corrs(X, Y) 
    N.append(X.shape[0])

N = int(np.round(np.mean(N)))
# Remove other
corrs = corrs[:, :, idx]

for sub_split in ['train', 'test']:
    if sub_split == 'train':
        corrs_av = corrs[0::2, :, :].mean(axis=0)
    elif sub_split == 'test':
        corrs_av = corrs[1::2, :, :].mean(axis=0)
    
    t_corrs = corrs_av * np.sqrt(N - 2) / np.sqrt(1 - corrs_av**2)
    t_corrs[t_corrs < 0] = 0
    p_corrs = stats.t.sf(np.abs(t_corrs), N - 1) * 2 
    hyp = (p_corrs < 0.05).astype(int)
    df = pd.DataFrame(hyp.T, columns=df_raw.columns[:33])
    df['sub_split'] = sub_split
    df['trial_split'] = 'train'
    df['emotion'] = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    dfs.append(df)

df = pd.concat(dfs, axis=0)
print(df.sort_values(by=['emotion', 'trial_split']))
df.to_csv('data/JackSchyns.tsv', sep='\t')

mapp = df.query("sub_split == 'train' & trial_split == 'train'")
mapp = mapp.drop(['sub_split', 'trial_split'], axis=1).set_index('emotion')
mapp_dict = {}
for emo, row in mapp.iterrows():
    mapp_dict[emo] = sorted(mapp.columns[mapp.loc[emo, :] == 1].tolist())
    print(f"'{emo}': {mapp_dict[emo]},")
