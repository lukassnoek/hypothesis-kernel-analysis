import numpy as np
import pandas as pd
from glob import glob
from scipy import stats
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder


def compute_corrs(X, Y):
    """ Compute Pearson correlation between each column in X
    and each column in Y. """
    corrs = np.zeros((X.shape[1], Y.shape[1]))  # 33 x 7
    pvals = np.zeros_like(corrs)
    for au_i in range(X.shape[1]):

        for emo_i in range(Y.shape[1]):
            corrs[au_i, emo_i], pvals[au_i, emo_i] = pearsonr(X[:, au_i], Y[:, emo_i])

    corrs[np.isnan(corrs)] = 0
    return corrs, pvals


# Load in data and split in AUs and emo ratings
#files = sorted(glob('data/ratings/sub*.tsv'))

# One-hot encode emotions
ohe = OneHotEncoder(sparse=False)
emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'other']
ohe.fit(np.array(emotions)[:, None])

# Remove "other" (but not from data)
idx = np.ones(7, dtype=bool)
cats = ohe.categories_[0]
idx[cats == 'other'] = False
cats = cats[idx]

files = sorted(glob('data/ratings/*/*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)

for ethn in ['WC', 'EA', '*']:
    if ethn != '*':
        df_l1 = mega_df.query("sub_ethnicity == @ethn")
    else:
        df_l1 = mega_df

    for sub_split in ['train', 'test']:
        df_l2 = df_l1.query("sub_split == @sub_split")
        for trial_split in ['train', 'test']:
            df_l3 = df_l2.query("trial_split == @trial_split")

            # Compute all correlations
            sub_ids = df_l3['sub'].unique()
            corrs = np.zeros((len(sub_ids), 33, 7))
            
            N = []
            for i, sub_id in enumerate(sub_ids):
                df = df_l3.query("sub == @sub_id")
                #df_raw = pd.read_csv(f, sep='\t', index_col=0).query("trial_split == @trial_split")
                X, Y = df.iloc[:, :33].to_numpy(), df.loc[:, 'emotion'].to_numpy()
                Y = ohe.transform(Y[:, None])
                corrs[i, :,:], pvals = compute_corrs(X, Y) 
                N.append(X.shape[0])

            N = int(np.round(np.mean(N)))
            
            # Remove other
            corrs = corrs[:, :, idx]
            corrs_av = corrs.mean(axis=0)
                
            t_corrs = corrs_av * np.sqrt(N - 2) / np.sqrt(1 - corrs_av**2)
            t_corrs[t_corrs < 0] = 0
            p_corrs = stats.t.sf(np.abs(t_corrs), N - 1) * 2 
            hyp = (p_corrs < 0.05).astype(int)
            df = pd.DataFrame(hyp.T, columns=df.columns[:33], index=emotions[:6])
            ethn = 'all' if ethn == '*' else ethn
            df.to_csv(f'data/JackSchyns_ethn-{ethn}_sub-{sub_split}_trial-{trial_split}.tsv', sep='\t')