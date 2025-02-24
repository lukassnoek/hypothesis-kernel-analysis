import numpy as np
import pandas as pd
from glob import glob
from scipy import stats
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder


def estimate_model(df, ohe, type_='emotion'):

    # One-hot encode emotions
    #ohe = OneHotEncoder(sparse=False)
    if type_ == 'emotion':
        target_col = 'emotion'
    else:
        target_col = 'state'

    labels = ohe.categories_[0]

    sub_ids = df['sub'].unique()
    models = np.zeros((len(sub_ids), len(labels), 33))
    pvalss = np.zeros((len(sub_ids), len(labels), 33))

    N = np.zeros(len(sub_ids))
    for i, sub_id in enumerate(sub_ids):
        df_l1 = df.query("sub == @sub_id")
        X = df_l1.iloc[:, :33].to_numpy()
        Y = ohe.transform(df_l1[target_col].to_numpy()[:, None])        
        
        corrs = np.zeros((Y.shape[1], X.shape[1]))  # 6 x 33
        pvals = np.zeros_like(corrs)
        for emo_i in range(Y.shape[1]):
            for au_i in range(X.shape[1]):
                if Y[:, emo_i].sum() == 0:
                    c, p = 0, 1
                else:
                    c, p = pearsonr(X[:, au_i], Y[:, emo_i])

                corrs[emo_i, au_i], pvals[emo_i, au_i] = c, p

        corrs[np.isnan(corrs)] = 0
        models[i, :, :] = corrs
        pvalss[i, :, :] = pvals
        N[i] = X.shape[0]

    N = int(np.round(np.mean(N)))
    #models = models[:, idx, :]
    model = models.mean(axis=0)
    t_model = model * np.sqrt(N - 2) / np.sqrt(1 - model **2)
    t_model[t_model < 0] = 0
    p_corrs = stats.t.sf(np.abs(t_model), N - 1) * 2 
    model = (p_corrs < 0.05).astype(int)

    # ALTERNATIVE
    #model = (pvalss[:, :, :].mean(axis=0) > 0.4).astype(int)
    #model = ((pvalss < 0.05).mean(axis=0) > 0.1).astype(int)
    model = pd.DataFrame(model, columns=df.columns[:33], index=labels)
    return model