import numpy as np
import pandas as pd
from glob import glob
from scipy import stats
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder


def estimate_model(files, prop_threshold=None, trial_split='train'):

    models = np.zeros((len(files), 7, 33))
    emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'other']
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(np.array(emotions)[:, None])

    N = np.zeros(len(files))

    for i, f in enumerate(files):
        df = pd.read_csv(f, sep='\t', index_col=0)
        df = df.query("trial_split == @trial_split")
        X = df.iloc[:, :33].to_numpy()
        Y = ohe.transform(df['emotion'].to_numpy()[:, None])        
        
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
        #models[i, :, :] = (pvals < 0.05).astype(int)
        models[i, :, :] = corrs
        N[i] = X.shape[0]

    N = int(np.round(np.mean(N)))
    model = models.mean(axis=0)
    t_model = model * np.sqrt(N - 2) / np.sqrt(1 - model **2)
    t_model[t_model < 0] = 0
    p_corrs = stats.t.sf(np.abs(t_model), N - 1) * 2 
    model = (p_corrs < 0.05).astype(int)

    #model = models.mean(axis=0)
    model = pd.DataFrame(model, columns=df.columns[:33], index=emotions)
    model = model.loc[emotions[:-1], :]
    if prop_threshold is not None:
        model = (model >= prop_threshold).astype(int)
    
    return model