import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def compute_noise_ceiling(y, scoring=roc_auc_score, soft=True, doubles_only=False, progbar=False):
    """ Computes the noise ceiling for data with repetitions.

    Parameters
    ----------
    y : pd.Series
        Series with numeric values and index corresponding to stim ID.
    scoring : func
        Scikit-learn scoring function to evaluate noise ceiling with
    soft : bool
        Whether to produce "soft" optimal labels (a probability) or hard (0/1)
    doubles_only : bool
        Whether to estimate the noise ceiling on the "doubles" only. Use this
        when only the doubles are used as a holdout test set.

    Returns
    -------
    ceiling : ndarray
        Numpy ndarray (shape: K,) with ceiling estimates
    
    """
    
    # Strings to nums
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=y.index)
    
    # Get unique indices (stim IDs)
    uniq_idx = y.index.unique()
    
    K = y.unique().size  # number of classes (for emotions: 6)
    counts = np.zeros((uniq_idx.size, K))
    to_iter = enumerate(tqdm(uniq_idx)) if progbar else enumerate(uniq_idx)
    is_double = np.zeros(len(uniq_idx), dtype=bool)
    for i, idx in to_iter:
        if isinstance(y.loc[idx], pd.core.series.Series):
            # If there are reps, count the labels across reps
            counts_df = y.loc[idx].value_counts()
            counts[i, counts_df.index] = counts_df.values
            is_double[i] = True
        else:
            # If no reps, just set label count to only label
            counts[i, y.loc[idx]] = 1
    
    if soft:
        optimal = counts / counts.sum(axis=1, keepdims=True)
    else:    
        # Pre-allocate best prediction array
        optimal = np.zeros_like(counts, dtype=float)

        for ii in range(counts.shape[0]):
            # Determine most frequent label across reps
            opt_class = np.where(counts[ii, :] == counts[ii, :].max())[0]
            rnd_class = np.random.choice(opt_class, size=1)  
            optimal[ii, rnd_class] = 1

    # Repeat best possible prediction R times
    optimal = pd.DataFrame(optimal, index=uniq_idx)
    optimal = optimal.loc[y.index.intersection(optimal.index), :]  # tile
    if doubles_only:
        optimal = optimal.loc[uniq_idx[is_double], :]
    
    optimal = optimal.sort_index()

    # Needed to convert 1D to 2D
    y_flat = y.loc[optimal.index.unique()].sort_index()
    ohe = OneHotEncoder(categories='auto', sparse=False)
    ohe.fit(np.arange(K)[:, np.newaxis])
    y_ohe = ohe.transform(y_flat[:, np.newaxis])
    
    # Compute best possible score ("ceiling") between actual labels (y_ohe)
    # and optimal labels (optimal)
    ceiling = np.zeros(6)
    ceiling[:] = np.nan
    idx = y_ohe.sum(axis=0) != 0
    ceiling[idx] = scoring(y_ohe[:, idx], optimal.values[:, idx], average=None)

    return ceiling
