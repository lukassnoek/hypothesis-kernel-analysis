import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def compute_noise_ceiling(y, scoring=roc_auc_score, soft=True, doubles_only=False, progbar=False, K=None,
                          return_number=False, bootstrap=False):
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
    le.fit(np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']))
    
    # convert y to numeric
    y = pd.Series(le.transform(y), index=y.index).sort_index()
    
    # Get unique indices (stim IDs)
    uniq_idx = y.index.unique()
    
    # If the number of classes is not explicitly given, infer from data
    if K is None:
        K = y.unique().size  # number of classes

    # Get 2D matrix with uniq_idx X classes
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

    # Remove
    if doubles_only:
        counts = counts[is_double, :]
        uniq_idx = uniq_idx[is_double]
        y = y.loc[y.loc[uniq_idx].index]

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

    optimal = pd.DataFrame(optimal, index=uniq_idx)
    #if subsample is not None:
    #    ss_idx = uniq_idx.to_series().sample(n=is_double.sum(), replace=True).index
    #    optimal = optimal.loc[ss_idx, :]
    #    y = y.loc[ss_idx]

    # This will repeat the optimal predictions R times
    optimal = optimal.loc[y.loc[uniq_idx].index, :].sort_index()

    # Needed to convert 1D to 2D
    ohe = OneHotEncoder(categories='auto', sparse=False)
    ohe.fit(np.arange(K)[:, np.newaxis])
    y_ohe = ohe.transform(y.values[:, np.newaxis])

    if bootstrap:
        bts_idx = uniq_idx.to_series().sample(frac=1, replace=True).index
        y_ohe_new, optimal_new = [], []
        for trial in sorted(bts_idx):
            idx = optimal.index == trial
            y_ohe_new.append(y_ohe[idx, :])
            optimal_new.append(optimal.loc[idx, :])

        y_ohe = np.vstack(y_ohe_new)
        optimal = pd.concat(optimal_new, axis=0)

    # Compute best possible score ("ceiling") between actual labels (y_ohe)
    # and optimal labels (optimal)
    ceiling = np.zeros(K)
    ceiling[:] = np.nan
    idx = y_ohe.sum(axis=0) != 0
    
    print(y_ohe[:10, idx])
    print(optimal.values[:10, idx])

    try:
        ceiling[idx] = scoring(y_ohe[:, idx], optimal.values[:, idx], average=None)
    except ValueError as e:
        raise(e)
        ceiling[idx] = np.nan

    if return_number:
        return ceiling, is_double.sum()
    else:
        return ceiling
