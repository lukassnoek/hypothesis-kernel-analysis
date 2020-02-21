import numpy as np
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
    for i, idx in to_iter:
        if isinstance(y.loc[idx], pd.core.series.Series):
            # If there are reps, count the labels across reps
            counts_df = y.loc[idx].value_counts()
            counts[i, counts_df.index] = counts_df.values
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
        doubles_idx = counts.sum(axis=1) != 1
        optimal = optimal.loc[uniq_idx[doubles_idx], :]
    
    optimal = optimal.sort_index()
    
    # Needed to convert 1D to 2D
    y_flat = y.loc[optimal.index.unique()].sort_index()
    ohe = OneHotEncoder(categories='auto', sparse=False)
    ohe.fit(np.arange(K)[:, np.newaxis])
    y_ohe = ohe.transform(y_flat[:, np.newaxis])
    
    # Compute best possible score ("ceiling") between actual labels (y_ohe)
    # and optimal labels (optimal)
    try:
        ceiling = scoring(y_ohe, optimal.values, average=None)
    except ValueError:
        ceiling = np.tile(np.nan, 6)

    return ceiling


if __name__ == '__main__':
    
    import pandas as pd
    from glob import glob
    from sklearn.metrics import f1_score

    # Load data from all subs
    ratings = []
    ncs = np.zeros((60, 6))
    for i, f in enumerate(tqdm(sorted(glob('data/ratings/sub*.tsv')))):
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
    idx = ['sub-{str(i+1).zfill(2)' for i in range(60)]
    ncs = pd.DataFrame(ncs, columns=emo_labels, index=idx)
    print(ncs.mean(axis=0))

    ratings = pd.concat(ratings, axis=0)
    nc = compute_noise_ceiling(
        ratings['emotion'],
        scoring=roc_auc_score,
        soft=True,
        progbar=True,
        doubles_only=True
    )
    nc = pd.DataFrame(nc, columns=[''], index=emo_labels)
    print(nc)
