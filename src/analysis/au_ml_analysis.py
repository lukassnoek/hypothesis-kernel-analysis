import numpy as np
import pandas as pd
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

model = LogisticRegression(class_weight='balanced')
files = sorted(glob('data/ratings/sub*.tsv'))
cv = RepeatedStratifiedKFold(n_repeats=2, n_splits=5)
for f in files:
    data = pd.read_csv(f, sep='\t', index_col=0).query("emotion != 'other'")
    X, y = data.iloc[:, :-2], data.iloc[:, -2]
    scores = np.zeros(6)
    for train_idx, test_idx in cv.split(X, y):  
        model.fit(X.iloc[train_idx, :], y[train_idx])
        preds = model.predict_proba(X.iloc[test_idx, :])
        scores += roc_auc_score(pd.get_dummies(y[test_idx]), preds, average=None)

    scores /= cv.get_n_splits()
    print(scores)