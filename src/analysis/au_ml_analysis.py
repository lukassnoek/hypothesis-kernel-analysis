import numpy as np
import pandas as pd
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict

model = LogisticRegression(class_weight='balanced')
files = sorted(glob('data/ratings/sub*.tsv'))
for f in files:
    data = pd.read_csv(f, sep='\t', index_col=0).query("emotion != 'other'")
    X, y = data.iloc[:, :-2], data.iloc[:, -2]
    X = np.random.randn(*X.shape)
    scores = cross_val_score(model, X, y, cv=20)