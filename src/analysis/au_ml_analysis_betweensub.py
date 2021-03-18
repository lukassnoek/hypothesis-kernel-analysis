import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

#model = LogisticRegression(class_weight='balanced')
model = RandomForestClassifier()
files = sorted(glob('data/ratings/sub*.tsv'))
cv = RepeatedStratifiedKFold(n_repeats=2, n_splits=5)

data_train = pd.concat(
    [pd.read_csv(f, sep='\t', index_col=0).query("emotion != 'other'")
     for f in files[::2]]
)

X_train, y_train = data_train.iloc[:, :-2], data_train.iloc[:, -2]
model.fit(X_train, y_train)

scores = np.zeros((len(files[1::2]), 6))
for i, f in enumerate(files[1::2]):
    data_test = pd.read_csv(f, sep='\t', index_col=0).query("emotion != 'other'")
    X_test, y_test = data_test.iloc[:, :-2], data_test.iloc[:, -2]
    y_pred = model.predict_proba(X_test)
    scores[i, :] = roc_auc_score(pd.get_dummies(y_test), y_pred, average=None)

scores = pd.DataFrame(
    scores,
    columns=['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'],
    index=[f'sub-{str(i+1).zfill(2)}' for i in np.arange(1, len(files), 2)]
)
scores = pd.melt(scores.reset_index(), id_vars='index', value_name='score', var_name='emotion')
scores.to_csv('results/method-ml_analysis-between_auroc.tsv', sep='\t')