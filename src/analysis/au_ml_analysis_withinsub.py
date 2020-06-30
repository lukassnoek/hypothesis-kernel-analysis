import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

model = LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='saga')
files = sorted(glob('data/ratings/sub*.tsv'))[:5]
cv = RepeatedStratifiedKFold(n_repeats=2, n_splits=5)
scores = np.zeros((len(files), 6))
coefs = np.zeros((len(files), 6, 34))
for i, f in enumerate(tqdm(files)):
    data = pd.read_csv(f, sep='\t', index_col=0).query("emotion != 'other'")
    X, y = data.iloc[:, :-2], data.iloc[:, -2]
    for train_idx, test_idx in cv.split(X, y):  
        model.fit(X.iloc[train_idx, :], y[train_idx])
        coefs[i, :, :] = model.coef_.copy()
        preds = model.predict_proba(X.iloc[test_idx, :])
        scores[i, :] += roc_auc_score(pd.get_dummies(y[test_idx]), preds, average=None)

    scores[i, :] /= cv.get_n_splits()
    coefs[i, :, :] /= cv.get_n_splits()

emo = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
scores = pd.DataFrame(
    scores,
    columns=emo,
    index=[f'sub-{str(i+1).zfill(2)}' for i in range(scores.shape[0])]
)
scores = pd.melt(scores.reset_index(), id_vars='index', value_name='score', var_name='emotion')
scores.to_csv('results/method-ml_analysis-within_auroc.tsv', sep='\t')

plt.imshow(coefs.mean(axis=0).T, aspect='auto')
plt.colorbar()
ax = plt.gca()
ax.set_xticks(range(6))
ax.set_xticklabels(emo)
ax.set_yticks(range(34))
ax.set_yticklabels(X.columns.tolist(), fontdict=dict(fontsize=8))
plt.savefig('results/au_ml_coefs.png', dpi=200)
