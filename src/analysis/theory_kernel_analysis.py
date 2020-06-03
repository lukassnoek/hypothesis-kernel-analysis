import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

sys.path.append('src')
from theories import THEORIES
from model import TheoryKernelClassifier

ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(np.arange(6)[:, np.newaxis])

scores_all = []
for tk_name, tk in THEORIES.items():
    model = TheoryKernelClassifier(kernel_dict=tk, binarize_X=False, normalize=True, scale_dist=False, beta_sm=50)

    subs = [str(s).zfill(2) for s in range(1, 61)]
    scores = np.zeros((len(subs), 6))
    for i, sub in enumerate(subs):
        data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
        X, y = data.iloc[:, :-2], data.iloc[:, -2]
        model.fit(X, y)
        y_pred = model.predict_proba(X, y)
        y_true = ohe.transform(y.values[:, np.newaxis])
        scores[i, :] = roc_auc_score(y_true, y_pred, average=None)
        
    #scores = pd.DataFrame(scores, columns=dl.le.classes_, index=subs).reset_index()
    #scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
    #scores = scores.rename({'index': 'sub'}, axis=1)
    #scores['tk'] = tk_name
    #scores_all.append(scores)

#scores = pd.concat(scores_all, axis=0)
#scores.to_csv('results/theorykernel_scores.tsv', sep='\t')
#print(scores.groupby(['tk', 'emotion']).mean())
