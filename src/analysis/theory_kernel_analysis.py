import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from theories import THEORIES
from model import TheoryKernelClassifier

PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(np.arange(6)[:, np.newaxis])

subs = [str(s).zfill(2) for s in range(1, 61)]
scores_all = []
for tk_name, tk in THEORIES.items():

    for kernel in ['linear', 'sigmoid', 'cosine']:
        for beta in [1, 10, 100, 1000]:
            model = TheoryKernelClassifier(au_cfg=tk, param_names=PARAM_NAMES, kernel=kernel, binarize_X=False, beta=beta)
            #model = GridSearchCV(model, param_grid={'beta': np.logspace(-1, 4, num=10)})
            scores = np.zeros((len(subs), 6))
            for i, sub in enumerate(subs):
                data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
                data = data.query("emotion != 'other'")
                X, y = data.iloc[:, :-2], data.iloc[:, -2]
                y_ohe = pd.get_dummies(y)
                model.fit(X, y)
                y_pred = model.predict_proba(X)
                scores[i, :] = roc_auc_score(y_ohe, y_pred, average=None)
            
            scores = pd.DataFrame(scores, columns=EMOTIONS, index=subs).reset_index()
            scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
            scores = scores.rename({'index': 'sub'}, axis=1)
            scores['tk'] = tk_name
            scores['kernel'] = kernel
            scores['beta'] = beta
            scores_all.append(scores)

scores = pd.concat(scores_all, axis=0)
scores.to_csv('results/theorykernel_scores.tsv', sep='\t')
print(scores.groupby(['tk', 'emotion', 'kernel', 'beta']).mean())
