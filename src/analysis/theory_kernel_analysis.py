import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from theories import THEORIES
from models import TheoryKernelClassifier

PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(np.arange(6)[:, np.newaxis])

beta = 1
kernel = 'sigmoid'
subs = [str(s).zfill(2) for s in range(1, 61)]
scores_all = []
preds_all = []
for tk_name, tk in tqdm(THEORIES.items()):
    ktype = 'similarity' if kernel in ['cosine', 'sigmoid'] else 'distance'
    model = TheoryKernelClassifier(au_cfg=tk, param_names=PARAM_NAMES, kernel=kernel, ktype=ktype,
                                    binarize_X=False, normalization='softmax', beta=beta)
    #model = GridSearchCV(model, param_grid={'normalization': ['softmax', 'linear']})
    scores = np.zeros((len(subs), 6))
    #scores_per_intensity = np.zeros(())
    preds = []
    for i, sub in enumerate(subs):
        data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
        data = data.query("emotion != 'other'")
        X, y = data.iloc[:, :-2], data.iloc[:, -2]
        y_ohe = pd.get_dummies(y)
        model.fit(X, y)
        #t_df = pd.DataFrame(model.Z_, columns=X.columns, index=np.array(EMOTIONS)[model.cls_idx_])
        #t_df.to_csv(f'data/{tk_name}.tsv', sep='\t')
        y_pred = pd.DataFrame(model.predict_proba(X), index=X.index, columns=EMOTIONS)
        scores[i, :] = roc_auc_score(y_ohe, y_pred, average=None)
        y_pred['sub'] = sub
        y_pred['intensity'] = data['intensity']
        y_pred['y_true'] = data['emotion']
        preds.append(y_pred)

    scores = pd.DataFrame(scores, columns=EMOTIONS, index=subs).reset_index()
    scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
    scores = scores.rename({'index': 'sub'}, axis=1)
    scores['tk'] = tk_name
    scores['kernel'] = kernel
    scores['beta'] = 1
    scores_all.append(scores)

    #preds = pd.melt(pd.concat(preds).reset_index(), id_vars=['index', 'sub', 'intensity'], value_name='pred', var_name='emotion')
    #preds.to_csv('results/theorkernel_predictions.tsv', sep='\t')
    preds = pd.concat(preds)
    preds['tk'] = tk_name
    preds['kernel'] = kernel
    preds['beta'] = beta
    preds_all.append(preds)

scores = pd.concat(scores_all, axis=0)
scores.to_csv('results/auroc.tsv', sep='\t')

preds = pd.concat(preds_all)
preds.to_csv('results/predictions.tsv', sep='\t')

mean_int = preds['intensity'].reset_index().groupby('index').mean()
preds.loc[mean_int.index, 'intensity'] = mean_int['intensity']

percentiles = preds['intensity'].quantile([0, .2, .4, .6, .8, 1.])
scores_int = pd.DataFrame(columns=['sub', 'emotion', 'tk', 'intensity', 'score'])
i = 0
for intensity in [1, 2, 3, 4, 5]:
    minn, maxx = percentiles.iloc[intensity-1], percentiles.iloc[intensity]
    preds_int = preds.query("@minn <= intensity & intensity <= @maxx")
    #print(preds_int)
    for sub in subs:
        for tk, _ in THEORIES.items():
            tmp_preds = preds_int.query("sub == @sub & tk == @tk")
            y_true = pd.get_dummies(tmp_preds['y_true'])
            score = roc_auc_score(y_true, tmp_preds.iloc[:, :6], average=None)
            for ii, s in enumerate(score):
                scores_int.loc[i, 'sub'] = sub
                scores_int.loc[i, 'emotion'] = EMOTIONS[ii]
                scores_int.loc[i, 'tk'] = tk
                scores_int.loc[i, 'intensity'] = intensity
                scores_int.loc[i, 'score'] = s
                i += 1

scores_int['score'] = scores_int['score'].astype(float)
scores_int.to_csv('results/auroc_per_intensity_quantile.tsv', sep='\t')
