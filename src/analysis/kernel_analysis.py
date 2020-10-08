import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from mappings import MAPPINGS
from models import KernelClassifier

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# One-hot encode target label
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(np.arange(6)[:, np.newaxis])

# Define analysis parameters
beta = 100
kernel = 'sigmoid'
subs = [str(s).zfill(2) for s in range(1, 61)]
scores_all, preds_all = [], []

# Loop across mappings
for mapp_name, mapp in tqdm(MAPPINGS.items()):
    # ktype = kernel type
    ktype = 'similarity' if kernel in ['cosine', 'sigmoid', 'linear'] else 'distance'
    model = KernelClassifier(au_cfg=mapp, param_names=PARAM_NAMES, kernel=kernel, ktype=ktype,
                             binarize_X=False, normalization='softmax', beta=beta)
    
    #model = GridSearchCV(model, param_grid={'normalization': ['softmax', 'linear']})
    scores = np.zeros((len(subs), len(EMOTIONS)))
    preds = []
    for i, sub in enumerate(subs):
        data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
        data = data.query("emotion != 'other'")
        X, y = data.iloc[:, :-2], data.iloc[:, -2]
        model.fit(X, y)

        # Predict data + compute performance (AUROC)
        y_pred = pd.DataFrame(model.predict_proba(X), index=X.index, columns=EMOTIONS)
        scores[i, :] = roc_auc_score(pd.get_dummies(y), y_pred, average=None)
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(pd.get_dummies(y).values.argmax(axis=1), y_pred.values.argmax(axis=1)))
        # Save results
        y_pred['sub'] = sub
        y_pred['intensity'] = data['intensity']
        y_pred['y_true'] = data['emotion']
        #print(y_pred)
        #exit()
        preds.append(y_pred)

    # Store scores and raw predictions
    scores = pd.DataFrame(scores, columns=EMOTIONS, index=subs).reset_index()
    scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
    scores = scores.rename({'index': 'sub'}, axis=1)
    scores['mapping'] = mapp_name
    scores['kernel'] = kernel
    scores['beta'] = beta
    scores_all.append(scores)

    preds = pd.concat(preds, axis=0)
    preds['mapping'] = mapp_name
    #preds = pd.melt(pd.concat(preds).reset_index(), id_vars=['index', 'sub', 'intensity', 'y_true'], value_name='pred', var_name='emotion')
    #preds['kernel'] = kernel
    #preds['beta'] = beta
    #print(preds.query("sub == '01'").sort_values('index'))
    
    preds_all.append(preds)

# Save scores and predictions
scores = pd.concat(scores_all, axis=0)
scores.to_csv('results/scores.tsv', sep='\t')
#print(scores.groupby(['emotion', 'mapping']).mean())

preds = pd.concat(preds_all)
preds.to_csv('results/predictions.tsv', sep='\t')

# Compute average intensity across repeated observations
mean_int = preds['intensity'].reset_index().groupby('index').mean()
preds.loc[mean_int.index, 'intensity'] = mean_int['intensity']

# Compute quantiles based on this mean intensity
percentiles = preds['intensity'].quantile([0, .2, .4, .6, .8, 1.])
scores_int = pd.DataFrame(columns=['sub', 'emotion', 'mapping', 'intensity', 'score'])
i = 0

# Loop over trials based on intensity levels
for intensity in tqdm([1, 2, 3, 4, 5]):
    # Get current set of trials based on `intensity`
    minn, maxx = percentiles.iloc[intensity-1], percentiles.iloc[intensity]
    preds_int = preds.query("@minn <= intensity & intensity <= @maxx")
    
    # Loop across subjects
    for sub in subs:
        for mapp_name, _ in MAPPINGS.items():
            tmp_preds = preds_int.query("sub == @sub & mapping == @mapp_name")
            y_true = pd.get_dummies(tmp_preds['y_true'])
            score = roc_auc_score(y_true, tmp_preds.iloc[:, :6], average=None)
            for ii, s in enumerate(score):
                scores_int.loc[i, 'sub'] = sub
                scores_int.loc[i, 'emotion'] = EMOTIONS[ii]
                scores_int.loc[i, 'mapping'] = mapp_name
                scores_int.loc[i, 'intensity'] = intensity
                scores_int.loc[i, 'score'] = s
                i += 1

scores_int['score'] = scores_int['score'].astype(float)
scores_int = scores_int.sort_values(['mapping', 'sub', 'emotion', 'intensity'])
scores_int.to_csv('results/score_per_intensity_quantile.tsv', sep='\t')