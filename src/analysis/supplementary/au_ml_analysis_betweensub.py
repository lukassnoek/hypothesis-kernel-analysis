import random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])
ohe = OneHotEncoder(sparse=False)
ohe.fit(EMOTIONS[:, np.newaxis])

model = LogisticRegression(class_weight='balanced')

files = sorted(glob('data/ratings/*/*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
mega_df = mega_df.query("emotion != 'other'")  # remove non-emo trials
mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs

train_df = mega_df.query("sub_split == 'train' & trial_split == 'train'")
test_df = mega_df.query("sub_split == 'train' & trial_split == 'test'")

scores_all = []
for ethn in ['all', 'WC', 'EA']:
    if ethn != 'all':
        train_df_l1 = train_df.query("sub_ethnicity == @ethn")
        test_df_l1 = test_df.query("sub_ethnicity == @ethn")
    else:
        train_df_l1 = train_df
        test_df_l1 = test_df

    for i, sub in enumerate(tqdm(test_df_l1['sub'].unique())):
        test_df_ = test_df_l1.query("sub == @sub")
        train_df_ = train_df_l1.query("sub != @sub")

        if ethn == 'all':
            n_ea = 19 if i % 2 == 0 else 20
            n_wc = 20 if i % 2 == 0 else 19
            subset_ea = random.sample(train_df_.query("sub_ethnicity == 'EA'")['sub'].unique().tolist(), n_ea)
            subset_wc = random.sample(train_df_.query("sub_ethnicity == 'WC'")['sub'].unique().tolist(), n_wc) 
            subset = subset_ea + subset_wc
            train_df_ = train_df_.query("sub in @subset")

        X_train, y_train = train_df_.iloc[:, :33], train_df_.loc[:, 'emotion']
        model.fit(X_train, y_train)

        X_test = test_df_.iloc[:, :33]
        y_test = test_df_.loc[:, 'emotion']
        y_pred = model.predict_proba(X_test)
        y_test = ohe.transform(y_test.to_numpy()[:, np.newaxis])
        idx = y_test.sum(axis=0) != 0

        scores = np.zeros(6)
        scores[:] = np.nan
        scores[idx] = roc_auc_score(y_test[:, idx], y_pred[:, idx], average=None)
        scores = pd.DataFrame(scores, columns=['score'])
        scores['emotion'] = EMOTIONS
        scores['sub'] = sub
        sub_ethn = test_df_['sub_ethnicity'].iloc[0]
        scores['sub_ethnicity'] = sub_ethn
        scores['model_ethnicity'] = ethn
        scores['sub_split'] = 'train'
        scores['trial_split'] = 'train'
        scores['mapping'] = 'ML'
        scores_all.append(scores)

scores = pd.concat(scores_all, axis=0)
scores.to_csv('results/scores_ml.tsv', sep='\t')
print(scores.groupby(['model_ethnicity', 'sub_ethnicity']).mean())