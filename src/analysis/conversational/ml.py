import numpy as np
import pandas as pd
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score


TARGET = 'emotion'#'conversational'
if TARGET == 'emotion':
    labels = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    files = sorted(glob(f'data/ratings/{TARGET}/*/*.tsv'))
else:
    labels = ['thinking', 'interested', 'bored', 'confused']
    files = sorted(glob(f'data/ratings/conversational/*/*.tsv'))

ohe = OneHotEncoder(sparse=False)
lenc = LabelEncoder()
lenc.fit(labels)
ohe.fit(np.array(lenc.classes_)[:, None])

mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
mega_df = mega_df.query(f"{TARGET} != 'other'")  # remove non-conv trials
mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs

scores_all = []

### Universal model
model = LogisticRegression()
scaler = StandardScaler()

train_df = mega_df.query("sub_split == 'train' & trial_split == 'train'")
X_train, y_train = train_df.iloc[:, :33].to_numpy(), train_df.loc[:, TARGET].to_numpy()
X_train = scaler.fit_transform(X_train)
y_train = lenc.transform(y_train)
model.fit(X_train, y_train)

test_df = mega_df.query("sub_split == 'train' & trial_split == 'test'")
for sub_id in test_df['sub'].unique():
    test_sub = test_df.query("sub == @sub_id")
    X_test, y_test = test_sub.iloc[:, :33].to_numpy(), test_sub.loc[:, TARGET].to_numpy()
    X_test = scaler.transform(X_test)
    y_pred = model.predict_proba(X_test)
    y_ohe = ohe.transform(y_test[:, None])
    idx = y_ohe.sum(axis=0) != 0
    
    scores = np.zeros(len(labels))
    scores[:] = np.nan
    scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)

    scores = pd.DataFrame(scores, columns=['score'])
    scores[TARGET] = labels
    scores['sub'] = sub_id
    scores['sub_ethnicity'] = test_sub['sub_ethnicity'].iloc[0]
    scores['model'] = 'universal'
    scores_all.append(scores)

### Culture-aware model
for ethn in ['WC', 'EA']:

    train_df = mega_df.query("sub_split == 'train' & trial_split == 'train' & sub_ethnicity == @ethn")
    X_train, y_train = train_df.iloc[:, :33].to_numpy(), train_df.loc[:, TARGET].to_numpy()
    X_train = scaler.fit_transform(X_train)
    y_train = lenc.transform(y_train)
    model.fit(X_train, y_train)

    test_df = mega_df.query("sub_split == 'train' & trial_split == 'test' & sub_ethnicity == @ethn")
    for sub_id in test_df['sub'].unique():
        test_sub = test_df.query("sub == @sub_id")
        X_test, y_test = test_sub.iloc[:, :33].to_numpy(), test_sub.loc[:, TARGET].to_numpy()
        X_test = scaler.transform(X_test)
        y_pred = model.predict_proba(X_test)
        y_ohe = ohe.transform(y_test[:, None])
        idx = y_ohe.sum(axis=0) != 0
        
        scores = np.zeros(len(labels))
        scores[:] = np.nan
        scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)

        scores = pd.DataFrame(scores, columns=['score'])
        scores[TARGET] = labels
        scores['sub'] = sub_id
        scores['sub_ethnicity'] = test_sub['sub_ethnicity'].iloc[0]
        scores['model'] = 'culture-aware'
        scores_all.append(scores)

### Individual model
for sub_id in mega_df['sub'].unique():

    train_df = mega_df.query("sub == @sub_id & trial_split == 'train'")
    X_train, y_train = train_df.iloc[:, :33].to_numpy(), train_df.loc[:, TARGET].to_numpy()
    X_train = scaler.fit_transform(X_train)
    y_train = lenc.transform(y_train)
    model.fit(X_train, y_train)

    test_sub = mega_df.query("sub == @sub_id & trial_split == 'test'")
    X_test, y_test = test_sub.iloc[:, :33].to_numpy(), test_sub.loc[:, TARGET].to_numpy()
    X_test = scaler.transform(X_test)
    y_pred = model.predict_proba(X_test)
    y_ohe = ohe.transform(y_test[:, None])
    idx = y_ohe.sum(axis=0) != 0
    
    scores = np.zeros(len(labels))
    scores[:] = np.nan
    scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)

    scores = pd.DataFrame(scores, columns=['score'])
    scores[TARGET] = labels
    scores['sub'] = sub_id
    scores['sub_ethnicity'] = test_sub['sub_ethnicity'].iloc[0]
    scores['model'] = 'individual'
    scores_all.append(scores)

scores = pd.concat(scores_all, axis=0)
print(scores.groupby([TARGET, 'model']).mean())
