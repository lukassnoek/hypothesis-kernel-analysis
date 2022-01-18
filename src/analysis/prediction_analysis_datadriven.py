import sys
import random
import numpy as np
import os.path as op
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

sys.path.append('.')
from src.datadriven import estimate_model
from src.models import KernelClassifier

# Initialize model!
model = KernelClassifier(au_cfg=None, param_names=None, kernel='cosine', ktype='similarity',
                         binarize_X=False, normalization='softmax', beta=1)
emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
ohe = OneHotEncoder(sparse=False)
ohe.fit(np.array(emotions)[:, None])

scores_all = []

for ethn in ['*', 'WC', 'EA']:
    files = sorted(glob(f'data/ratings/{ethn}/*.tsv'))
    files = [f for i, f in enumerate(files) if i % 3 != 0]  # only fit model on train subs!
    
    for test_file in tqdm(files, desc=f'ethn = {ethn}'):
        train_files = [f for f in files if f != test_file]
        if ethn == '*':
            train_files = random.sample(train_files, 29)

        Z = estimate_model(train_files, prop_threshold=None)
        model.add_Z(Z)
        #lmodel = LogisticRegression()
        #X = np.vstack([pd.read_csv(f, sep='\t', index_col=0).query("trial_split == 'train' & emotion != 'other'").iloc[:, :33].to_numpy()
        #               for f in train_files])
        #Y = np.concatenate([ohe.transform(pd.read_csv(f, sep='\t', index_col=0).query("trial_split == 'train' & emotion != 'other'").loc[:, 'emotion'].to_numpy()[:, None])
        #               for f in train_files])
        #Y = Y.argmax(axis=1)
        #lmodel.fit(X, Y)

        df = pd.read_csv(test_file, sep='\t', index_col=0)
        df = df.query("emotion != 'other' & trial_split == 'test'")  # remove non-emo trials
        df = df.loc[df.index != 'empty', :]  # remove trials w/o AUs

        for face_gender in [0, 1, 'all']:
            if face_gender != 'all':
                this_data = df.query("face_gender == @face_gender")
            else:
                this_data = df

            X, y = this_data.iloc[:, :33], this_data.loc[:, 'emotion']
            X = this_data.iloc[:, :33].to_numpy()
            y_ohe = ohe.transform(this_data['emotion'].to_numpy()[:, None])

            y_pred = model.predict_proba(X)
            #y_pred = lmodel.predict_proba(X)
            idx = y_ohe.sum(axis=0) != 0
            
            scores = np.zeros(6)
            scores[:] = np.nan
            scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)
            scores = pd.DataFrame(scores, columns=['score'])
            scores['emotion'] = emotions
            scores['sub'] = op.basename(test_file).split('_')[0].split('-')[1]
            sub_ethn = this_data['sub_ethnicity'].unique()[0]
            scores['sub_ethnicity'] = {0: 'WC', 1: 'EA'}[sub_ethn]
            scores['model_ethnicity'] = 'all' if ethn == '*' else ethn
            scores['sub_split'] = this_data['sub_split'].unique()[0]
            scores['face_gender'] = {0: 'F', 1: 'M', 'all': 'all'}[face_gender]
            scores['trial_split'] = 'train'
            scores['mapping'] = 'JackSchyns'
            scores_all.append(scores)

scores = pd.concat(scores_all, axis=0)
print(scores.groupby(['model_ethnicity', 'sub_ethnicity', 'emotion']).mean())
scores.to_csv('results/scores_js.tsv', sep='\t')