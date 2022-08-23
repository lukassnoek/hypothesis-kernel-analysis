# Fully cross-validated, data-driven model analysis (Jack et al., 2012, model)

import sys
import random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
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

files = sorted(glob('data/ratings/*/*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
mega_df = mega_df.query("sub_split == 'train'")
mega_df = mega_df.query("emotion != 'other'")  # remove non-emo trials
mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs

scores_all = []
models_all = []
for model_ethn in ['all', 'WC', 'EA']:
    if model_ethn != 'all':
        df_l1 = mega_df.query("sub_ethnicity == @model_ethn")
    else:
        df_l1 = mega_df

    sub_ids = df_l1['sub'].unique().tolist()
    for i, test_id in enumerate(tqdm(sub_ids, desc=f'ethn = {model_ethn}')):
        df_test = df_l1.query("sub == @test_id & trial_split == 'test'")
        df_train = df_l1.query("sub != @test_id & trial_split == 'train'")

        if model_ethn == 'all':
            n_ea = 19 if i % 2 == 0 else 20
            n_wc = 20 if i % 2 == 0 else 19
            subset_ea = random.sample(df_train.query("sub_ethnicity == 'EA'")['sub'].unique().tolist(), n_ea)
            subset_wc = random.sample(df_train.query("sub_ethnicity == 'WC'")['sub'].unique().tolist(), n_wc) 
            subset = subset_ea + subset_wc
            df_train = df_train.query("sub in @subset")

        Z = estimate_model(df_train)
        model.add_Z(Z.copy())
        Z['fold'] = i
        Z['model_ethnicity'] = model_ethn 
        models_all.append(Z)

        X, y = df_test.iloc[:, :33].to_numpy(), df_test.loc[:, 'emotion'].to_numpy()
        y_ohe = ohe.transform(y[:, None])

        y_pred = model.predict_proba(X)
        idx = y_ohe.sum(axis=0) != 0
        
        scores = np.zeros(6)
        scores[:] = np.nan
        scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)
        scores = pd.DataFrame(scores, columns=['score'])
        scores['emotion'] = emotions
        scores['sub'] = test_id
        scores['sub_ethnicity'] = df_test['sub_ethnicity'].iloc[0]
        scores['model_ethnicity'] = model_ethn
        scores['mapping'] = 'JackSchyns'
        scores_all.append(scores)

scores = pd.concat(scores_all, axis=0)
print(scores.groupby(['model_ethnicity', 'sub_ethnicity', 'emotion']).mean())
scores.to_csv('results/scores_js.tsv', sep='\t')

model = pd.concat(models_all, axis=0)
for ethn in ['all', 'WC', 'EA']:
    model_ = model.query("model_ethnicity == @ethn").drop('model_ethnicity', axis=1)
    model_ = model_.reset_index().groupby('index').mean().drop('fold', axis=1)
    model_.index.name = ''
    model_.to_csv(f'data/JackSchyns_ethn-{ethn}_CV.tsv', sep='\t')