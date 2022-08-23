import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

sys.path.append('src')
from mappings import MAPPINGS
from models import KernelClassifier

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])
ohe = OneHotEncoder(sparse=False)
ohe.fit(EMOTIONS[:, np.newaxis])

le = LabelEncoder()
le.fit(EMOTIONS)

# Define analysis parameters
beta = 1
kernel = 'cosine'
ktype = 'similarity'

scores_all = []

# Loop across mappings (Darwin, Ekman, etc.)
mappings = ['Cordaro2018IPC', 'Cordaro2018ref', 'Darwin', 'Ekman', 'Keltner2019', 'Matsumoto2008',
            'JackSchyns_ethn-all_CV'  # also use previously fitted data-driven model
]

files = sorted(glob('data/ratings/*/*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
mega_df = mega_df.query("sub_split == 'train' & trial_split == 'train'")
mega_df = mega_df.query("emotion != 'other'")  # remove non-emo trials
mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs

scores_all = []
for method in ['predict', 'predict_proba']:
    # Loop across mappings (Darwin, Ekman, etc.)

    for mapp_name in mappings:

        # Initialize model!
        model = KernelClassifier(au_cfg=None, param_names=None, kernel=kernel, ktype=ktype,
                                binarize_X=False, normalization='softmax', beta=beta)

        # Note that there is no "fitting" of the model! The mappings themselves
        # can be interpreted as already-fitted models
        model.add_Z(pd.read_csv(f'data/{mapp_name}.tsv', sep='\t', index_col=0))
        
        # Compute model performance per subject!
        for sub_id in tqdm(mega_df['sub'].unique(), desc=mapp_name):
            df_l1 = mega_df.query("sub == @sub_id")

            # Initialize with NaNs in case of no trials for a
            # given emotion category
            scores = np.zeros(len(EMOTIONS))
            scores[:] = np.nan
        
            X, y = df_l1.iloc[:, :33], df_l1.loc[:, 'emotion']
            y_ohe = ohe.transform(y.to_numpy()[:, None])
            y_pred = getattr(model, method)(X)
            y_ohe = ohe.transform(y.to_numpy()[:, np.newaxis])
            idx = y_ohe.sum(axis=0) != 0
            if method == 'predict':
                y_pred = le.inverse_transform(y_pred)
                y_pred = ohe.transform(y_pred[:, None])
                scores[idx] = roc_auc_score(y_ohe, y_pred, average=None)
            else:
                scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)

            # Store scores and raw predictions
            scores = pd.DataFrame(scores, columns=['score'])
            scores['emotion'] = EMOTIONS
            scores['sub'] = sub_id
            ethn = df_l1['sub_ethnicity'].unique()[0]
            scores['sub_ethnicity'] = ethn
            scores['mapping'] =  mapp_name
            scores['method'] = method
            scores_all.append(scores)

# Save scores and predictions
scores = pd.concat(scores_all, axis=0)
print(scores.groupby(['method', 'mapping']).mean())
scores.to_csv('results/prob_vs_discrete_scores.tsv', sep='\t')