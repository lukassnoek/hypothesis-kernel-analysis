import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from models import KernelClassifier

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])

# One-hot encode target label
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(EMOTIONS[:, None])

# Define analysis parameters
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

# Loop across mappings (Darwin, Ekman, etc.)
for kernel in tqdm(['cosine', 'sigmoid', 'linear', 'euclidean', 'l1', 'l2']):
    for beta in [1, 10, 100, 1000, 10000]:
        for mapp_name in mappings:
            # ktype = kernel type (infer from kernel name)
            ktype = 'similarity' if kernel in ['cosine', 'sigmoid', 'linear'] else 'distance'

            # Initialize model!
            model = KernelClassifier(au_cfg=None, param_names=PARAM_NAMES, kernel=kernel, ktype=ktype,
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
                y_pred = model.predict_proba(X)
                y_ohe = ohe.transform(y.to_numpy()[:, np.newaxis])
                idx = y_ohe.sum(axis=0) != 0
                scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)

                # Store scores and raw predictions
                scores = pd.DataFrame(scores, columns=['score'])
                scores['emotion'] = EMOTIONS
                scores['sub'] = sub_id
                scores['kernel'] = kernel
                scores['beta'] = beta
                ethn = df_l1['sub_ethnicity'].unique()[0]
                scores['mapping'] =  mapp_name
                scores_all.append(scores)

# Save scores and predictions
scores = pd.concat(scores_all, axis=0)
print(scores.groupby(['kernel', 'beta']).mean())
scores.to_csv('results/scores_hyperparameters.tsv', sep='\t')