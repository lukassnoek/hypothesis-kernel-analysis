import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
import pandas as pd
import numpy as np
import os.path as op
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from models import KernelClassifier


def _parallel_analysis(au, mappings, EMOTIONS):

    scores_all = []
    for mapp_name in tqdm(mappings, desc=au):

        model = KernelClassifier(au_cfg=None, param_names=None, kernel='cosine', ktype='similarity',
                                binarize_X=False, normalization='softmax', beta=1)

        # Save Z_orig for later!
        Z_orig = pd.read_csv(f'data/{mapp_name}.tsv', sep='\t', index_col=0)
        model.add_Z(Z_orig.copy())

        for emo in EMOTIONS:  # ablate au from each emotion in turn!
            sub_files = sorted(glob(f'data/ratings/*/*.tsv'))
            for f in sub_files:
                df = pd.read_csv(f, sep='\t', index_col=0)
                df = df.query("emotion != 'other'")  # remove non-emo trials
                df = df.loc[df.index != 'empty', :]  # remove trials w/o AUs

                # Again, we're applying the ablation analysis on both train and test
                # trials (for supp materials), but we're only using the test results
                # to construct 'optimal' models
                for trial_split in ['train', 'test', 'all']:
                    if trial_split != 'all':
                        data = df.query("trial_split == @trial_split")
                    else:
                        data = df

                    scores = np.zeros(len(EMOTIONS))
                    scores[:] = np.nan
                    X, y = data.iloc[:, :33], data.loc[:, 'emotion']

                    # First compute original (unablated) model performance                    
                    y_ohe = ohe.transform(y.to_numpy()[:, None])
                    y_pred = model.predict_proba(X)
                    idx = y_ohe.sum(axis=0) != 0
                    scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)

                    # Ablate `au` from `emo`
                    model.Z_.loc[emo, au] = 0  # here is where the ablation happens!
                    y_pred = model.predict_proba(X)
                    new = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)
                    scores[idx] = (new - scores[idx])  # Store the absolute AUROC difference (orig - ablated)
                    model.Z_ = Z_orig.copy()  # restore original model (otherwise ablations 'add up')

                    scores = pd.DataFrame(scores, columns=['score'])
                    scores['emotion'] = EMOTIONS
                    scores['sub'] = op.basename(f).split('_')[0].split('-')[1]
                    scores['sub_ethnicity'] = df['sub_ethnicity'].unique()[0]
                    scores['sub_split'] = df['sub_split'].unique()[0]
                    scores['trial_split'] = trial_split
                    scores['mapping'] =  mapp_name
                    scores['ablated_au'] = au
                    scores['ablated_from'] = emo
                    scores_all.append(scores)
    
    scores = pd.concat(scores_all, axis=0)
    return scores


from joblib import Parallel, delayed

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])
ohe = OneHotEncoder(sparse=False)
ohe.fit(EMOTIONS[:, np.newaxis])


mappings = ['Cordaro2018IPC', 'Cordaro2018ref', 'Darwin', 'Ekman', 'Keltner2019', 'Matsumoto2008',
            'JackSchyns_ethn-WC_sub-train_trial-train',
            'JackSchyns_ethn-EA_sub-train_trial-train',
            'JackSchyns_ethn-all_sub-train_trial-train',
            ]

scores = Parallel(n_jobs=33)(delayed(_parallel_analysis)(
    au, mappings, EMOTIONS)
    for au in PARAM_NAMES
)

scores = pd.concat(scores, axis=0)
scores.to_csv('results/scores_ablation.tsv', sep='\t')