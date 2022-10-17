import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from models import KernelClassifier
from mappings_conversational import MAPPINGS


def _parallel_analysis(df, au, mappings):

    scores_all = []
    for mapp_name in tqdm(mappings, desc=au):

        model = KernelClassifier(au_cfg=None, param_names=None, kernel='cosine', ktype='similarity',
                                binarize_X=False, normalization='softmax', beta=1)

        # Save Z_orig for later!
        Z_orig = pd.read_csv(f'data/{mapp_name}_conversational.tsv', sep='\t', index_col=0)
        model.add_Z(Z_orig.copy())
        defined_states = sorted(list(set(model.Z_.index.tolist())))
    
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(np.array(defined_states)[:, None])
        
        for state in defined_states:  # ablate au from each emotion in turn!

            sub_ids = df['sub'].unique()
            for sub_id in sub_ids:
                df_l1 = df.query("sub == @sub_id & state in @defined_states")

                scores = np.zeros(len(defined_states))
                scores[:] = np.nan
                X, y = df_l1.iloc[:, :33], df_l1.loc[:, 'state']

                # First compute original (unablated) model performance                    
                y_ohe = ohe.transform(y.to_numpy()[:, None])
                y_pred = model.predict_proba(X)
                idx = y_ohe.sum(axis=0) != 0
                scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)

                # Ablate `au` from `emo`
                model.Z_.loc[state, au] = 0  # here is where the ablation happens!
                y_pred = model.predict_proba(X)
                new = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)
                scores[idx] = (new - scores[idx])  # Store the absolute AUROC difference (orig - ablated)
                model.Z_ = Z_orig.copy()  # restore original model (otherwise ablations 'add up')

                scores = pd.DataFrame(scores, columns=['score'])
                scores['state'] = defined_states
                scores['sub'] = sub_id
                scores['sub_ethnicity'] = df_l1['sub_ethnicity'].unique()[0]
                scores['sub_split'] = 'train'
                scores['trial_split'] = 'train'
                scores['mapping'] =  mapp_name
                scores['ablated_au'] = au
                scores['ablated_from'] = state
                scores_all.append(scores)

    scores = pd.concat(scores_all, axis=0)
    return scores


from joblib import Parallel, delayed

files = sorted(glob('data/ratings/conversational/*/*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
mega_df = mega_df.query("sub_split == 'train' & trial_split == 'train'")
mega_df = mega_df.query("emotion != 'other'")  # remove non-emo trials
mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()

scores = Parallel(n_jobs=33)(delayed(_parallel_analysis)(
    mega_df, au, list(MAPPINGS.keys())) for au in PARAM_NAMES
)

scores = pd.concat(scores, axis=0)
scores.to_csv('results/scores_ablation_conversational.tsv', sep='\t')