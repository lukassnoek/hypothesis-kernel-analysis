import os
import os.path as op
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from models import KernelClassifier
from mappings_conversational import MAPPINGS

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()

df_abl = pd.read_csv('results/scores_ablation_conversational.tsv', sep='\t', index_col=0)

files = sorted(glob('data/ratings/conversational/*/*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
mega_df = mega_df.query("sub_split == 'test' & trial_split == 'test'")
mega_df = mega_df.query("emotion != 'other'")  # remove non-emo trials
mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs

scores_all, cm_all, opt_models = [], [], []    
for mapp_name in tqdm(list(MAPPINGS.keys())):

    model = KernelClassifier(au_cfg=None, param_names=None, kernel='cosine', ktype='similarity',
                            binarize_X=False, normalization='softmax', beta=1)

    # Save Z_orig for later!
    Z_orig = pd.read_csv(f'data/{mapp_name}_conversational.tsv', sep='\t', index_col=0)
    defined_states = sorted(list(set(Z_orig.index.tolist())))
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(np.array(defined_states)[:, None])

    for model_ethn in ['all', 'WC', 'EA']:

        if model_ethn != 'all':
            df_abl_l1 = df_abl.query(f"sub_ethnicity == '{model_ethn}'")
        else:
            df_abl_l1 = df_abl
        
        df_abl_l1 = df_abl_l1.query("state in @defined_states")
        Z_opt = Z_orig.copy()

        for state in defined_states:    
            df_abl_l2 = df_abl_l1.query("ablated_from == @state & state == @state & score != 0")
            aus = df_abl_l2.groupby('ablated_au').mean().query("score < 0").index.tolist()
            
            for au in aus:
                Z_opt.loc[state, au] = 1

            # Which AUs decrease prediction? 
            aus = df_abl_l2.groupby('ablated_au').mean().query("score > 0").index.tolist()
            for au in aus:
                Z_opt.loc[state, au] = 0

        Z_opt2save = Z_opt.copy()
        Z_opt2save['mapping'] = mapp_name
        Z_opt2save['model_ethnicity'] = model_ethn
        opt_models.append(Z_opt2save)

        for sub_ethn in ['all', 'WC', 'EA']:
            if sub_ethn != 'all':
                sub_ids = mega_df.query("sub_ethnicity == @sub_ethn")['sub']
            else:
                sub_ids = mega_df['sub']

            sub_ids = sub_ids.unique().tolist()
            cm_old, cm_new = np.zeros((len(defined_states), len(defined_states))), np.zeros((len(defined_states), len(defined_states)))
            for sub_id in sub_ids:
                df = mega_df.query("sub == @sub_id & state in @defined_states")

                scores = np.zeros(len(defined_states))
                scores[:] = np.nan
                
                scores_new = np.zeros(len(defined_states))
                scores_new[:] = np.nan
            
                X, y = df.iloc[:, :33], df.loc[:, 'state']
                y_ohe = ohe.transform(y.to_numpy()[:, None])
                
                # Original prediction
                model.add_Z(Z_orig)
                y_pred = model.predict_proba(X)
                idx = y_ohe.sum(axis=0) != 0
                scores[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)
                cm_old += confusion_matrix(y_ohe.argmax(axis=1), y_pred.argmax(axis=1))

                model.add_Z(Z_opt)
                y_pred = model.predict_proba(X)
                scores_new[idx] = roc_auc_score(y_ohe[:, idx], y_pred[:, idx], average=None)
                cm_new += confusion_matrix(y_ohe.argmax(axis=1), y_pred.argmax(axis=1))
                
                scores_diff = scores_new - scores
                scores = pd.DataFrame(np.c_[scores, scores_new, scores_diff], columns=['orig_score', 'opt_score', 'diff_score'])
                scores['state'] = defined_states
                scores['sub'] = sub_id
                scores['sub_ethnicity'] = df['sub_ethnicity'].unique()[0]
                scores['sub_split'] = 'test'
                scores['trial_split'] = 'test'
                scores['mapping'] =  mapp_name
                scores['model_ethnicity'] = model_ethn
                scores_all.append(scores)
                
            cm_old = pd.DataFrame(cm_old, index=defined_states, columns=defined_states)
            cm_old['mapping'] = mapp_name
            cm_old['type'] = 'orig'
            cm_old['model_ethnicity'] = model_ethn
            cm_old['sub_ethnicity'] = sub_ethn
            
            cm_new = pd.DataFrame(cm_new, index=defined_states, columns=defined_states)
            cm_new['mapping'] = mapp_name
            cm_new['type'] = 'opt'
            cm_new['model_ethnicity'] = model_ethn
            cm_new['sub_ethnicity'] = sub_ethn
            cm_all.append(pd.concat((cm_old, cm_new), axis=0))
            
            
scores = pd.concat(scores_all, axis=0)
print(scores.groupby(['model_ethnicity', 'sub_ethnicity']).mean())
scores.to_csv('results/scores_optimal_conversational.tsv', sep='\t')

cm = pd.concat(cm_all, axis=0)
cm.to_csv('results/cm_optimal_conversational.tsv', sep='\t')

opt_models = pd.concat(opt_models, axis=0)
for m in opt_models['mapping'].unique():
    om = opt_models.query("mapping == @m")
    om.to_csv(f'results/opt_models/{m}_conversational.tsv', sep='\t')