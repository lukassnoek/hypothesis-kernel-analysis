import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from models import KernelClassifier
from sim_mappings import simulate_aus, simulate_configs


MAPPINGS_c, list_configs = simulate_configs(1000, 10)
MAPPINGS_a, list_aus = simulate_aus(1000, 10)

to_iterate = dict(n_configs=MAPPINGS_c, n_aus=MAPPINGS_a)

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])

# One-hot encode target label
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(EMOTIONS[:, None])

files = sorted(glob('data/ratings/*/*.tsv'))
mega_df = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in files], axis=0)
mega_df = mega_df.query("sub_split == 'train' & trial_split == 'train'")
mega_df = mega_df.query("emotion != 'other'")  # remove non-emo trials
mega_df = mega_df.loc[mega_df.index != 'empty', :]  # remove trials w/o AUs

# Loop across mappings (Darwin, Ekman, etc.)
for sim_type, MAPPINGS in to_iterate.items():
    # Define analysis parameters
    scores_all = []

    print(f"Running simulation analysis for {sim_type} ...")
    
    for mapp_name, mapp in tqdm(MAPPINGS.items()):

        # Initialize model!
        model = KernelClassifier(au_cfg=mapp, param_names=PARAM_NAMES, kernel='cosine', ktype='similarity',
                                binarize_X=False, normalization='softmax', beta=1)
        
        model.fit(None, None)

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
            scores['mapping'] = mapp_name
            if sim_type == 'n_configs':
                scores['n_configs'] = [len(v.keys()) for k, v in mapp.items()]
            else:
                scores['n_aus'] = len(mapp['anger'])  # same for every emotion
            scores_all.append(scores)

    # Save scores and predictions
    scores = pd.concat(scores_all, axis=0)
    scores.to_csv(f'results/scores_bias_simulation_{sim_type}.tsv', sep='\t')
