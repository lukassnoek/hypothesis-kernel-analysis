import pandas as pd
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from noiseceiling.utils import _find_repeats

mat = loadmat('data/AU_data_for_Lukas.mat')
au_names = [n[0] for n in mat['AUnames'][0]]
rename_au = {'AU10Open': 'AU10', 'AU10LOpen': 'AU10L', 'AU10ROpen': 'AU10R', 'AU16Open': 'AU16', 'AU27i': 'AU27'}
au_names = [rename_au[name] if name in rename_au.keys() else name for name in au_names]

emo_names = [e[0] for e in mat['expnames'][0]]

au_data = mat['data_AUamp']
au_data[np.isnan(au_data)] = 0
au_data_onoff = mat['data_AUon']

# Check whether the on/off data is indeed amplitude > 0
au_data_bin = (au_data > 0).astype(int)
np.testing.assert_array_equal(au_data_bin, au_data_onoff)

# Make sure dimensions match
au_data = np.moveaxis(au_data, -1, 0)              # 60 x 42 x 2400
au_model = np.moveaxis(mat['models_AUon'], -1, 1)  # 60 x 42 x 6
emo_rating = np.moveaxis(mat['data_cat'], -1, 0)   # 60 x 2400 x 7
intensity = mat['data_rat'].T                      # 60 x 2400

# load face identities
mat = loadmat('data/cluster_data_ID.mat')['id'].squeeze()
f_ids = np.stack([mat[i].T for i in range(len(mat))])  # 60 x 2400 x 8
f_ids = f_ids.round(1)  # round to one decimal to reduce precision

# Last 45 participants saw one of 8 faces
f_ids_45 = f_ids[15:, :, :].argmax(axis=-1)

# First 15 participants saw a weighted face
f_ids_df = pd.DataFrame(f_ids[:15, :, :].reshape((15*2400, 8)), columns=[f'fp_{i}' for i in range(8)])
uniq_face_ids, _ = _find_repeats(f_ids_df, progress_bar=True)
uniq_face_ids = np.vstack((uniq_face_ids.reshape((15, 2400)) + 7, f_ids_45))  # 60 x 2400
gender = (f_ids.argmax(axis=2) > 3).astype(int)  # 0 = female, 1 = male
gender = gender.reshape((60, 2400))

for i in tqdm(range(au_data.shape[0])):
    idx = []
    for ii in range(au_data.shape[2]):
        au_on = np.where(au_data[i, :, ii] > 0)[0]
        this_idx= '_'.join(
            [f'{au_names[iii]}-{int(100 * au_data[i, iii, ii])}'
             for iii in au_on]
        )
        
        if not this_idx:
            this_idx = 'empty'

        idx.append(this_idx)

    this_dat = np.c_[au_data[i, :, :].T, uniq_face_ids[i, :], gender[i, :]]
    df = pd.DataFrame(this_dat, columns=au_names + ['face_id', 'gender'], index=idx)

    # Let's do some cleaning. First, remove the bilateral AUs that *also*
    # are activated unilaterally
    for au in ['2', '6', '7', '10', '12', '14', '20']:
        L = 'AU' + au + 'L'
        R = 'AU' + au + 'R'
        act = df['AU' + au].values
        df[L] = np.c_[act, df[L].values].max(axis=1)
        if au != '2':
            df[R] = np.c_[act, df[R].values].max(axis=1)
        else:
            df[R] = act

        # Remove the bilateral one
        df = df.drop('AU' + au, axis=1)

    # Now, let's "remove" (recode) compound AUs
    for aus in ['1-2', '25-12', '12-6']:
        act = df['AU' + aus].values
        for au in aus.split('-'):
            if au in ['1', '25']:
                df['AU' + au] = np.c_[act, df['AU' + au]].max(axis=1)
            else:
                df['AU' + au + 'L'] = np.c_[act, df['AU' + au + 'L']].max(axis=1)
                df['AU' + au + 'R'] = np.c_[act, df['AU' + au + 'R']].max(axis=1)

        df = df.drop('AU' + aus, axis=1)

    new_cols = []
    for col in df.columns:
        if col in ['face_id', 'gender']:
            new_col = col
        elif 'L' in col or 'R' in col:
            new_col = col.replace('L', '').replace('R', '')
            new_col = 'AU' + new_col[2:].zfill(2) + col[-1]
        else:
            new_col = 'AU' + col[2:].zfill(2)
        
        new_cols.append(new_col)

    df.columns = new_cols
    df = df.loc[:, sorted(df.columns)]
    au_cols = [col for col in df.columns if 'AU' in col]
    
    if i == 0:
        np.savetxt('data/au_names_new.txt', au_cols, fmt='%s')

    # Merge activation 0.0666 and 0.1333
    vals = df.to_numpy()
    vals = np.round(vals, 1)
    df.loc[:] = vals
    
    new_idx = []
    for _, row in df.iterrows():
        au_on = sorted(np.where(row.iloc[:33] > 0)[0])
        this_idx = '_'.join(
            [f'{df.columns[i]}-{int(100 * row[i])}'
             for i in au_on]
        )
        if not this_idx:
            this_idx = 'empty'

        new_idx.append(this_idx)

    df.loc[:, :] = np.round(vals, 2)
    df.index = new_idx

    # Determine train & test for JS models
    #rep_ids, _ = _find_repeats(df)
    #df.loc[:, 'data_split'] = ['train' if ri in np.unique(rep_ids)[0::2] else 'test'
    #                           for ri in rep_ids]
    
    df['emotion'] = [emo_names[idx] for idx in emo_rating[i, :, :].argmax(axis=1)]
    df['intensity'] = intensity[i, :]
    f_out = f'data/ratings/sub-{str(i+1).zfill(2)}_ratings.tsv'
    df.to_csv(f_out, sep='\t')