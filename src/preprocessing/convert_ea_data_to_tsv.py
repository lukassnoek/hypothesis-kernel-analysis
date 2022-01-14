import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

ORIG_AU_NAMES = [
    'AU1', 'AU1-2', 'AU2', 'AU2L', 'AU4', 'AU5', 'AU6', 'AU6L', 'AU6R', 'AU7L', 'AU7R', 'AU9',
    'AU10Open', 'AU10LOpen', 'AU10ROpen', 'AU11L', 'AU11R', 'AU12', 'AU25-12', 'AU12L', 'AU12R',
    'AU13', 'AU14', 'AU14L', 'AU14R', 'AU15', 'AU16Open', 'AU17', 'AU20', 'AU20L', 'AU20R', 'AU22',
    'AU23', 'AU24', 'AU25', 'AU26', 'AU27i', 'AU38', 'AU39', 'AU43', 'AU7', 'AU12-6'
]
rename_au = {'AU10Open': 'AU10', 'AU10LOpen': 'AU10L', 'AU10ROpen': 'AU10R', 'AU16Open': 'AU16', 'AU27i': 'AU27'}
au_names = [rename_au[name] if name in rename_au.keys() else name for name in ORIG_AU_NAMES]
emo_names = ['happy', 'surprise', 'fear', 'disgust', 'anger', 'sadness', 'other']
resp_mapper = {}
i = 1
for emo in emo_names:
    for inten in range(1, 6):
        resp_mapper[i] = {'emotion': emo, 'intensity': inten}
        i += 1

mat = loadmat('data/raw/EA_data_Lukas.mat')
resp = mat['responses'].squeeze()
stim_au = mat['stim_au_patterns']
stim_gender = mat['stim_gend'].squeeze()
stim_id = mat['stim_id'].squeeze()

sub_idx = mat['participants'].squeeze()
subs = np.unique(sub_idx)
for i, sub in tqdm(enumerate(subs)):
    idx = sub_idx == sub
    au_data = stim_au[idx, :]
    
    idx = []
    for ii in range(au_data.shape[0]):
        au_on = np.where(au_data[ii, :] > 0)[0]
        this_idx= '_'.join(
            [f'{au_names[iii]}-{int(100 * au_data[ii, iii])}'
             for iii in au_on]
        )
        
        if not this_idx:
            this_idx = 'empty'

        idx.append(this_idx)

    df = pd.DataFrame(au_data, columns=au_names, index=idx)
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
        if 'L' in col or 'R' in col:
            new_col = col.replace('L', '').replace('R', '')
            new_col = 'AU' + new_col[2:].zfill(2) + col[-1]
        else:
            new_col = 'AU' + col[2:].zfill(2)
        
        new_cols.append(new_col)

    df.columns = new_cols
    df = df.loc[:, sorted(df.columns)]
    vals = df.to_numpy()
    vals = np.round(vals, 1)
    df.loc[:] = vals
   
    these_id = stim_id[sub_idx == sub]
    df['face_id'] = these_id
    these_gend = stim_gender[sub_idx == sub]
    df['face_gender'] = these_gend

    these_resp = resp[sub_idx == sub]
    emo_resp = [resp_mapper[x]['emotion'] for x in these_resp]
    inten_resp = [resp_mapper[x]['intensity'] for x in these_resp]
    df['emotion'] = emo_resp
    df['intensity'] = inten_resp

    df['sub_nr'] = i+1
    df['sub_ethnicity'] = 1
    f_out = f'data/ratings/EA/sub-{str(i+1).zfill(2)}_ratings.tsv'
    df.to_csv(f_out, sep='\t')