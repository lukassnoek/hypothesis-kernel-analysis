import pandas as pd
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


mat = loadmat('data/AU_data_for_Lukas.mat')
au_names = [n[0] for n in mat['AUnames'][0]]
rename_au = {'AU10Open': 'AU10', 'AU10LOpen': 'AU10L', 'AU10ROpen': 'AU10R', 'AU16Open': 'AU16', 'AU27i': 'AU27'}
au_names = [rename_au[name] if name in rename_au.keys() else name for name in au_names]

np.savetxt('data/au_names.txt', au_names, fmt='%s')
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

    df = pd.DataFrame(au_data[i, :, :].T, columns=au_names, index=idx)
   
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
        if 'L' in col or 'R' in col:
            new_col = col.replace('L', '').replace('R', '')
            new_col = 'AU' + new_col[2:].zfill(2) + col[-1]
        else:
            new_col = 'AU' + col[2:].zfill(2)
        
        new_cols.append(new_col)

    df.columns = new_cols
    df = df.loc[:, sorted(df.columns)]
    
    if i == 0:
        np.savetxt('data/au_names_new.txt', df.columns, fmt='%s')

    # Merge activation 0.0666 and 0.1333
    vals = df.to_numpy()
    #print(np.unique(np.round(vals, 3)))
    vals = np.round(vals, 1)
    print(np.unique(vals))
    #vals[(0 < vals) & (vals < 0.334)] = 0.25
    #vals[(0.334 < vals) & (vals < 0.667)] = 0.5
    #vals[vals >= 0.667] = 0.75
    df.loc[:] = vals
    
    new_idx = []
    for _, row in df.iterrows():
        au_on = sorted(np.where(row > 0)[0])
        this_idx = '_'.join(
            [f'{df.columns[i]}-{int(100 * row[i])}'
             for i in au_on]
        )
        if not this_idx:
            this_idx = 'empty'

        new_idx.append(this_idx)

    df.loc[:, :] = np.round(vals, 2)
    df.index = new_idx
    
    df['emotion'] = [emo_names[idx] for idx in emo_rating[i, :, :].argmax(axis=1)]
    df['intensity'] = intensity[i, :]
    f_out = f'data/ratings/sub-{str(i+1).zfill(2)}_ratings.tsv'
    df.to_csv(f_out, sep='\t')