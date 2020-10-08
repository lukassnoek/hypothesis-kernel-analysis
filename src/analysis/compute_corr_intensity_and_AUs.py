import numpy as np
import pandas as pd
from scipy.stats import pearsonr

subs = [str(s).zfill(2) for s in range(1, 61)]
c = []
for sub in subs:
    data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
    data = data.query("emotion != 'other'")
    corr = pearsonr(data.iloc[:, :-2].sum(axis=1), data['intensity'])
    c.append(corr)

corr = np.mean(corr)
print(corr)
