import numpy as np
import pandas as pd

subs = [str(s).zfill(2) for s in range(1, 61)]
N = []
for sub in subs:
    data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
    data = data.query("emotion != 'other'")
    N.append(data.shape[0])

print(np.sum(N))
print(np.mean(N))
print(np.std(N))