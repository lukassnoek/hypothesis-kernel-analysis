import pandas as pd


df = pd.read_csv('data/JackSchyns.tsv', sep='\t', index_col=0)
df = df.query("sub == 'average_even' & trial_split == 'even'")
for emo in df['emotion']:
    mapp = df.query("emotion == @emo").iloc[0]
    mapp = mapp.index[mapp == 1].tolist()
    print(f"{emo}: {mapp}")