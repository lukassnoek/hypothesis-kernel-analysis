import pandas as pd
from scipy.io import loadmat

mat = loadmat('data/raw/emotion/EA_data_Lukas.mat')
participants60 = mat['participants'].squeeze()

info = pd.read_csv('data/raw/emotion/A197.tsv', sep='\t')
info = info.query("Participant in @participants60")
#info = info.query("Participant == 10301")
for part in info['Participant'].unique():
    info_ = info.query("Participant == @part").copy()
    for exp in info_['Experiment'].unique():
        info__ = info_.query("Experiment == @exp").copy()
        print(info__['Timestamp'])
        exit()
        for block in info__['Block'].unique():
            info___ = info__.query("Block == @block").copy()
            print(info___['Trial'].unique().size)
        #print(info__)
            exit()
    #print(info_['Experiment'].unique().size)
    #exit()
    #print(info_['Timestamp'].apply(lambda x: x.split(' ')[0]).unique().size)