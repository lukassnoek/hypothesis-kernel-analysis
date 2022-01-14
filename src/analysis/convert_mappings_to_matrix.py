# This script converts the 'verbal' mappings from
# mappings.py into a 2D (emotions x AUs) embedding matrix
import sys
import numpy as np
sys.path.append('src')
from mappings import MAPPINGS
from models import KernelClassifier


PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
#EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])


for mapp_name, mapp in MAPPINGS.items():
    # Initialize model!

    model = KernelClassifier(au_cfg=mapp, param_names=PARAM_NAMES, kernel='cosine', ktype='similarity',
                             binarize_X=False, normalization='softmax', beta=1)
    
    # Technically, we're not "fitting" anything, but this will set up the mapping matrix (self.Z_)
    model.fit(None, None)
    model.Z_ = model.Z_.astype(int)
    model.Z_.to_csv(f'data/{mapp_name}.tsv', sep='\t')
