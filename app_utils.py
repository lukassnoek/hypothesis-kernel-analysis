import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix
from matplotlib.lines import Line2D

sys.path.append('src')
from models import KernelClassifier

# Define parameter names (AUs) and target label (EMOTIONS)
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = np.array(['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])
SUBS = [str(s).zfill(2) for s in range(1, 61)]
plt.style.use('dark_background')


@st.cache(show_spinner=False)
def _load_data(n_subs=60):
    X_all, y_all = [], []
    for sub in SUBS[:n_subs]:
        data = pd.read_csv(f'data/ratings/sub-{sub}_ratings.tsv', sep='\t', index_col=0)
        data = data.query("emotion != 'other'")
        data = data.loc[data.index != 'empty', :]
        X, y = data.iloc[:, :-2], data.iloc[:, -2]
        X_all.append(X)
        y_all.append(y)
    
    return X_all, y_all


@st.cache(show_spinner=False)
def _run_analysis(mapp, X_all, y_all, beta, kernel, analysis_type):
    ktype = 'similarity' if kernel in ['cosine', 'sigmoid', 'linear'] else 'distance'
    model = KernelClassifier(au_cfg=mapp, param_names=PARAM_NAMES, kernel=kernel, ktype=ktype,
                             binarize_X=False, normalization='softmax', beta=beta)
    model.fit(None, None)
    scores = np.zeros((len(SUBS), len(EMOTIONS)))
    confmat = np.zeros((6, 6))

    # Compute model performance per subject!
    for i, (X, y) in enumerate(zip(X_all, y_all)):
        # Predict data + compute performance (AUROC)
        y_pred = model.predict_proba(X)
        y_ohe = pd.get_dummies(y).to_numpy()
        scores[i, :] = roc_auc_score(y_ohe, y_pred, average=None)
        confmat += confusion_matrix(y_ohe.argmax(axis=1), y_pred.argmax(axis=1))
        
    # Store scores and raw predictions
    scores = pd.DataFrame(scores, columns=EMOTIONS, index=SUBS).reset_index()
    scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
    scores = scores.rename({'index': 'sub'}, axis=1)
    scores.loc[:, 'analysis_type'] = analysis_type
    
    return scores, confmat, model.Z_.sort_index()


def _plot_results(scores, hue=None, diff=False):

    emo_scores = scores.groupby('emotion').mean()
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax = sns.barplot(x='emotion', y='score', hue=hue, data=scores, ax=ax)
    ax = sns.stripplot(x='emotion', y='score', hue=hue, edgecolor='black', linewidth=0.4,
                       data=scores, ax=ax, dodge=True, jitter=True)
  
    ax.set_xlabel('')
    if diff:
        ax.set_ylabel('AUROC (adj. - orig.)', fontsize=15)
        ax.set_ylim(-0.2, 0.2)
        ax.axhline(0, ls='--', c='k')
        y_txt = 0.18
    else:
        ax.set_ylabel('AUROC', fontsize=15)
        ax.set_ylim(0.35, 1.0)
        ax.axhline(0.5, ls='--', c='k')
        y_txt = 0.95

    for i, (emo, row) in enumerate(emo_scores.iterrows()):
        ax.text(i, y_txt, np.round(row.loc['score'], 2), ha='center')

    if hue is not None:
        cmap = sns.color_palette()
        labels = scores[hue].unique()
        handles = [Line2D([0], [0], color='k', marker='o', label=labels[i], markerfacecolor=cmap[i])
                   for i in range(len(labels))]
        ax.legend(handles=handles, bbox_to_anchor=(1, 1.2), ncol=len(labels))

    sns.despine()
    return fig, ax


_map2leg = {
    'Darwin': 'Darwin (1886)',
    'Matsumoto2008': 'Matsumoto et al. (2008)',
    'Keltner2019': 'Keltner et al. (2019)',
    'Cordaro2018ref': 'Cordaro et al.\n(2008; ref.)',
    'Cordaro2018IPC': 'Cordaro et al.\n(2008; ICP)',
    'Ekman': 'EMFACS',
    'JS': 'Jack & Schyns'
}

_leg2map = {v: k for k, v in _map2leg.items()}