import sys
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from copy import deepcopy

sys.path.append('src')
from mappings import MAPPINGS
from app_utils import _load_data, _run_analysis, _map2leg, _leg2map, _plot_results

EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()

st.title('Hypothesis kernel analysis')
st.write("""
An app to interactively test hypotheses about AU-emotion mappings.
""")

st.header("Test existing mappings")
st.write("""
Here, you can test one or more existing mappings. Pick one from the dropdown list below and the app will
automatically evaluate this mapping against the ratings from 60 participants. The average AUROC
score across participants (SD in parentheses) is shown when the analysis is done. To show a bar graph
with the model performance, enable the 'Show orig. performance plot' option in the sidebar. When adjusting
existing mappings (see next section), it's best to leave this option off (because plotting cannot be cached).

To show the AUs contained in the selected mapping, enable the 'Show config' setting in the sidebar.
""")


mapp = st.selectbox('Select an existing mapping', list(_map2leg.values()))

st.sidebar.header("Display options")
show_perf_plot = st.sidebar.checkbox("Show orig. performance plot")
show_adjust = st.sidebar.checkbox("Show adjust app")
show_corr_plot = st.sidebar.checkbox("Show mapping corr. plot")    

# Analysis settings
st.sidebar.header("Analysis parameters")
n_sub = st.sidebar.slider("N subj.", min_value=1, max_value=60, value=60, step=1)
kernel = st.sidebar.selectbox("Kernel", ['cosine', 'euclidean'])
beta = st.sidebar.slider("Beta", min_value=1, max_value=1000, value=1, step=1)

X_all, y_all = _load_data(n_sub)
scores, conf_mat, Z = _run_analysis(MAPPINGS[_leg2map[mapp]], X_all, y_all, beta, kernel, analysis_type='original')
mean, sd = scores['score'].mean(), scores['score'].std()
st.write(f"Mean AUROC across participants and emotions: {mean:.3f} (SD = {sd:.3f})")

if show_perf_plot:
    fig, ax = _plot_results(scores)
    st.pyplot(fig)

if show_adjust:
    st.header("Adjust existing mappings")
    st.write("""
    Here, you can adjust the selected mappings and subsequently test it.
    To reset the configuration, uncheck and check the 'Show adjust app' option
    in the sidebar.
    """)
    current_mapp = MAPPINGS[_leg2map[mapp]]
    
    new_mapp = {}
    for i, (emo, cfg) in enumerate(current_mapp.items()):
        with st.beta_expander(f"{emo.capitalize()} config(s)"):
            if isinstance(cfg, dict):
                new_mapp[emo] = {}
                for nr, this_cfg in cfg.items():
                    new_mapp[emo][nr] = st.multiselect(f'{emo} ({nr}) AUs', PARAM_NAMES, default=this_cfg)
            else:
                new_mapp[emo] = st.multiselect(f'{emo} AUs', PARAM_NAMES, default=cfg)
    
    st.write('')
    st.write("When you're happy with the adjusted configurations, click the 'Run analysis' button below.")
    
    cols = st.beta_columns(2)
    with cols[0]:
        run = st.button("Run analysis")

    with cols[1]:
        plot_type = st.radio('', ['Side-by-side', 'Difference'])

    ### CONF MATRIX STUFF
    st.sidebar.header("Confusion matrix params")
    metric = st.sidebar.selectbox('Metric', ['recall', 'precision'])
    vmin = st.sidebar.slider('Minimum', min_value=0., max_value=1., value=0., step=0.01)
    vmax = st.sidebar.slider('Maximum', min_value=0., max_value=1., value=1., step=0.01)
    
    if run:
        scores_adj, conf_mat_adj, Z_adj = _run_analysis(new_mapp, X_all, y_all, beta, kernel, analysis_type='adjusted')
        if plot_type == 'Side-by-side':
            scores_comb = pd.concat((scores, scores_adj))
            fig, ax = _plot_results(scores_comb, hue='analysis_type')
        else:
            scores_diff = scores.copy()
            scores_diff['score'] = scores_adj['score'] - scores['score']
            fig, ax = _plot_results(scores_diff, hue=None, diff=True)
    
        st.pyplot(fig)

        fig, axes = plt.subplots(figsize=(8, 4.3), ncols=2, sharex=True, sharey=True,
                                 constrained_layout=True)

        if metric == 'recall':
            conf_mat_norm = conf_mat / conf_mat.sum(axis=1)
            conf_mat_norm[np.isnan(conf_mat_norm)] = 0
            conf_mat_adj_norm = conf_mat_adj / conf_mat_adj.sum(axis=1)
            conf_mat_adj_norm[np.isnan(conf_mat_adj_norm)] = 0
        else:
            conf_mat_norm = conf_mat / conf_mat.sum(axis=0)
            conf_mat_norm[np.isnan(conf_mat_norm)] = 0
            conf_mat_adj_norm = conf_mat_adj / conf_mat_adj.sum(axis=0)
            conf_mat_adj_norm[np.isnan(conf_mat_adj_norm)] = 0

        axes[0] = sns.heatmap(conf_mat_norm, vmin=vmin, vmax=vmax, ax=axes[0], annot=True,
                              cbar=False, square=True)
        axes[0].set_title("Original", fontsize=15)
        axes[1] = sns.heatmap(conf_mat_adj_norm, vmin=vmin, vmax=vmax, ax=axes[1], annot=True,
                              cbar=False, square=True)
        axes[1].set_title("Adjusted", fontsize=15)
        
        axes[0].set_xticklabels(EMOTIONS, rotation=90)
        axes[0].set_ylabel('Y true', fontsize=12)
        axes[0].set_xlabel('Y pred.', fontsize=12)
        
        axes[1].set_xticklabels(EMOTIONS, rotation=90)
        
        axes[0].set_yticklabels(EMOTIONS, rotation=0)
        st.pyplot(fig)


        fig, axes = plt.subplots(figsize=(8, 4.3), ncols=2, sharex=True, sharey=True,
                                 constrained_layout=True)
        
        axes[0] = sns.heatmap(Z.T.corr(), vmin=vmin, vmax=vmax, ax=axes[0], annot=True,
                              cbar=False, square=True, annot_kws={'fontsize': 6})
        axes[0].set_title("Original", fontsize=15)
        axes[1] = sns.heatmap(Z_adj.T.corr(), vmin=vmin, vmax=vmax, ax=axes[1], annot=True,
                              cbar=False, square=True, annot_kws={'fontsize': 6})
        axes[1].set_title("Adjusted", fontsize=15)
        
        axes[0].set_xticklabels(Z.index.tolist(), rotation=90)
        axes[1].set_xticklabels(Z.index.tolist(), rotation=90)
        axes[0].set_yticklabels(Z.index.tolist(), rotation=0)
        st.pyplot(fig)

