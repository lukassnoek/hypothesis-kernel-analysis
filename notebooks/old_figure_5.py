df_o = pd.read_csv('../results/scores_optimal.tsv', sep='\t', index_col=0)
df_o = df_o.query("model_ethnicity == 'all'")
df_o = df_o.replace(map2leg)
df_o = df_o.replace({'JackSchyns_ethn-all_CV': 'Jack et al.\n(2012)'})
df_o = pd.concat([
    df_o,
    df_o.groupby(['mapping', 'sub', 'sub_ethnicity']).mean().reset_index().assign(emotion='average')],
    axis=0)

df_o_av = df_o.groupby(['mapping', 'emotion']).mean().reset_index()

# OLD FIGURE
fig = plt.figure(figsize=(15, 10), constrained_layout=True)
gs = fig.add_gridspec(3, 8, height_ratios=[3, 2, 2], width_ratios=[1, 1, 1, 1, 1, 1, 1, 0.2])

ax1 = fig.add_subplot(gs[0, :])
ax1 = sns.barplot(x='emotion', y='diff_score', data=df_o, hue='mapping',
                 ax=ax1, ci='sd')

ax1 = sns.stripplot(x='emotion', y='diff_score', hue='mapping', ax=ax1,
                    edgecolor='black', linewidth=0.4, data=df_o,
                    dodge=True, jitter=True)

# # hack to add whitespace between row 1 and 2
ax1.set_xlabel('bla', labelpad=25)
ax1.xaxis.label.set_color('white')
            
ax1.set_ylabel(r'$\Delta$ AUROC', fontsize=20, labelpad=10)
nc_ = nc.query("sub_ethnicity == 'all' & sub_split == 'test' & trial_split == 'test'")
for i, emo in enumerate(emo_names + ['average']):
    orig = df_o_av.query("emotion == @emo").set_index('mapping').loc[map_names, 'orig_score'].to_numpy()
    this_nc = nc_.query("emotion == @emo")['noise_ceiling'].item() - orig
    this_sd = nc_.query("emotion == @emo")['sd'].item()
    ax1.plot(np.linspace(i - 0.35, i + 0.35, num=7), this_nc, ls='--', c='k')
    ax1.fill_between(
        np.linspace(i-0.35, i+0.35, num=7),
        this_nc - this_sd, this_nc + this_sd,
        color='gray', alpha=0.3
    )

ax1.set_facecolor('white')
ax1.set_ylim(-0.1, 0.4)
ax1.axhline(0.5, c='k', ls='-')
ax1.legend_.remove()
colors = {mapp: sns.color_palette()[i] for i, mapp in enumerate(map_names)}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
ax1.legend(handles, labels, ncol=7, loc=(-0.025, 1.05), frameon=False, fontsize=13)

ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=20)
ax1.set_xlim(-0.5, 6.75)
sns.despine()

# CM PLOTS
axes = [[fig.add_subplot(gs[ii, i]) for i in range(7)] for ii in range(1, 3)]
cbar_ax = fig.add_subplot(gs[1:, 7])
I = np.eye(6)
for i, mapp in enumerate(cm_df['mapping'].unique()):
    bg = np.zeros(2)
    for ii, tpe in enumerate(['orig', 'opt']):
        cm = cm_df.query("mapping == @mapp & type == @tpe").drop(['mapping', 'type'], axis=1)
        cm = cm.loc[emo_reorder, emo_reorder].to_numpy()
        bg[ii] = (cm[np.triu_indices_from(cm, k=1)].sum() + cm[np.tril_indices_from(cm, k=-1)].sum()) / 30

        cm /= cm.sum(axis=1)
        
        xtl = emo_short if ii == 1 else False
        ytl = emo_short if i == 0 else False
        cbar = True if i == 6 and ii == 0 else False
        axes[ii][i] = sns.heatmap(
            cm, ax=axes[ii][i], vmin=0, vmax=0.8, square=True, cbar=cbar,
            xticklabels=xtl, yticklabels=ytl, cbar_ax=cbar_ax
        )
        if cbar:
            cbar_ax.set_title('Prop.\ntrials', size=15, pad=10)
            cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size=12)
            cbar_ax.set_box_aspect(12.4)
            for axis in ['top','bottom','left','right']:
                cbar_ax.spines[axis].set_visible(True)
        
        if ii == 0:
            axes[ii][i].set_title(mapp, fontsize=15, pad=10)
        
        if ii == 1:
            axes[ii][i].set_xticklabels(axes[ii][i].get_xticklabels(), rotation=90, fontsize=15)

        if i == 0:
            axes[ii][i].set_yticklabels(axes[ii][i].get_yticklabels(), rotation=0, fontsize=15)

        axes[ii][i].tick_params(axis='x', which='both', bottom=False)
        axes[ii][i].tick_params(axis='y', which='both', left=False)
        #bg[ii] = 1 - (np.sum((cm - I) ** 2) / np.sum((1 - I) ** 2))
    
    delta_bg = (bg[1] - bg[0]) / bg[0] * 100
    axes[1][i].text(3, -1.4, f"{delta_bg:.1f}%", ha='center', va='center', fontsize=18)

axes[1][0].set_ylabel('True emotion', fontsize=20, ha='center', va='center')
axes[1][0].yaxis.set_label_coords(-0.5, 1.25)
axes[1][3].set_xlabel('Predicted emotion', fontsize=20, labelpad=15)
#axes[1][0].text(-0.5, -1.3, 'Conf.:', fontsize=20, ha='center', va='center')
axes[0][0].text(-1.75, 3, 'Original', fontsize=20, ha='center', va='center', rotation=90)
axes[1][0].text(-1.75, 3, 'Optimal', fontsize=20, ha='center', va='center', rotation=90)

plt.savefig('figure_5.png', dpi=400)