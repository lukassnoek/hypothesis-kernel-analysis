### PER INTENSITY LEVEL
mean_intensity = ratings['intensity'].reset_index().groupby('index').mean()
ratings.loc[mean_intensity.index, 'intensity'] = mean_intensity['intensity']
percentiles = ratings['intensity'].quantile([0, .25, 0.5, 0.75, 1.])
nc_df = pd.DataFrame(columns=[
    ['participant_id', 'emotion', 'intensity', 'noise_ceiling', 'sd']
])
dfs = []
i = 0
for intensity in [1, 2, 3, 4]:
    minn, maxx = percentiles.iloc[intensity-1], percentiles.iloc[intensity]
    tmp_ratings = ratings.query("@minn <= intensity & intensity <= @maxx")
    nc = compute_nc_classification(
        tmp_ratings.iloc[:, :33], tmp_ratings['emotion'], **kwargs
    )
    nc_b = run_bootstraps_nc(tmp_ratings.iloc[:, :33], tmp_ratings['emotion'], kwargs=kwargs, n_bootstraps=100)
    nc = np.c_[nc.to_numpy().squeeze(), nc_b.std(axis=0).to_numpy()]
    nc = pd.DataFrame(nc, columns=['noise_ceiling', 'sd'])
    nc['emotion'] = emotions
    nc['intensity_level'] = intensity
    dfs.append(nc)

nc_df = pd.concat(dfs, axis=0)
nc_df.to_csv('results/noise_ceiling_intensity_stratified.tsv', sep='\t', index=True)


### PER INTENSITY LEVEL
mean_intensity = ratings['intensity'].reset_index().groupby('index').mean()
ratings.loc[mean_intensity.index, 'intensity'] = mean_intensity['intensity']
percentiles = ratings['intensity'].quantile([0, .25, 0.5, 0.75, 1.])
nc_df = pd.DataFrame(columns=[
    ['participant_id', 'emotion', 'intensity', 'noise_ceiling', 'sd']
])
dfs = []
i = 0
for intensity in [1, 2, 3, 4]:
    minn, maxx = percentiles.iloc[intensity-1], percentiles.iloc[intensity]
    tmp_ratings = ratings.query("@minn <= intensity & intensity <= @maxx")
    nc = compute_nc_classification(
        tmp_ratings.iloc[:, :33], tmp_ratings['emotion'], **kwargs
    )
    nc_b = run_bootstraps_nc(tmp_ratings.iloc[:, :33], tmp_ratings['emotion'], kwargs=kwargs, n_bootstraps=100)
    nc = np.c_[nc.to_numpy().squeeze(), nc_b.std(axis=0).to_numpy()]
    nc = pd.DataFrame(nc, columns=['noise_ceiling', 'sd'])
    nc['emotion'] = emotions
    nc['intensity_level'] = intensity
    dfs.append(nc)

nc_df = pd.concat(dfs, axis=0)
nc_df.to_csv('results/noise_ceiling_intensity_stratified_test.tsv', sep='\t', index=True)
