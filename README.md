# hypothesis-kernel-analysis
Code for the "emotion hypothesis kernel analysis" project.

## Order to run scripts
The following data files are needed to reproduce the analyses:

* `data/raw/AU_data_for_Lukas.mat` and `data/raw/cluster_data_ID.mat` (WC data)
* `data/raw/EA_data_lukas.mat` (EA data)

Then, the following scripts can be run to preprocess the data (i.e., convert mat-files to TSV files):

1. `src/preprocessing/convert_wc_data_to_tsv`  
2. `src/preprocessing/convert_ea_data_to_tsv`

To determine the train/test splits, then run:

3. `src/preprocessing/determine_train_and_test.py`

The data-driven models (per split) are estimated by running:

4. `src/analysis/estimate_datadriven_models.py`

To convert the mappings into 2D embedding matrices that the kernel classifier can use, run:

5. `src/analysis/convert_mappings_to_matrix.py`

Note that the `JackSchyns` models (i.e., the data-driven models) are already in the right 2D-matrix format.

Finally, run the three-stage analysis:

6. `src/analysis/prediction_analysis.py`
7. `src/analysis/explanation_analysis.py`
8. `src/analysis/exploration_analysis.py`

To reproduce the paper's figures, execute the following notebook:

9. `notebooks/figures.ipynb`