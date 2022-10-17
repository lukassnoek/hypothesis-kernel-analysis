# hypothesis-kernel-analysis
Code for the "emotion hypothesis kernel analysis" project.

## Order to run scripts
Download the data from [Figshare](https://doi.org/10.21942/uva.21261885) and store in `data/`. These data represent the preprocessed data.

The data-driven models (per split) are estimated by running:

4. `src/analysis/estimate_datadriven_models.py`

To convert the mappings into 2D embedding matrices that the kernel classifier can use, run:

5. `src/analysis/convert_mappings_to_matrix.py`

Finally, run the three-stage analysis:

6. `src/analysis/prediction_analysis.py`
7. `src/analysis/explanation_analysis.py`
8. `src/analysis/exploration_analysis.py`

To reproduce the paper's figures, execute the following notebook:

9. `notebooks/figures.ipynb`
