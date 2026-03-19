# Action-augmented HMM: Full Data Requirements

This repository assumes a simple local layout:

- shared input data in `data/`
- model logs and model files under `models/`

Below is the full set of data products expected by the cleaned notebooks. Some of these files are intentionally not tracked in the public GitHub repository because they are too large for practical git hosting.

## Small Shared Files

These are the compact files that are reasonable to keep in the public repository:

- `data/sessions_combined.pkl`
- `data/neural_covariances.mat`
- `data/df_25_rand.pkl`
- `data/dfv_rand_25.pkl`
- `data/rand_train_val_cg_rank1_pde_behavior_cross_compare_results_25_states.pkl`

## Large External Files

These are still required for some workflows, but should be treated as local external data rather than version-controlled repository contents.

### Per-Animal Processed Neural/Session Files

- `data/G375_all_session_processed_data.pkl`
- `data/G386_all_session_processed_data.pkl`
- `data/G402_all_session_processed_data.pkl`
- `data/G410_all_session_processed_data.pkl`
- `data/G420_all_session_processed_data.pkl`
- `data/G506_all_session_processed_data.pkl`

### Per-Animal Neural PV Representation Files

- `data/G375_all_sessions_neural_pv_representation.pkl`
- `data/G386_all_sessions_neural_pv_representation.pkl`
- `data/G402_all_sessions_neural_pv_representation.pkl`
- `data/G410_all_sessions_neural_pv_representation.pkl`
- `data/G420_all_sessions_neural_pv_representation.pkl`
- `data/G506_all_sessions_neural_pv_representation.pkl`

### Optional Analysis Product

- `data/sim_df_25_rand.pkl`

## Model Summary Files

Currently available model-summary files intended to stay under `models/`:

- `models/25/model_log.pkl`
- `models/25/new_emit_model_log.pkl`

These are summary tables. Figure notebooks that load actual AHMMs also require the referenced `.npz` model files to exist somewhere under `models/` or another searchable location that the path-repair helper can resolve.

## Full Local Inventory Reference

For completeness, the local working layout may include all of the following under `data/`:

- `data/sessions_combined.pkl`
- `data/neural_covariances.mat`
- `data/df_25_rand.pkl`
- `data/dfv_rand_25.pkl`
- `data/sim_df_25_rand.pkl`
- `data/rand_train_val_cg_rank1_pde_behavior_cross_compare_results_25_states.pkl`
- `data/G375_all_session_processed_data.pkl`
- `data/G386_all_session_processed_data.pkl`
- `data/G402_all_session_processed_data.pkl`
- `data/G410_all_session_processed_data.pkl`
- `data/G420_all_session_processed_data.pkl`
- `data/G506_all_session_processed_data.pkl`
- `data/G375_all_sessions_neural_pv_representation.pkl`
- `data/G386_all_sessions_neural_pv_representation.pkl`
- `data/G402_all_sessions_neural_pv_representation.pkl`
- `data/G410_all_sessions_neural_pv_representation.pkl`
- `data/G420_all_sessions_neural_pv_representation.pkl`
- `data/G506_all_sessions_neural_pv_representation.pkl`

## Notebook Requirements

### `ahmm_session.ipynb`

Uses:

- `data/sessions_combined.pkl`

Writes or updates:

- model outputs and logs under `models/`

### `ahmm_session_analysis_final.ipynb`

Uses:

- `data/sessions_combined.pkl`
- `data/neural_covariances.mat`
- model summary logs such as `models/25/model_log.pkl`
- the actual AHMM `.npz` model files referenced by those logs

Writes analysis products such as:

- `data/df_25_rand.pkl`
- `data/dfv_rand_25.pkl`
- `data/sim_df_25_rand.pkl`
- `data/rand_train_val_cg_rank1_pde_behavior_cross_compare_results_25_states.pkl`

### Figure Notebooks

The split figure notebooks use different subsets of the shared files:

#### `ahmm_final_figures_panels_4b_4d_4e.ipynb`

Uses:

- `data/sessions_combined.pkl`
- `data/df_25_rand.pkl`
- `data/dfv_rand_25.pkl`
- actual AHMM model files referenced by `df_25_rand.pkl`

#### `ahmm_final_figures_panel_4g.ipynb`

Uses:

- `data/sessions_combined.pkl`
- `data/df_25_rand.pkl`
- actual AHMM model files referenced by `df_25_rand.pkl`

#### `ahmm_final_figures_panel_4h.ipynb`

Uses:

- `data/sessions_combined.pkl`
- `data/df_25_rand.pkl`
- `data/dfv_rand_25.pkl`
- actual AHMM model files referenced by `df_25_rand.pkl`

#### `ahmm_final_figures_panels_4i_4j_pv_cov.ipynb`

Uses:

- `data/sessions_combined.pkl`
- `data/df_25_rand.pkl`
- actual AHMM model files referenced by `df_25_rand.pkl`

#### `ahmm_final_figures_panels_4k_4i_4n_session_pair.ipynb`

Uses:

- `data/sessions_combined.pkl`
- `data/df_25_rand.pkl`
- `data/dfv_rand_25.pkl`
- `data/G*_all_session_processed_data.pkl`
- `data/G*_all_sessions_neural_pv_representation.pkl`
- actual AHMM model files referenced by `df_25_rand.pkl`

#### `ahmm_final_figures_extended.ipynb`

Uses:

- `data/dfv_rand_25.pkl`

This notebook can run from the saved analysis table alone and does not require model files.

## Compatibility Note

The `.pkl` files expected by this repository should be Python 3.12-compatible so the notebooks and shared modules can use plain `pd.read_pickle(...)`.

## Recommended Minimum Share Set

If you want someone else to reproduce the cleaned analysis and figures from this repository layout, the minimum practical share set is:

- all files currently under `data/`
- all referenced AHMM `.npz` model files
- the model summary logs under `models/`

Without the actual model `.npz` files, only the saved-table-based notebooks or notebook sections will run.
