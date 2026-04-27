# Action-augmented HMM

This repository contains the cleaned Action-augmented HMM training, analysis, and figure-plotting workflow for the project. The code is organized around three shared Python modules:

- `ahmm_utils.py`: data preparation, session handling, model training, and saved-model I/O
- `ahmm_eval.py`: analysis helpers, similarity metrics, dataframe/model lookup helpers, and evaluation utilities
- `ahmm_plotting.py`: plotting helpers for transitions, emissions, rasters, tuning heatmaps, PV representations, and figure panels

## Repository Layout

- `run_demo.py`
  Compact single-session smoke test for training one model and plotting the core diagnostics.
- `ahmm_session.ipynb`
  Multi-session model training notebook.
- `ahmm_session_analysis_final.ipynb`
  Analysis notebook for the model sets in the manuscript.
- `ahmm_final_figures_panels_4b_4d_4e.ipynb`
- `ahmm_final_figures_panel_4g.ipynb`
- `ahmm_final_figures_panel_4h.ipynb`
- `ahmm_final_figures_panels_4i_4j_pv_cov.ipynb`
- `ahmm_final_figures_panels_4k_4i_4n_session_pair.ipynb`
- `ahmm_final_figures_extended.ipynb`
  Panel-focused figure notebooks. These now load shared helpers from the Python modules instead of redefining large helper blocks inside the notebooks.
- `data/`
  Small shared input data used by the demo, analysis notebook, and figure notebooks.
- `models/`
  Saved model logs and related model-summary files.
- `demo_outputs/`
  Outputs produced by `run_demo.py`.
- `FULL_DATA_REQUIREMENTS.md`
  Detailed description of the data files currently expected by the notebooks.

## Data Layout

The repository is set up to read data primarily from `data/`. For the public repository, only the compact shared inputs are intended to live in git:

- `sessions_combined.pkl`
- `neural_covariances.mat`
- `df_25_rand.pkl`
- `dfv_rand_25.pkl`
- `rand_train_val_cg_rank1_pde_behavior_cross_compare_results_25_states.pkl`

Large per-animal processed session files and neural PV representation pickles are treated as local-only external data because they exceed practical GitHub repository limits. See `FULL_DATA_REQUIREMENTS.md` for the full list.

Model summary logs currently live under `models/`, for example:

- `models/25/model_log.pkl`
- `models/25/new_emit_model_log.pkl`

## Environment Setup

### Using pip (venv)

Create a virtual environment with Python 3.12 or 3.11:

- **macOS / Linux**
  ```bash
  python3.12 -m venv your_env_name
  ```

- **Windows (Command Prompt / PowerShell)**
  ```bash
  py -3.12 -m venv your_env_name
  ```

Activate the environment:

- **macOS / Linux**
  ```bash
  source your_env_name/bin/activate
  ```

- **Windows (Command Prompt)**
  ```bat
  your_env_name\Scripts\activate
  ```

- **Windows (PowerShell)**
  ```powershell
  .\your_env_name\Scripts\Activate.ps1
  ```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

### Using conda

Create the environment:
```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate your_environment
```

> Replace `your_env_name` / `your_environment` with any name you prefer.  
> You can also edit the `name:` field inside `environment.yml`.

Tested locally with Python 3.12 on macOS.

## Demo

Run (expected runtime within 3 mins):

```bash
python run_demo.py
```

The demo:

1. loads one bundled session from `data/sessions_combined.pkl`
2. trains one 25-state AHMM
3. computes validation NLL and PDE
4. plots the main model summary figures

Outputs are written to `demo_outputs/`:

- `demo_summary.json`
- `demo_transition_matrices.png`
- `demo_policy_matrix.png`
- `demo_emission_matrix.png`
- `demo_pv_representation_matrix.png`
- `demo_transition_graph_pies.png`

## Main Workflows

### Train Models

Use `ahmm_session.ipynb` to train models across sessions.

### Build Analysis Tables

Use `ahmm_session_analysis_final.ipynb` for the cleaned random train/val split 25-state analysis path. This notebook expects the session data, neural covariance file, and saved model summaries to already be present.

### Plot Figures

Use the split figure notebooks rather than the original monolithic figure notebook:

- `ahmm_final_figures_panels_4b_4d_4e.ipynb`
- `ahmm_final_figures_panel_4g.ipynb`
- `ahmm_final_figures_panel_4h.ipynb`
- `ahmm_final_figures_panels_4i_4j_pv_cov.ipynb`
- `ahmm_final_figures_panels_4k_4i_4n_session_pair.ipynb`
- `ahmm_final_figures_extended.ipynb`

These notebooks are set up to display figures inline rather than saving them automatically.

## Notes

- The figure and analysis notebooks use `pd.read_pickle(...)` directly. The saved pickle files expected by this public repository should be Python 3.12-compatible.
- Path handling for saved models is centralized in `ahmm_eval.py`, so stale `save_path` / `model_path` entries from older logs are repaired automatically against the current `models/` layout when possible.
