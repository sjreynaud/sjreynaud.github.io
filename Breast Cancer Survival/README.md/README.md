# METABRIC survival modeling and interpretability

## Overview
This repository contains a reproducible pipeline for survival analysis and interpretability on the METABRIC dataset, using the file `data/METABRIC_cleaned_imputed.csv`. The workflow emphasizes clinical relevance, transparent preprocessing, and publication-ready outputs. Models include Kaplan–Meier and Cox proportional hazards, with SHAP-based interpretability for selected predictive models.

## Dataset
- **Source file:** `data/METABRIC_cleaned_imputed.csv`
- **Note:** This is a processed version prepared for survival modeling with imputation. Raw data access may be restricted; ensure appropriate approvals and data governance.

## Environment setup
- **Colab (recommended for quick start):**
  1. Upload/clone the repo into Colab.
  2. Run `!pip install -r requirements.txt`.
  3. Ensure `data/METABRIC_cleaned_imputed.csv` is present under `data/`.
- **Local (Conda):**
  1. Install Conda (miniconda or mamba).
  2. `conda env create -f environment.yml`
  3. `conda activate metabric-survival`
  4. `python -m ipykernel install --user --name metabric-survival --display-name "metabric-survival"`

## Project structure
- **`notebooks/`**: Numbered, modular pipeline notebooks (00–09) for provenance.
- **`src/`**: Reusable modules (data loading, preprocessing, modeling, evaluation, plots).
- **`data/`**: Input dataset (not tracked or tracked via LFS). Place `METABRIC_cleaned_imputed.csv` here.
- **`outputs/`**: Saved tables, figures, metrics, and reports for publication.
- **`configs/`**: YAML configs controlling experiments and feature handling.
- **`environment.yml`, `requirements.txt`**: Environment definitions.
- **`LICENSE`, `README.md`, `.gitignore`**: Documentation and governance.

## Reproducible pipeline
- **00_schema_and_missingness:** Summarize columns, types, missingness, and basic stats; save `outputs/00_schema_summary.md`.
- **01_split_and_seed_control:** Fixed seeds; train-test split strategies; save `outputs/01_split_summary.json`.
- **02_preprocessing:** Encode categorical, scale numeric, reconstruct feature names; save `outputs/02_feature_map.csv`.
- **03_eda:** Survival curves per key strata; correlation heatmaps; save `outputs/03_eda_plots/`.
- **04_models_km_cox:** Kaplan–Meier and CoxPH fits; assumptions checks; save `outputs/04_survival_models/`.
- **05_ml_models:** Elastic Net/GBM with survival-compatible approaches; save `outputs/05_ml_models/`.
- **06_interpretability:** SHAP summaries and dependence plots; save `outputs/06_shap/`.
- **07_validation:** Bootstrap/CV; calibration; save `outputs/07_validation/`.
- **08_reporting:** Auto-generate `outputs/report.md` with figures/tables.
- **09_packaging:** Export artifacts and environment fingerprints.

## Usage
- **Run notebooks in order** (00 → 09). Each notebook writes outputs with matching IDs.
- **Configs**: Adjust `configs/default.yml` to change feature sets, encoding, and seeds.
- **CLI (optional)**: `python -m src.pipeline --config configs/default.yml`

## Outputs and traceability
- **Provenance:** Every figure/table is mapped to a code cell ID and notebook number.
- **Artifacts:** Versioned under `outputs/` with timestamps and config hashes.

## Contributing
- **Issues/PRs:** Please include data governance notes and reproducibility checks.
- **Style:** `black`, `isort`, `flake8` enforced via pre-commit (optional).

## Citation
> Sidney J Reynaud,Jr., "METABRIC Survival Modeling and Interpretability," GitHub repository, 2025.

## License
See `LICENSE` for terms (MIT recommended for broad reuse).
