# Project overview

## Context and objectives
This repository supports a clinical data science workflow using the METABRIC_cleaned_imputed.csv dataset to explore prognostic factors in breast cancer and build interpretable survival models. The aim is to produce reproducible analyses, transparent methodology, and publication-ready figures/tables for collaboration.

## Clinical questions
- **Primary:** Which clinicopathologic features are associated with overall survival in METABRIC?
- **Secondary:** How do model-based risk scores compare to traditional staging (e.g., tumor size, nodal status), and can we present interpretable feature effects?

## Dataset summary
- **Source:** METABRIC (cleaned + imputed version used here)
- **Rows/columns:** See data profile in outputs/data_profile/summary.md
- **Key variables:** Overall survival time (months), event indicator, age at diagnosis, tumor size, nodal status, grade, ER/PR/HER2 status, molecular subtype, treatment indicators, and genomic features if available.
- **Missingness handling:** Imputation documented in docs/methods.md; per-variable notes in outputs/data_quality/missingness.md.

## Reproducible workflow
- **Environment:** Google Colab + pinned Python package versions
- **Pipeline:** Numbered notebook sections (00–99) for setup, EDA, preprocessing, modeling, evaluation, and reporting
- **Outputs:** All figures/tables saved under outputs/ with traceable file stems and code cell references
- **Versioning:** GitHub + commit messages describing changes to data processing and modeling

## Deliverables
- **EDA figures:** Histograms, correlation heatmaps, Kaplan–Meier curves
- **Model artifacts:** Trained Cox PH / RSF models + serialized coefficients/feature importances
- **Reports:** Markdown summaries under docs/; per-run metrics under outputs/metrics/
- **Interpretability:** Coefficient tables, partial dependence/ICE, SHAP for survival models where applicable

## Directory map
- **docs/** Project overview, clinical and statistical methodology, references
- **notebooks/** Numbered Colab notebooks with step-by-step sections
- **outputs/** Saved plots, tables, metrics, and logs with run IDs
- **data/** METABRIC_cleaned_imputed.csv (never committed if restricted)
- **src/** Reusable functions for preprocessing, modeling, and reporting

## Governance and collaboration
- **Provenance tracking:** Each output records dataset hash, git commit, and code cell ID
- **Clinical review loop:** Analysts prepare interpretable summaries; clinicians review figures/tables; feedback integrated into subsequent runs
- **Ethics:** Non-identifiable, secondary analysis; methods emphasize transparency and limitations

## Quick start
1. **Open Colab:** notebooks/00_setup_environment.ipynb
2. **Load data:** data/METABRIC_cleaned_imputed.csv
3. **Run EDA:** notebooks/10_eda_profile.ipynb (outputs/data_profile/*)
4. **Preprocess + encode:** notebooks/20_preprocess_encode.ipynb (outputs/preprocessing/*)
5. **Train survival models:** notebooks/30_modeling_survival.ipynb (outputs/models/*, outputs/metrics/*)
6. **Generate report:** notebooks/90_reporting.ipynb writes docs/* from outputs/*
