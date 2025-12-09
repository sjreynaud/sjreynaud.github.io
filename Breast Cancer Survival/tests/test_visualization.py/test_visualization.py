import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA_PATH = os.environ.get("DATA_PATH", "/content/METABRIC_cleaned_imputed.csv")
FIG_DIR = os.environ.get("FIG_DIR", "/content/outputs/figures")
TIME_CANDIDATES = ["Overall_Survival", "OS_MONTHS", "survival_time", "OS"]
EVENT_CANDIDATES = ["Overall_Survival_Status", "OS_EVENT", "event", "Death_Event"]

def load_data(path=DATA_PATH):
    assert os.path.exists(path), f"Dataset not found at {path}"
    df = pd.read_csv(path)
    assert len(df) > 0, "Dataset is empty"
    return df

def detect_survival_columns(df):
    time_col = next((c for c in TIME_CANDIDATES if c in df.columns), None)
    event_col = next((c for c in EVENT_CANDIDATES if c in df.columns), None)
    assert time_col is not None, "Missing time column"
    assert event_col is not None, "Missing event column"
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    return time_col, event_col

def select_stratifier(df, max_levels=4):
    # Pick a small-cardinality categorical feature for KM stratification
    candidates = [c for c in df.columns if df[c].dtype == "object"]
    for c in candidates:
        levels = df[c].dropna().unique()
        if 2 <= len(levels) <= max_levels:
            return c
    # Fallback: create a binned stratifier from a numeric feature
    num_cols = [c for c in df.columns if df[c].dtype != "object"]
    assert len(num_cols) > 0, "No numeric columns to bin for stratifier"
    q = pd.qcut(df[num_cols[0]], q=3, labels=["low", "mid", "high"])
    df["__binned__"] = q.astype(str)
    return "__binned__"

def preprocess_for_cox(df, time_col, event_col):
    feature_df = df.drop(columns=[time_col, event_col])
    categorical_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    X = pre.fit_transform(feature_df)
    X = np.asarray(X)
    dfX = pd.DataFrame(X)
    dfX["__time__"] = df[time_col].to_numpy()
    dfX["__event__"] = df[event_col].to_numpy()
    dfX = dfX.replace([np.inf, -np.inf], np.nan).dropna(subset=["__time__", "__event__"])
    dfX = dfX[dfX["__time__"] > 0]
    return dfX

def test_kaplan_meier_plot_saved():
    df = load_data()
    time_col, event_col = detect_survival_columns(df)
    strat_col = select_stratifier(df)

    # Ensure output dir exists
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, f"km_by_{strat_col}.png")

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6,4))
    for level in sorted(df[strat_col].dropna().unique()):
        mask = (df[strat_col] == level) & df[time_col].notna() & df[event_col].notna()
        if mask.sum() < 5:
            continue
        kmf.fit(durations=df.loc[mask, time_col].astype(float),
                event_observed=df.loc[mask, event_col].astype(int),
                label=str(level))
        kmf.plot(ci_show=False)
    plt.title(f"Kaplan–Meier survival by {strat_col}")
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    assert os.path.exists(out_path), f"KM figure not saved to {out_path}"

def test_cox_coefficients_barplot_saved():
    df = load_data()
    time_col, event_col = detect_survival_columns(df)
    dfX = preprocess_for_cox(df, time_col, event_col)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(dfX, duration_col="__time__", event_col="__event__")

    coef = cph.params_.sort_values(ascending=False)
    # Plot top 20 coefficients for readability
    topk = coef.head(20)
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "cox_top_coefficients.png")

    plt.figure(figsize=(7,5))
    topk.plot(kind="bar")
    plt.title("Top Cox coefficients")
    plt.xlabel("Feature")
    plt.ylabel("Coefficient")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    assert os.path.exists(out_path), f"Coefficient figure not saved to {out_path}"
