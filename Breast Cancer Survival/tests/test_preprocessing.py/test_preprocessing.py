import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA_PATH = os.environ.get("DATA_PATH", "/content/METABRIC_cleaned_imputed.csv")

# Common candidate names across METABRIC derivatives
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
    assert time_col is not None, f"Could not find a time column among: {TIME_CANDIDATES}"
    assert event_col is not None, f"Could not find an event column among: {EVENT_CANDIDATES}"
    # Ensure numeric types
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    return time_col, event_col

def build_preprocessing(df, time_col, event_col):
    # Exclude survival columns from features
    feature_df = df.drop(columns=[time_col, event_col])
    # Basic dtype detection
    categorical_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

    # Pipelines
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
    return pre, numeric_cols, categorical_cols

def get_feature_names(pre, numeric_cols, categorical_cols, df):
    pre.fit(df[numeric_cols + categorical_cols])
    num_names = numeric_cols
    cat_encoder = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    return num_names + cat_feature_names

def test_load_and_schema():
    df = load_data()
    # Basic sanity checks
    assert df.isnull().sum().sum() >= 0, "Null check failed"
    assert df.shape[1] > 5, "Too few columns to be METABRIC-like"
    time_col, event_col = detect_survival_columns(df)
    # Event should be binary {0,1} in most cleaned datasets
    unique_events = set(df[event_col].dropna().unique())
    assert unique_events.issubset({0,1}), f"Event column not binary: {unique_events}"

def test_preprocessing_pipeline_runs():
    df = load_data()
    time_col, event_col = detect_survival_columns(df)
    pre, num_cols, cat_cols = build_preprocessing(df, time_col, event_col)
    # Fit-transform features only
    X = pre.fit_transform(df[num_cols + cat_cols])
    assert isinstance(X, np.ndarray), "Preprocessing did not produce a dense ndarray"
    assert X.shape[0] == df.shape[0], "Row count changed unexpectedly"
    assert not np.isnan(X).any(), "Preprocessed matrix contains NaNs"

def test_feature_name_reconstruction():
    df = load_data()
    time_col, event_col = detect_survival_columns(df)
    pre, num_cols, cat_cols = build_preprocessing(df, time_col, event_col)
    names = get_feature_names(pre, num_cols, cat_cols, df)
    assert len(names) > 0, "No feature names reconstructed"
    # Names should be unique and interpretable (e.g., col_value for one-hots)
    assert len(names) == len(set(names)), "Feature names are not unique"
