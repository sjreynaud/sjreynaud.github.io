import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA_PATH = os.environ.get("DATA_PATH", "/content/METABRIC_cleaned_imputed.csv")
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
    assert time_col is not None, f"Missing time column among {TIME_CANDIDATES}"
    assert event_col is not None, f"Missing event column among {EVENT_CANDIDATES}"
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    return time_col, event_col

def build_preprocessor(df, time_col, event_col):
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
    return pre, numeric_cols, categorical_cols

def to_cox_frame(X, time, event):
    # lifelines expects a DataFrame with numeric columns + time + event
    dfX = pd.DataFrame(X)
    dfX["__time__"] = time
    dfX["__event__"] = event
    # Remove rows with non-positive time or missing
    dfX = dfX.replace([np.inf, -np.inf], np.nan).dropna(subset=["__time__", "__event__"])
    dfX = dfX[dfX["__time__"] > 0]
    return dfX

def test_cox_training_and_cindex():
    df = load_data()
    time_col, event_col = detect_survival_columns(df)
    pre, num_cols, cat_cols = build_preprocessor(df, time_col, event_col)

    X = pre.fit_transform(df[num_cols + cat_cols])
    y_time = df[time_col].to_numpy()
    y_event = df[event_col].to_numpy()

    X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
        X, y_time, y_event, test_size=0.25, random_state=42, stratify=y_event
    )

    train_df = to_cox_frame(X_train, t_train, e_train)
    test_df = to_cox_frame(X_test, t_test, e_test)

    cph = CoxPHFitter(penalizer=0.1)  # light ridge for stability
    cph.fit(train_df, duration_col="__time__", event_col="__event__")

    # Concordance should be within [0,1] and ideally above a weak baseline
    c_index = cph.concordance_index_
    assert 0.0 <= c_index <= 1.0, f"Invalid c-index: {c_index}"
    assert c_index >= 0.55, f"c-index too low ({c_index}); check features or preprocessing"

    # Evaluate on test set (not a strict threshold, but should compute without error)
    test_score = cph.score(test_df, scoring_method="concordance_index")
    assert 0.0 <= test_score <= 1.0, "Test concordance index out of bounds"

def test_reproducibility_fixed_seed():
    df = load_data()
    time_col, event_col = detect_survival_columns(df)
    pre, num_cols, cat_cols = build_preprocessor(df, time_col, event_col)

    X = pre.fit_transform(df[num_cols + cat_cols])
    y_time = df[time_col].to_numpy()
    y_event = df[event_col].to_numpy()

    split1 = train_test_split(X, y_time, y_event, test_size=0.25, random_state=123, stratify=y_event)
    split2 = train_test_split(X, y_time, y_event, test_size=0.25, random_state=123, stratify=y_event)

    X_train1, _, t_train1, _, e_train1, _ = split1
    X_train2, _, t_train2, _, e_train2, _ = split2

    # Same seed => same partitions
    assert np.array_equal(X_train1, X_train2), "Train features differ despite fixed seed"
    assert np.array_equal(t_train1, t_train2), "Train times differ despite fixed seed"
    assert np.array_equal(e_train1, e_train2), "Train events differ despite fixed seed"
