"""Load and validate SECOM dataset from Kaggle CSV."""
from __future__ import annotations

import pandas as pd

from src.paths import DATA_RAW

CSV_NAME = "uci-secom.csv"
LABEL_COL = "Pass/Fail"
TIME_COL = "Time"


def load_secom(csv_path=None) -> tuple[pd.DataFrame, pd.Series]:
    path = csv_path or (DATA_RAW / CSV_NAME)
    if not path.exists():
        raise FileNotFoundError(f"SECOM data not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected column '{LABEL_COL}' in {path}")

    labels = df[LABEL_COL].astype(str).str.strip()
    valid = labels.isin(["-1", "1"])
    if not valid.all():
        bad = labels[~valid].unique()[:10]
        raise ValueError(f"Invalid labels found: {bad}")

    y = (labels == "1").astype(int)
    feature_cols = [c for c in df.columns if c not in (TIME_COL, LABEL_COL)]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    return X, y


def load_with_metadata(csv_path=None) -> pd.DataFrame:
    path = csv_path or (DATA_RAW / CSV_NAME)
    df = pd.read_csv(path, low_memory=False)
    df["y"] = (df[LABEL_COL].astype(str).str.strip() == "1").astype(int)
    return df
