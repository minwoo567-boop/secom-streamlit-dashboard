"""Preprocessing pipeline factory by prep_id."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.base import BaseEstimator, TransformerMixin


class DropHighMissingColumns(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.keep_cols_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        miss = X.isna().mean(axis=0)
        self.keep_cols_ = miss[miss <= self.threshold].index.tolist()
        if len(self.keep_cols_) == 0:
            self.keep_cols_ = list(X.columns[: min(10, X.shape[1])])
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return X[self.keep_cols_].values


class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.bounds_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.bounds_ = (
            np.nanquantile(X, self.lower, axis=0),
            np.nanquantile(X, self.upper, axis=0),
        )
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        lo, hi = self.bounds_
        return np.clip(X, lo, hi)


class IQRClipTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanquantile(X, 0.25, axis=0)
        q3 = np.nanquantile(X, 0.75, axis=0)
        iqr = q3 - q1
        self.bounds_ = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        lo, hi = self.bounds_
        return np.clip(X, lo, hi)


def build_prep_pipeline(prep_id: str, n_features_hint=500) -> Pipeline:
    steps = []

    if prep_id in ("P1_drop20_median", "P2_drop40_median", "P3_drop40_knn",
                   "P4_drop40_winsor", "P5_drop40_iqr", "P6_full_mi40",
                   "P7_full_mi80", "P8_miss_indicator"):
        thr = 0.2 if prep_id == "P1_drop20_median" else 0.4
        steps.append(("drop_missing", DropHighMissingColumns(threshold=thr)))

    imputer = SimpleImputer(strategy="median")
    steps.append(("impute", imputer))

    if prep_id == "P4_drop40_winsor" or prep_id in ("P6_full_mi40", "P7_full_mi80"):
        steps.append(("winsor", WinsorizeTransformer()))
    elif prep_id == "P5_drop40_iqr":
        steps.append(("iqr_clip", IQRClipTransformer()))

    if prep_id in ("P1_drop20_median", "P2_drop40_median", "P3_drop40_knn",
                   "P4_drop40_winsor", "P6_full_mi40", "P7_full_mi80", "P8_miss_indicator"):
        steps.append(("scale", RobustScaler()))
    elif prep_id == "P5_drop40_iqr":
        steps.append(("scale", StandardScaler()))

    if prep_id == "P2_drop40_median":
        steps.append(("variance", VarianceThreshold(threshold=0.0)))

    if prep_id == "P6_full_mi40":
        steps.append(("select_k", SelectKBest(mutual_info_classif, k=min(40, n_features_hint))))
    elif prep_id == "P7_full_mi80":
        steps.append(("select_k", SelectKBest(mutual_info_classif, k=min(80, n_features_hint))))
    elif prep_id == "P8_miss_indicator":
        steps.append(("select_k", SelectKBest(mutual_info_classif, k=min(40, n_features_hint))))

    if not steps:
        steps = [("impute", SimpleImputer(strategy="median"))]

    return Pipeline(steps)
