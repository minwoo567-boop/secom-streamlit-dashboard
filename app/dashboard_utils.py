# -*- coding: utf-8 -*-
"""Helpers for Streamlit dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.inspection import permutation_importance
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]


def load_paths():
    return {
        "root": ROOT,
        "exp": ROOT / "artifacts" / "experiments",
        "stats": ROOT / "artifacts" / "stats",
        "models": ROOT / "models",
        "reports": ROOT / "reports",
        "eda": ROOT / "artifacts" / "eda",
    }


def load_leaderboard_with_stats() -> pd.DataFrame:
    paths = load_paths()
    lb = pd.read_csv(paths["exp"] / "experiments_leaderboard.csv")
    sp = paths["stats"] / "pvalues_matrix.csv"
    if sp.exists():
        stats = pd.read_csv(sp)
        lb = lb.merge(stats, on="experiment_id", how="left")
    else:
        lb["p_adj"] = np.nan
        lb["significant"] = False
        lb["mean_delta"] = np.nan
    return lb.sort_values("pr_auc_mean", ascending=False)


def load_final_bundle():
    paths = load_paths()
    cfg = json.loads((paths["models"] / "final_config.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(paths["models"] / "predictions_test.csv")
    bundle = None
    mp = paths["models"] / "final_model.joblib"
    if mp.exists():
        bundle = load(mp)
    return cfg, preds, bundle


def load_raw_data():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.load_secom import load_secom, load_with_metadata
    from src.splits import load_splits

    X, y = load_secom()
    meta = load_with_metadata()
    _, test_idx, _ = load_splits()
    return X, y, meta, test_idx


def get_feature_importance(n_repeats=8, max_samples=800):
    import sys
    sys.path.insert(0, str(ROOT))
    from sklearn.pipeline import Pipeline

    from src.models import build_model
    from src.prep_pipelines import build_prep_pipeline
    from src.sampling import apply_sampling
    from src.splits import load_splits

    cfg, _, _ = load_final_bundle()
    spec = cfg["spec"]
    X, y, _, _ = load_raw_data()
    dev_idx, _, _ = load_splits()
    X_dev, y_dev = X.iloc[dev_idx], y.iloc[dev_idx]

    if len(X_dev) > max_samples:
        rng = np.random.RandomState(42)
        sub = rng.choice(len(X_dev), max_samples, replace=False)
        X_dev = X_dev.iloc[sub]
        y_dev = y_dev.iloc[sub]

    from sklearn.base import clone

    prep = build_prep_pipeline(spec["prep_id"], n_features_hint=X_dev.shape[1])
    model = build_model(
        spec["model_id"], spec["sample_id"], y_train=y_dev.values, hp=spec.get("hp", {})
    )
    pipe = Pipeline([("prep", prep), ("model", clone(model))])
    pipe.fit(X_dev, y_dev)

    perm = permutation_importance(
        pipe,
        X_dev,
        y_dev,
        n_repeats=n_repeats,
        random_state=42,
        scoring="average_precision",
        n_jobs=1,
    )
    imp = pd.DataFrame({
        "feature": X_dev.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return imp, spec


def compare_feature_groups(X: pd.DataFrame, y: pd.Series, importance: pd.DataFrame, top_k=15, bottom_k=15):
    top_feats = importance.head(top_k)["feature"].tolist()
    bot_feats = importance.tail(bottom_k)["feature"].tolist()
    rows = []
    for group_name, feats in [("high_importance", top_feats), ("low_importance", bot_feats)]:
        for f in feats:
            if f not in X.columns:
                continue
            a = X.loc[y == 0, f].dropna()
            b = X.loc[y == 1, f].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            try:
                _, pval = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                pval = 1.0
            rows.append({
                "group": group_name,
                "feature": f,
                "pass_mean": a.mean(),
                "fail_mean": b.mean(),
                "pass_std": a.std(),
                "fail_std": b.std(),
                "mean_diff": abs(a.mean() - b.mean()),
                "p_value": pval,
                "pass_missing_pct": X.loc[y == 0, f].isna().mean(),
                "fail_missing_pct": X.loc[y == 1, f].isna().mean(),
            })
    return pd.DataFrame(rows)


def enrich_predictions(preds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    thr = cfg.get("threshold", 0.5)
    out = preds.copy()
    out["margin"] = np.where(
        out["y_true"] == 1,
        thr - out["y_prob"],  # FN: how far below threshold
        out["y_prob"] - thr,  # FP: how far above
    )
    out["abs_margin"] = (out["y_prob"] - thr).abs()
    out["confidence"] = np.where(out["y_pred"] == 1, out["y_prob"], 1 - out["y_prob"])
    out["is_error"] = out["error_type"].isin(["FN", "FP"])
    out["near_threshold"] = out["is_error"] & (out["abs_margin"] < 0.08)
    out["high_confidence_error"] = out["is_error"] & (out["confidence"] > 0.7)
    return out


def label_noise_candidates(X: pd.DataFrame, y: pd.Series, error_preds: pd.DataFrame, k=5) -> pd.DataFrame:
    """kNN label consistency on misclassified test samples."""
    if error_preds.empty:
        return pd.DataFrame(columns=["idx", "reason", "labels"])

    idx = error_preds["idx"].values
    X_sub = X.loc[idx].fillna(X.loc[idx].median())
    suspects = []

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X_sub)))
    nn.fit(X_sub.values)
    _, nbrs = nn.kneighbors(X_sub.values)

    for j, row_idx in enumerate(idx):
        neighbors = [X.index[i] for i in nbrs[j, 1:]]
        neighbor_labels = y.loc[neighbors].values
        own = y.loc[row_idx]
        disagree = (neighbor_labels != own).mean()
        if disagree >= 0.6:
            suspects.append({
                "idx": int(row_idx),
                "reason": f"knn_neighbor_disagree_{disagree:.0%}",
                "labels": f"true={own}, neighbors={neighbor_labels.tolist()[:5]}",
            })

    # Global duplicate rows with conflicting labels (full dataset)
    dup = X.duplicated(keep=False)
    if dup.any():
        for _, grp in X[dup].groupby(X[dup].apply(lambda r: hash(tuple(r.fillna(-999).values[:30])), axis=1)):
            labs = y.loc[grp.index].unique()
            if len(labs) > 1:
                for i in grp.index:
                    if i in idx:
                        suspects.append({
                            "idx": int(i),
                            "reason": "duplicate_row_conflicting_label",
                            "labels": str(labs.tolist()),
                        })

    return pd.DataFrame(suspects).drop_duplicates(subset=["idx"]) if suspects else pd.DataFrame(
        columns=["idx", "reason", "labels"]
    )


def method_labels():
    return {
        "P0_minimal": "결측 중앙값 대체만",
        "P2_drop40_median": "결측 40% 초과 컬럼 제거 + 중앙값 대체",
        "P4_drop40_winsor": "결측 40% 제거 + Winsor(1–99%) + RobustScaler",
        "P6_full_mi40": "결측 40% 제거 + Winsor + MI 상위 40특성 선택",
        "S0_none": "샘플링 없음",
        "S1_weight": "class_weight=balanced",
        "S2_smote": "SMOTE (train fold only)",
        "B0_dummy": "Stratified Dummy",
        "B1_logistic_l2": "Logistic Regression L2",
        "M1_random_forest": "Random Forest",
        "M2_xgboost": "XGBoost",
        "M3_lightgbm": "LightGBM",
    }
