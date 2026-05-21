"""Select best experiment by OOF PR-AUC and retrain on full dev, evaluate test once."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from scipy.stats import binom

from src.load_secom import load_secom
from src.metrics_utils import compute_metrics, optimize_threshold
from src.models import build_model
from src.paths import ARTIFACTS_EXP, ARTIFACTS_STATS, CONFIGS, MODELS_DIR, REPORTS
from src.prep_pipelines import build_prep_pipeline
from src.run_experiments import run_single_experiment
from src.sampling import apply_sampling
from src.splits import load_splits
from src.stat_tests import run_statistical_comparison


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=500, random_state=42):
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_prob[idx]))
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def select_and_train(best_id: str | None = None, test_evaluated: bool = False):
    with open(CONFIGS / "cv_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lb = pd.read_csv(ARTIFACTS_EXP / "experiments_leaderboard.csv")
    lb = lb.dropna(subset=["pr_auc_mean"]).sort_values("pr_auc_mean", ascending=False)

    if best_id is None:
        stats_path = ARTIFACTS_STATS / "pvalues_matrix.csv"
        if stats_path.exists():
            stats = pd.read_csv(stats_path)
            sig = stats[stats["significant"].astype(str).str.lower() == "true"]
            if len(sig) > 0:
                best_id = sig.iloc[0]["experiment_id"]
        if best_id is None:
            # Prefer valid model/hp pairing on leaderboard top rows
            for _, row in lb.iterrows():
                eid = row["experiment_id"]
                mid = row.get("model_id", eid.split("__")[2] if "__" in eid else "")
                hid = row.get("hp_id", "")
                if mid == "M2_xgboost" and hid == "tune_lgbm":
                    continue
                if mid == "M3_lightgbm" and hid == "tune_xgb":
                    continue
                best_id = eid
                break
            if best_id is None:
                best_id = lb.iloc[0]["experiment_id"]

    parts = best_id.split("__")
    spec = {
        "experiment_id": best_id,
        "prep_id": parts[0],
        "sample_id": parts[1],
        "model_id": parts[2],
        "hp_id": parts[3],
        "hp": {},
    }
    if len(parts) > 4 and parts[4].startswith("hp"):
        with open(ARTIFACTS_EXP / f"{best_id}.json", encoding="utf-8") as f:
            saved = json.load(f)
            spec["hp"] = saved.get("hp", {})

    X, y = load_secom()
    dev_idx, test_idx, folds = load_splits()

    # Assert test not used before
    flag_path = MODELS_DIR / "test_evaluated.flag"
    if flag_path.exists() and not test_evaluated:
        raise RuntimeError("Test set already evaluated once. Delete flag to override.")

    X_dev, y_dev = X.iloc[dev_idx], y.iloc[dev_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    prep = build_prep_pipeline(spec["prep_id"], n_features_hint=X_dev.shape[1])
    X_dev_p = prep.fit_transform(X_dev, y_dev)
    X_test_p = prep.transform(X_test)

    X_tr_s, y_tr_s, _ = apply_sampling(X_dev_p, y_dev.values, spec["sample_id"])
    model = build_model(spec["model_id"], spec["sample_id"], y_train=y_tr_s, hp=spec["hp"])
    model.fit(X_tr_s, y_tr_s)

    prob_test = model.predict_proba(X_test_p)[:, 1]
    thr = optimize_threshold(y_dev.values, model.predict_proba(X_dev_p)[:, 1])
    test_metrics = compute_metrics(y_test.values, prob_test, threshold=thr)

    pr_lo, pr_hi = bootstrap_ci(
        y_test.values,
        prob_test,
        lambda yt, yp: compute_metrics(yt, yp, threshold=thr)["pr_auc"],
    )
    test_metrics["pr_auc_ci_low"] = pr_lo
    test_metrics["pr_auc_ci_high"] = pr_hi

    bundle = {
        "spec": spec,
        "prep": prep,
        "model": model,
        "threshold": thr,
        "test_metrics": test_metrics,
    }
    dump(bundle, MODELS_DIR / "final_model.joblib")
    with open(MODELS_DIR / "final_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {"best_id": best_id, "spec": spec, "threshold": thr, "test_metrics": test_metrics},
            f,
            indent=2,
            default=str,
        )

    pred_df = pd.DataFrame({
        "idx": test_idx,
        "y_true": y_test.values,
        "y_prob": prob_test,
        "y_pred": (prob_test >= thr).astype(int),
    })
    pred_df["error_type"] = "TN"
    pred_df.loc[(pred_df.y_true == 1) & (pred_df.y_pred == 1), "error_type"] = "TP"
    pred_df.loc[(pred_df.y_true == 0) & (pred_df.y_pred == 1), "error_type"] = "FP"
    pred_df.loc[(pred_df.y_true == 1) & (pred_df.y_pred == 0), "error_type"] = "FN"
    pred_df.to_csv(MODELS_DIR / "predictions_test.csv", index=False)

    flag_path.write_text("evaluated_once", encoding="utf-8")

    summary = REPORTS / "final_model_summary.md"
    summary.write_text(
        f"# Final Model\n\n"
        f"- Best experiment: `{best_id}`\n"
        f"- Test PR-AUC: {test_metrics['pr_auc']:.4f} "
        f"(95% bootstrap CI: {pr_lo:.4f}–{pr_hi:.4f})\n"
        f"- Test Recall: {test_metrics['recall']:.4f}\n"
        f"- Test F1: {test_metrics['f1']:.4f}\n",
        encoding="utf-8",
    )
    return bundle


if __name__ == "__main__":
    run_statistical_comparison()
    select_and_train()
