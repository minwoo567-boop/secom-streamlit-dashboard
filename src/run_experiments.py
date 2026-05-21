"""Run staged experiment grid with 5-fold OOF CV."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone

from src.load_secom import load_secom
from src.metrics_utils import compute_metrics, optimize_threshold
from src.models import build_model, expand_hp_grid
from src.paths import ARTIFACTS_EXP, ARTIFACTS_OOF, CONFIGS
from src.prep_pipelines import build_prep_pipeline
from src.sampling import apply_sampling
from src.splits import load_splits


def _experiment_id(prep_id, sample_id, model_id, hp_id, hp_suffix=""):
    base = f"{prep_id}__{sample_id}__{model_id}__{hp_id}"
    return f"{base}__{hp_suffix}" if hp_suffix else base


def _load_stage_config(stage_file: str):
    with open(CONFIGS / stage_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _hp_list_for_model(model_id: str, hp_id: str, hp_profiles: dict) -> list[dict]:
    if hp_id == "default":
        return [{}]
    if hp_id == "tune_xgb" and model_id != "M2_xgboost":
        return []
    if hp_id == "tune_lgbm" and model_id != "M3_lightgbm":
        return []
    return expand_hp_grid(model_id, hp_id, hp_profiles)


def _iter_experiments(stage_cfg: dict):
    # Explicit list mode (e.g. experiment_grid_quick.yaml — max N runs)
    if "experiments" in stage_cfg:
        for exp in stage_cfg["experiments"]:
            prep_id = exp["prep_id"]
            sample_id = exp["sample_id"]
            model_id = exp["model_id"]
            hp_id = exp.get("hp_id", "default")
            hp = exp.get("hp", {})
            yield {
                "stage": stage_cfg.get("stage_name", "quick"),
                "prep_id": prep_id,
                "sample_id": sample_id,
                "model_id": model_id,
                "hp_id": hp_id,
                "hp": hp,
                "experiment_id": _experiment_id(prep_id, sample_id, model_id, hp_id),
            }
        return

    hp_profiles = stage_cfg.get("hp_profiles", {})
    for stage in stage_cfg.get("stages", []):
        for prep_id, sample_id, model_id, hp_id in itertools.product(
            stage["prep_ids"],
            stage["sample_ids"],
            stage["model_ids"],
            stage["hp_ids"],
        ):
            hp_list = _hp_list_for_model(model_id, hp_id, hp_profiles)
            if not hp_list:
                continue
            for i, hp in enumerate(hp_list):
                suffix = f"hp{i}" if len(hp_list) > 1 else ""
                yield {
                    "stage": stage["name"],
                    "prep_id": prep_id,
                    "sample_id": sample_id,
                    "model_id": model_id,
                    "hp_id": hp_id,
                    "hp": hp,
                    "experiment_id": _experiment_id(prep_id, sample_id, model_id, hp_id, suffix),
                }


def run_single_experiment(X, y, folds, spec: dict, threshold_strategy="optimize_f1_val", random_state=42):
    exp_id = spec["experiment_id"]
    fold_metrics = []
    oof_prob = np.zeros(len(y))
    oof_mask = np.zeros(len(y), dtype=bool)

    for fold_i, fold in enumerate(folds):
        tr_idx = fold["train_idx"]
        va_idx = fold["val_idx"]

        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        prep = build_prep_pipeline(spec["prep_id"], n_features_hint=X_tr.shape[1])
        X_tr_p = prep.fit_transform(X_tr, y_tr)
        X_va_p = prep.transform(X_va)

        X_tr_s, y_tr_s, samp_info = apply_sampling(
            X_tr_p, y_tr.values, spec["sample_id"], random_state=random_state
        )

        model = build_model(
            spec["model_id"],
            spec["sample_id"],
            y_train=y_tr_s,
            hp=spec["hp"],
            random_state=random_state,
        )
        model.fit(X_tr_s, y_tr_s)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_va_p)[:, 1]
        else:
            prob = model.predict(X_va_p).astype(float)

        oof_prob[va_idx] = prob
        oof_mask[va_idx] = True

        thr = optimize_threshold(y_va.values, prob, strategy=threshold_strategy)
        m = compute_metrics(y_va.values, prob, threshold=thr)
        m["fold"] = fold_i
        m["sampling"] = samp_info
        fold_metrics.append(m)

    dev_idx = np.concatenate([f["train_idx"] for f in folds[:1]] + [f["val_idx"] for f in folds])
    # OOF only on val portions
    y_dev_oof = y.iloc[np.where(oof_mask)[0]]
    prob_dev_oof = oof_prob[oof_mask]

    thr_global = optimize_threshold(y_dev_oof.values, prob_dev_oof, strategy=threshold_strategy)
    agg = {
        "experiment_id": exp_id,
        "stage": spec["stage"],
        "prep_id": spec["prep_id"],
        "sample_id": spec["sample_id"],
        "model_id": spec["model_id"],
        "hp_id": spec["hp_id"],
        "hp": spec["hp"],
        "pr_auc_mean": float(np.mean([m["pr_auc"] for m in fold_metrics])),
        "pr_auc_std": float(np.std([m["pr_auc"] for m in fold_metrics])),
        "recall_mean": float(np.mean([m["recall"] for m in fold_metrics])),
        "f1_mean": float(np.mean([m["f1"] for m in fold_metrics])),
        "balanced_acc_mean": float(np.mean([m["balanced_acc"] for m in fold_metrics])),
        "fold_metrics": fold_metrics,
        "oof_threshold": thr_global,
    }
    for k in ["pr_auc", "recall", "f1", "balanced_acc"]:
        agg[f"{k}_folds"] = [m[k] for m in fold_metrics]

    oof_df = pd.DataFrame({
        "idx": np.where(oof_mask)[0],
        "y_true": y_dev_oof.values,
        "y_prob": prob_dev_oof,
    })
    oof_df.to_parquet(ARTIFACTS_OOF / f"{exp_id}.parquet", index=False)

    with open(ARTIFACTS_EXP / f"{exp_id}.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=str)

    return agg


def run_stages(stage_files=None, existing_skip=True):
    if stage_files is None:
        stage_files = ["experiment_grid_stage1.yaml", "experiment_grid_stage2.yaml"]

    X, y = load_secom()
    _, _, folds = load_splits()

    all_results = []
    seen_ids = set()

    for sf in stage_files:
        cfg = _load_stage_config(sf)
        for spec in _iter_experiments(cfg):
            eid = spec["experiment_id"]
            if eid in seen_ids:
                continue
            seen_ids.add(eid)

            out_path = ARTIFACTS_EXP / f"{eid}.json"
            if existing_skip and out_path.exists():
                with open(out_path, encoding="utf-8") as f:
                    all_results.append(json.load(f))
                continue

            print(f"Running {eid}...")
            try:
                agg = run_single_experiment(X, y, folds, spec)
                all_results.append(agg)
            except Exception as e:
                print(f"  FAILED {eid}: {e}")
                err = {**spec, "error": str(e), "experiment_id": eid}
                all_results.append(err)
                with open(ARTIFACTS_EXP / f"{eid}_error.json", "w", encoding="utf-8") as f:
                    json.dump(err, f, indent=2)

    rows = []
    for r in all_results:
        if "error" in r:
            continue
        rows.append({
            "experiment_id": r["experiment_id"],
            "stage": r.get("stage"),
            "prep_id": r.get("prep_id"),
            "sample_id": r.get("sample_id"),
            "model_id": r.get("model_id"),
            "hp_id": r.get("hp_id"),
            "pr_auc_mean": r.get("pr_auc_mean"),
            "pr_auc_std": r.get("pr_auc_std"),
            "recall_mean": r.get("recall_mean"),
            "f1_mean": r.get("f1_mean"),
            "balanced_acc_mean": r.get("balanced_acc_mean"),
        })
    lb = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    lb.to_csv(ARTIFACTS_EXP / "experiments_leaderboard.csv", index=False)
    return lb


if __name__ == "__main__":
    run_stages()
