"""Model factory by model_id."""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def build_model(model_id: str, sample_id: str, y_train=None, hp: dict | None = None, random_state=42):
    hp = hp or {}
    cw = "balanced" if sample_id == "S1_weight" else None

    if model_id == "B0_dummy":
        return DummyClassifier(strategy="stratified", random_state=random_state)

    if model_id == "B1_logistic_l2":
        return LogisticRegression(
            penalty="l2",
            C=hp.get("C", 1.0),
            solver="saga",
            max_iter=2000,
            class_weight=cw,
            random_state=random_state,
        )

    if model_id == "B2_logistic_l1":
        return LogisticRegression(
            penalty="l1",
            C=hp.get("C", 1.0),
            solver="saga",
            max_iter=2000,
            class_weight=cw,
            random_state=random_state,
        )

    if model_id == "M1_random_forest":
        return RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 300),
            max_depth=hp.get("max_depth", 15),
            min_samples_leaf=hp.get("min_samples_leaf", 2),
            class_weight=cw,
            random_state=random_state,
            n_jobs=-1,
        )

    scale_pos = 1.0
    if y_train is not None and model_id in ("M2_xgboost", "M3_lightgbm"):
        y_train = np.asarray(y_train)
        n_pos = max((y_train == 1).sum(), 1)
        n_neg = max((y_train == 0).sum(), 1)
        scale_pos = n_neg / n_pos

    if model_id == "M2_xgboost":
        spw = hp.get("scale_pos_weight", scale_pos)
        return XGBClassifier(
            max_depth=hp.get("max_depth", 5),
            learning_rate=hp.get("learning_rate", 0.05),
            n_estimators=hp.get("n_estimators", 400),
            subsample=hp.get("subsample", 0.8),
            colsample_bytree=hp.get("colsample_bytree", 0.8),
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_id == "M3_lightgbm":
        return LGBMClassifier(
            num_leaves=hp.get("num_leaves", 31),
            learning_rate=hp.get("learning_rate", 0.05),
            n_estimators=hp.get("n_estimators", 400),
            min_child_samples=hp.get("min_child_samples", 10),
            class_weight=cw,
            is_unbalance=hp.get("is_unbalance", False),
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_id: {model_id}")


def expand_hp_grid(model_id: str, hp_id: str, hp_profiles: dict | None = None) -> list[dict]:
    if hp_id == "default":
        return [{}]

    profiles = hp_profiles or {}
    if hp_id not in profiles:
        return [{}]

    grid = profiles[hp_id]
    keys = list(grid.keys())
    combos: list[dict] = []

    def _expand(acc: dict, idx: int):
        if idx == len(keys):
            combos.append(acc.copy())
            return
        k = keys[idx]
        for v in grid[k]:
            acc[k] = v
            _expand(acc, idx + 1)

    _expand({}, 0)
    return combos if combos else [{}]
