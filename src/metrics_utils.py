"""Classification metrics and threshold tuning."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def optimize_threshold(y_true, y_prob, strategy="optimize_f1_val") -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if strategy == "fixed_0.5":
        return 0.5

    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_score = 0.5, -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if strategy == "optimize_f1_val":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif strategy == "optimize_recall_at_precision_0.3":
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            score = rec if prec >= 0.3 else -1.0
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t
