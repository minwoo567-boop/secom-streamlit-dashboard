"""Evaluate saved final model on hold-out test (single use)."""
from __future__ import annotations

import json

from joblib import load

from src.paths import MODELS_DIR


def evaluate_final():
    flag = MODELS_DIR / "test_evaluated.flag"
    bundle = load(MODELS_DIR / "final_model.joblib")
    with open(MODELS_DIR / "final_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "best_id": cfg.get("best_id"),
        "test_metrics": cfg.get("test_metrics"),
        "threshold": cfg.get("threshold"),
        "evaluated": flag.exists(),
    }


if __name__ == "__main__":
    print(evaluate_final())
