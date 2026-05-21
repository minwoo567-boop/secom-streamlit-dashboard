"""Fixed train/dev/test and CV fold indices."""
from __future__ import annotations

import pickle

import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.load_secom import load_secom
from src.paths import CONFIGS, DATA_SPLITS


def _load_cv_config():
    with open(CONFIGS / "cv_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_and_save_splits(force=False):
    cfg = _load_cv_config()
    rs = cfg["random_state"]
    test_size = cfg["test_size"]
    n_splits = cfg["n_splits"]

    test_path = DATA_SPLITS / "test_indices.pkl"
    fold_path = DATA_SPLITS / "fold_indices.pkl"
    dev_path = DATA_SPLITS / "dev_indices.pkl"

    if test_path.exists() and fold_path.exists() and not force:
        return

    X, y = load_secom()
    n = len(y)
    all_idx = np.arange(n)

    dev_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        stratify=y,
        random_state=rs,
    )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=cfg.get("shuffle", True),
        random_state=rs,
    )
    folds = []
    y_dev = y.iloc[dev_idx].values
    for train_rel, val_rel in skf.split(dev_idx, y_dev):
        folds.append({
            "train_idx": dev_idx[train_rel],
            "val_idx": dev_idx[val_rel],
        })

    with open(test_path, "wb") as f:
        pickle.dump(test_idx, f)
    with open(dev_path, "wb") as f:
        pickle.dump(dev_idx, f)
    with open(fold_path, "wb") as f:
        pickle.dump(folds, f)

    meta = {
        "n_total": n,
        "n_dev": len(dev_idx),
        "n_test": len(test_idx),
        "n_fail_total": int(y.sum()),
        "random_state": rs,
    }
    with open(DATA_SPLITS / "split_meta.yaml", "w", encoding="utf-8") as f:
        yaml.dump(meta, f)


def load_splits():
    create_and_save_splits()
    with open(DATA_SPLITS / "test_indices.pkl", "rb") as f:
        test_idx = pickle.load(f)
    with open(DATA_SPLITS / "dev_indices.pkl", "rb") as f:
        dev_idx = pickle.load(f)
    with open(DATA_SPLITS / "fold_indices.pkl", "rb") as f:
        folds = pickle.load(f)
    return dev_idx, test_idx, folds
