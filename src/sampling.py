"""Imbalance sampling — train fold only."""
from __future__ import annotations

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from imblearn.under_sampling import RandomUnderSampler


def apply_sampling(X, y, sample_id: str, random_state=42):
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if sample_id in ("S0_none", "S1_weight"):
        return X, y, {"applied": sample_id, "resampled": False}

    if n_pos < 2:
        return X, y, {"applied": sample_id, "resampled": False, "skip": "too_few_positives"}

    k = min(3, n_pos - 1)
    rs = random_state

    try:
        if sample_id == "S2_smote":
            sampler = SMOTE(random_state=rs, k_neighbors=k)
        elif sample_id == "S3_adasyn":
            sampler = ADASYN(random_state=rs, n_neighbors=k)
        elif sample_id == "S4_borderline":
            sampler = BorderlineSMOTE(random_state=rs, k_neighbors=k)
        elif sample_id == "S5_under_smote":
            rus = RandomUnderSampler(
                sampling_strategy={0: min(n_neg, n_pos * 3), 1: n_pos},
                random_state=rs,
            )
            X_u, y_u = rus.fit_resample(X, y)
            smote = SMOTE(random_state=rs, k_neighbors=min(k, int((y_u == 1).sum()) - 1) or 1)
            X_r, y_r = smote.fit_resample(X_u, y_u)
            return X_r, y_r, {"applied": sample_id, "resampled": True}
        else:
            return X, y, {"applied": sample_id, "resampled": False}

        X_r, y_r = sampler.fit_resample(X, y)
        return X_r, y_r, {"applied": sample_id, "resampled": True}
    except Exception as e:
        return X, y, {"applied": sample_id, "resampled": False, "error": str(e)}
