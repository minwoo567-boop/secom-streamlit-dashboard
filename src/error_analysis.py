"""FN/FP profiling and improvement roadmap."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.load_secom import load_secom
from src.paths import MODELS_DIR, REPORTS


def run_error_analysis():
    pred_path = MODELS_DIR / "predictions_test.csv"
    if not pred_path.exists():
        raise FileNotFoundError("Run select_best.py first.")

    preds = pd.read_csv(pred_path)
    X, y = load_secom()
    X = X.loc[preds["idx"]]

    with open(MODELS_DIR / "final_config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    fn = preds[preds["error_type"] == "FN"]
    fp = preds[preds["error_type"] == "FP"]
    tp = preds[preds["error_type"] == "TP"]
    tn = preds[preds["error_type"] == "TN"]

    # Feature profile: mean abs diff FN vs TP on test
    lines = [
        "# Error Analysis Report\n\n",
        f"Final model: `{cfg.get('best_id')}`\n\n",
        "## Confusion summary (hold-out test)\n\n",
        f"| Type | Count |\n|------|-------|\n",
        f"| TP | {len(tp)} |\n| TN | {len(tn)} |\n| FP | {len(fp)} |\n| FN | {len(fn)} |\n\n",
        "## Business interpretation\n\n",
        "- **FN (missed defects)**: Highest operational risk — wafer passes inspection but is actually fail.\n",
        "- **FP (false alarm)**: Triggers unnecessary rework or line stop.\n\n",
    ]

    if len(fn) > 0 and len(tp) > 0:
        X_fn = X.loc[fn["idx"].values].astype(float)
        X_tp = X.loc[tp["idx"].values].astype(float)
        diff = (X_fn.mean() - X_tp.mean()).abs().sort_values(ascending=False)
        top = diff.head(15)
        lines.append("## Top sensor gaps (FN mean vs TP mean)\n\n")
        lines.append("| Feature | |mean_diff| |\n|---------|------------|\n")
        for feat, val in top.items():
            lines.append(f"| {feat} | {val:.4f} |\n")

    lines.extend([
        "\n## Model improvement roadmap\n\n",
        "1. **Threshold tuning**: Lower cutoff if FN cost dominates (see Streamlit slider).\n",
        "2. **Cost-sensitive learning**: Increase `scale_pos_weight` or FN-weighted loss.\n",
        "3. **Feature engineering**: Missingness indicators per sensor group; rolling stats if timestamp used.\n",
        "4. **Ensemble**: Combine top-2 OOF models (e.g. LightGBM + XGBoost) if statistically significant.\n",
        "5. **Sampling**: If FN remain high, compare S2/S4 SMOTE variants on new production window.\n",
    ])

    sig_path = REPORTS.parent / "reports" / "statistical_comparison.md"
    if (REPORTS / "statistical_comparison.md").exists():
        lines.append("\n## Statistically validated factors\n\n")
        lines.append("See `reports/statistical_comparison.md` for prep/sample/model combos "
                     "that significantly beat baseline.\n")

    out = REPORTS / "error_analysis.md"
    out.write_text("".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    run_error_analysis()
