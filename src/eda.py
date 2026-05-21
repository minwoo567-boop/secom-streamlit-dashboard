"""EDA and data quality reports for SECOM."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from src.load_secom import load_secom, load_with_metadata
from src.paths import ARTIFACTS_EDA, REPORTS
from src.splits import create_and_save_splits, load_splits


def run_eda():
    create_and_save_splits()
    X, y = load_secom()
    df = load_with_metadata()
    dev_idx, test_idx, folds = load_splits()

    n, p = X.shape
    n_fail = int(y.sum())
    miss_col = X.isna().mean()
    miss_row = X.isna().mean(axis=1)
    const_cols = (X.std(skipna=True) == 0).sum()
    dup_mask = X.duplicated(keep=False)

    quality_items = []

    def add_item(type_, severity, count, rule, disposition, experiment_ids):
        quality_items.append({
            "type": type_,
            "severity": severity,
            "count": int(count),
            "rule": rule,
            "disposition": disposition,
            "experiment_ids": experiment_ids,
        })

    add_item("class_imbalance", "Major", n_fail,
             f"Fail rate {n_fail/n:.2%}", "compare_in_grid",
             ["S0_none", "S1_weight", "S2_smote", "S3_adasyn"])
    add_item("high_missing_columns", "Major", int((miss_col > 0.4).sum()),
             "columns with >40% missing", "compare_in_grid",
             ["P1_drop20_median", "P2_drop40_median", "P6_full_mi40"])
    add_item("constant_columns", "Major", int(const_cols),
             "zero variance columns", "transform",
             ["P2_drop40_median", "P6_full_mi40"])
    if (miss_row > 0.5).any():
        add_item("high_missing_rows", "Major", int((miss_row > 0.5).sum()),
                 "rows with >50% missing", "compare_in_grid", ["prep_row_drop50"])
    else:
        add_item("high_missing_rows", "Minor", 0, "none above 50%", "log_only", [])

    fold_fail_min = []
    for fold in folds:
        yv = y.iloc[fold["val_idx"]]
        fold_fail_min.append(int((yv == 1).sum()))
    min_fold_fail = min(fold_fail_min)
    sev = "Minor" if min_fold_fail >= 10 else "Major"
    add_item("fold_fail_count", sev, min_fold_fail,
             "min Fail in CV val fold", "compare_in_grid" if sev == "Major" else "log_only",
             ["S1_weight", "S2_smote"])

    if dup_mask.any():
        add_item("duplicate_rows", "Major", int(dup_mask.sum()),
                 "duplicate feature rows", "compare_in_grid", ["prep_dedup_fail_priority"])
    else:
        add_item("duplicate_rows", "Minor", 0, "no full duplicates", "log_only", [])

    # Mann-Whitney top features
    top_feats = []
    for col in X.columns[:200]:  # sample for speed
        a = X.loc[y == 0, col].dropna()
        b = X.loc[y == 1, col].dropna()
        if len(a) > 10 and len(b) > 5:
            try:
                _, pval = mannwhitneyu(a, b, alternative="two-sided")
                top_feats.append((col, pval))
            except ValueError:
                pass
    top_feats.sort(key=lambda x: x[1])
    top10 = top_feats[:10]

    # Figures
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    y.value_counts().plot(kind="bar", ax=axes[0, 0], color=["#4c78a8", "#e45756"])
    axes[0, 0].set_title("Pass (0) vs Fail (1)")
    axes[0, 0].set_xticklabels(["Pass", "Fail"], rotation=0)

    miss_col.hist(bins=50, ax=axes[0, 1], color="#72b7b2")
    axes[0, 1].set_title("Column missing rate distribution")
    axes[0, 1].axvline(0.4, color="red", linestyle="--", label="40%")
    axes[0, 1].legend()

    miss_row.hist(bins=50, ax=axes[1, 0], color="#f58518")
    axes[1, 0].set_title("Row missing rate distribution")

    if top10:
        cols_plot = [t[0] for t in top10[:3]]
        for i, c in enumerate(cols_plot):
            X.loc[y == 0, c].dropna().plot(
                kind="density", ax=axes[1, 1], label=f"Pass-{c}", alpha=0.6
            )
            X.loc[y == 1, c].dropna().plot(
                kind="density", ax=axes[1, 1], label=f"Fail-{c}", alpha=0.6
            )
        axes[1, 1].set_title("Top discriminative features (density)")
        axes[1, 1].legend(fontsize=7)
    plt.tight_layout()
    fig_path = ARTIFACTS_EDA / "eda_overview.png"
    fig.savefig(fig_path, dpi=120)
    plt.close()

    report_json = {
        "n_rows": n,
        "n_features": p,
        "n_fail": n_fail,
        "fail_rate": n_fail / n,
        "n_missing_cols_gt_40pct": int((miss_col > 0.4).sum()),
        "n_constant_cols": int(const_cols),
        "min_fold_val_fail": min_fold_fail,
        "quality_items": quality_items,
        "top_mannwhitney_features": [{"feature": f, "p_value": float(p)} for f, p in top10],
    }
    with open(ARTIFACTS_EDA / "eda_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)

    decision_rows = [
        ("결측률>40% 컬럼 다수", "Major", "compare_in_grid", "P2_drop40_median, P6_full_mi40"),
        ("Fail ~6.6% 불균형", "Major", "compare_in_grid", "S1_weight, S2_smote, S3_adasyn"),
        ("상수/저분산 특성", "Major", "transform", "P2_drop40_median, VarianceThreshold"),
        ("이상치 꼬리", "Major", "compare_in_grid", "P4_drop40_winsor vs P5_drop40_iqr"),
        ("고차원 591 features", "Major", "compare_in_grid", "P6_full_mi40, P7_full_mi80"),
        (f"CV fold 최소 Fail={min_fold_fail}", sev, "compare_in_grid" if sev == "Major" else "log_only", "S1_weight"),
    ]

    md = ["# EDA Decision Log\n", f"- Samples: {n}, Features: {p}, Fail: {n_fail} ({n_fail/n:.2%})\n"]
    md.append("\n| EDA 발견 | 심각도 | 조치 | 실험 축 ID |\n|----------|--------|------|------------|\n")
    for row in decision_rows:
        md.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |\n")

    md_path = REPORTS / "eda_decision_log.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("".join(md), encoding="utf-8")

    qc_md = ["# EDA Quality Report\n\n"]
    for item in quality_items:
        qc_md.append(f"- **{item['type']}** ({item['severity']}): {item['count']} — {item['rule']} → `{item['disposition']}`\n")
    (REPORTS / "eda_quality_report.md").write_text("".join(qc_md), encoding="utf-8")

    return report_json


if __name__ == "__main__":
    run_eda()
