"""Statistical comparison vs baseline (Wilcoxon + Holm)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from src.paths import ARTIFACTS_EXP, ARTIFACTS_STATS, CONFIGS, REPORTS


def holm_correction(p_values: list[float], alpha=0.05) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.ones(m) * 1.0
    sorted_p = np.array(p_values)[order]
    for i, p in enumerate(sorted_p):
        adjusted[order[i]] = min(1.0, p * (m - i))
    return adjusted.tolist()


def run_statistical_comparison(
    baseline_id: str | None = None,
    metric_key: str = "pr_auc",
):
    with open(CONFIGS / "cv_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    min_gain = cfg.get("min_practical_gain", 0.02)
    alpha = cfg.get("alpha", 0.05)

    lb_path = ARTIFACTS_EXP / "experiments_leaderboard.csv"
    if not lb_path.exists():
        raise FileNotFoundError("Run experiments first.")

    lb = pd.read_csv(lb_path)
    results = {}
    for _, row in lb.iterrows():
        eid = row["experiment_id"]
        jpath = ARTIFACTS_EXP / f"{eid}.json"
        if not jpath.exists():
            continue
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        fold_key = f"{metric_key}_folds"
        if fold_key in data:
            results[eid] = data[fold_key]
        elif "fold_metrics" in data:
            results[eid] = [m[metric_key.replace("_folds", "")] for m in data["fold_metrics"]]

    if not results:
        return

    if baseline_id is None:
        # Stage0 dummy or lowest tier logistic P0
        for cand in [
            "P0_minimal__S0_none__B0_dummy__default",
            "P0_minimal__S0_none__B1_logistic_l2__default",
        ]:
            if cand in results:
                baseline_id = cand
                break
        if baseline_id is None:
            baseline_id = list(results.keys())[0]

    base_scores = np.array(results[baseline_id])
    rows = []
    raw_p = []
    eids = []

    for eid, scores in results.items():
        if eid == baseline_id:
            continue
        scores = np.array(scores)
        if len(scores) != len(base_scores):
            continue
        try:
            stat, p = wilcoxon(scores, base_scores, alternative="greater")
        except ValueError:
            p = 1.0
            stat = 0.0
        delta = float(scores.mean() - base_scores.mean())
        rows.append({
            "experiment_id": eid,
            "metric": metric_key,
            "baseline_id": baseline_id,
            "mean_delta": delta,
            "wilcoxon_stat": float(stat),
            "p_raw": float(p),
        })
        raw_p.append(float(p))
        eids.append(eid)

    if raw_p:
        p_adj_list = holm_correction(raw_p, alpha=alpha)
        for i, eid in enumerate(eids):
            idx = [r["experiment_id"] for r in rows].index(eid)
            rows[idx]["p_adj"] = p_adj_list[i]
            rows[idx]["significant"] = (
                p_adj_list[i] < alpha
                and rows[idx]["mean_delta"] >= min_gain
            )

    df = pd.DataFrame(rows).sort_values("mean_delta", ascending=False)
    df.to_csv(ARTIFACTS_STATS / "pvalues_matrix.csv", index=False)

    md = [
        "# Statistical Comparison\n\n",
        f"Baseline: `{baseline_id}`\n",
        f"Metric: {metric_key} (fold-wise Wilcoxon vs baseline, Holm correction α={alpha})\n",
        f"Practical gain threshold: {min_gain}\n\n",
        "| experiment_id | mean_Δ | p_adj | significant |\n",
        "|---------------|--------|-------|-------------|\n",
    ]
    for _, r in df.head(30).iterrows():
        sig = "yes" if r.get("significant") else "no"
        md.append(
            f"| {r['experiment_id']} | {r['mean_delta']:.4f} | {r.get('p_adj', float('nan')):.4f} | {sig} |\n"
        )
    (REPORTS / "statistical_comparison.md").write_text("".join(md), encoding="utf-8")
    return df


if __name__ == "__main__":
    run_statistical_comparison()
