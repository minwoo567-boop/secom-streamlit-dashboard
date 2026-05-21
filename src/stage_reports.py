"""Write per-stage experiment summaries."""
import json
from pathlib import Path

import pandas as pd

from src.paths import ARTIFACTS_EXP, REPORTS


def write_stage_summaries():
    rows = []
    for p in ARTIFACTS_EXP.glob("*.json"):
        if "_error" in p.name:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "pr_auc_mean" in d:
            rows.append(d)
    df = pd.DataFrame(rows)
    for stage in sorted(df["stage"].dropna().unique()):
        sub = df[df.stage == stage].sort_values("pr_auc_mean", ascending=False).head(3)
        lines = [f"# Stage {stage} Summary\n\n", "Top 3 by OOF PR-AUC:\n\n"]
        for _, r in sub.iterrows():
            lines.append(
                f"- **{r['experiment_id']}**: PR-AUC {r['pr_auc_mean']:.4f} "
                f"(+/- {r['pr_auc_std']:.4f}), Recall {r['recall_mean']:.4f}\n"
            )
        (REPORTS / f"stage_{stage}_summary.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    write_stage_summaries()
