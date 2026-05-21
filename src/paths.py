from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS_EDA = ARTIFACTS / "eda"
ARTIFACTS_EXP = ARTIFACTS / "experiments"
ARTIFACTS_OOF = ARTIFACTS / "oof"
ARTIFACTS_STATS = ARTIFACTS / "stats"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

for p in [
    DATA_PROCESSED,
    DATA_SPLITS,
    ARTIFACTS_EDA,
    ARTIFACTS_EXP,
    ARTIFACTS_OOF,
    ARTIFACTS_STATS,
    MODELS_DIR,
    REPORTS,
]:
    p.mkdir(parents=True, exist_ok=True)
