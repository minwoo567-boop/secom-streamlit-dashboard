"""Download SECOM dataset from Kaggle."""
import os
from pathlib import Path

from src.paths import DATA_RAW, ROOT


def _load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def download():
    _load_env()
    from kaggle.api.kaggle_api_extended import KaggleApi

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        "paresh2047/uci-semcom",
        path=str(DATA_RAW),
        unzip=True,
        quiet=False,
    )
    print(f"Downloaded to {DATA_RAW}")


if __name__ == "__main__":
    download()
