"""
download.py - fetch just the two CSVs we use from Kaggle into data/raw/.

The Kaggle dataset also holds the Medium and Large sets (many GB). kagglehub's
`path=` pulls one file at a time, which keeps the download to ~1 GB. kagglehub
caches under ~/.cache, so we move the file out rather than copy it - otherwise
every CSV sits on disk twice.

Usage: python src/download.py
"""
import shutil
from pathlib import Path

import kagglehub

HANDLE = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
FILES = ["HI-Small_Trans.csv", "LI-Small_Trans.csv"]
RAW_DIR = Path("data/raw")

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = RAW_DIR / name
        if dst.exists():
            print(f"{dst} already there, skipping")
            continue
        cached = Path(kagglehub.dataset_download(HANDLE, path=name))
        shutil.move(cached, dst)
        print(f"{dst}  ({dst.stat().st_size / 1e6:,.0f} MB)")
