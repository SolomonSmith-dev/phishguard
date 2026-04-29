"""Dataset loading and preprocessing.

Pulls PhiUSIIL phishing URLs (Kaggle) and Tranco top-1M (benign) and writes
parquet splits for training the URL model.

Run via:
    python -m phishguard.data.load --download

Authentication:
    Set KAGGLE_USERNAME and KAGGLE_KEY env vars (see ~/.kaggle/kaggle.json).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

PHIUSIIL_DATASET = "ndarvind/phiusiil-phishing-url-dataset"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"


def ensure_dirs() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)


def download_phiusiil() -> Path:
    """Use Kaggle CLI; assumes credentials configured."""
    target = RAW / "phiusiil"
    target.mkdir(exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", PHIUSIIL_DATASET, "-p", str(target), "--unzip"],
        check=True,
    )
    csv = next(target.glob("*.csv"))
    return csv


def download_tranco() -> Path:
    import httpx
    target = RAW / "tranco_top1m.csv"
    if target.exists():
        return target
    zip_path = RAW / "tranco.zip"
    with httpx.Client(follow_redirects=True, timeout=60.0) as c:
        r = c.get(TRANCO_URL)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW)
    extracted = RAW / "top-1m.csv"
    extracted.rename(target)
    zip_path.unlink()
    return target


def build_url_dataset(phish_csv: Path, tranco_csv: Path, n_benign: int = 200_000) -> pd.DataFrame:
    """Combine phishing and benign URLs into a labeled dataframe."""
    phish = pd.read_csv(phish_csv)
    # PhiUSIIL has many engineered cols; we only need the URL string and label.
    url_col = next(c for c in phish.columns if c.lower() in {"url", "domain"})
    label_col = next(c for c in phish.columns if c.lower() in {"label", "is_phishing", "type"})
    phish = phish[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label_raw"})
    phish["is_phish"] = phish["label_raw"].apply(_normalize_label)
    phish = phish.drop(columns="label_raw")
    phish = phish[phish["is_phish"] == 1]

    tranco = pd.read_csv(tranco_csv, header=None, names=["rank", "domain"])
    tranco["url"] = "https://" + tranco["domain"]
    tranco = tranco.head(n_benign).assign(is_phish=0)[["url", "is_phish"]]

    df = pd.concat([phish, tranco], ignore_index=True)
    df = df.drop_duplicates(subset="url").sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def _normalize_label(v: object) -> int:
    s = str(v).strip().lower()
    if s in {"1", "phishing", "phish", "bad", "malicious"}:
        return 1
    if s in {"0", "benign", "legitimate", "good", "safe"}:
        return 0
    return -1  # filtered out by callers


def split_and_save(df: pd.DataFrame) -> None:
    train, temp = train_test_split(df, test_size=0.2, stratify=df["is_phish"], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["is_phish"], random_state=42)
    train.to_parquet(PROCESSED / "url_train.parquet", index=False)
    val.to_parquet(PROCESSED / "url_val.parquet", index=False)
    test.to_parquet(PROCESSED / "url_test.parquet", index=False)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    if args.download:
        if not os.getenv("KAGGLE_KEY"):
            raise SystemExit("Set KAGGLE_USERNAME and KAGGLE_KEY first.")
        phish_csv = download_phiusiil()
        tranco_csv = download_tranco()
    else:
        phish_csv = next((RAW / "phiusiil").glob("*.csv"))
        tranco_csv = RAW / "tranco_top1m.csv"

    df = build_url_dataset(phish_csv, tranco_csv)
    split_and_save(df)


if __name__ == "__main__":
    main()
