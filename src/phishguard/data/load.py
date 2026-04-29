"""Dataset loading and preprocessing.

Uses PhiUSIIL for BOTH phishing and benign URLs. Both classes have full URLs
with paths and realistic subdomain structure, so the model cannot rely on
trivial shortcuts like 'has-subdomain' or 'has-path'.

Tranco top sites are kept as a held-out 'in-the-wild benign' probe set,
NOT as training data.

Run via:
    python -m phishguard.data.load --download
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
KAGGLE_DIR = Path.home() / ".kaggle"
KAGGLE_JSON = KAGGLE_DIR / "kaggle.json"
KAGGLE_TOKEN = KAGGLE_DIR / "access_token"


def ensure_dirs() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)


def have_kaggle_creds() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return KAGGLE_JSON.exists() or KAGGLE_TOKEN.exists()


def download_phiusiil() -> Path:
    target = RAW / "phiusiil"
    target.mkdir(exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", PHIUSIIL_DATASET, "-p", str(target), "--unzip"],
        check=True,
    )
    return next(target.glob("*.csv"))


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
    (RAW / "top-1m.csv").rename(target)
    zip_path.unlink()
    return target


def _normalize_label(v: object) -> int:
    s = str(v).strip().lower()
    if s in {"1", "phishing", "phish", "bad", "malicious"}:
        return 1
    if s in {"0", "benign", "legitimate", "good", "safe"}:
        return 0
    return -1


def build_url_dataset(phish_csv: Path) -> pd.DataFrame:
    """Use PhiUSIIL's own labels for BOTH classes (no Tranco mixing)."""
    df = pd.read_csv(phish_csv)
    url_col = next(c for c in df.columns if c.lower() in {"url", "domain"})
    label_col = next(c for c in df.columns if c.lower() in {"label", "is_phishing", "type"})
    df = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label_raw"})
    df["is_phish"] = df["label_raw"].apply(_normalize_label)
    df = df[df["is_phish"].isin([0, 1])].drop(columns="label_raw")
    df = df.drop_duplicates(subset="url").sample(frac=1, random_state=42).reset_index(drop=True)
    n_phish = int(df["is_phish"].sum())
    n_benign = int((df["is_phish"] == 0).sum())
    print(f"PhiUSIIL labels: phish={n_phish}  benign={n_benign}")
    return df


def build_tranco_probe(tranco_csv: Path, n: int = 5000) -> pd.DataFrame:
    """Held-out probe of in-the-wild benign sites. NOT training data."""
    tranco = pd.read_csv(tranco_csv, header=None, names=["rank", "domain"])
    tranco["url"] = "https://" + tranco["domain"]
    return tranco.head(n).assign(is_phish=0)[["url", "is_phish"]]


def split_and_save(df: pd.DataFrame, probe: pd.DataFrame) -> None:
    train, temp = train_test_split(df, test_size=0.2, stratify=df["is_phish"], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["is_phish"], random_state=42)
    train.to_parquet(PROCESSED / "url_train.parquet", index=False)
    val.to_parquet(PROCESSED / "url_val.parquet", index=False)
    test.to_parquet(PROCESSED / "url_test.parquet", index=False)
    probe.to_parquet(PROCESSED / "url_probe_tranco.parquet", index=False)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  tranco_probe={len(probe)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    if args.download:
        if not have_kaggle_creds():
            raise SystemExit("Kaggle credentials not found at ~/.kaggle/")
        phish_csv = download_phiusiil()
        tranco_csv = download_tranco()
    else:
        phish_csv = next((RAW / "phiusiil").glob("*.csv"))
        tranco_csv = RAW / "tranco_top1m.csv"

    df = build_url_dataset(phish_csv)
    probe = build_tranco_probe(tranco_csv)
    split_and_save(df, probe)


if __name__ == "__main__":
    main()
