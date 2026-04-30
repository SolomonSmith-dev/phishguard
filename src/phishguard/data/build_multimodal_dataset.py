"""Build HTML and screenshot training splits from scraped snapshots.

Reads `data/processed/snapshots/manifest.jsonl`, joins back to a label source
(parquet with `url, is_phish`), and emits:

    data/processed/html_{train,val,test}.parquet  (with html_text + is_phish)
    data/processed/screenshots/{train,val,test}/{phish,benign}/*.png  (ImageFolder)

Usage:
    python -m phishguard.data.build_multimodal_dataset \
        --manifest data/processed/snapshots/manifest.jsonl \
        --labels data/processed/url_train.parquet \
        --output-html data/processed \
        --output-img  data/processed/screenshots
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_manifest(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def join_with_labels(manifest: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = manifest.merge(labels[["url", "is_phish"]], on="url", how="inner")
    merged = merged[merged["ok"]]  # drop failed scrapes
    return merged.reset_index(drop=True)


def write_html_splits(df: pd.DataFrame, out_dir: Path, seed: int = 42) -> None:
    rows = []
    for _, r in df.iterrows():
        try:
            text = Path(r["html_path"]).read_text(encoding="utf-8", errors="ignore")
        except (OSError, FileNotFoundError):
            continue
        rows.append({"url": r["url"], "html_text": text, "is_phish": int(r["is_phish"])})
    if not rows:
        print("no html rows after filtering -- skipping html splits")
        return
    full = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    if full["is_phish"].nunique() < 2 or len(full) < 6:
        # Not enough signal/rows for stratified split. Dump all to train.
        print(
            f"only {len(full)} rows / {full['is_phish'].nunique()} classes -- writing as train-only"
        )
        full.to_parquet(out_dir / "html_train.parquet", index=False)
        return

    train, temp = train_test_split(
        full, test_size=0.3, stratify=full["is_phish"], random_state=seed
    )
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["is_phish"], random_state=seed)
    train.to_parquet(out_dir / "html_train.parquet", index=False)
    val.to_parquet(out_dir / "html_val.parquet", index=False)
    test.to_parquet(out_dir / "html_test.parquet", index=False)
    print(f"html  train={len(train)} val={len(val)} test={len(test)} -> {out_dir}")


def write_image_splits(df: pd.DataFrame, out_dir: Path, seed: int = 42) -> None:
    df = df.dropna(subset=["screenshot_path"])
    if df.empty:
        print("no screenshots -- skipping img splits")
        return
    if df["is_phish"].nunique() < 2 or len(df) < 6:
        n_classes = df["is_phish"].nunique()
        print(f"only {len(df)} screenshots / {n_classes} classes -- writing all to train")
        train, val, test = df, df.iloc[:0], df.iloc[:0]
    else:
        train, temp = train_test_split(
            df, test_size=0.3, stratify=df["is_phish"], random_state=seed
        )
        val, test = train_test_split(
            temp, test_size=0.5, stratify=temp["is_phish"], random_state=seed
        )

    for split_name, split_df in (("train", train), ("val", val), ("test", test)):
        for label_name in ("benign", "phish"):
            (out_dir / split_name / label_name).mkdir(parents=True, exist_ok=True)
        for _, r in split_df.iterrows():
            label_name = "phish" if int(r["is_phish"]) else "benign"
            src = Path(r["screenshot_path"])
            if not src.exists():
                continue
            dst = out_dir / split_name / label_name / src.name
            shutil.copy2(src, dst)

    n_train = sum(1 for _ in (out_dir / "train").rglob("*.png"))
    n_val = sum(1 for _ in (out_dir / "val").rglob("*.png"))
    n_test = sum(1 for _ in (out_dir / "test").rglob("*.png"))
    print(f"img   train={n_train} val={n_val} test={n_test} -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output-html", type=Path, default=Path("data/processed"))
    ap.add_argument("--output-img", type=Path, default=Path("data/processed/screenshots"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"manifest rows: {len(manifest)}  ok: {int(manifest['ok'].sum())}")
    labels = pd.read_parquet(args.labels)
    df = join_with_labels(manifest, labels)
    print(f"joined rows (with labels and ok): {len(df)}")

    write_html_splits(df, args.output_html, seed=args.seed)
    write_image_splits(df, args.output_img, seed=args.seed)


if __name__ == "__main__":
    main()
