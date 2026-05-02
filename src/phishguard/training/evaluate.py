"""Evaluate the URL model on a dataset split and optionally run a Tranco probe.

Loads the URL model artifacts from disk, runs evaluation on the requested
split, and writes a report. If the benign-only Tranco probe dataset is
present, this module also reports false-positive rates at fixed thresholds.

Usage:
    python -m phishguard.training.evaluate --url-config configs/url_model.yaml
    python -m phishguard.training.evaluate --url-config configs/url_model_v0_2.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from phishguard.features import URLFeatureExtractor


def _load_url_artifacts(cfg: dict[str, Any]) -> tuple[lgb.Booster, Any, list[str]]:
    art = cfg["artifacts"]
    booster = lgb.Booster(model_file=art["model_path"])
    calibrator = joblib.load(art["calibrator_path"])
    feat_names = json.loads(Path(art["feature_names_path"]).read_text())
    return booster, calibrator, feat_names


def _featurize(urls: list[str], feat_names: list[str]) -> pd.DataFrame:
    rows = URLFeatureExtractor().transform(urls)
    df = pd.DataFrame(rows)
    if "tld" in df.columns:
        df["tld"] = df["tld"].astype("category")
    return df[feat_names]


def evaluate_url(cfg: dict[str, Any], split: str = "test") -> dict[str, float]:
    booster, calibrator, feat_names = _load_url_artifacts(cfg)
    df = pd.read_parquet(cfg["data"][f"{split}_path"])
    label_col = cfg["data"]["label_col"]
    X = _featurize(df["url"].astype(str).tolist(), feat_names)
    y = df[label_col].astype(int).values

    raw = booster.predict(X)
    cal = calibrator.transform(raw)
    preds = (cal >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y, cal)),
        "ap": float(average_precision_score(y, cal)),
        "f1": float(f1_score(y, preds)),
        "brier": float(brier_score_loss(y, cal)),
        "n": int(len(y)),
    }


def evaluate_tranco_probe(cfg: dict[str, Any]) -> dict[str, float]:
    """Tranco is benign-only -- report FPR@thresholds, not AUC."""
    booster, calibrator, feat_names = _load_url_artifacts(cfg)
    probe_path = Path("data/processed/url_probe_tranco.parquet")
    if not probe_path.exists():
        return {}
    df = pd.read_parquet(probe_path)
    X = _featurize(df["url"].astype(str).tolist(), feat_names)
    raw = booster.predict(X)
    cal = np.asarray(calibrator.transform(raw))
    return {
        "n": int(len(cal)),
        "median_p_phish": float(np.median(cal)),
        "fpr_at_0.5": float((cal >= 0.5).mean()),
        "fpr_at_0.7": float((cal >= 0.7).mean()),
        "fpr_at_0.9": float((cal >= 0.9).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--url-config",
        type=Path,
        default=Path("configs/url_model.yaml"),
        help="URL model config to evaluate.",
    )
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", type=Path, default=Path("reports/evaluation.md"))
    args = ap.parse_args()

    url_cfg = yaml.safe_load(args.url_config.read_text())

    sections: list[str] = ["# PhishGuard evaluation", ""]

    sections.append(f"## URL model (`{url_cfg['artifacts']['model_path']}`)")
    sections.append("")
    url_metrics = evaluate_url(url_cfg, split=args.split)
    sections.append(f"`{args.split}` split (n={url_metrics['n']}):")
    sections.append("")
    sections.append("| metric | value |")
    sections.append("|---|---|")
    for k in ("auc", "ap", "f1", "brier"):
        sections.append(f"| {k} | {url_metrics[k]:.4f} |")
    sections.append("")

    probe = evaluate_tranco_probe(url_cfg)
    if probe:
        sections.append("Tranco top-5000 probe (benign-only, FPR-style):")
        sections.append("")
        sections.append("| metric | value |")
        sections.append("|---|---|")
        for k, v in probe.items():
            if k == "n":
                sections.append(f"| n | {v} |")
            else:
                sections.append(f"| {k} | {v:.4f} |")
        sections.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections) + "\n")
    print("\n".join(sections))
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
