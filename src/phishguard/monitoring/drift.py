"""Drift monitoring report.

Compares a recent prediction window vs. a reference (training-time) snapshot.
Outputs an Evidently HTML dashboard plus a JSON summary for alerting.

Usage:
    python -m phishguard.monitoring.drift --reference data/processed/url_train.parquet \
        --current data/predictions/last_7d.parquet --out reports/drift.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Evidently 0.7+ moved the v0 preset/report API under `evidently.legacy.*`.
# We use it deliberately because the new pipeline-style API is overkill for
# a single-shot drift check. Pin or migrate when Evidently removes legacy.
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.report import Report

from phishguard.features import URLFeatureExtractor


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    feats = URLFeatureExtractor().transform(df["url"].astype(str).tolist())
    out = pd.DataFrame(feats)
    out["is_phish"] = df["is_phish"].values if "is_phish" in df.columns else None
    return out.drop(columns=["tld"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--current", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/drift.html"))
    args = ap.parse_args()

    ref = featurize(pd.read_parquet(args.reference))
    cur = featurize(pd.read_parquet(args.current))

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(args.out))

    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(json.dumps(report.as_dict(), indent=2, default=str))
    print(f"drift report -> {args.out}")
    print(f"summary json -> {summary_path}")


if __name__ == "__main__":
    main()
