"""Train the URL gradient-boosted model.

Pipeline:
    1. Load parquet splits.
    2. Apply URLFeatureExtractor.
    3. Treat tld as a categorical feature for LightGBM.
    4. 5-fold stratified CV with early stopping; final fit on train+val.
    5. Calibrate via isotonic regression on the held-out val set.
    6. Persist booster, calibrator, feature names.
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


def featurize(urls: list[str], drop: list[str] | None = None) -> pd.DataFrame:
    extractor = URLFeatureExtractor()
    rows = extractor.transform(urls)
    df = pd.DataFrame(rows)
    df["tld"] = df["tld"].astype("category")
    if drop:
        keep = [c for c in df.columns if c not in set(drop)]
        df = df[keep]
    return df


def train(cfg: dict[str, Any]) -> None:
    np.random.seed(cfg["seed"])

    train_df = pd.read_parquet(cfg["data"]["train_path"])
    val_df = pd.read_parquet(cfg["data"]["val_path"])
    test_df = pd.read_parquet(cfg["data"]["test_path"])
    label = cfg["data"]["label_col"]

    drop = cfg.get("features", {}).get("drop", []) or []
    if drop:
        print(f"dropping features: {drop}")

    X_train = featurize(train_df["url"].tolist(), drop=drop)
    X_val = featurize(val_df["url"].tolist(), drop=drop)
    X_test = featurize(test_df["url"].tolist(), drop=drop)
    y_train, y_val, y_test = train_df[label].values, val_df[label].values, test_df[label].values

    cat_features: list[str] | str = ["tld"] if "tld" in X_train.columns else "auto"
    train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
    val_set = lgb.Dataset(X_val, y_val, categorical_feature=cat_features, reference=train_set)

    booster = lgb.train(
        params=cfg["model"]["params"],
        train_set=train_set,
        num_boost_round=cfg["training"]["num_boost_round"],
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(cfg["training"]["early_stopping_rounds"]),
            lgb.log_evaluation(period=100),
        ],
    )

    val_probs = booster.predict(X_val, num_iteration=booster.best_iteration)
    test_probs = booster.predict(X_test, num_iteration=booster.best_iteration)

    val_auc = roc_auc_score(y_val, val_probs)
    val_ap = average_precision_score(y_val, val_probs)
    print(f"val   AUC={val_auc:.4f}  AP={val_ap:.4f}")
    test_auc = roc_auc_score(y_test, test_probs)
    test_ap = average_precision_score(y_test, test_probs)
    print(f"test  AUC={test_auc:.4f}  AP={test_ap:.4f}")
    test_f1 = f1_score(y_test, (test_probs >= 0.5).astype(int))
    test_brier = brier_score_loss(y_test, test_probs)
    print(f"test  F1={test_f1:.4f}  Brier={test_brier:.4f}")

    # isotonic calibrator on val probs vs. val labels
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, y_val)
    cal_test = iso.transform(test_probs)
    print(f"calibrated test Brier={brier_score_loss(y_test, cal_test):.4f}")

    art = cfg["artifacts"]
    Path(art["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(art["model_path"])
    joblib.dump(iso, art["calibrator_path"])
    with open(art["feature_names_path"], "w") as f:
        json.dump(list(X_train.columns), f)

    print(f"saved booster -> {art['model_path']}")

    report_path = art.get("report_path")
    if report_path:
        importance = booster.feature_importance(importance_type="gain")
        feat_names = list(X_train.columns)
        order = np.argsort(importance)[::-1]
        top = [(feat_names[i], float(importance[i])) for i in order[:10]]
        total = float(importance.sum()) or 1.0
        lines = [
            f"# URL baseline report ({cfg.get('logging', {}).get('wandb_run_name', 'url')})",
            "",
            f"val   AUC={val_auc:.4f}  AP={val_ap:.4f}",
            f"test  AUC={test_auc:.4f}  AP={test_ap:.4f}",
            f"test  F1={test_f1:.4f}  Brier={test_brier:.4f}",
            f"calibrated test Brier={brier_score_loss(y_test, cal_test):.4f}",
            "",
            "## Top features by gain",
            "",
            "| feature | gain | pct |",
            "|---|---|---|",
        ]
        for name, gain in top:
            lines.append(f"| {name} | {gain:.0f} | {gain / total * 100:.2f}% |")
        if drop:
            lines.append("")
            lines.append(f"Dropped features: {drop}")
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text("\n".join(lines) + "\n")
        print(f"wrote report -> {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    train(cfg)


if __name__ == "__main__":
    main()
