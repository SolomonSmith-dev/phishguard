"""Train the late-fusion meta-learner.

Inputs are per-modality probabilities produced by the three models on a held-out
fusion split. Outputs a calibrated fused probability and a tuned decision threshold.

Threshold selection respects min_recall constraint from config because in
phishing detection a missed phish is much costlier than a false alarm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, roc_auc_score,
)

from phishguard.models.fusion import FEATURE_ORDER, FusionModel


def load_xy(path: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    p_url = df[cfg["inputs"]["url_prob_col"]].values
    p_html = df[cfg["inputs"]["html_prob_col"]]
    p_img = df[cfg["inputs"]["img_prob_col"]]

    missing_html = p_html.isna().astype(int).values
    missing_img = p_img.isna().astype(int).values
    p_html = p_html.fillna(0.5).values
    p_img = p_img.fillna(0.5).values

    X = np.stack([p_url, p_html, p_img, missing_html, missing_img], axis=1).astype(np.float32)
    y = df["is_phish"].astype(int).values
    return X, y


def pick_threshold(y_true: np.ndarray, probs: np.ndarray, min_recall: float, mode: str) -> float:
    p, r, thr = precision_recall_curve(y_true, probs)
    if mode == "f1":
        f1s = 2 * p * r / (p + r + 1e-12)
        best = int(np.argmax(f1s[:-1])) if len(thr) > 0 else 0
        return float(thr[best]) if len(thr) > 0 else 0.5
    # min_recall constraint, maximize precision
    mask = r[:-1] >= min_recall
    if not mask.any():
        return 0.5
    candidates = np.where(mask)[0]
    best = candidates[np.argmax(p[candidates])]
    return float(thr[best])


def train(cfg: dict) -> None:
    np.random.seed(cfg["seed"])

    X_train, y_train = load_xy(cfg["data"]["fusion_train"], cfg)
    X_val, y_val = load_xy(cfg["data"]["fusion_val"], cfg)
    X_test, y_test = load_xy(cfg["data"]["fusion_test"], cfg)

    model = FusionModel()
    model.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))

    val_probs = model.predict_proba(X_val)
    test_probs = model.predict_proba(X_test)

    print(f"val   AUC={roc_auc_score(y_val, val_probs):.4f}  AP={average_precision_score(y_val, val_probs):.4f}")
    print(f"test  AUC={roc_auc_score(y_test, test_probs):.4f}  AP={average_precision_score(y_test, test_probs):.4f}")

    thr = pick_threshold(
        y_val, val_probs,
        min_recall=cfg["decision_threshold"]["min_recall"],
        mode=cfg["decision_threshold"]["optimize_for"],
    )
    model.threshold = thr
    test_preds = (test_probs >= thr).astype(int)
    print(f"threshold={thr:.4f}  test F1={f1_score(y_test, test_preds):.4f}  Brier={brier_score_loss(y_test, test_probs):.4f}")

    art = cfg["artifacts"]
    Path(art["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    model.save(Path(art["model_path"]))
    Path(art["threshold_path"]).write_text(json.dumps({"threshold": thr, "feature_order": list(FEATURE_ORDER)}))
    print(f"saved -> {art['model_path']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    train(cfg)


if __name__ == "__main__":
    main()
