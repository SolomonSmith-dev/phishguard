"""Late-fusion meta-learner over per-modality probabilities.

Takes p_url, p_html, p_img and outputs a calibrated phishing probability.
Logistic regression is the right pick here because:
    1. The inputs are already probabilities, so the linear combination is
       directly interpretable as a stacked classifier.
    2. We have plenty of training rows but few features, so simple beats fancy.
    3. Easy to retrain on drift.

For graceful degradation when a modality is missing (image fetch failed,
HTML scrape timed out), we impute the missing probability with 0.5 and
add a binary 'missing' indicator. The model learns to discount accordingly.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

FEATURE_ORDER = (
    "p_url",
    "p_html",
    "p_img",
    "missing_html",
    "missing_img",
)


@dataclass
class FusionInputs:
    p_url: float
    p_html: float | None
    p_img: float | None

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.p_url,
                0.5 if self.p_html is None else self.p_html,
                0.5 if self.p_img is None else self.p_img,
                int(self.p_html is None),
                int(self.p_img is None),
            ],
            dtype=np.float32,
        )


class FusionModel:
    """Wraps a calibrated logistic regression over modality probabilities."""

    def __init__(self, base: LogisticRegression | None = None) -> None:
        self.base = base or LogisticRegression(
            C=1.0, penalty="l2", class_weight="balanced", max_iter=1000
        )
        self.calibrated: CalibratedClassifierCV | None = None
        self.threshold: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.calibrated = CalibratedClassifierCV(self.base, method="isotonic", cv=5)
        self.calibrated.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.calibrated is not None, "Call fit() before predict_proba()."
        return self.calibrated.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"calibrated": self.calibrated, "threshold": self.threshold}, f)

    @classmethod
    def load(cls, path: Path) -> FusionModel:
        with path.open("rb") as f:
            blob = pickle.load(f)
        m = cls()
        m.calibrated = blob["calibrated"]
        m.threshold = blob["threshold"]
        return m

    def save_threshold(self, path: Path) -> None:
        path.write_text(json.dumps({"threshold": self.threshold}))
