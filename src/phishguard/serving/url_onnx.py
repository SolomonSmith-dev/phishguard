"""Shared helpers for running the URL LightGBM model through ONNX Runtime.

The ONNX graph takes one float64 tensor named ``input`` with one column per
feature in ``url_features_v0_2.json``. LightGBM handled ``tld`` as a pandas
categorical; the ONNX graph only sees numbers, so the serving path has to
map each TLD string to the integer code LightGBM assigned at training time
(``Booster.pandas_categorical``). Unknown TLDs become NaN, which LightGBM and
the ONNX TreeEnsemble both route down the missing-value branch.

The isotonic calibrator is NOT folded into the graph. It is applied in
Python on top of the raw ONNX probability, exactly as it was applied on top
of the native booster. Folding it in would mean exporting a scikit-learn
IsotonicRegression through skl2onnx and re-verifying a second converter for
no latency benefit (the calibrator is a 1-D lookup).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INPUT_NAME = "input"
PROBABILITIES_OUTPUT = "probabilities"


def tld_code_map(tld_categories: list[str]) -> dict[str, int]:
    return {str(t): i for i, t in enumerate(tld_categories)}


def encode_features(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    tld_codes: dict[str, int],
) -> np.ndarray:
    """Turn extractor rows into the float64 matrix the ONNX graph expects."""
    df = pd.DataFrame(rows)[feature_names]
    if "tld" in df.columns:
        df["tld"] = df["tld"].astype(str).map(tld_codes).astype("float64")
    return df.to_numpy(dtype=np.float64)


def booster_tld_categories(booster: Any) -> list[str]:
    """Pull the TLD category list LightGBM stored alongside the booster."""
    cats = getattr(booster, "pandas_categorical", None) or []
    return [str(c) for c in cats[0]] if cats else []


class UrlOnnxModel:
    """ONNX Runtime session plus the metadata needed to feed it URLs."""

    def __init__(
        self,
        onnx_path: Path,
        feature_names: list[str],
        tld_categories: list[str],
    ) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.feature_names = feature_names
        self.tld_codes = tld_code_map(tld_categories)
        self.input_name = self.session.get_inputs()[0].name
        in_type = self.session.get_inputs()[0].type
        self.input_dtype = np.float64 if "double" in in_type else np.float32

    @classmethod
    def from_paths(cls, onnx_path: Path, features_path: Path, tld_path: Path) -> UrlOnnxModel:
        return cls(
            onnx_path,
            json.loads(features_path.read_text()),
            json.loads(tld_path.read_text()),
        )

    def predict_raw(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Uncalibrated p(phish) for each extractor row, from the ONNX session."""
        x = encode_features(rows, self.feature_names, self.tld_codes).astype(self.input_dtype)
        probs = self.session.run([PROBABILITIES_OUTPUT], {self.input_name: x})[0]
        return np.asarray(probs[:, 1], dtype=np.float64)
