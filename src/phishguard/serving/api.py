"""FastAPI service for PhishGuard.

Endpoints:
    GET  /healthz           Liveness probe.
    POST /predict           Multi-modal phishing prediction.

Design rules:
    1. URL model is always required and runs synchronously inline (it's cheap).
    2. HTML and screenshot are optional. If absent, fusion model degrades gracefully.
    3. All ML artifacts loaded once at startup (lazy via lifespan).
    4. Predictions logged to a Postgres table for the drift dashboard.

Latency budget on CPU:
    URL only:        < 5ms
    URL + HTML:      < 200ms
    URL + HTML + IMG < 300ms
"""

from __future__ import annotations

import base64
import io
import json
import logging
import pickle
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from phishguard.features import URLFeatureExtractor
from phishguard.models.fusion import FusionInputs, FusionModel
from phishguard.serving.prediction_log import PredictionLogger

logger = logging.getLogger("phishguard")
logging.basicConfig(level=logging.INFO)


CKPT_DIR = Path("models/checkpoints")
URL_MODEL = CKPT_DIR / "url_model.lgb"
URL_CALIB = CKPT_DIR / "url_calibrator.pkl"
URL_FEATS = CKPT_DIR / "url_features.json"
HTML_ONNX = CKPT_DIR / "html_model.onnx"
IMG_ONNX = CKPT_DIR / "screenshot_model.onnx"
FUSION = CKPT_DIR / "fusion_model.pkl"


class PredictRequest(BaseModel):  # type: ignore[misc]
    url: str = Field(..., min_length=4, max_length=4096)
    html: str | None = Field(None, description="Raw HTML text. Optional.")
    screenshot_b64: str | None = Field(None, description="Base64-encoded PNG. Optional.")


class ModalityProb(BaseModel):  # type: ignore[misc]
    p: float | None = None
    latency_ms: float | None = None
    available: bool = False


class PredictResponse(BaseModel):  # type: ignore[misc]
    p_phish: float
    is_phish: bool
    threshold: float
    modalities: dict[str, ModalityProb]
    latency_ms_total: float


# globals populated at startup
_state: dict[str, Any] = {}


def _pick_url_artifact(v0_2_name: str, v0_1_path: Path) -> Path:
    """Prefer v0.2 ablation artifact when present; fall back to v0.1."""
    v0_2 = CKPT_DIR / v0_2_name
    return v0_2 if v0_2.exists() else v0_1_path


def _load_calibrator(path: Path) -> Any:
    """Load a sklearn calibrator serialized with joblib or plain pickle.

    New artifacts are written with ``joblib.dump``; this fallback ensures
    v0.1 artifacts written with ``pickle.dump`` still load without error.
    """
    try:
        return joblib.load(path)
    except (ValueError, EOFError, pickle.UnpicklingError) as exc:
        logger.warning(
            "joblib.load failed (%s); retrying with pickle for v0.1 artifact: %s",
            type(exc).__name__,
            exc,
        )
        with open(path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301 – trusted local artifact


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("loading artifacts ...")
    url_model_path = _pick_url_artifact("url_model_v0_2.lgb", URL_MODEL)
    url_calib_path = _pick_url_artifact("url_calibrator_v0_2.pkl", URL_CALIB)
    url_feats_path = _pick_url_artifact("url_features_v0_2.json", URL_FEATS)
    logger.info("url model: %s", url_model_path)
    _state["url_booster"] = lgb.Booster(model_file=str(url_model_path))
    _state["url_calibrator"] = _load_calibrator(url_calib_path)
    _state["url_feature_names"] = json.loads(url_feats_path.read_text())
    _state["url_extractor"] = URLFeatureExtractor()

    if HTML_ONNX.exists():
        _state["html_session"] = ort.InferenceSession(
            str(HTML_ONNX), providers=["CPUExecutionProvider"]
        )
        from transformers import AutoTokenizer

        _state["html_tokenizer"] = AutoTokenizer.from_pretrained("models/checkpoints/html_model")
    if IMG_ONNX.exists():
        _state["img_session"] = ort.InferenceSession(
            str(IMG_ONNX), providers=["CPUExecutionProvider"]
        )

    _state["fusion"] = FusionModel.load(FUSION) if FUSION.exists() else None
    _state["pred_log"] = PredictionLogger()
    threshold = _state["fusion"].threshold if _state["fusion"] is not None else 0.5
    logger.info("ready (threshold=%.4f)", threshold)
    yield
    logger.info("shutting down")
    log = _state.get("pred_log")
    if log is not None:
        log.close()


app = FastAPI(title="PhishGuard", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")  # type: ignore[misc]
def healthz() -> dict[str, Any]:
    return {"status": "ok", "models_loaded": list(_state.keys())}


def _predict_url(url: str) -> float:
    rows = _state["url_extractor"].transform([url])
    df = pd.DataFrame(rows)[_state["url_feature_names"]]
    df["tld"] = df["tld"].astype("category")
    raw = _state["url_booster"].predict(df)[0]
    return float(_state["url_calibrator"].transform([raw])[0])


def _predict_html(html: str) -> float | None:
    sess = _state.get("html_session")
    tok = _state.get("html_tokenizer")
    if sess is None or tok is None:
        return None
    from phishguard.training.train_html import clean_html

    encoded = tok(clean_html(html), truncation=True, max_length=512, return_tensors="np")
    out = sess.run(
        None, {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
    )
    logits = out[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return float(probs[1])


def _predict_image(b64: str) -> float | None:
    sess = _state.get("img_session")
    if sess is None:
        return None
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, ...]
    out = sess.run(None, {"pixel_values": arr.astype(np.float32)})
    logits = out[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return float(probs[1])


@app.post("/predict", response_model=PredictResponse)  # type: ignore[misc]
def predict(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()
    modalities: dict[str, ModalityProb] = {}

    try:
        t = time.perf_counter()
        p_url = _predict_url(req.url)
        modalities["url"] = ModalityProb(
            p=p_url, latency_ms=(time.perf_counter() - t) * 1000, available=True
        )
    except Exception as e:
        logger.exception("url model failed")
        raise HTTPException(status_code=500, detail=f"url model failure: {e}") from e

    p_html = None
    if req.html:
        t = time.perf_counter()
        p_html = _predict_html(req.html)
        modalities["html"] = ModalityProb(
            p=p_html, latency_ms=(time.perf_counter() - t) * 1000, available=p_html is not None
        )
    else:
        modalities["html"] = ModalityProb(available=False)

    p_img = None
    if req.screenshot_b64:
        t = time.perf_counter()
        try:
            p_img = _predict_image(req.screenshot_b64)
        except Exception:
            logger.exception("screenshot decode/inference failed; degrading")
            p_img = None
        modalities["img"] = ModalityProb(
            p=p_img, latency_ms=(time.perf_counter() - t) * 1000, available=p_img is not None
        )
    else:
        modalities["img"] = ModalityProb(available=False)

    fusion: FusionModel | None = _state.get("fusion")
    if fusion is not None:
        X = FusionInputs(p_url=p_url, p_html=p_html, p_img=p_img).to_vector().reshape(1, -1)
        p_phish = float(fusion.predict_proba(X)[0])
        threshold = fusion.threshold
    else:
        # No fusion artifact yet -- URL-only passthrough. HTML/img get averaged in
        # if available so the response is still informative on partial modality input.
        parts = [p_url] + [p for p in (p_html, p_img) if p is not None]
        p_phish = float(sum(parts) / len(parts))
        threshold = 0.5

    is_phish = bool(p_phish >= threshold)
    latency_total = (time.perf_counter() - t0) * 1000

    log = _state.get("pred_log")
    if log is not None:
        log.log(
            {
                "url": req.url,
                "p_url": p_url,
                "p_html": p_html,
                "p_img": p_img,
                "p_phish": p_phish,
                "threshold": threshold,
                "is_phish": is_phish,
                "latency_ms_total": latency_total,
                "had_html": req.html is not None,
                "had_img": req.screenshot_b64 is not None,
            }
        )

    return PredictResponse(
        p_phish=p_phish,
        is_phish=is_phish,
        threshold=threshold,
        modalities=modalities,
        latency_ms_total=latency_total,
    )
