"""FastAPI service for PhishGuard.

Endpoints:
    GET  /                  Demo page (static HTML that calls /predict).
    GET  /healthz           Liveness probe, plus which URL model file is loaded.
    POST /predict           Multi-modal phishing prediction.

Design rules:
    1. URL model is always required and runs synchronously inline (it's cheap).
       It is served through ONNX Runtime: the LightGBM booster is exported by
       ``phishguard.training.export_url_onnx`` and the isotonic calibrator is
       applied in Python on top of the ONNX probability.
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
import logging
import pickle
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from phishguard.features import URLFeatureExtractor
from phishguard.models.fusion import FusionInputs, FusionModel
from phishguard.serving.limits import BodySizeLimit
from phishguard.serving.prediction_log import PredictionLogger
from phishguard.serving.url_onnx import UrlOnnxModel

logger = logging.getLogger("phishguard")
logging.basicConfig(level=logging.INFO)


CKPT_DIR = Path("models/checkpoints")
URL_ONNX_NAME = "url_model.onnx"
URL_CALIB_NAME = "url_calibrator.pkl"
URL_FEATS_NAME = "url_features.json"
URL_TLDS_NAME = "url_tld_categories.json"
HTML_ONNX_NAME = "html_model.onnx"
IMG_ONNX_NAME = "screenshot_model.onnx"
FUSION_NAME = "fusion_model.pkl"
DEMO_PAGE = Path(__file__).with_name("static") / "demo.html"
# Written by scripts/deploy_hf_space.py into the image; first line is "commit <sha>".
BUILD_MANIFEST = Path("ARTIFACTS.sha256")

# Upper bounds on the optional payload fields. The service is public, so an
# unbounded string field lets one request pin memory while pydantic parses it.
# 2 MB of HTML covers real pages; 10 MB of base64 is a ~7.5 MB PNG.
MAX_HTML_CHARS = 2_000_000
MAX_SCREENSHOT_B64_CHARS = 10_000_000
# Whole-body cap enforced before parsing (see serving/limits.py): room for a URL,
# ASCII HTML and a screenshot at their field caps plus JSON overhead. HTML that is
# mostly escaped non-ASCII can hit this cap below 2,000,000 characters.
MAX_BODY_BYTES = 13 * 1024 * 1024


class PredictRequest(BaseModel):  # type: ignore[misc]
    url: str = Field(..., min_length=4, max_length=4096)
    html: str | None = Field(
        None, max_length=MAX_HTML_CHARS, description="Raw HTML text. Optional."
    )
    screenshot_b64: str | None = Field(
        None, max_length=MAX_SCREENSHOT_B64_CHARS, description="Base64-encoded PNG. Optional."
    )


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
# what is serving: URL model file name and the deployed source commit
_meta: dict[str, str | None] = {"url_model": None, "build": None}


def _read_build_commit(path: Path) -> str | None:
    """Return the commit recorded by the deploy script, or None outside a deploy image."""
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    parts = first.split()
    return parts[1] if len(parts) == 2 and parts[0] == "commit" else None


def _pick_url_artifact(v0_1_name: str) -> Path:
    """Prefer the v0.2 ablation artifact when present; fall back to v0.1.

    ``url_model.onnx`` -> ``url_model_v0_2.onnx`` and so on. Resolved against
    ``CKPT_DIR`` at call time so tests can point the app at a fixture directory.
    """
    stem, suffix = v0_1_name.rsplit(".", 1)
    v0_2 = CKPT_DIR / f"{stem}_v0_2.{suffix}"
    return v0_2 if v0_2.exists() else CKPT_DIR / v0_1_name


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
    url_onnx_path = _pick_url_artifact(URL_ONNX_NAME)
    url_calib_path = _pick_url_artifact(URL_CALIB_NAME)
    url_feats_path = _pick_url_artifact(URL_FEATS_NAME)
    url_tlds_path = _pick_url_artifact(URL_TLDS_NAME)
    if not url_onnx_path.exists():
        raise FileNotFoundError(
            f"{url_onnx_path} missing. Run `make export-url-onnx` (or `make train-url-v0_2`, "
            "which exports as its last step) to produce the ONNX URL model."
        )
    logger.info("url model (onnxruntime): %s", url_onnx_path)
    _meta["url_model"] = url_onnx_path.name
    _meta["build"] = _read_build_commit(BUILD_MANIFEST)
    _state["url_session"] = UrlOnnxModel.from_paths(url_onnx_path, url_feats_path, url_tlds_path)
    _state["url_calibrator"] = _load_calibrator(url_calib_path)
    _state["url_feature_names"] = _state["url_session"].feature_names
    _state["url_extractor"] = URLFeatureExtractor()

    html_onnx = CKPT_DIR / HTML_ONNX_NAME
    img_onnx = CKPT_DIR / IMG_ONNX_NAME
    fusion_path = CKPT_DIR / FUSION_NAME
    if html_onnx.exists():
        _state["html_session"] = ort.InferenceSession(
            str(html_onnx), providers=["CPUExecutionProvider"]
        )
        from transformers import AutoTokenizer

        _state["html_tokenizer"] = AutoTokenizer.from_pretrained("models/checkpoints/html_model")
    if img_onnx.exists():
        _state["img_session"] = ort.InferenceSession(
            str(img_onnx), providers=["CPUExecutionProvider"]
        )

    _state["fusion"] = FusionModel.load(fusion_path) if fusion_path.exists() else None
    _state["pred_log"] = PredictionLogger()
    threshold = _state["fusion"].threshold if _state["fusion"] is not None else 0.5
    logger.info("ready (threshold=%.4f)", threshold)
    yield
    logger.info("shutting down")
    log = _state.get("pred_log")
    if log is not None:
        log.close()


app = FastAPI(title="PhishGuard", version="0.1.0", lifespan=lifespan)
app.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)


@app.exception_handler(RequestValidationError)  # type: ignore[misc]
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 body echoes the rejected input back, so a 2 MB
    # field comes back as a 2 MB error. Keep only where and why it failed.
    errors = [{k: e[k] for k in ("loc", "msg", "type") if k in e} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/healthz")  # type: ignore[misc]
def healthz() -> dict[str, Any]:
    # url_model names the ONNX file actually loaded, so a deploy check can
    # tell the v0.2 model from the v0.1 fallback in _pick_url_artifact.
    # build is the deployed source commit (None outside a deploy image).
    return {"status": "ok", "models_loaded": list(_state.keys()), **_meta}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)  # type: ignore[misc]
def demo_page() -> HTMLResponse:
    return HTMLResponse(DEMO_PAGE.read_text(encoding="utf-8"))


def _predict_url(url: str) -> float:
    """Featurize, run the ONNX session, then apply the isotonic calibrator."""
    rows = _state["url_extractor"].transform([url])
    raw = _state["url_session"].predict_raw(rows)[0]
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
        # Full exception is in the server log; the public response stays generic
        # so it never carries file paths or internals.
        raise HTTPException(status_code=500, detail="url model failure") from e

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
