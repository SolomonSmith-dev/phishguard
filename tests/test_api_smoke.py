"""Smoke tests for the FastAPI service. Uses TestClient so no live server required.

Skipped automatically if the URL artifacts are missing (fresh-clone / CI).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_CKPT = Path("models/checkpoints")
# Accept v0.1 (url_model.lgb) or any versioned artifact (url_model_v*.lgb).
_URL_ARTIFACTS_PRESENT = bool(
    list(_CKPT.glob("url_model*.lgb")) if _CKPT.is_dir() else []
)
pytestmark = pytest.mark.skipif(not _URL_ARTIFACTS_PRESENT, reason="URL artifacts not built yet")


@pytest.fixture(scope="module")
def client():
    from phishguard.serving.api import app

    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "url_booster" in body["models_loaded"]


def test_predict_benign_tranco(client):
    r = client.post("/predict", json={"url": "https://google.com"})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["p_phish"] <= 1.0
    assert body["modalities"]["url"]["available"] is True
    assert body["modalities"]["html"]["available"] is False
    assert body["modalities"]["img"]["available"] is False


def test_predict_obvious_phish(client):
    r = client.post(
        "/predict",
        json={"url": "https://hd74-nb-5d.web.app/"},
    )
    assert r.status_code == 200
    body = r.json()
    # This one is from the PhiUSIIL phish class -- model should at least
    # rank it above the very-low benign band. We don't assert >= threshold
    # because the URL-only baseline is documented to be brittle.
    assert body["p_phish"] > 0.1


def test_predict_rejects_too_short_url(client):
    r = client.post("/predict", json={"url": "ab"})
    assert r.status_code == 422  # pydantic min_length
