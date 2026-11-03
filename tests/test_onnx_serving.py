"""The URL route is served by ONNX Runtime, not the native LightGBM booster.

Builds a tiny LightGBM URL model in a fixture (no checkpoint on disk needed),
exports it with ``phishguard.training.export_url_onnx``, points the FastAPI
app at the fixture directory and checks that:

    1. the loaded URL model object wraps an ``onnxruntime.InferenceSession``;
    2. ``POST /predict`` goes through ``InferenceSession.run`` (monkeypatched);
    3. the exported graph agrees with the native booster within 1e-5.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient
from phishguard.serving import api
from phishguard.serving.url_onnx import UrlOnnxModel
from phishguard.training.export_url_onnx import export_booster, native_predict
from phishguard.training.train_url import featurize
from sklearn.isotonic import IsotonicRegression

DROP = ["has_https", "has_http", "url_length", "host_length", "path_length"]

_BENIGN_HOSTS = ["example", "wikipedia", "github", "python", "mozilla", "nytimes", "bbc", "apache"]
_BENIGN_TLDS = ["com", "org", "net", "edu", "io", "gov"]
_PHISH_HOSTS = ["secure-login", "account-verify", "paypal-update", "signin-alert", "wallet-confirm"]
_PHISH_TLDS = ["tk", "xyz", "top", "ml", "ga", "click"]


def _synthetic_urls() -> tuple[list[str], np.ndarray]:
    urls: list[str] = []
    labels: list[int] = []
    for i, host in enumerate(_BENIGN_HOSTS):
        for tld in _BENIGN_TLDS:
            urls.append(f"https://{host}{i}.{tld}")
            labels.append(0)
            urls.append(f"https://{host}.{tld}/about")
            labels.append(0)
    for i, host in enumerate(_PHISH_HOSTS):
        for tld in _PHISH_TLDS:
            urls.append(f"http://{host}{i}.app-{i}.{tld}/login.php?id={i}")
            labels.append(1)
            urls.append(f"http://verify.{host}.{tld}/account/update")
            labels.append(1)
    return urls, np.asarray(labels)


@pytest.fixture(scope="module")
def tiny_checkpoints(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ckpt = tmp_path_factory.mktemp("checkpoints")
    urls, y = _synthetic_urls()
    x = featurize(urls, drop=DROP)
    booster = lgb.train(
        {
            "objective": "binary",
            "num_leaves": 7,
            "min_data_in_leaf": 5,
            "learning_rate": 0.2,
            "verbose": -1,
            "seed": 42,
        },
        lgb.Dataset(x, y, categorical_feature=["tld"]),
        num_boost_round=20,
    )
    raw = booster.predict(x)
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y)

    booster.save_model(ckpt / "url_model_v0_2.lgb")
    joblib.dump(iso, ckpt / "url_calibrator_v0_2.pkl")
    (ckpt / "url_features_v0_2.json").write_text(json.dumps(list(x.columns)))
    export_booster(
        booster,
        list(x.columns),
        ckpt / "url_model_v0_2.onnx",
        ckpt / "url_tld_categories_v0_2.json",
    )
    return ckpt


@pytest.fixture()
def client(tiny_checkpoints: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api, "CKPT_DIR", tiny_checkpoints)
    api._state.clear()
    with TestClient(api.app) as c:
        yield c
    api._state.clear()


def test_url_model_is_an_onnxruntime_session(client: TestClient) -> None:
    model = api._state["url_session"]
    assert isinstance(model, UrlOnnxModel)
    assert isinstance(model.session, ort.InferenceSession)
    assert "url_session" in client.get("/healthz").json()["models_loaded"]
    assert "url_booster" not in api._state


def test_predict_route_runs_the_onnx_session(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = api._state["url_session"].session
    calls: list[dict[str, np.ndarray]] = []
    real_run = session.run

    def spy_run(output_names, input_feed, run_options=None):  # type: ignore[no-untyped-def]
        calls.append(input_feed)
        return real_run(output_names, input_feed, run_options)

    monkeypatch.setattr(session, "run", spy_run)
    r = client.post("/predict", json={"url": "http://secure-login9.app-9.tk/login.php?id=9"})
    assert r.status_code == 200
    assert len(calls) == 1, "URL route did not call InferenceSession.run exactly once"
    (feed,) = calls
    assert set(feed) == {api._state["url_session"].input_name}
    assert feed[api._state["url_session"].input_name].shape == (1, 42)
    body = r.json()
    assert 0.0 <= body["p_phish"] <= 1.0
    assert body["modalities"]["url"]["available"] is True


def test_predict_without_onnx_artifact_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "CKPT_DIR", tmp_path)
    api._state.clear()
    with pytest.raises(FileNotFoundError, match="export-url-onnx"), TestClient(api.app):
        pass


def test_onnx_matches_native_booster(tiny_checkpoints: Path) -> None:
    booster = lgb.Booster(model_file=str(tiny_checkpoints / "url_model_v0_2.lgb"))
    model = UrlOnnxModel.from_paths(
        tiny_checkpoints / "url_model_v0_2.onnx",
        tiny_checkpoints / "url_features_v0_2.json",
        tiny_checkpoints / "url_tld_categories_v0_2.json",
    )
    urls, _ = _synthetic_urls()
    # Include URLs the model never saw, with a TLD outside the training categories.
    urls += ["https://unseen-host.example.museum/path/a/b", "http://10.0.0.1/login"]
    rows = api.URLFeatureExtractor().transform(urls)
    native = native_predict(booster, rows, model.feature_names)
    onnx = model.predict_raw(rows)
    assert np.abs(native - onnx).max() < 1e-5


def test_onnx_input_is_float64_with_exact_thresholds(tiny_checkpoints: Path) -> None:
    sess = ort.InferenceSession(
        str(tiny_checkpoints / "url_model_v0_2.onnx"), providers=["CPUExecutionProvider"]
    )
    (inp,) = sess.get_inputs()
    assert inp.type == "tensor(double)"
    feats = json.loads((tiny_checkpoints / "url_features_v0_2.json").read_text())
    assert inp.shape == [None, len(feats)]
    tlds = json.loads((tiny_checkpoints / "url_tld_categories_v0_2.json").read_text())
    assert set(tlds) >= set(_BENIGN_TLDS + _PHISH_TLDS)
