"""Tests for scripts/smoke_live.py: the live check must pass on a healthy service and
fail on the failure shapes this project has actually had (inverted labels, the v0.1
fallback model, train/serve skew)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_live", ROOT / "scripts" / "smoke_live.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up while exec runs
    spec.loader.exec_module(mod)
    return mod


smoke = _load()
FIXTURES: dict[str, Any] = json.loads((ROOT / "scripts" / "smoke_fixtures.json").read_text())
GOOD_HEALTH = {"status": "ok", "url_model": "url_model_v0_2.onnx"}


def _expected_scores() -> dict[str, float | None]:
    return {r["url"]: r["expected"] for r in [*FIXTURES["phish"], *FIXTURES["tranco"]]}


def _failed(checks: list[Any]) -> set[str]:
    return {c.name for c in checks if not c.ok}


def test_fixture_shape() -> None:
    assert FIXTURES["model_file"] == smoke.EXPECTED_MODEL_FILE
    assert len(FIXTURES["phish"]) >= 20
    assert len(FIXTURES["tranco"]) >= 50
    urls = [r["url"] for r in [*FIXTURES["phish"], *FIXTURES["tranco"]]]
    assert len(urls) == len(set(urls)), "phish and tranco fixtures overlap"


def test_healthy_service_passes() -> None:
    checks = smoke.evaluate(GOOD_HEALTH, True, 422, _expected_scores(), FIXTURES)
    assert _failed(checks) == set(), [c.detail for c in checks if not c.ok]


def test_inverted_labels_fail() -> None:
    # The PhiUSIIL polarity bug: every score flipped.
    flipped = {u: 1.0 - float(p) for u, p in _expected_scores().items() if p is not None}
    failed = _failed(smoke.evaluate(GOOD_HEALTH, True, 422, flipped, FIXTURES))  # type: ignore[arg-type]
    assert {"phish", "tranco", "parity"} <= failed


def test_v0_1_fallback_fails_health() -> None:
    health = {"status": "ok", "url_model": "url_model.onnx"}
    assert "health" in _failed(smoke.evaluate(health, True, 422, _expected_scores(), FIXTURES))


def test_small_skew_fails_parity_only() -> None:
    skewed = dict(_expected_scores())
    url = FIXTURES["tranco"][3]["url"]
    skewed[url] = float(skewed[url] or 0.0) + 1e-3
    assert _failed(smoke.evaluate(GOOD_HEALTH, True, 422, skewed, FIXTURES)) == {"parity"}


def test_stale_build_fails_commit_check() -> None:
    # A Space still serving the previous deploy right after an upload.
    health = {**GOOD_HEALTH, "build": "aaaa"}
    checks = smoke.evaluate(health, True, 422, _expected_scores(), FIXTURES, expect_commit="bbbb")
    assert _failed(checks) == {"build"}
    ok = smoke.evaluate(health, True, 422, _expected_scores(), FIXTURES, expect_commit="aaaa")
    assert _failed(ok) == set()


def test_build_manifest_parsing(tmp_path: Path) -> None:
    from phishguard.serving.api import _read_build_commit

    manifest = tmp_path / "ARTIFACTS.sha256"
    manifest.write_text("commit abc123\nfff  models/checkpoints/x.onnx\n")
    assert _read_build_commit(manifest) == "abc123"
    assert _read_build_commit(tmp_path / "missing") is None
    manifest.write_text("")
    assert _read_build_commit(manifest) is None
    manifest.write_text("not a manifest line\n")
    assert _read_build_commit(manifest) is None


def test_unreachable_service_fails_everything() -> None:
    scores: dict[str, float | None] = dict.fromkeys(_expected_scores())
    failed = _failed(smoke.evaluate(None, False, None, scores, FIXTURES))
    assert failed == {"health", "demo", "validation", "parity", "phish", "tranco"}


_CKPT = ROOT / "models" / "checkpoints" / "url_model_v0_2.onnx"


@pytest.mark.skipif(not _CKPT.exists(), reason="URL v0.2 ONNX artifact not built")
def test_fixtures_match_local_artifacts() -> None:
    """The committed expected scores are what the local served path produces today."""
    from fastapi.testclient import TestClient
    from phishguard.serving.api import app

    with TestClient(app) as c:
        for row in [*FIXTURES["phish"], *FIXTURES["tranco"]]:
            p = c.post("/predict", json={"url": row["url"]}).json()["p_phish"]
            assert abs(p - row["expected"]) <= smoke.PARITY_TOL, row["url"]
