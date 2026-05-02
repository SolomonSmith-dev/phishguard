"""Prediction log behavior: redaction of credential params and graceful
fallback when DATABASE_URL is unset."""

from __future__ import annotations

import json

from phishguard.serving.prediction_log import PredictionLogger, redact


def test_redacts_token_param() -> None:
    out = redact("https://x.com/login?token=abc123def&u=alice")
    assert "abc123def" not in out
    assert "u=alice" in out


def test_redacts_multiple_credential_params() -> None:
    out = redact("https://x.com/?api_key=KKK&access_token=TTT&safe=ok")
    assert "KKK" not in out and "TTT" not in out
    assert "safe=ok" in out


def test_redacts_case_insensitive() -> None:
    out = redact("https://x.com/?Token=ABC&PASSWORD=xyz")
    assert "ABC" not in out
    assert "xyz" not in out


def test_passthrough_when_no_credentials() -> None:
    url = "https://example.com/page?q=hello"
    assert redact(url) == url


def test_logger_falls_back_to_jsonl(tmp_path, monkeypatch) -> None:
    """Logger writes to /tmp JSONL when DATABASE_URL is missing."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    fallback = tmp_path / "predictions.jsonl"
    monkeypatch.setattr("phishguard.serving.prediction_log._FALLBACK_PATH", fallback)

    log = PredictionLogger()
    log.log(
        {
            "url": "https://x.com/?token=secret",
            "p_url": 0.1,
            "p_html": None,
            "p_img": None,
            "p_phish": 0.1,
            "threshold": 0.5,
            "is_phish": False,
            "latency_ms_total": 1.5,
            "had_html": False,
            "had_img": False,
        }
    )

    assert fallback.exists()
    rec = json.loads(fallback.read_text().strip())
    assert rec["url_redacted"].endswith("token=<redacted>")
    assert rec["url"] == "https://x.com/?token=secret"  # raw url stays for ops
    assert rec["p_phish"] == 0.1
