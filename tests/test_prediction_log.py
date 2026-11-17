"""Prediction log behavior: redaction of credential params and graceful
fallback when DATABASE_URL is unset."""

from __future__ import annotations

import json

import pytest
from phishguard.serving.prediction_log import PredictionLogger, host_only, redact


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


_ROW = {
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


def test_redacts_userinfo_and_fragment_tokens() -> None:
    out = redact("https://bob:hunter2@x.com/cb#access_token=TTT&scope=read")
    assert "hunter2" not in out and "bob" not in out
    assert "TTT" not in out
    assert "scope=read" in out
    assert out.startswith("https://<redacted>@x.com/")


def test_redacts_oauth_and_signed_url_params() -> None:
    out = redact("https://x.com/cb?code=C0DE&state=S7&X-Amz-Signature=SIG&page=2")
    assert "C0DE" not in out and "S7" not in out and "SIG" not in out
    assert "page=2" in out


def test_public_mode_never_persists_raw_url(tmp_path, monkeypatch) -> None:
    """PHISHGUARD_STORE_RAW_URL=0 (the public Space) keeps only scheme://host."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PHISHGUARD_STORE_RAW_URL", "0")
    fallback = tmp_path / "predictions.log"
    monkeypatch.setattr("phishguard.serving.prediction_log._FALLBACK_PATH", fallback)

    PredictionLogger().log({"url": "https://u:pw@x.com:8443/reset/S3CR3T?token=secret", **_ROW})

    text = fallback.read_text()
    assert "secret" not in text and "S3CR3T" not in text and "pw" not in text
    rec = json.loads(text.strip())
    assert rec["url"] == rec["url_redacted"] == "https://x.com"


@pytest.mark.parametrize(
    ("url", "secret"),
    [
        ("https://x.com/cb?%74oken=ENC1", "ENC1"),  # percent-encoded key
        ("https://x.com/cb?my_token=SUF2", "SUF2"),  # key not on the denylist
        ("https://x.com/r?next=https%3A%2F%2Fy.com%3Ftoken%3DNEST3", "NEST3"),  # nested URL
        ("https://x.com/magic/PATH4", "PATH4"),  # secret in the path
        ("https://a@b:PW5@evil.com/", "PW5"),  # double @
        ("  javascript:alert('XSS6')", "XSS6"),  # no host at all
        ("x.com/login?pass=NOSCHEME7", "NOSCHEME7"),  # no scheme
        ("http://[::1/TOKEN8", "TOKEN8"),  # unparseable authority
        ("https://x.com\\reset\\BSLASH9", "BSLASH9"),  # backslash: path to a browser
        ("https://x.com%2fENC10/", "ENC10"),  # percent-encoded slash in the host
        ("https://x.com;SEMI11/", "SEMI11"),
    ],
)
def test_host_only_drops_every_secret_bearing_part(url: str, secret: str) -> None:
    out = host_only(url)
    assert secret not in out
    assert out.startswith(("http://", "https://", "<"))


def test_host_only_keeps_last_host_after_at() -> None:
    assert host_only("http://paypal.com@evil.example/login") == "http://evil.example"


def test_host_only_normal_hosts() -> None:
    assert host_only("https://Example.COM:8443/x") == "https://example.com"
    assert host_only("http://192.168.0.1/admin") == "http://192.168.0.1"
    assert host_only("http://[2001:db8::1]/x") == "http://2001:db8::1"
    assert host_only("http://例え.テスト/ログイン") == "http://xn--r8jz45g.xn--zckzah"


def test_fallback_stops_at_size_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    fallback = tmp_path / "predictions.log"
    monkeypatch.setattr("phishguard.serving.prediction_log._FALLBACK_PATH", fallback)
    monkeypatch.setattr("phishguard.serving.prediction_log._FALLBACK_MAX_BYTES", 10)
    fallback.write_text("x" * 10)

    PredictionLogger().log({"url": "https://x.com/", **_ROW})

    assert fallback.read_text() == "x" * 10


def test_fallback_write_error_does_not_raise(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # A directory where the file should be: open("a") raises IsADirectoryError, an OSError.
    fallback = tmp_path / "predictions.log"
    fallback.mkdir()
    monkeypatch.setattr("phishguard.serving.prediction_log._FALLBACK_PATH", fallback)
    monkeypatch.setattr("phishguard.serving.prediction_log._FALLBACK_MAX_BYTES", 1 << 40)

    PredictionLogger().log({"url": "https://x.com/", **_ROW})  # must not raise
