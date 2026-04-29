"""Tests for URL feature extraction.

These lock in feature contracts so future changes don't silently shift the
GBDT input distribution.
"""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st
from phishguard.features import URLFeatureExtractor, extract_url_features


def test_obvious_phish_signals():
    f = extract_url_features("http://paypal-secure-login.tk/account/verify?id=1")
    assert f.has_https == 0
    assert f.has_http == 1
    assert f.is_suspicious_tld == 1
    # paypal-secure-login is the registered domain (no subdomain), so check domain
    assert f.brand_in_domain == 1
    assert f.brand_count_total >= 1
    assert f.suspicious_keyword_count >= 2
    assert f.num_hyphens >= 2


def test_brand_in_subdomain():
    f = extract_url_features("http://paypal.evil-host.tk/login")
    assert f.brand_in_subdomain == 1


def test_benign_url_clean():
    f = extract_url_features("https://www.example.com/")
    assert f.has_https == 1
    assert f.is_suspicious_tld == 0
    assert f.is_ip_host == 0
    assert f.brand_count_total == 0


def test_ip_host_detected():
    f = extract_url_features("http://192.168.1.1/admin")
    assert f.is_ip_host == 1


def test_at_symbol_redirect():
    f = extract_url_features("http://google.com@evil.com/")
    assert f.has_at_symbol == 1


def test_punycode_detected():
    f = extract_url_features("https://xn--pple-43d.com/login")
    assert f.is_punycode == 1


def test_shortener_detected():
    f = extract_url_features("https://bit.ly/abc123")
    assert f.has_shortener == 1


def test_extractor_batch():
    ex = URLFeatureExtractor()
    rows = ex.transform(["https://a.com", "http://b.tk/login"])
    assert len(rows) == 2
    assert all("url_length" in r for r in rows)


def test_entropy_is_finite():
    f = extract_url_features(
        "https://verylongdomainnamewithlotsofcharacters.example.com/very/deep/path"
    )
    assert math.isfinite(f.url_entropy)
    assert f.url_entropy > 0


def test_handles_malformed_urls():
    """Regression: hypothesis found ValueError on '[' (Invalid IPv6 URL)."""
    for bad in ["[", "]", "[::", "http://[bad", "://", "", "   ", "%"]:
        f = extract_url_features(bad)
        assert f.url_length >= 0


@given(st.text(min_size=1, max_size=200))
def test_never_raises(s):
    f = extract_url_features(s)
    assert f.url_length >= 0
