"""URL canonicalization is applied at both train and serve time, so it has
to be deterministic and order-preserving for the failure modes that matter."""

from __future__ import annotations

from phishguard.features import canonicalize


def test_strips_leading_www() -> None:
    assert canonicalize("https://www.example.com") == "https://example.com"
    assert canonicalize("http://www.foo.bar") == "http://foo.bar"


def test_strips_trailing_root_slash_only() -> None:
    assert canonicalize("https://example.com/") == "https://example.com"
    assert canonicalize("https://example.com/path/") == "https://example.com/path/"


def test_drops_fragment() -> None:
    assert canonicalize("https://x.com/page#section") == "https://x.com/page"


def test_adds_default_scheme() -> None:
    assert canonicalize("example.com").startswith("http://")


def test_handles_empty() -> None:
    assert canonicalize("") == "http://"
    assert canonicalize("   ") == "http://"


def test_idempotent() -> None:
    """canonicalize(canonicalize(x)) == canonicalize(x)."""
    samples = [
        "https://www.example.com/",
        "http://WWW.foo.bar/path/",
        "example.com",
        "https://x.com/page#frag",
    ]
    for u in samples:
        once = canonicalize(u)
        assert canonicalize(once) == once, f"{u!r} not idempotent: {once!r}"


def test_preserves_non_root_paths() -> None:
    assert (
        canonicalize("https://github.com/anthropics/claude-code")
        == "https://github.com/anthropics/claude-code"
    )


def test_case_insensitive_www_strip() -> None:
    """`WWW.test.com` should canonicalize to drop the www regardless of case."""
    assert canonicalize("https://WWW.test.com") == "https://test.com"
