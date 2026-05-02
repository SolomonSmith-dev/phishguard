"""URL feature engineering.

~63 numeric features per URL. Pure-Python, dependency-light, fast enough
to run inline in the API request path.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from urllib.parse import ParseResult, parse_qs, urlparse

import tldextract

_BRAND_KEYWORDS = (
    "paypal",
    "apple",
    "microsoft",
    "amazon",
    "google",
    "facebook",
    "instagram",
    "netflix",
    "chase",
    "wellsfargo",
    "bankofamerica",
    "citi",
    "office365",
    "outlook",
    "gmail",
    "linkedin",
    "dropbox",
    "docusign",
    "irs",
    "usps",
    "fedex",
    "dhl",
    "ups",
    "binance",
    "coinbase",
    "metamask",
)
_SUSPICIOUS_TLDS = frozenset(
    {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "country", "support", "loan", "online"}
)
_SHORTENERS = frozenset(
    {
        "bit.ly",
        "tinyurl.com",
        "goo.gl",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "adf.ly",
        "rebrand.ly",
        "cutt.ly",
        "shorturl.at",
    }
)
_SUSPICIOUS_KEYWORDS = (
    "secure",
    "account",
    "update",
    "verify",
    "login",
    "signin",
    "confirm",
    "banking",
    "wallet",
    "webscr",
    "billing",
    "password",
    "alert",
)

_IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$|^\[?(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\]?$")
_HEX_RE = re.compile(r"%[0-9a-fA-F]{2}")
_PUNYCODE_RE = re.compile(r"xn--", re.IGNORECASE)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _safe_urlparse(url: str) -> ParseResult:
    """urlparse can raise ValueError on malformed bracketed input. Fall back gracefully."""
    candidate = url if "://" in url else f"http://{url}"
    try:
        return urlparse(candidate)
    except ValueError:
        return urlparse("http://invalid.invalid")


def canonicalize(url: str) -> str:
    """Make trivially-different URLs land in the same feature bucket.

    PhiUSIIL legit URLs are 100% `https://www.*` while Tranco probe URLs are
    bare hosts. Without normalization the model becomes a `www.` detector
    instead of a phishing detector. We strip `www.` and trailing slashes at
    both train and serve time so distributions match.

    Steps:
        1. Strip leading/trailing whitespace.
        2. If host has no scheme prefix, default to http.
        3. Drop a single leading `www.` from the host.
        4. Drop trailing slash when the path is just '/'.
        5. Drop URL fragment (#section).
    """
    s = (url or "").strip()
    if "://" not in s:
        s = f"http://{s}"
    if "#" in s:
        s = s.split("#", 1)[0]
    # Strip a single leading www. from the host (case-insensitive).
    for prefix in ("https://www.", "http://www."):
        if s.lower().startswith(prefix):
            scheme_end = len(prefix) - len("www.")
            s = s[:scheme_end] + s[scheme_end + 4 :]
            break
    if s.endswith("/") and s.count("/") == 3:
        s = s.rstrip("/")
    return s


@dataclass(slots=True)
class URLFeatures:
    url_length: int
    host_length: int
    path_length: int
    query_length: int
    num_dots: int
    num_hyphens: int
    num_underscores: int
    num_slashes: int
    num_digits: int
    num_at: int
    num_question: int
    num_equals: int
    num_ampersand: int
    num_percent: int
    num_special: int
    digit_ratio: float
    letter_ratio: float
    special_ratio: float
    has_https: int
    has_http: int
    has_port: int
    port_is_nonstandard: int
    is_ip_host: int
    is_punycode: int
    has_at_symbol: int
    has_double_slash_in_path: int
    has_hex_encoding: int
    hex_encoding_ratio: float
    has_shortener: int
    subdomain_depth: int
    subdomain_length: int
    domain_length: int
    tld: str
    is_suspicious_tld: int
    path_depth: int
    num_query_params: int
    longest_path_token_length: int
    has_file_extension: int
    suspicious_keyword_count: int
    brand_in_subdomain: int
    brand_in_domain: int
    brand_in_path: int
    brand_count_total: int
    url_entropy: float
    host_entropy: float
    path_entropy: float
    has_unicode: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {f: getattr(self, f) for f in self.__slots__}


def extract_url_features(url: str) -> URLFeatures:
    url = canonicalize(url)
    parsed = _safe_urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain or ""
    domain = extracted.domain or ""
    tld = (extracted.suffix or "").lower()

    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    specials = sum(not c.isalnum() for c in url)
    n = max(len(url), 1)

    hex_matches = _HEX_RE.findall(url)
    has_hex = int(bool(hex_matches))
    hex_ratio = (sum(len(m) for m in hex_matches) / n) if hex_matches else 0.0

    suspicious_kw = sum(1 for kw in _SUSPICIOUS_KEYWORDS if kw in url.lower())

    brand_in_sub = sum(1 for b in _BRAND_KEYWORDS if b in subdomain.lower())
    brand_in_dom = sum(1 for b in _BRAND_KEYWORDS if b in domain.lower())
    brand_in_pth = sum(1 for b in _BRAND_KEYWORDS if b in path.lower())

    path_tokens = [t for t in path.split("/") if t]
    longest_path_token = max((len(t) for t in path_tokens), default=0)
    file_ext = bool(re.search(r"\.[a-z0-9]{1,5}$", path.lower()))

    is_ip = int(bool(_IP_RE.match(host)))
    is_puny = int(bool(_PUNYCODE_RE.search(host)))
    has_at = int("@" in url)
    has_double_slash = int("//" in path[1:])

    has_shortener = int(any(host.endswith(s) for s in _SHORTENERS))

    try:
        port = parsed.port
    except ValueError:
        port = None
    has_port = int(port is not None)
    port_nonstd = int(has_port and port not in (80, 443))

    has_unicode = int(any(ord(c) > 127 for c in url))

    return URLFeatures(
        url_length=len(url),
        host_length=len(host),
        path_length=len(path),
        query_length=len(query),
        num_dots=url.count("."),
        num_hyphens=url.count("-"),
        num_underscores=url.count("_"),
        num_slashes=url.count("/"),
        num_digits=digits,
        num_at=url.count("@"),
        num_question=url.count("?"),
        num_equals=url.count("="),
        num_ampersand=url.count("&"),
        num_percent=url.count("%"),
        num_special=specials,
        digit_ratio=digits / n,
        letter_ratio=letters / n,
        special_ratio=specials / n,
        has_https=int(parsed.scheme == "https"),
        has_http=int(parsed.scheme == "http"),
        has_port=has_port,
        port_is_nonstandard=port_nonstd,
        is_ip_host=is_ip,
        is_punycode=is_puny,
        has_at_symbol=has_at,
        has_double_slash_in_path=has_double_slash,
        has_hex_encoding=has_hex,
        hex_encoding_ratio=hex_ratio,
        has_shortener=has_shortener,
        subdomain_depth=len([s for s in subdomain.split(".") if s]),
        subdomain_length=len(subdomain),
        domain_length=len(domain),
        tld=tld,
        is_suspicious_tld=int(tld in _SUSPICIOUS_TLDS),
        path_depth=len(path_tokens),
        num_query_params=len(parse_qs(query)),
        longest_path_token_length=longest_path_token,
        has_file_extension=int(file_ext),
        suspicious_keyword_count=suspicious_kw,
        brand_in_subdomain=int(brand_in_sub > 0),
        brand_in_domain=int(brand_in_dom > 0),
        brand_in_path=int(brand_in_pth > 0),
        brand_count_total=brand_in_sub + brand_in_dom + brand_in_pth,
        url_entropy=_shannon_entropy(url),
        host_entropy=_shannon_entropy(host),
        path_entropy=_shannon_entropy(path),
        has_unicode=has_unicode,
    )


class URLFeatureExtractor:
    NUMERIC_FIELDS = tuple(f for f in URLFeatures.__slots__ if f != "tld")
    CATEGORICAL_FIELDS = ("tld",)

    def transform(self, urls: list[str]) -> list[dict[str, float | int | str]]:
        return [extract_url_features(u).to_dict() for u in urls]
