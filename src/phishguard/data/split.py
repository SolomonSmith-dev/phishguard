"""Domain-grouped train/val/test split.

Rows that share a registered domain (eTLD+1) always land in the same split, so a site
seen in training can never also be in test. Uses the tldextract bundled suffix snapshot
only: no network call is made.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

import pandas as pd
import tldextract

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def registered_domain(url: str) -> str:
    """Return the registered domain of a URL, or the bare host for IPs and unknown suffixes."""
    if not isinstance(url, str) or not url:
        return ""
    try:
        ext = _EXTRACT(url)
    except Exception:
        return ""
    if ext.top_domain_under_public_suffix:
        return str(ext.top_domain_under_public_suffix).lower()
    return str(ext.domain or ext.suffix or "").lower()


def split_by_domain(
    df: pd.DataFrame,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    url_col: str = "url",
) -> dict[str, pd.DataFrame]:
    """Split rows so no registered domain appears in more than one split."""
    domains = df[url_col].map(registered_domain)
    uniq = sorted(set(domains))
    random.Random(seed).shuffle(uniq)
    sizes = domains.value_counts()
    total = len(df)
    names = ("train", "val", "test")
    targets = [f * total for f in fractions]
    assign: dict[str, str] = {}
    filled = [0, 0, 0]
    for dom in uniq:
        # Put the domain in the split that is furthest below its target share.
        idx = max(range(3), key=lambda i: targets[i] - filled[i])
        assign[dom] = names[idx]
        filled[idx] += int(sizes[dom])
    which = domains.map(assign)
    parts = {n: df[which == n].reset_index(drop=True) for n in names}
    assert_no_domain_leak(parts, url_col=url_col)
    return parts


def assert_no_domain_leak(parts: Mapping[str, pd.DataFrame], url_col: str = "url") -> None:
    """Raise AssertionError if any registered domain is present in two splits."""
    seen: dict[str, str] = {}
    for name, part in parts.items():
        for dom in set(part[url_col].map(registered_domain)):
            if dom in seen and seen[dom] != name:
                raise AssertionError(f"domain leak: {dom!r} in {seen[dom]} and {name}")
            seen[dom] = name
