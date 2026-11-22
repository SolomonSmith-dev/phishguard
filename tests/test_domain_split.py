"""Domain-grouped split: no registered domain may appear in two splits."""

import pandas as pd
import pytest
from phishguard.data.split import registered_domain, split_by_domain


def _frame() -> pd.DataFrame:
    rows = []
    for i in range(60):
        dom = f"site{i}.com"
        for j in range(3):
            rows.append({"url": f"http://www{j}.{dom}/p{j}", "is_phish": i % 2, "html_text": "x"})
    return pd.DataFrame(rows)


def test_registered_domain_collapses_subdomains() -> None:
    assert registered_domain("http://a.b.example.co.uk/x") == "example.co.uk"
    assert registered_domain("https://login.evil.com:8080/p?q=1") == "evil.com"


def test_registered_domain_ip_and_garbage_do_not_raise() -> None:
    assert registered_domain("http://10.0.0.1/x") == "10.0.0.1"
    assert registered_domain("") == ""


def test_no_domain_in_two_splits() -> None:
    df = _frame()
    parts = split_by_domain(df, seed=42)
    doms = {k: set(v["url"].map(registered_domain)) for k, v in parts.items()}
    assert doms["train"].isdisjoint(doms["val"])
    assert doms["train"].isdisjoint(doms["test"])
    assert doms["val"].isdisjoint(doms["test"])
    assert sum(len(v) for v in parts.values()) == len(df)


def test_split_is_deterministic_and_nonempty() -> None:
    df = _frame()
    a = split_by_domain(df, seed=42)
    b = split_by_domain(df, seed=42)
    for k in ("train", "val", "test"):
        assert len(a[k]) > 0
        assert list(a[k]["url"]) == list(b[k]["url"])


def test_leak_check_raises_on_overlap() -> None:
    from phishguard.data.split import assert_no_domain_leak

    df = _frame()
    parts = split_by_domain(df, seed=42)
    parts["test"] = pd.concat([parts["test"], parts["train"].head(1)])
    with pytest.raises(AssertionError):
        assert_no_domain_leak(parts)
