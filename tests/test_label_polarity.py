"""Lock in the PhiUSIIL label polarity.

Regression test for the v0.1/v0.2 label-inversion bug discovered via
end-to-end smoke testing. PhiUSIIL's `label` column uses 1=legitimate,
0=phishing, which is the opposite of what the column name implies. Our
internal `is_phish` polarity is 1=phishing, 0=legitimate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from phishguard.data.load import _normalize_phiusiil_label

TRAIN = Path("data/processed/url_train.parquet")


def test_normalize_phiusiil_inverts_polarity() -> None:
    # PhiUSIIL raw label=1 -> legitimate (our is_phish=0)
    assert _normalize_phiusiil_label(1) == 0
    # PhiUSIIL raw label=0 -> phishing (our is_phish=1)
    assert _normalize_phiusiil_label(0) == 1
    # Word-form variants follow the same polarity
    assert _normalize_phiusiil_label("legitimate") == 0
    assert _normalize_phiusiil_label("phishing") == 1


@pytest.mark.skipif(not TRAIN.exists(), reason="train split not built yet")
def test_train_split_phish_class_smells_phishy() -> None:
    """Soft smoke check: a sample of is_phish=1 URLs should look more phishy
    than is_phish=0 URLs by simple heuristics. This catches future inversions
    without depending on the model itself."""
    df = pd.read_parquet(TRAIN)
    phish = df[df["is_phish"] == 1].sample(n=500, random_state=0)
    legit = df[df["is_phish"] == 0].sample(n=500, random_state=0)

    # Phishing URLs in PhiUSIIL skew toward HTTP, not HTTPS, and toward suspicious
    # hosting platforms (firebaseapp.com, weebly, .xyz). Use a vocabulary check.
    suspicious_tokens = (
        "firebaseapp",
        "weebly",
        ".xyz",
        ".tk",
        ".ga",
        ".cf",
        "login.php",
        ".cloudwaysapps",
        "accounts-",
        "verify",
        "secure-",
    )

    def hit_rate(urls: pd.Series) -> float:
        s = urls.str.lower()
        hits = pd.Series(False, index=s.index)
        for tok in suspicious_tokens:
            hits = hits | s.str.contains(tok, regex=False, na=False)
        return float(hits.mean())

    phish_rate = hit_rate(phish["url"])
    legit_rate = hit_rate(legit["url"])

    assert phish_rate > legit_rate, (
        f"is_phish=1 token-hit rate ({phish_rate:.2%}) should exceed "
        f"is_phish=0 ({legit_rate:.2%}) -- labels likely inverted again"
    )
