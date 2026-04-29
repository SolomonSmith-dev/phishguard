"""Assert no single feature dominates the URL model's gain (data-leakage proxy)."""

import json
from pathlib import Path

import pytest

MODEL_PATH = Path("models/checkpoints/url_model.lgb")
FEATURES_PATH = Path("models/checkpoints/url_features.json")
MAX_SINGLE_FEATURE_GAIN_FRACTION = 0.30


@pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="url_model.lgb not present -- skip on fresh clone"
)
def test_no_feature_dominates_gain() -> None:
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(MODEL_PATH))
    importance = booster.feature_importance(importance_type="gain")

    total = importance.sum()
    assert total > 0, "Model reports zero total gain -- something is wrong with the artifact"

    fractions = importance / total
    worst = fractions.max()
    worst_idx = int(fractions.argmax())

    feature_names = booster.feature_name()
    worst_name = feature_names[worst_idx] if feature_names else str(worst_idx)

    assert worst < MAX_SINGLE_FEATURE_GAIN_FRACTION, (
        f"Feature '{worst_name}' accounts for {worst:.1%} of total gain "
        f"(limit {MAX_SINGLE_FEATURE_GAIN_FRACTION:.0%}). "
        "Likely a leakage indicator -- audit feature importance before proceeding."
    )


@pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="url_model.lgb not present -- skip on fresh clone"
)
def test_feature_names_match_saved_json() -> None:
    """Feature names in the booster must match the saved url_features.json."""
    if not FEATURES_PATH.exists():
        pytest.skip("url_features.json not present")

    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(MODEL_PATH))
    saved: list[str] = json.loads(FEATURES_PATH.read_text())

    assert booster.feature_name() == saved, (
        "Booster feature names differ from url_features.json -- "
        "model and feature list are out of sync."
    )
