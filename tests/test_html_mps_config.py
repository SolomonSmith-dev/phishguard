"""MPS smoke config and the optional wandb / memory guard in train_html."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "html_model_mps.yaml"


def test_mps_config_values() -> None:
    cfg = yaml.safe_load(CFG.read_text())
    assert cfg["model"]["max_length"] == 512
    assert cfg["training"]["batch_size"] == 4
    assert cfg["training"]["grad_accum_steps"] == 8
    assert cfg["training"]["fp16"] is False
    assert cfg["training"]["epochs"] == 1
    assert cfg["logging"]["report_to"] == "none"
    assert cfg["training"]["max_rss_gb"] <= 6


def test_original_config_untouched() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "html_model.yaml").read_text())
    assert cfg["training"]["batch_size"] == 16


def test_wandb_off_by_default() -> None:
    from phishguard.training.train_html import build_report_to

    assert build_report_to({}) == []
    assert build_report_to({"logging": {"report_to": "none"}}) == []
    assert build_report_to({"logging": {"report_to": "wandb"}}) == ["wandb"]


def test_rss_guard_aborts_over_limit() -> None:
    from phishguard.training.train_html import MemoryLimitExceededError, check_rss

    check_rss(limit_gb=1000.0)
    with pytest.raises(MemoryLimitExceededError):
        check_rss(limit_gb=0.0001)


def test_warmup_kwargs_follow_installed_transformers() -> None:
    import inspect

    from phishguard.training.train_html import warmup_kwargs
    from transformers import TrainingArguments

    kw = warmup_kwargs(0.1)
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    assert len(kw) == 1 and set(kw) <= accepted
