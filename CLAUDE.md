# PhishGuard - Claude Code context

Multi-modal phishing detector. Late fusion of URL gradient boosting + DistilBERT
(HTML) + EfficientNet-B0 (screenshots). Calibrated probabilities throughout.

## Current state (2026-04-29)

- Tagged: `v0.1-url-baseline` (URL model only, AUC=0.999 on PhiUSIIL)
- Tagged: `v0.1-hardened` (pre-commit gate + leakage test added)
- Top-3 features account for 95% of gain (`has_https`, `path_length`,
  `has_http`). Documented in `LIMITATIONS.md`. Multi-modal v1.0 is the answer.
- Tranco top-5000 probe: median p_phish=0.008, FPR@0.7=2.78%
- `test_no_feature_dominates_gain` is marked `known_fail` -- it catches the
  documented has_https leak (47.6%) and will flip to pass once v0.2 retrains
  without the leakage shortcut.

## Stack

- Python 3.12 (pyenv), `>=3.11` required
- LightGBM (URL), DistilBERT/HuggingFace (HTML), EfficientNet-B0 (screenshots),
  scikit-learn (fusion + calibration)
- FastAPI + ONNX Runtime (serving), Playwright (scraping)
- pytest + hypothesis (tests), Evidently AI (drift)
- Wandb is optional. Set `WANDB_MODE=disabled` to skip.

## Conventions

- Training scripts take `--config <path>` and read Hydra YAML
- Artifacts go to `models/checkpoints/`
- Metrics written to `reports/<modality>_baseline_<tag>.txt`
- Calibrated probabilities everywhere (isotonic on val set)
- Modalities are independent; serving degrades gracefully if any are missing

## Don't repeat

- AUC=1.0 is a red flag, not a celebration. Always inspect feature importance.
- Tranco mixed with PhiUSIIL creates a trivial subdomain shortcut. Use
  PhiUSIIL's native benign labels. Tranco is a held-out probe, not training.
- Hypothesis property-based tests catch input bugs example tests miss.
- Pre-commit is now fully configured (`.pre-commit-config.yaml`). Never
  uninstall or skip hooks. ruff, ruff-format, mypy, and file checks all run.
- `kaggle` CLI v2 uses `~/.kaggle/access_token`, not `kaggle.json`.
- Playwright Chromium is installed at `~/Library/Caches/ms-playwright/chromium-1208/`.
  Verify with `ls ~/Library/Caches/ms-playwright/chromium-1208/INSTALLATION_COMPLETE`
  before running any scraping steps.

## Next executable steps

1. ~~Add `.pre-commit-config.yaml` and reinstall the hook.~~ Done (v0.1-hardened)
2. ~~`tests/test_no_leakage.py`: assert no single feature gain exceeds 30%.~~ Done (known_fail on v0.1)
3. `make scrape` smoke test on 100 URLs from `data/processed/url_test.parquet`.
   Verify HTML + PNG land in `data/processed/snapshots/`. Playwright Chromium
   confirmed installed.
4. `make train-html` once scraper smoke test passes.
5. `make train-img` once enough screenshots exist.

## Quick verify

    git status
    git log --oneline -5
    git tag -l
    .venv/bin/python scripts/inspect_url_baseline.py
