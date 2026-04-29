# PhishGuard - Claude Code context

Multi-modal phishing detector. Late fusion of URL gradient boosting + DistilBERT
(HTML) + EfficientNet-B0 (screenshots). Calibrated probabilities throughout.

## Current state (2026-04-29)

- Tagged: `v0.1-url-baseline` (URL model only, AUC=0.999 on PhiUSIIL)
- Top-3 features account for 95% of gain (`has_https`, `path_length`,
  `has_http`). Documented in `LIMITATIONS.md`. Multi-modal v1.0 is the answer.
- Tranco top-5000 probe: median p_phish=0.008, FPR@0.7=2.78%

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
- Pre-commit installed without a config silently blocks every commit. Already
  uninstalled in this repo. Re-install only with a real `.pre-commit-config.yaml`.
- `kaggle` CLI v2 uses `~/.kaggle/access_token`, not `kaggle.json`.

## Next executable steps

1. Add `.pre-commit-config.yaml` (ruff, ruff-format, end-of-file-fixer,
   check-large-files, mypy on src) and reinstall the hook.
2. `tests/test_no_leakage.py`: assert no single feature gain exceeds 30%.
3. `make scrape` smoke test on 100 URLs from `data/processed/url_test.parquet`.
   Verify HTML + PNG land in `data/processed/snapshots/`.
4. `make train-html` once scraper smoke test passes.
5. `make train-img` once enough screenshots exist.

## Quick verify

    git status
    git log --oneline -5
    git tag -l
    .venv/bin/python scripts/inspect_url_baseline.py
