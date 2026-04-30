# PhishGuard - Claude Code context

Multi-modal phishing detector. Late fusion of URL gradient boosting + DistilBERT
(HTML) + EfficientNet-B0 (screenshots). Calibrated probabilities throughout.

## Current state (2026-04-30)

- Tags: `v0.1-url-baseline`, `v0.1-hardened`, `v0.2-url-ablation` (planned)
- Three load-bearing PhiUSIIL methodology issues found via E2E smoke testing:
  1. Label polarity inverted (raw `label=1` is *legitimate* in PhiUSIIL).
     Fixed in `data/load.py::_normalize_phiusiil_label`.
  2. 100% of legit URLs are `https://www.*`, fixed by
     `features.canonicalize` (strips www, normalizes trailing slash).
  3. 0% of legit URLs have paths in PhiUSIIL -- this is intrinsic to the
     dataset and a key motivation for multi-modal. Documented only.
- v0.1 URL baseline (post-fix): test AUC=0.9956, Tranco FPR@0.7=0.22%.
  `has_https` is 34.91% of gain. `xfail` regression marker on strict bar.
- v0.2 URL ablation (production): test AUC=0.9943, Tranco FPR@0.7=1.54%.
  Top feature subdomain_depth at 27.16%, top-3=57.9%. All leakage
  tests pass. The FastAPI service auto-loads v0.2 when present.
- 34 tests pass, 1 xfailed (intentional). Lint clean.
- Scraper smoke test: 22/50 successful on Tranco (timeouts on flaky sites
  expected). End-to-end pipeline validated.

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
2. ~~Leakage test, label fix, canonicalization, v0.2 ablation.~~ Done (v0.2-url-ablation)
3. `make scrape` at scale (a few thousand URLs, mixed phish + legit) on a host
   with reliable network and Playwright headless. Smoke test confirmed pipeline
   works on 22/50 Tranco URLs (timeouts are normal on flaky sites).
4. `make build-multimodal` to convert manifest into html_train/val/test parquets
   and screenshots/{train,val,test}/{phish,benign}/ ImageFolder layout.
5. `make train-html` on a CUDA host. ONNX export at the end.
6. `make train-img` on a CUDA host. ONNX export at the end.
7. `make train-fusion` once at least URL + one of HTML/img exists.

## Quick verify

    git status
    git log --oneline -5
    git tag -l
    .venv/bin/python scripts/inspect_url_baseline.py
