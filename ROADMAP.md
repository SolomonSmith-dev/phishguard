# PhishGuard Roadmap

The original 4-phase execution plan for the multi-modal phishing detector. Status updates land in the README's Status section as each phase ships.

## Phase 1: Foundations + URL model

1. Set up repo, pyproject, pre-commit, ruff, mypy, pytest.
2. Pull PhiUSIIL and Tranco. Build `src/phishguard/data/load.py`.
3. Engineer 60+ URL features in `features/url_features.py`.
4. Train LightGBM with stratified k-fold, log to W&B.
5. Ship calibrated isotonic regression on top.
6. **Target:** F1 > 0.95 on URL-only baseline.

**Status:** done. See README for v0.1 and v0.2 numbers.

## Phase 2: HTML model

1. Write `data/scrape.py` with async Playwright. Be polite, rate limit.
2. Snapshot HTML and PNG together for every URL.
3. Tokenize HTML with DistilBERT tokenizer, truncate to 512.
4. Fine-tune DistilBERT classifier on cleaned HTML text.
5. Export to ONNX.
6. **Target:** F1 > 0.92 on HTML-only.

**Status:** scaffolded, awaits scraped data and GPU.

## Phase 3: Screenshot model + fusion

1. Train EfficientNet-B0 on 224x224 screenshots, augment with random crop and color jitter.
2. Run all three models on a held-out fusion set.
3. Train logistic meta-learner over their probability outputs.
4. Apply Platt scaling or isotonic calibration on the final score.
5. Add adversarial robustness eval (FGSM on screenshot model).
6. **Target:** AUC > 0.99 on the fused holdout.

## Phase 4: Serving, monitoring, polish

1. FastAPI service `/predict` accepting URL, optional HTML, optional screenshot URI.
2. Docker compose with the API plus a Postgres for prediction logs.
3. Evidently AI dashboard reading from the prediction log.
4. Browser extension stub that calls your local API.
5. README finalization, model card, training reproducibility doc.
6. Public deploy on Fly.io or Cloud Run.

**Status:** Service + Postgres logging + browser extension stub + drift dashboard done. Public deploy pending.
