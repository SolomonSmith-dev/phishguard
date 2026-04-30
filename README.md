# PhishGuard

Multi-modal phishing URL detector. Three independent models (URL features, HTML/DOM, page screenshot) fused via a stacked meta-learner. Deployed as a FastAPI service with a browser extension client and drift monitoring.

This is a portfolio-grade ML project. Read this README as the spec, not just docs.

---

## Why this exists

Phishing detection is a multi-modal problem. URL strings leak signal, HTML structure leaks more, and rendered screenshots leak the most because attackers visually clone real brands. Most public detectors use one of those signals. Few candidates ship a fused, calibrated, monitored system. This is the project that fills that gap.

---

## Architecture

```
                  ┌────────────────────┐
   URL string  ─► │  GBDT url-model    │ ─► p_url
                  └────────────────────┘
                  ┌────────────────────┐
   HTML text   ─► │  DistilBERT html   │ ─► p_html
                  └────────────────────┘
                  ┌────────────────────┐
   Screenshot  ─► │  EfficientNet-B0   │ ─► p_img
                  └────────────────────┘
                                       │
                              ┌────────▼─────────┐
                              │ Logistic meta    │ ─► p_phish (calibrated)
                              │ + isotonic cal   │
                              └──────────────────┘
```

Late fusion was chosen over early fusion because the modalities have very different sample availability. URLs are cheap. HTML requires a fetch. Screenshots require a headless browser render. Fusing late lets the system gracefully degrade when slow modalities time out.

---

## Stack

| Layer        | Tool                                                 |
|--------------|------------------------------------------------------|
| Modeling     | PyTorch, HuggingFace, LightGBM                       |
| Tracking     | Weights and Biases, MLflow (model registry)          |
| Serving      | FastAPI, ONNX Runtime, Uvicorn                       |
| Pipelines    | Hydra configs, DVC for data versioning               |
| Data scrape  | Playwright (async)                                   |
| Container    | Docker, docker compose                               |
| Tests        | pytest, hypothesis                                   |
| Monitoring   | Evidently AI for drift, Prometheus for latency       |
| Deployment   | Fly.io or GCP Cloud Run                              |

---

## Datasets

1. PhiUSIIL Phishing URL dataset (Kaggle, ~235k URLs, balanced)
2. PhishTank verified feed (live, hourly snapshots)
3. Tranco top 1M for benign URLs
4. CIRCL phishing screenshots
5. Self-collected via Playwright over a window of suspect URLs

You will write the scrape pipeline. Do not skip this. Real data engineering is half the resume signal.

---

## Repo layout

```
phishguard/
├── configs/                  # Hydra YAML
├── docker/
├── notebooks/                # exploration only, not production
├── src/phishguard/
│   ├── data/                 # ingest, clean, scrape
│   ├── features/             # URL feature engineering
│   ├── models/               # 4 model definitions
│   ├── training/             # train loops per modality
│   ├── serving/              # FastAPI app
│   └── monitoring/           # drift, calibration
├── tests/
├── Makefile
└── pyproject.toml
```

---

## 4 week execution plan

### Week 1 - foundations and URL model
1. Set up repo, pyproject, pre-commit, ruff, mypy, pytest
2. Pull PhiUSIIL and Tranco. Build `src/phishguard/data/load.py`
3. Engineer 60+ URL features in `features/url_features.py`
4. Train LightGBM with stratified k-fold, log to W and B
5. Ship calibrated isotonic regression on top
6. **Target:** F1 > 0.95 on URL-only baseline

### Week 2 - HTML model
1. Write `data/scrape.py` with async Playwright. Be polite, rate limit
2. Snapshot HTML and PNG together for every URL
3. Tokenize HTML with DistilBERT tokenizer, truncate to 512
4. Fine-tune DistilBERT classifier on cleaned HTML text
5. Export to ONNX
6. **Target:** F1 > 0.92 on HTML-only

### Week 3 - screenshot model + fusion
1. Train EfficientNet-B0 on 224x224 screenshots, augment with random crop and color jitter
2. Run all three models on a held-out fusion set
3. Train logistic meta-learner over their probability outputs
4. Apply Platt scaling or isotonic calibration on the final score
5. Add adversarial robustness eval (FGSM on screenshot model)
6. **Target:** AUC > 0.99 on the fused holdout

### Week 4 - serving, monitoring, polish
1. FastAPI service `/predict` accepting URL, optional HTML, optional screenshot URI
2. Docker compose with the API plus a Postgres for prediction logs
3. Evidently AI dashboard reading from the prediction log
4. Browser extension stub that calls your local API
5. README finalization, model card, training reproducibility doc
6. Public deploy on Fly.io or Cloud Run

---

## Killer features that recruiters notice

1. **Calibrated probabilities**, not just predictions. Use Brier score and reliability diagrams in the README.
2. **Model cards** documenting data, intended use, fairness considerations, known failure modes.
3. **Adversarial section** showing how robust the screenshot CNN is under perturbation.
4. **Latency budget** documented (URL: under 5ms, HTML: under 200ms, image: under 100ms on CPU via ONNX).
5. **Reproducibility:** make train-all rebuilds every model from scratch in under 4 hours on a single GPU.
6. **Drift dashboard** screenshot in the README.

---

## Quick start

```bash
make setup            # creates venv, installs deps, sets up pre-commit
make data             # downloads PhiUSIIL and Tranco
make train-url        # trains URL GBDT v0.1
make train-url-v0_2   # trains v0.2 ablation (drops has_https/has_http/length)
make scrape           # Playwright-renders snapshots into data/processed/snapshots
make build-multimodal # joins manifest with labels into html/image splits
make train-html       # trains DistilBERT (needs GPU)
make train-img        # trains EfficientNet (needs GPU)
make train-fusion     # trains stacked meta-learner
make serve            # runs the FastAPI service locally (auto-loads v0.2 if present)
make eval             # writes reports/evaluation_v0_{1,2}.md
make drift            # writes reports/drift.html via Evidently
make docker-up        # API + Postgres via docker compose
make test             # full suite (34 passing, 1 xfailed)
```

## Status (2026-04-30)

| Phase | Status |
|---|---|
| URL data pipeline + canonicalization | done |
| URL v0.1 baseline (leaky, documented) | done |
| URL v0.2 ablation (production) | done |
| Pre-commit + lint + test gates | done |
| Scrape pipeline (Playwright) | smoke-tested |
| Multimodal dataset assembly | done |
| FastAPI service + Postgres logging | done |
| Browser extension stub | done |
| Drift dashboard (Evidently) | done |
| HTML model training | scaffolded, awaits scraped data + GPU |
| Screenshot model training | scaffolded, awaits scraped data + GPU |
| Late-fusion training | scaffolded, awaits modality outputs |

See `MODEL_CARD.md` for metrics and `LIMITATIONS.md` for the methodology
findings (label-polarity inversion in PhiUSIIL, www-prefix shortcut, path-presence
shortcut). All three were caught by end-to-end smoke tests, not by reading
holdout numbers.

---

## Stretch goals (after the 4 weeks)

1. Real-time browser extension that flags suspect pages while browsing
2. Active learning loop to retrain on hard examples from production
3. LLM-based explanation layer that generates a human-readable reason
4. Multi-label expansion: phishing, malware, scam, credential harvest
5. Write it up as a paper or technical blog post for arXiv or your portfolio site

---

## License

MIT. Phishing detection should be open.
