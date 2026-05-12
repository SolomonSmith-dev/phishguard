# PhishGuard

Multi-modal phishing URL detector. Three independent models (URL features, HTML/DOM, page screenshot) fused via a stacked meta-learner with calibrated probabilities. Deployed as a FastAPI service with a browser extension client and drift monitoring.

## TL;DR

- **Production model:** `v0.2-url-ablation`. LightGBM URL classifier, test AUC 0.9943, Tranco top-5000 FPR at 0.7 of 1.54%. Leakage-free. Top feature `subdomain_depth` at 27.16% of gain. 34 tests passing.
- **Earlier baselines:** `v0.1-url-baseline` (AUC 0.9956 post-fix) and `v0.1-hardened` (added pre-commit + leakage tests). Kept for ablation comparison.
- **In progress:** HTML (DistilBERT) and screenshot (EfficientNet-B0) modalities, then stacked fusion.
- **Honest about limits:** `LIMITATIONS.md` documents the methodology issues found in PhiUSIIL (inverted label polarity, 100% `https://www.*` legits, 0% paths on legits) and how each was fixed.

See the detailed phase table further down for full status.

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

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for the full 4-phase execution plan and per-phase status. Current shipped state is summarized in the Status section below.

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
