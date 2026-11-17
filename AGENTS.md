# AGENTS.md

Multi-modal phishing detector: a gradient-boosted URL model, a DistilBERT HTML model and an EfficientNet-B0 screenshot model, combined by late fusion with calibrated probabilities. The URL model is the shipped baseline.

## Layout

- `src/phishguard/`: package (`data`, `features`, `models`, `training`, `serving`, `monitoring`)
- `tests/`: pytest suite, including the leakage and label tests
- `configs/`: training configs (YAML)
- `scripts/`, `docker/`, `deploy/`: utilities and packaging
- `extension/`: browser extension
- `reports/`: written metrics; `docs/`: deploy notes
- `LIMITATIONS.md`, `MODEL_CARD.md`: known limits and the leakage write-up

## Setup and checks

- Install: `pip install -e ".[dev]"` (.github/workflows/ci.yml, line 20; `make setup` in Makefile does this in a venv)
- Test: `pytest` (ci.yml, line 24; `make test`, Makefile line 74)
- Lint and types: `ruff check src tests` and `mypy src` (Makefile, lines 78 and 79; `make lint`)
- Format: `make format` (Makefile)
- Pre-commit hooks are configured in `.pre-commit-config.yaml`. Do not skip them.

## Conventions

- Training scripts take `--config <path>` and read YAML from `configs/`.
- Write metrics to `reports/`. Report calibrated probabilities, and state which dataset and split a number comes from.
- Modalities are independent; serving must degrade gracefully when a model is missing.
- Weights & Biases logging is optional; set `WANDB_MODE=disabled` to turn it off.

## Data leakage

- Never loosen the leakage tests (`tests/test_no_leakage.py`, `tests/test_label_polarity.py`, `tests/test_canonicalize.py`) to make a model pass. A near-perfect score is a warning sign: inspect feature importance first.
- Do not mix the benign URL probe set into training data. Keep it as a held-out probe, since it creates a trivial subdomain shortcut.

## Safety

- Never commit secrets or credentials; use `.env.example` as the template.
- Do not run deploy targets (`make deploy-*`) unless asked.

## Done means

`pytest`, `ruff check src tests` and `mypy src` pass, and any changed metric is written to `reports/` with its source.
