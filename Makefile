.PHONY: setup data clean scrape build-multimodal train-url train-url-v0_2 export-url-onnx train-html train-img train-fusion train-all serve eval eval-v0_1 eval-v0_2 drift test lint format docker docker-up docker-down check-python deploy-home deploy-hf deploy-hf-stage docker-hf smoke-live smoke-fixtures

PYTHON := $(shell command -v python3.11 2>/dev/null || command -v python3.12 2>/dev/null || command -v python3.13 2>/dev/null || command -v python3 2>/dev/null)
VENV := .venv
ACTIVATE := . $(VENV)/bin/activate

check-python:
	@if [ -z "$(PYTHON)" ]; then echo "ERROR: no python3 found."; exit 1; fi
	@$(PYTHON) -c 'import sys; assert sys.version_info >= (3,11), f"need >=3.11, got {sys.version}"' \
	  || (echo "ERROR: Python >=3.11 required. Found: $$($(PYTHON) --version)"; exit 1)
	@echo "using $(PYTHON) ($$($(PYTHON) --version))"

setup: check-python
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -e ".[dev]"
	$(ACTIVATE) && playwright install chromium
	$(ACTIVATE) && pre-commit install

data:
	$(ACTIVATE) && python -m phishguard.data.load --download

scrape:
	$(ACTIVATE) && python -m phishguard.data.scrape --input data/raw/urls.parquet --output data/processed/snapshots/

build-multimodal:
	$(ACTIVATE) && python -m phishguard.data.build_multimodal_dataset \
	  --manifest data/processed/snapshots/manifest.jsonl \
	  --labels data/processed/url_train.parquet \
	  --output-html data/processed \
	  --output-img data/processed/screenshots

train-url:
	$(ACTIVATE) && python -m phishguard.training.train_url --config configs/url_model.yaml

train-url-v0_2:
	$(ACTIVATE) && WANDB_MODE=disabled python -m phishguard.training.train_url --config configs/url_model_v0_2.yaml
	$(MAKE) export-url-onnx

# Export the trained v0.2 booster to ONNX (what the API serves) and verify it
# against the native booster. Writes reports/url_onnx_<today>.md.
export-url-onnx:
	$(ACTIVATE) && python -m phishguard.training.export_url_onnx --config configs/url_model_v0_2.yaml \
	  --report reports/url_onnx_$$(date +%Y-%m-%d).md

train-html:
	$(ACTIVATE) && python -m phishguard.training.train_html --config configs/html_model.yaml

train-img:
	$(ACTIVATE) && python -m phishguard.training.train_screenshot --config configs/screenshot_model.yaml

train-fusion:
	$(ACTIVATE) && python -m phishguard.training.train_fusion --config configs/fusion.yaml

train-all: train-url train-html train-img train-fusion

serve:
	$(ACTIVATE) && uvicorn phishguard.serving.api:app --host 0.0.0.0 --port 8000 --reload

eval: eval-v0_1 eval-v0_2

eval-v0_1:
	$(ACTIVATE) && python -m phishguard.training.evaluate --url-config configs/url_model.yaml --out reports/evaluation_v0_1.md

eval-v0_2:
	$(ACTIVATE) && python -m phishguard.training.evaluate --url-config configs/url_model_v0_2.yaml --out reports/evaluation_v0_2.md

drift:
	$(ACTIVATE) && python -m phishguard.monitoring.drift \
	  --reference data/processed/url_train.parquet \
	  --current   data/processed/url_test.parquet \
	  --out reports/drift.html

test:
	$(ACTIVATE) && pytest

lint:
	$(ACTIVATE) && ruff check src tests
	$(ACTIVATE) && mypy src

format:
	$(ACTIVATE) && ruff format src tests
	$(ACTIVATE) && ruff check --fix src tests

docker:
	docker build -t phishguard:latest -f docker/Dockerfile .

docker-up:
	docker compose -f docker/docker-compose.yml up -d --build

docker-down:
	docker compose -f docker/docker-compose.yml down

# ---- Public deploy. See docs/deploy.md.
# Live: home-server container behind the Cloudflare tunnel.
HOME_HOST ?= home-server
LIVE_URL ?= https://phishguard.soloserver.xyz

# Build HEAD on the home server and swap the container; waits for /healthz.
deploy-home:
	$(ACTIVATE) && python scripts/deploy_home_server.py --host $(HOME_HOST)

# Alternative: Hugging Face Docker Space. Needs a PRO plan since 2026-11 (402 otherwise).
HF_SPACE ?= SolomonSmith-dev/phishguard

# Write the Space build context to build/hf-space without uploading.
deploy-hf-stage:
	$(ACTIVATE) && python scripts/deploy_hf_space.py --repo-id $(HF_SPACE) --stage-only

# Stage, upload in one commit, and wait until the Space reports RUNNING.
deploy-hf:
	$(ACTIVATE) && python scripts/deploy_hf_space.py --repo-id $(HF_SPACE) --wait

# Build and run the exact Space image locally on :7860.
docker-hf: deploy-hf-stage
	docker build -t phishguard-space:local build/hf-space
	docker run --rm -p 7860:7860 phishguard-space:local

# Live checks: health and model file, demo page, 422, parity, phish recall, Tranco flag rate.
smoke-live:
	$(ACTIVATE) && python scripts/smoke_live.py --base-url $(LIVE_URL)

# Re-score scripts/smoke_fixtures.json with the local artifacts (after a retrain or re-export).
smoke-fixtures:
	$(ACTIVATE) && python scripts/smoke_live.py --regen

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
