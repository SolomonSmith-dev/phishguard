# Multimodal training status and what it needs

**Status: staged, not trained.** The DistilBERT (HTML) and EfficientNet-B0
(screenshot) branches have runnable training scripts and configs, but no
labeled multimodal corpus exists yet, so there are no checkpoints and no
reported metrics. Nothing here is faked: this file records what actually
exists on disk today and exactly what has to be added to produce a real,
measured model.

## What is actually on disk (verified 2026-11-16)

### Scraped snapshots
- `data/processed/snapshots/manifest.jsonl`: 50 scrape attempts, `22 ok`.
- `data/processed/snapshots/html/`: 22 HTML files.
- `data/processed/snapshots/img/`: 22 PNG screenshots.
- These 22 successful scrapes are all **benign** top sites (google.com,
  facebook.com, github.com, wikipedia.org, ...), scraped from a Tranco probe,
  not from a phishing corpus. There are **zero** phishing pages among them.

### Labels
- `data/processed/url_{train,val,test}.parquet`: 188,296 / 23,537 / 23,537
  URL rows with an `is_phish` label, covering the URL model only.
- Joining those label files back to the 22 snapshot URLs yields **0 rows**
  (`build_multimodal_dataset` reports `joined rows: 0`). None of the
  scraped snapshot URLs exist in the URL label set, and even the ones with
  labels would only give the benign class. There is no labeled row that pairs
  a snapshot with a phishing/benign ground truth.

### What the builder produced
```
$ python -m phishguard.data.build_multimodal_dataset \
    --manifest data/processed/snapshots/manifest.jsonl \
    --labels data/processed/url_train.parquet \
    --output-html data/processed --output-img data/processed/screenshots
manifest rows: 50  ok: 22
joined rows (with labels and ok): 0
no html rows after filtering -- skipping html splits
no screenshots -- skipping img splits
```

### Why training cannot run on this data
- Both trainers need **both classes** in labeled splits: the screenshot
  trainer reads an `ImageFolder` of `{train,val,test}/{phish,benign}/*.png`,
  and the HTML trainer reads labeled parquet rows. With one class and no
  labels, neither can assemble a valid train/val/test split (`roc_auc_score`
  needs both classes, the `ImageFolder` has no `phish/` directory).
- A GPU is not the blocker. Even on CPU, a tiny run is impossible without a
  second class and ground-truth labels.

## What training needs

### 1. A labeled multimodal corpus (the hard requirement)
A per-sample pairing of **URL -> (screenshot PNG, HTML text, is_phish label)**
with enough of **both** classes to produce a real split. Two viable sources:

- **PhishTank + self-scrape:** pull the PhishTank live feed (has
  phish URLs), pair each with a benign URL (e.g. Tranco top-N), render
  screenshots and capture HTML with the existing Playwright scrape pipeline
  (`src/phishguard/data/scrape.py`), then label by source. This gives real
  phish pages but requires either a headless-render budget or a stash of
  already-captured phish pages.
- **CIRCL screenshot dataset:** CIRCL publishes labeled phishing page
  screenshots; format it into the expected `{phish,benign}` `ImageFolder`
  layout and the same HTML/URL pairing where available.

Minimum honest bar before calling anything "trained": a held-out test split
the model never trained on, with both classes, and the metric written to
`reports/`. Follow the same anti-leakage audit applied to the URL model
(`tests/test_no_leakage.py`, `tests/test_label_polarity.py`,
`tests/test_canonicalize.py`) for any modality that could cheat.

### 2. A compute budget
- **GPU run (recommended):** any CUDA instance. `train_html.py` and
  `train_screenshot.py` already select CUDA if available and enable fp16.
  This is minutes of wall time per training run.
- **CPU run (viable at small scale):** the scripts already fall back to CPU.
  EfficientNet-B0 at 224px on an M4 is a few minutes per epoch on a few
  hundred images; DistilBERT over short HTML is a few GPUs-allowed minutes
  per epoch or slower on CPU. A small corpus is CPU-trainable in under an
  hour; a large corpus is not. Set `data.image_size` down (e.g. 128) in
  `configs/screenshot_model.yaml` for a cheaper CPU smoke before a real run.

### 3. Runtime jars
Environment for `make build-multimodal` (pandas, pyarrow, scikit-learn) and
`make train-html` / `make train-img` (torch, torchvision, transformers). The
torch/transformers dependency tree is the heavy one. `uv venv && uv pip
install -e ".[dev]"` covers the project deps.

## Definition of done for the TRAIN decision (issue #6)
Each branch has one of:
- a **measured** metric in `reports/` from a held-out test split (preferred),
  verified through the same ONNX + calibrated serving path the URL model is
  audited on, or
- an explicit statement that it **remains scaffolded** because the corpus is
  not available, with this file as the record.

The failure mode being guarded against is a claimed model with no artifact.
This project's standard, set in the URL branch, is "measured, not claimed":
no number is published until a report file exists for it.
