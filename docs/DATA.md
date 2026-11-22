# Data sources for the HTML branch

## PhreshPhish

- Dataset: PhreshPhish, Hugging Face id `phreshphish/phreshphish`
- Source: https://huggingface.co/datasets/phreshphish/phreshphish
- Paper: arXiv 2507.10854
- Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The dataset is public and not gated.
- Attribution: "PhreshPhish" by the PhreshPhish authors, from https://huggingface.co/datasets/phreshphish/phreshphish, licensed under CC BY 4.0. Changes: we use a small subset, truncate each page to 200,000 characters, and re-split it by registered domain.

### Subset used here

- Loader: `python -m phishguard.data.phreshphish` (streams parquet shards, never downloads the full set).
- Shards read: `data/train-000.parquet` and `data/train-001.parquet` of the train split, in file order, no shuffle.
- Size: at most 2,000 phishing and 2,000 benign records, and a hard cap of 2 GiB of HTML text.
- Seed: 42 (row shuffle and domain split).
- Split: 70/15/15, grouped by registered domain, so no domain appears in two splits (`phishguard.data.split`).
- Because the subset comes from the first two shards, it is not a random sample of the full set. Treat any number from it as a smoke result, not a benchmark.

### Handling and redistribution

- Raw pages are NOT redistributed. The cache lives in `data/raw/phreshphish/` and `data/processed/phreshphish/`, both git-ignored. No page, URL list or model weight is committed.
- Records are treated as hostile data. The code reads strings only. It never renders a page, runs a script, or fetches a URL found in a record.
- Run the smoke profile with `python -m phishguard.training.train_html --config configs/html_model_mps.yaml`. It logs to a local JSONL file only; wandb is off unless `logging.report_to: wandb` is set.
