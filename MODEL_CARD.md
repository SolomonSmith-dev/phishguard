# PhishGuard model card

## Model summary

Multi-modal phishing URL classifier. Three independent modality models
fused by a calibrated logistic meta-learner. URL-only baseline ships
today; HTML and screenshot models are scaffolded and produce ONNX
artifacts after their respective `make train-*` targets run.

| Model | Modality | Algorithm | Status |
|---|---|---|---|
| `url_model` | URL string | LightGBM GBDT | v0.1 (leaky) and v0.2 (ablated) shipped |
| `html_model` | Raw HTML text | DistilBERT fine-tune | scaffolded, awaits scraping |
| `screenshot_model` | Rendered PNG | EfficientNet-B0 | scaffolded, awaits scraping |
| `fusion_model` | All of above | Calibrated logistic regression | scaffolded |

## Intended use

* Pre-warning users about likely phishing pages, primarily as a browser
  extension or browser-side proxy.
* Triage signal for SOC analysts working a queue of suspicious URLs.

## Out of scope

* High-stakes blocking without human review. The model is calibrated but
  not perfect. Expect ~3% FPR at threshold 0.7 even on benign Tranco URLs.
* Detecting domain-takeover attacks where the URL is on a legitimate domain.
* Detecting voice/SMS phishing -- this model only sees URLs/HTML/screenshots.

## Training data

| Dataset | Use | Size |
|---|---|---|
| PhiUSIIL (Kaggle) | URL train/val/test | ~235k labeled URLs (balanced) |
| Tranco top 1M | Held-out probe (benign-only) | top 5000 |
| Self-scraped via Playwright | HTML/screenshot training | (in progress) |

PhiUSIIL provides BOTH phishing and benign labels, which avoids the trivial
"is-this-from-Tranco" shortcut that arises from mixing phishing-only feeds
with Tranco-as-benign. See `LIMITATIONS.md`.

## URL model (LightGBM GBDT)

### v0.1 (`url_model.lgb`)

* 47 numeric URL-string features + 1 categorical (TLD), URL canonicalization
  applied at both train and serve time.
* Training: 5-fold stratified CV on PhiUSIIL train, 100-round early stop on val,
  isotonic calibration on val.
* Test AUC = 0.9956, AP = 0.9961, F1 = 0.9861, calibrated Brier = 0.0101.
* Tranco probe (live benign): FPR@0.7 = 0.22%, median p_phish = 0.0114.
* **Documented leakage:** `has_https` accounts for 34.9% of gain.
  See `LIMITATIONS.md`. The strict no-leakage test
  (`test_v0_1_no_feature_dominates_strict`) is intentionally `xfail`.

### v0.2 (`url_model_v0_2.lgb`) -- production

* Same feature pipeline with `has_https`, `has_http`, `url_length`,
  `host_length`, `path_length` ablated.
* Test AUC = 0.9943, AP = 0.9948, F1 = 0.9782, calibrated Brier = 0.0157.
* Tranco probe FPR@0.7 = 1.54%, median p_phish = 0.0125.
* Top feature: `subdomain_depth` at 27.2% of gain. Top-3 = 57.9%. The
  leakage tests pass.
* This is the model the FastAPI service auto-loads when present.

### Why we ship v0.2 even though v0.1 has higher Tranco AUC

v0.1's `has_https` shortcut is a real signal in the training distribution
but not a robust phishing indicator -- attackers trivially serve HTTPS. The
model card commits us to honesty about training-time shortcuts, not
optimizing for in-distribution AUC. v0.2 is the model whose features we can
defend.

## Calibration

Isotonic regression fit on the val split, applied at prediction time. Brier
on the test set is reported above. Reliability diagrams are not yet
generated; planned in `notebooks/url_calibration.ipynb`.

## Evaluation methodology

* `make eval` runs `phishguard.training.evaluate` for both URL models and
  writes `reports/evaluation_v0_{1,2}.md`.
* The held-out Tranco probe is the most interesting metric: training data
  comes from PhiUSIIL, but Tranco is independent and live. FPR there is the
  honest signal of how often we will yell at a real user.

## Failure modes

1. **Brand-cloning on legitimate TLDs:** an attacker who buys
   `paypal-secure.com` (not a suspicious TLD, has https, normal length)
   slips through both v0.1 and v0.2. Multi-modal HTML/screenshot is the
   intended fix.
2. **Punycode IDN homoglyphs:** detected as `is_punycode` but the model
   weights this lightly relative to other features.
3. **URL shorteners:** `bit.ly` etc are flagged as shorteners but cannot
   be verified without a fetch (HTML modality required).
4. **Long benign URLs with many slashes:** v0.2's reliance on `num_slashes`
   means deeply-nested benign URLs (CDN paths, GitHub permalinks) may have
   elevated p_phish.

## Fairness considerations

URL features are essentially script-agnostic but PhiUSIIL is Western/
English-centric in TLD distribution. International phishing campaigns
targeting `.cn`, `.ru`, `.in` TLDs are under-represented in training.
Production deployments outside that locale should retrain or supplement.

## Privacy

The browser-extension stub sends only the URL of the active page to a
local API. HTML capture is opt-in and never leaves localhost in the stub.
For production: log retention should be capped to 30 days, and URLs
containing query-string credentials (?token=, ?session=) should be
redacted before persistence.

## Reproducibility

`make data` then `make train-url` rebuilds v0.1 from scratch.
`make train-url-v0_2` rebuilds the ablation. Both are deterministic at
seed=42. End-to-end runs in under 10 minutes on a 2024 MacBook M-series.
HTML/screenshot models will require a CUDA host.

## Versioning

* `v0.1-url-baseline` -- initial URL model with documented leakage.
* `v0.1-hardened` -- added pre-commit gate and the no-leakage test.
* `v0.2-url-ablation` -- v0.2 model trained, leakage test now passing.
