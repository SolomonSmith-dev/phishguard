# Known limitations

## tl;dr

PhiUSIIL is a richly-featured but methodologically-skewed phishing dataset.
A URL-only model trained on it will hit AUC > 0.99 in-distribution but
generalize poorly. Three load-bearing findings, all surfaced by end-to-end
testing rather than by reading numbers off a holdout:

1. **PhiUSIIL `label` polarity is inverted** -- raw `label=1` means
   *legitimate*, not phishing. Caught when manually-crafted phishing URLs
   were predicted as 0.0. Fixed in `data/load.py::_normalize_phiusiil_label`.
   Locked in by `tests/test_label_polarity.py`.
2. **PhiUSIIL legit URLs are 100% `https://www.*` and 0% have paths**.
   Phishing URLs commonly have neither shape. Without normalization the
   model becomes a `www.` and `path-presence` detector. Fixed by
   `features.canonicalize` (strips `www.`, normalizes trailing slash) at
   both train and serve time.
3. **`has_https` was 47.6% of v0.1 gain** before fixes. Documented and
   ablated in v0.2. Marked `known_fail` on the strict no-leakage test.

## v0.1 URL baseline (post-fix)

| Metric | Value |
|---|---|
| Test AUC | 0.9956 |
| Test AP | 0.9961 |
| Test F1 | 0.9861 |
| Calibrated test Brier | 0.0101 |

Top features by gain (v0.1, after all fixes):

| Feature | Gain % |
|---|---|
| has_https | 34.91 |
| subdomain_length | 34.69 |
| num_slashes | 14.53 |
| tld | 4.58 |
| num_special | 3.20 |

`has_https` still dominates (now at 35%, was 48%). `tests/test_no_leakage.py::test_v0_1_no_feature_dominates_strict` is marked `known_fail` to lock this in -- the strict bar is 30% gain.

### Held-out Tranco probe (v0.1)

| Threshold | FPR |
|---|---|
| 0.5 | 0.82% |
| 0.7 | 0.22% |
| 0.9 | 0.08% |

Median p_phish on Tranco top-5000: 0.0114.

## v0.2 ablation (post-fix, ships as production)

Drops `has_https`, `has_http`, and length features. Forces structural learning.

| Metric | Value |
|---|---|
| Test AUC | 0.9943 |
| Test AP | 0.9948 |
| Test F1 | 0.9782 |
| Calibrated test Brier | 0.0157 |

Top features by gain:

| Feature | Gain % |
|---|---|
| subdomain_depth | 27.16 |
| tld | 19.23 |
| num_slashes | 11.51 |
| domain_length | 10.31 |
| special_ratio | 9.32 |

No single feature dominates. Top-3 = 57.9%, well under the 80% concentration limit. The leakage tests `test_v0_2_no_egregious_leakage` and `test_v0_2_top3_concentration` pass.

### Held-out Tranco probe (v0.2)

| Threshold | FPR |
|---|---|
| 0.5 | 2.52% |
| 0.7 | 1.54% |
| 0.9 | 0.90% |

v0.2 has slightly higher FPR than v0.1 because the structural features it
relies on are noisier than the binary `has_https` shortcut. The trade is
worth it: v0.2 generalizes more honestly because no single feature is
load-bearing.

## What this means in production

The URL-only model is a useful pre-filter, **not a final verdict**. It is
calibrated, it works on PhiUSIIL-shaped URLs, and it correctly flags the
same kinds of phishing kits PhiUSIIL was scraped from (Firebase hosting,
Cloudways apps, suspicious TLDs with explicit paths). It will struggle on:

* Long benign URLs with many path segments (GitHub permalinks, CDN paths).
  These look phishy on `num_slashes` because PhiUSIIL legit URLs are
  bare-domain.
* Phishing campaigns hosted on legit TLDs with no obvious URL tells. The
  attacker buys a clean domain, hosts a credential form. URL features
  cannot see the page.
* IDN homoglyphs (punycode) -- detected as a feature but lightly weighted.

Multi-modal fusion (HTML + screenshot) is the intended fix. The
infrastructure is in place; HTML and screenshot models train via
`make train-html` and `make train-img` once `make scrape` produces enough
labeled snapshots and a CUDA host is available.

## v0.3 candidates (not started)

* Add Common Crawl URLs as a third class of training data so the legit
  distribution includes path-bearing URLs and bare domains alike.
* Strip subdomains entirely from feature inputs so the model can't latch
  onto www-vs-no-www anymore.
* Train on URL string with a small character-level CNN/transformer instead
  of hand-engineered features. Likely to discover and ignore the
  shortcuts a featurizer cannot.
