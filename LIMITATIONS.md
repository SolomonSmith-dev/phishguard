# Known limitations

## v0.1 URL baseline (single modality)

The URL-only LightGBM model achieves AUC=0.999 on PhiUSIIL test.
Feature importance is concentrated:

| Feature       | Gain % |
|---------------|--------|
| has_https     | 47.6   |
| path_length   | 44.3   |
| has_http      | 3.5    |

This reflects PhiUSIIL collection methodology: phishing URLs sampled from
PhishTank/OpenPhish (HTTP-heavy older campaigns), benign URLs from
established sites (HTTPS-default). A determined attacker could trivially
spoof both signals.

### What this means for production

A single-modality URL classifier is insufficient. The multi-modal fusion
in v1.0 (URL + HTML + screenshot) addresses this by requiring the
attacker to also clone visual identity, which is significantly harder.

### Held-out probe (Tranco top-5000)

| Metric         | Value  |
|----------------|--------|
| median p_phish | 0.008  |
| FPR @ 0.5      | 3.64%  |
| FPR @ 0.7      | 2.78%  |
| FPR @ 0.9      | 1.32%  |

## v0.2 ablation (planned)

Retrain URL model with `has_https`, `has_http`, and length features dropped
to force structural learning. Target AUC: 0.94 to 0.96 with flatter
importance distribution. This will be added as a comparison row in the
README, not as a replacement for v0.1.
