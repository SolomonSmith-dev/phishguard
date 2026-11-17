---
title: PhishGuard
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: URL-only phishing classifier, LightGBM on ONNX Runtime
---

# PhishGuard

A URL-only phishing classifier. `POST /predict` with `{"url": "..."}` returns a
calibrated `p_phish`. The page at `/` is a demo form over the same endpoint.
The model reads the URL string only and never fetches the page.

Measured on a held-out test split of 23,537 URLs through this served path
(ONNX Runtime plus isotonic calibrator): ROC AUC 0.9942. On the Tranco top
5,000 sites, never trained on, 1.54% are flagged at threshold 0.7 and 2.52% at
the served threshold of 0.5.

Source, model card and the leakage write-up:
https://github.com/SolomonSmith-dev/phishguard

This Space is deployed by `scripts/deploy_hf_space.py` in that repository.
Do not edit files here by hand; the next deploy replaces them.
