# PhishGuard evaluation

## URL model (`models/checkpoints/url_model_v0_2.lgb`)

`test` split (n=23537):

| metric | value |
|---|---|
| auc | 0.9942 |
| ap | 0.9941 |
| f1 | 0.9782 |
| brier | 0.0157 |

Tranco top-5000 probe (benign-only, FPR-style):

| metric | value |
|---|---|
| n | 5000 |
| median_p_phish | 0.0125 |
| fpr_at_0.5 | 0.0252 |
| fpr_at_0.7 | 0.0154 |
| fpr_at_0.9 | 0.0090 |
