# PhishGuard evaluation

## URL model (`models/checkpoints/url_model.lgb`)

`test` split (n=23537):

| metric | value |
|---|---|
| auc | 0.9954 |
| ap | 0.9952 |
| f1 | 0.9863 |
| brier | 0.0101 |

Tranco top-5000 probe (benign-only, FPR-style):

| metric | value |
|---|---|
| n | 5000 |
| median_p_phish | 0.0114 |
| fpr_at_0.5 | 0.0082 |
| fpr_at_0.7 | 0.0022 |
| fpr_at_0.9 | 0.0008 |
