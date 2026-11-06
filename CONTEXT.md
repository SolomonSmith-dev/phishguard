# phishguard

Multi-modal phishing URL detector (URL + HTML/DOM + screenshot) with FastAPI service and browser extension.

## Task Routing

| Task Type | Go To | Description |
|-----------|-------|-------------|
| Feature extract / models | src/ · models/ | URL/HTML/vision + meta-learner |
| Configs | `configs/` | Train/serve configs |
| API / docker | docker/ · Makefile | Service packaging |
| Reports / evals | reports/ · tests/ | Metrics and tests |
| Limits | `LIMITATIONS.md` | Known failure modes |

## Shared Resources

| Resource | Location | Contains |
|----------|----------|----------|
| CLAUDE | `CLAUDE.md` | Project contract |
| Docs | `docs/` | Design notes |
| Graph | `graphify-out/` | God node: extract_url_features |
