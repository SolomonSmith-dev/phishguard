# Deploy: Hugging Face Docker Space

The public service is a Docker Space on the free CPU basic tier ($0).

- Space page: https://huggingface.co/spaces/SolomonSmith-dev/phishguard
- App: https://solomonsmith-dev-phishguard.hf.space (demo at `/`, API at `/predict`, health at `/healthz`)

## What ships

`scripts/deploy_hf_space.py` builds `build/hf-space/` from the repo and uploads
it in one commit that deletes any remote file it did not upload:

- `deploy/hf-space/Dockerfile`, `README.md` (Space card), `requirements.txt` (pinned serve deps, no torch)
- `src/phishguard/`
- the four v0.2 URL artifacts: `url_model_v0_2.onnx`, `url_calibrator_v0_2.pkl`, `url_features_v0_2.json`, `url_tld_categories_v0_2.json`
- `ARTIFACTS.sha256`: the git commit and the sha256 of each artifact

`models/checkpoints/` is git-ignored, so the artifacts come from the local
machine. Produce them with `make train-url-v0_2` (or `make export-url-onnx`
if the booster already exists) and keep `reports/url_onnx_<date>.md`.

scikit-learn is pinned to 1.8.0 in the image because `url_calibrator_v0_2.pkl`
was pickled with it. The public suffix list is fetched once at image build and
baked in, so serving makes no network calls.

## Deploy

1. One time: a Hugging Face token with write access, stored with `hf auth login`
   (or `HF_TOKEN` in the environment).
2. Commit first. The script refuses a dirty tree unless `--allow-dirty`.
3. `make deploy-hf`. It prints the commit URL and waits until the Space is
   `RUNNING` and `/healthz` reports the commit just uploaded (`build`), because
   right after an upload the old container can still report `RUNNING`. It exits
   1 on `BUILD_ERROR`, `RUNTIME_ERROR` or a 15-minute timeout.
4. `make smoke-live`. Six checks (seven with `--expect-commit <sha>`), exit 0 only if all pass:
   - `health`: `/healthz` is ok and reports `url_model_v0_2.onnx` (not the v0.1 fallback)
   - `demo`: `/` serves the demo page
   - `validation`: a 2-character URL gets 422
   - `parity`: all 75 fixture URLs score within 1e-6 of the verified local model
   - `phish`: at least 90% of 25 held-out PhiUSIIL phish URLs are flagged at 0.5
   - `tranco`: at most 6% of the Tranco top 50 are flagged at 0.5 (measured: 1 of 50, `apple-dns.net`)

Before a deploy, `make docker-hf` builds and runs the same image locally on
:7860, and `python scripts/smoke_live.py --base-url http://localhost:7860`
runs the same checks against it.

## After a retrain or re-export

Run `make smoke-fixtures` to re-score `scripts/smoke_fixtures.json`, and update
the numbers in `src/phishguard/serving/static/demo.html` (see `DESIGN.md`).
`tests/test_smoke_live.py::test_fixtures_match_local_artifacts` fails until the
fixtures match the artifacts on disk.

## Rollback

Check out the last good commit (with its artifacts) and run `make deploy-hf`
again. Every deploy commit on the Space names the source commit, and
`ARTIFACTS.sha256` in the Space repo records exactly what is serving.

## Limits of the free tier

- Free Spaces sleep after a period without traffic. The first request after
  that wakes the container, which takes about a minute; `smoke_live.py` waits
  up to 240 s for `/healthz`.
- Disk is not persistent. The prediction log (`/tmp/phishguard_predictions.jsonl`,
  credential-like query parameters redacted) is lost on every restart and is
  not published.
- No Postgres: `DATABASE_URL` is unset, so the logger uses the file fallback.
