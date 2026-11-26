# Deploy

Live at **https://phishguard.soloserver.xyz** (demo at `/`, API at `/predict`,
health at `/healthz`). Cost: $0.

It runs as a Docker container on the home server, reachable only through the
existing Cloudflare tunnel. A Hugging Face Docker Space was the first target;
Hugging Face now answers `402 Payment Required` for Docker Spaces on free
hardware ("requires a PRO subscription"), so that path is kept but not used.

## The image

`scripts/deploy_hf_space.py --stage-only` writes the build context to
`build/hf-space/` (the home-server deploy writes the same thing to
`build/home-server/`):

- `deploy/hf-space/Dockerfile`, `requirements.txt` (pinned serve deps, no torch), `README.md` (Space card)
- `src/phishguard/`
- the four v0.2 URL artifacts: `url_model_v0_2.onnx`, `url_calibrator_v0_2.pkl`, `url_features_v0_2.json`, `url_tld_categories_v0_2.json`
- `ARTIFACTS.sha256`: the git commit and the sha256 of each artifact; `/healthz` reports the commit as `build`

`models/checkpoints/` is git-ignored, so the artifacts come from the local
machine. Produce them with `make train-url-v0_2` (or `make export-url-onnx`
if the booster exists) and keep the `reports/url_onnx_<date>.md` it writes.

scikit-learn is pinned to 1.8.0 because `url_calibrator_v0_2.pkl` was pickled
with it. The public suffix list is fetched once at image build and copied to
`/tmp` at container start, so serving makes no network calls and works on a
read-only filesystem.

## Deploy to the home server

1. Commit first. The script refuses tracked changes and untracked files under `src/` or `deploy/`.
2. `make deploy-home`. It rsyncs the build context to `home-server:~/phishguard-deploy/context`,
   builds `phishguard:<short sha>` there, replaces the `phishguard` container and waits
   until `/healthz` on the server reports this commit.
3. `make smoke-live` (add `--expect-commit $(git rev-parse HEAD)` by running
   `python scripts/smoke_live.py --base-url https://phishguard.soloserver.xyz --expect-commit <sha>`).
   Exit 0 only if every check passes:
   - `health`: `/healthz` is ok and reports `url_model_v0_2.onnx` (not the v0.1 fallback)
   - `demo`: `/` serves the demo page
   - `validation`: a 2-character URL gets 422
   - `parity`: all 75 fixture URLs score within 1e-6 of the verified local model
   - `phish`: at least 90% of 25 held-out PhiUSIIL phish URLs are flagged at 0.5
   - `tranco`: at most 6% of the Tranco top 50 are flagged at 0.5 (measured: 1 of 50, `apple-dns.net`)
   - `build` (with `--expect-commit`): the server reports that commit

Before a deploy, `make docker-hf` builds and runs the same image locally on
:7860, and `python scripts/smoke_live.py --base-url http://localhost:7860` runs
the same checks against it.

### How the container runs

`127.0.0.1:7860` only, so the tunnel is the only way in. Read-only root
filesystem, `/tmp` as a 64 MB tmpfs, all Linux capabilities dropped,
`no-new-privileges`, 1 CPU, 1 GB memory, 128 pids, UID 1000, restart
`unless-stopped`.

### Tunnel

`/etc/cloudflared/config.yml` on the home server has an ingress rule
`phishguard.soloserver.xyz -> http://127.0.0.1:7860` above the 404 catch-all,
and DNS has a CNAME from `phishguard.soloserver.xyz` to the tunnel. This adds a
second public hostname next to Jellyfin on the same tunnel.

### Rollback

Every deploy keeps the previous image tag. On the server:
`docker images phishguard` to list tags, then `docker rm -f phishguard` and the
`docker run` line from `scripts/deploy_home_server.py` with the older tag.
Or check out the last good commit and run `make deploy-home`.

### Take it offline

`ssh home-server docker rm -f phishguard`. The tunnel then returns 502 for the
hostname. Remove the ingress rule and the DNS record to retire it fully.

## Hugging Face Space (needs PRO)

`make deploy-hf` stages, uploads in one commit that deletes stale remote files,
and waits until the Space's `/healthz` reports the uploaded commit (right after
an upload the old container can still report `RUNNING`). Needs `hf auth login`
with write access and, since 2026-11, a PRO plan.

## After a retrain or re-export

Run `make smoke-fixtures` to re-score `scripts/smoke_fixtures.json`, and update
the numbers in `src/phishguard/serving/static/demo.html` (see `DESIGN.md`).
`tests/test_smoke_live.py::test_fixtures_match_local_artifacts` fails until the
fixtures match the artifacts on disk.

## Data and limits

- The prediction log (`/tmp/phishguard_predictions.jsonl`) lives in the tmpfs:
  lost on every restart, never published, capped at 50 MB. The image sets
  `PHISHGUARD_STORE_RAW_URL=0`, so only `scheme://host` is stored (path, query,
  fragment, port and userinfo dropped; see `host_only()` in
  `serving/prediction_log.py`). Local runs keep the raw URL, as before.
- Request bodies over 13 MB get 413 before they are read; 422 and 500
  responses do not echo input or internals.
- No Postgres: `DATABASE_URL` is unset, so the logger uses the file fallback.
