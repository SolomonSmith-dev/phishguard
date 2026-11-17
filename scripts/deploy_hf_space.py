"""Stage and deploy the PhishGuard URL model to a Hugging Face Docker Space.

The Space repo is a build output, not a second source tree. This script
assembles it from the repo every time:

    build/hf-space/
      Dockerfile, README.md, requirements.txt   from deploy/hf-space/
      src/phishguard/                           the package, minus caches
      models/checkpoints/                       the four v0.2 URL artifacts
      ARTIFACTS.sha256                          checksums plus the git commit

then uploads it in one commit that also deletes any remote file it did not
upload, so the Space always matches one commit of this repo.

Usage:
    python scripts/deploy_hf_space.py --stage-only   # build/hf-space only
    python scripts/deploy_hf_space.py --wait         # stage, upload, wait until it serves HEAD

Needs a Hugging Face token with write access, from `hf auth login` or HF_TOKEN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "deploy" / "hf-space"
CKPT_DIR = ROOT / "models" / "checkpoints"
DEFAULT_STAGE = ROOT / "build" / "hf-space"
DEFAULT_REPO = "SolomonSmith-dev/phishguard"

# Exactly what api.py loads for the v0.2 URL model. Anything else in
# models/checkpoints (v0.1 files, the native .lgb booster) stays out of the image.
ARTIFACTS = (
    "url_model_v0_2.onnx",
    "url_calibrator_v0_2.pkl",
    "url_features_v0_2.json",
    "url_tld_categories_v0_2.json",
)
DEPLOY_FILES = ("Dockerfile", "README.md", "requirements.txt")
FAIL_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "DELETING"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def stage(stage_dir: Path) -> str:
    """Write the Space build context to stage_dir and return the manifest text."""
    missing = [name for name in ARTIFACTS if not (CKPT_DIR / name).is_file()]
    if missing:
        sys.exit(
            f"missing artifacts in {CKPT_DIR}: {', '.join(missing)}. "
            "Run `make export-url-onnx` first."
        )
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    (stage_dir / "models" / "checkpoints").mkdir(parents=True)

    for name in DEPLOY_FILES:
        shutil.copy2(DEPLOY_DIR / name, stage_dir / name)
    shutil.copytree(
        ROOT / "src" / "phishguard",
        stage_dir / "src" / "phishguard",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".DS_Store"),
    )
    lines = [f"commit {git('rev-parse', 'HEAD')}"]
    for name in ARTIFACTS:
        dst = stage_dir / "models" / "checkpoints" / name
        shutil.copy2(CKPT_DIR / name, dst)
        lines.append(f"{sha256(dst)}  models/checkpoints/{name}")
    manifest = "\n".join(lines) + "\n"
    (stage_dir / "ARTIFACTS.sha256").write_text(manifest)
    return manifest


def serving_commit(app_url: str) -> str | None:
    """The source commit the running container reports at /healthz, if reachable."""
    req = urllib.request.Request(
        f"{app_url}/healthz", headers={"User-Agent": "phishguard-deploy/1"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            build = json.loads(r.read()).get("build")
            return str(build) if build else None
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def wait_until_serving(api: HfApi, repo_id: str, app_url: str, commit: str, timeout_s: int) -> str:
    """Wait for the new build, not just any RUNNING container.

    Right after an upload the Space can still report RUNNING for the previous
    build, so RUNNING alone is not proof. Done means /healthz reports `commit`.
    """
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        stage_now = str(api.get_space_runtime(repo_id).stage)
        if stage_now != last:
            print(f"space stage: {stage_now}", flush=True)
            last = stage_now
        if stage_now in FAIL_STAGES:
            return stage_now
        if stage_now == "RUNNING" and serving_commit(app_url) == commit:
            return "SERVING"
        time.sleep(10)
    return f"TIMEOUT after {timeout_s}s (last stage {last}, serving {serving_commit(app_url)})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE)
    ap.add_argument("--stage-only", action="store_true", help="write the build context, no upload")
    ap.add_argument("--allow-dirty", action="store_true", help="deploy with uncommitted changes")
    ap.add_argument("--wait", action="store_true", help="wait until /healthz reports this commit")
    ap.add_argument("--wait-timeout", type=int, default=900)
    args = ap.parse_args()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not args.stage_only and not args.allow_dirty:
        sys.exit(
            f"working tree has uncommitted changes; commit first or pass --allow-dirty:\n{dirty}"
        )

    manifest = stage(args.stage_dir)
    print(f"staged {args.stage_dir}\n{manifest}", end="")
    if args.stage_only:
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()["name"]
    print(f"authenticated as {user}")
    api.create_repo(
        args.repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True
    )
    commit = git("rev-parse", "HEAD")
    info = api.upload_folder(
        folder_path=str(args.stage_dir),
        repo_id=args.repo_id,
        repo_type="space",
        commit_message=f"deploy phishguard {commit[:7]}" + (" (dirty)" if dirty else ""),
        delete_patterns="*",
    )
    print(f"uploaded: {info.commit_url}")
    host = getattr(api.space_info(args.repo_id), "host", None)
    owner, name = args.repo_id.split("/")
    app_url = (host or f"https://{owner.lower()}-{name.lower()}.hf.space").rstrip("/")
    print(f"space page: https://huggingface.co/spaces/{args.repo_id}")
    print(f"app url:    {app_url}")

    if args.wait:
        final = wait_until_serving(api, args.repo_id, app_url, commit, args.wait_timeout)
        print(f"final: {final}")
        return 0 if final == "SERVING" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
