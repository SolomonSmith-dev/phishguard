"""Deploy the PhishGuard image to the home server, served through the Cloudflare tunnel.

Same build context as the Hugging Face path (scripts/deploy_hf_space.py stage()),
built on the server itself (x86_64) and run as a locked-down container:

    127.0.0.1:7860 only (the tunnel is the only way in), read-only root
    filesystem, /tmp as a 64 MB tmpfs (prediction log), no Linux capabilities,
    no-new-privileges, 1 CPU, 1 GB memory, 128 pids, restart unless-stopped.

Each deploy tags the image phishguard:<short sha> and keeps older tags, so a
rollback is `docker run` of the previous tag (see docs/deploy.md).

Usage:
    python scripts/deploy_home_server.py              # deploy HEAD, wait for /healthz
    python scripts/deploy_home_server.py --host home-server --port 7860

Needs: ssh access to the host (key auth), docker and rsync there, a clean tree.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_hf_space import ROOT, dirty_state, git, stage  # noqa: E402

CONTAINER = "phishguard"
REMOTE_DIR = "phishguard-deploy/context"  # relative to the remote $HOME


def ssh(host: str, command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        capture_output=True,
        text=True,
        check=check,
    )


def remote_build(host: str, port: int) -> str | None:
    """The build commit the container reports at /healthz, read on the host itself."""
    r = ssh(host, f"curl -fsS --max-time 5 http://127.0.0.1:{port}/healthz", check=False)
    if r.returncode != 0:
        return None
    try:
        build = json.loads(r.stdout).get("build")
    except ValueError:
        return None
    return str(build) if build else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="home-server")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--wait-timeout", type=int, default=180)
    args = ap.parse_args()

    dirty = dirty_state()
    if dirty and not args.allow_dirty:
        sys.exit(f"working tree has uncommitted changes; commit first:\n{dirty}")

    commit = git("rev-parse", "HEAD")
    tag = f"phishguard:{commit[:7]}"
    stage_dir = ROOT / "build" / "home-server"
    print(stage(stage_dir), end="")

    print(f"syncing build context to {args.host}:~/{REMOTE_DIR}", flush=True)
    ssh(args.host, f"mkdir -p {REMOTE_DIR}")
    subprocess.run(
        ["rsync", "-a", "--delete", f"{stage_dir}/", f"{args.host}:{REMOTE_DIR}/"], check=True
    )

    print(f"building {tag} on {args.host}", flush=True)
    build = ssh(args.host, f"docker build -q -t {tag} {REMOTE_DIR}", check=False)
    if build.returncode != 0:
        print(build.stdout[-2000:], build.stderr[-4000:], sep="\n")
        return 1

    run = " ".join(
        [
            "docker run -d",
            f"--name {CONTAINER}",
            "--restart unless-stopped",
            f"-p 127.0.0.1:{args.port}:7860",
            "--read-only",
            "--tmpfs /tmp:rw,noexec,nosuid,size=64m",
            "--cap-drop ALL",
            "--security-opt no-new-privileges",
            "--cpus 1 --memory 1g --pids-limit 128",
            shlex.quote(tag),
        ]
    )
    print(f"replacing container {CONTAINER}", flush=True)
    ssh(args.host, f"docker rm -f {CONTAINER} >/dev/null 2>&1; {run}")

    deadline = time.monotonic() + args.wait_timeout
    while time.monotonic() < deadline:
        serving = remote_build(args.host, args.port)
        if serving == commit:
            print(f"SERVING {commit} on {args.host}:127.0.0.1:{args.port} as {tag}")
            return 0
        time.sleep(3)
    logs = ssh(args.host, f"docker logs --tail 40 {CONTAINER}", check=False)
    print(f"TIMEOUT: /healthz did not report {commit}\n{logs.stdout}{logs.stderr}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
