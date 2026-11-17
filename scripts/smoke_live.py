"""Smoke test a running PhishGuard service (local Docker or the live Space).

Checks, each of which can fail:
    health     GET /healthz is ok and serves the v0.2 ONNX file, not the v0.1 fallback
    demo       GET / returns the demo page
    validation POST /predict with a 2-character URL is rejected with 422
    parity     every fixture URL scores within PARITY_TOL of the score the verified
               local artifacts gave it (catches skew from library or suffix-list drift)
    phish      at least PHISH_MIN_RECALL of held-out PhiUSIIL phish URLs are flagged
    tranco     at most TRANCO_MAX_FLAG_RATE of Tranco top-50 sites are flagged

Stdlib only, so it runs from any python3 >= 3.11:

    python scripts/smoke_live.py --base-url https://solomonsmith-dev-phishguard.hf.space
    python scripts/smoke_live.py --regen   # rewrite fixtures from local artifacts + data

Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "scripts" / "smoke_fixtures.json"

PARITY_TOL = 1e-6
PHISH_MIN_RECALL = 0.90
TRANCO_MAX_FLAG_RATE = 0.06  # measured 0.02 on the top 50 (1 of 50), 0.0252 on the top 5,000
EXPECTED_MODEL_FILE = "url_model_v0_2.onnx"
USER_AGENT = "phishguard-smoke/1"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


# ---------------------------------------------------------------- evaluation


def evaluate(
    health: dict[str, Any] | None,
    demo_ok: bool,
    invalid_status: int | None,
    scores: dict[str, float | None],
    fixtures: dict[str, Any],
    expect_commit: str | None = None,
) -> list[Check]:
    """Pure function over what the service returned. Unit-tested in tests/test_smoke_live.py."""
    checks: list[Check] = []
    if expect_commit:
        build = (health or {}).get("build")
        checks.append(
            Check("build", build == expect_commit, f"serving {build} (want {expect_commit})")
        )

    model = (health or {}).get("url_model")
    checks.append(
        Check(
            "health",
            bool(health) and health.get("status") == "ok" and model == fixtures["model_file"],
            f"status={(health or {}).get('status')} url_model={model} "
            f"(want {fixtures['model_file']})",
        )
    )
    checks.append(
        Check("demo", demo_ok, "GET / returned the demo page" if demo_ok else "GET / failed")
    )
    checks.append(
        Check(
            "validation", invalid_status == 422, f"2-char URL -> HTTP {invalid_status} (want 422)"
        )
    )

    threshold = float(fixtures["threshold"])
    rows = [*fixtures["phish"], *fixtures["tranco"]]
    missing = [r["url"] for r in rows if scores.get(r["url"]) is None]
    drift = [
        (r["url"], abs(float(s) - float(r["expected"])))
        for r in rows
        if (s := scores.get(r["url"])) is not None
        and abs(float(s) - float(r["expected"])) > PARITY_TOL
    ]
    worst = max((d for _, d in drift), default=0.0)
    checks.append(
        Check(
            "parity",
            not missing and not drift,
            f"{len(rows) - len(missing) - len(drift)}/{len(rows)} within {PARITY_TOL:g}"
            + (f"; {len(missing)} unscored" if missing else "")
            + (f"; {len(drift)} drifted, worst {worst:.3g} ({drift[0][0]})" if drift else ""),
        )
    )

    phish = [scores.get(r["url"]) for r in fixtures["phish"]]
    phish_scored = [float(p) for p in phish if p is not None]
    recall = sum(p >= threshold for p in phish_scored) / len(phish) if phish else 0.0
    checks.append(
        Check(
            "phish",
            recall >= PHISH_MIN_RECALL,
            f"{sum(p >= threshold for p in phish_scored)}/{len(phish)} flagged at {threshold:.2f} "
            f"(recall {recall:.2f}, want >= {PHISH_MIN_RECALL:.2f}; "
            f"median {statistics.median(phish_scored) if phish_scored else float('nan'):.3f})",
        )
    )

    tranco = [scores.get(r["url"]) for r in fixtures["tranco"]]
    tranco_scored = [float(p) for p in tranco if p is not None]
    flagged = [
        r["url"]
        for r in fixtures["tranco"]
        if (s := scores.get(r["url"])) is not None and s >= threshold
    ]
    rate = len(flagged) / len(tranco) if tranco else 1.0
    checks.append(
        Check(
            "tranco",
            bool(tranco) and len(tranco_scored) == len(tranco) and rate <= TRANCO_MAX_FLAG_RATE,
            f"{len(flagged)}/{len(tranco)} flagged at {threshold:.2f} "
            f"(rate {rate:.2f}, want <= {TRANCO_MAX_FLAG_RATE:.2f}; "
            f"median {statistics.median(tranco_scored) if tranco_scored else float('nan'):.3f})"
            + (f"; flagged: {', '.join(flagged)}" if flagged else ""),
        )
    )
    return checks


# ---------------------------------------------------------------- HTTP


def _request(
    url: str, body: dict[str, Any] | None = None, timeout: float = 30
) -> tuple[int, bytes]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def wait_healthy(base: str, wait_s: int) -> dict[str, Any] | None:
    """Free Spaces sleep when idle; the first request can take a minute to wake one."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            status, raw = _request(f"{base}/healthz", timeout=20)
            if status == 200:
                return dict(json.loads(raw))
            print(f"healthz HTTP {status}; waiting", file=sys.stderr)
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            print(f"healthz not reachable ({type(e).__name__}); waiting", file=sys.stderr)
        if time.monotonic() >= deadline:
            return None
        time.sleep(10)


def run_live(
    base: str, fixtures: dict[str, Any], wait_s: int, expect_commit: str | None = None
) -> list[Check]:
    base = base.rstrip("/")
    health = wait_healthy(base, wait_s)
    try:
        status, raw = _request(f"{base}/")
        demo_ok = status == 200 and b"PhishGuard" in raw and b'id="score"' in raw
    except (urllib.error.URLError, TimeoutError):
        demo_ok = False
    try:
        invalid_status: int | None = _request(f"{base}/predict", {"url": "ab"})[0]
    except (urllib.error.URLError, TimeoutError):
        invalid_status = None

    scores: dict[str, float | None] = {}
    for row in [*fixtures["phish"], *fixtures["tranco"]]:
        try:
            status, raw = _request(f"{base}/predict", {"url": row["url"]})
            scores[row["url"]] = float(json.loads(raw)["p_phish"]) if status == 200 else None
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError):
            scores[row["url"]] = None
    return evaluate(health, demo_ok, invalid_status, scores, fixtures, expect_commit)


# ---------------------------------------------------------------- fixtures


def regen_fixtures(phish_n: int, tranco_n: int) -> dict[str, Any]:
    """Score fixture URLs in-process with the verified local artifacts (needs .venv + data)."""
    import pandas as pd
    from fastapi.testclient import TestClient
    from phishguard.serving.api import app

    test = pd.read_parquet(ROOT / "data/processed/url_test.parquet")
    probe = pd.read_parquet(ROOT / "data/processed/url_probe_tranco.parquet")
    phish_urls = test[test["is_phish"] == 1].sample(phish_n, random_state=7)["url"].tolist()
    tranco_urls = probe["url"].head(tranco_n).tolist()
    with TestClient(app) as c:
        health = c.get("/healthz").json()

        def score(u: str) -> float:
            return float(c.post("/predict", json={"url": u}).json()["p_phish"])

        threshold = float(c.post("/predict", json={"url": tranco_urls[0]}).json()["threshold"])
        return {
            "about": (
                "Expected scores from the verified local artifacts. phish: held-out PhiUSIIL "
                f"test-split phish, sample({phish_n}, random_state=7). tranco: top {tranco_n} of "
                "the Tranco probe, never trained on. Regenerate with "
                "`python scripts/smoke_live.py --regen` after retraining or re-exporting."
            ),
            "model_file": health["url_model"],
            "threshold": threshold,
            "phish": [{"url": u, "expected": score(u)} for u in phish_urls],
            "tranco": [{"url": u, "expected": score(u)} for u in tranco_urls],
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-url", help="service root, e.g. http://localhost:7860")
    ap.add_argument("--fixtures", type=Path, default=FIXTURES)
    ap.add_argument("--wait", type=int, default=240, help="seconds to wait for /healthz")
    ap.add_argument("--expect-commit", help="also fail unless /healthz reports this commit")
    ap.add_argument("--regen", action="store_true", help="rewrite fixtures from local artifacts")
    ap.add_argument("--phish-n", type=int, default=25)
    ap.add_argument("--tranco-n", type=int, default=50)
    args = ap.parse_args()

    if args.regen:
        fx = regen_fixtures(args.phish_n, args.tranco_n)
        args.fixtures.write_text(json.dumps(fx, indent=1) + "\n")
        print(f"wrote {args.fixtures}: {len(fx['phish'])} phish, {len(fx['tranco'])} tranco")
        return 0
    if not args.base_url:
        ap.error("--base-url is required unless --regen")

    fixtures = json.loads(args.fixtures.read_text())
    checks = run_live(args.base_url, fixtures, args.wait, args.expect_commit)
    for c in checks:
        print(f"{'PASS' if c.ok else 'FAIL'}  {c.name:<10} {c.detail}")
    failed = [c.name for c in checks if not c.ok]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed against {args.base_url}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
