#!/usr/bin/env python3
"""Capture a truthful detector demo frame for the README first screen (issue #7).

Runs the REAL served detector on a target and renders one frame showing the
per-modality probabilities and the fused verdict. It does NOT fabricate a
detector: if the ONNX checkpoints for the HTML/screenshot models do not exist,
it exits non-zero with a clear "nothing is trained yet" message.

Usage:
    python scripts/capture_detector_demo.py \
        --url https://example.com \
        --out docs/detector_demo.png \
        --server http://localhost:8000

The script drives the SAME /predict endpoint the browser extension calls, so
whatever it draws is whatever the served model actually returns. Checkpoints
that must exist for each modality to appear:
    models/checkpoints/html_model.onnx
    models/checkpoints/screenshot_model.onnx
    models/checkpoints/fusion_model.pkl   (optional; without it, URL-only path)
If only the URL checkpoint exists, the frame is drawn URL-only and the HTML and
IMG panels say "not trained" instead of inventing a number.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_NAMES = {
    "html_model.onnx": "HTML (DistilBERT)",
    "screenshot_model.onnx": "Screenshot (EfficientNet-B0)",
}


def check_trained() -> tuple[list[str], list[str]]:
    ckpt_dir = Path("models/checkpoints")
    present: list[str] = []
    absent: list[str] = []
    for fname in MODEL_NAMES:
        p = ckpt_dir / fname
        (present if p.exists() else absent).append(fname)
    return present, absent


def call_predict(url: str, server: str, screenshot_b64: str | None) -> dict:
    resp = httpx.post(
        f"{server.rstrip('/')}/predict",
        json={
            "url": url,
            "html": None,
            "screenshot_b64": screenshot_b64,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def draw_frame(result: dict, out: Path, url: str) -> None:
    modalities = result["modalities"]
    threshold = result["threshold"]
    p_phish = result["p_phish"]
    is_phish = result["is_phish"]

    labels = ["URL", "HTML", "IMG"]
    probs = [
        modalities.get("url", {}).get("p"),
        modalities.get("html", {}).get("p"),
        modalities.get("img", {}).get("p"),
    ]
    available = [
        modalities.get("url", {}).get("available", False),
        modalities.get("html", {}).get("available", False),
        modalities.get("img", {}).get("available", False),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bar_colors = []
    shown = []
    for i, (name, p, av) in enumerate(zip(labels, probs, available, strict=False)):
        if av and p is not None:
            ax.bar(i, p, width=0.5, color="#d9534f" if p >= threshold else "#5cb85c")
            bar_colors.append("#d9534f" if p >= threshold else "#5cb85c")
            shown.append((name, p))
        else:
            ax.bar(i, 0.0, width=0.5, color="#cccccc")
            shown.append((name, None))
    ax.axhline(threshold, color="#0275d8", linestyle="--", linewidth=1)
    ax.text(1.98, threshold, f" threshold {threshold:.2f}", va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(phish)")
    ax.set_title(
        f"PhishGuard detector demo — {url}\nP(phish)={p_phish:.3f}  "
        f"verdict: {'PHISH' if is_phish else 'benign'}",
        fontsize=11,
    )

    # annotate missing modalities honestly
    for i, (_, p) in enumerate(shown):
        if p is None:
            ax.text(i, 0.02, "not trained", ha="center", fontsize=9, color="#777777")
        else:
            ax.text(i, p + 0.02, f"{p:.3f}", ha="center", fontsize=9)

    ax.set_xlim(-0.5, len(labels) - 0.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"frame written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="https://accounts-verify.secure-login.xyz/login.php")
    ap.add_argument("--out", type=Path, default=Path("docs/detector_demo.png"))
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="PNG screenshot to send as the IMG modality (optional)",
    )
    args = ap.parse_args()

    present, absent = check_trained()
    if not present:
        print("ERROR: no detector checkpoints exist under models/checkpoints/.")
        print(
            "Nothing has been trained for the HTML or screenshot modality, "
            "so no demo can be produced. See docs/multimodal-training-status.md."
        )
        sys.exit(1)
    print(f"trained checkpoints found: {present}")
    if absent:
        print(f"absent (will show as 'not trained'): {absent}")

    screenshot_b64 = None
    if args.screenshot:
        screenshot_b64 = base64.b64encode(args.screenshot.read_bytes()).decode()

    print(f"calling served detector at {args.server} ...")
    result = call_predict(args.url, args.server, screenshot_b64)
    draw_frame(result, args.out, args.url)
    print(f"latency_ms_total={result.get('latency_ms_total')}")


if __name__ == "__main__":
    main()
