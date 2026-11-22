"""Stream a small, capped subset of PhreshPhish into a git-ignored local cache.

Source: https://huggingface.co/datasets/phreshphish/phreshphish (CC BY 4.0).
Records are hostile data. This module only reads strings out of them. It never renders
HTML, never runs scripts, and never fetches any URL found in a record.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ID = "phreshphish/phreshphish"
DEFAULT_CACHE_DIR = Path("data/raw/phreshphish")  # git-ignored via data/raw/
BYTE_CAP = 2 * 1024**3
PER_CLASS = 2000
MAX_HTML_CHARS = 200_000  # per-record cap; training reads far less than this
MAX_SCAN = 200_000  # hard stop on records examined


class ByteCapExceededError(RuntimeError):
    """A single record is larger than the whole byte cap."""


def build_subset(
    records: Iterable[Mapping[str, Any]],
    out_dir: Path,
    per_class: int = PER_CLASS,
    byte_cap: int = BYTE_CAP,
    seed: int = 42,
    max_scan: int = MAX_SCAN,
) -> dict[str, Any]:
    """Collect up to per_class phish and benign records, never exceeding byte_cap of HTML."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {1: 0, 0: 0}
    rows: list[dict[str, Any]] = []
    total = 0
    scanned = 0
    stopped_by = "exhausted"
    for rec in records:
        scanned += 1
        if scanned > max_scan:
            stopped_by = "max_scan"
            break
        label = rec.get("label")
        if label not in ("phish", "benign"):
            continue
        is_phish = 1 if label == "phish" else 0
        if counts[is_phish] >= per_class:
            continue
        html = rec.get("html")
        if not isinstance(html, str) or not html:
            continue
        html = html[:MAX_HTML_CHARS]
        size = len(html.encode("utf-8"))
        if size > byte_cap:
            raise ByteCapExceededError(f"one record is {size} bytes, cap is {byte_cap}")
        if total + size > byte_cap:
            stopped_by = "byte_cap"
            break
        total += size
        counts[is_phish] += 1
        rows.append(
            {
                "sha256": rec.get("sha256", ""),
                "url": rec.get("url", ""),
                "html_text": html,
                "is_phish": is_phish,
            }
        )
        if counts[1] >= per_class and counts[0] >= per_class:
            stopped_by = "per_class_reached"
            break
    random.Random(seed).shuffle(rows)
    pd.DataFrame(rows, columns=["sha256", "url", "html_text", "is_phish"]).to_parquet(
        out_dir / "subset.parquet", index=False
    )
    meta = {
        "source": REPO_ID,
        "licence": "cc-by-4.0",
        "seed": seed,
        "per_class_cap": per_class,
        "byte_cap": byte_cap,
        "n_phish": counts[1],
        "n_benign": counts[0],
        "html_bytes": total,
        "scanned": scanned,
        "stopped_by": stopped_by,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def stream_records(shards: tuple[int, ...] = (0, 1)) -> Iterator[Mapping[str, Any]]:
    """Stream PhreshPhish train records from a few parquet shards, in order (no full download).

    Shards are read sequentially and lazily, so transfer stays near the bytes actually used.
    Imported lazily so tests never need the network.
    """
    from datasets import load_dataset

    base = f"hf://datasets/{REPO_ID}/data"
    files = [f"{base}/train-{i:03d}.parquet" for i in shards]
    ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    ds = ds.select_columns(["sha256", "url", "html", "label"])
    yield from ds


def main() -> None:
    from phishguard.data.split import split_by_domain

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--splits-out", type=Path, default=Path("data/processed/phreshphish"))
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    ap.add_argument("--byte-cap", type=int, default=BYTE_CAP)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    meta = build_subset(
        stream_records(tuple(args.shards)),
        args.out,
        args.per_class,
        min(args.byte_cap, BYTE_CAP),
        args.seed,
    )
    df = pd.read_parquet(args.out / "subset.parquet")
    parts = split_by_domain(df, seed=args.seed)
    args.splits_out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(args.splits_out / f"html_{name}.parquet", index=False)
        meta[f"split_{name}"] = {"rows": len(part), "phish": int(part["is_phish"].sum())}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
