"""Loader tests use fake in-memory records. Nothing is downloaded or fetched."""

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from phishguard.data.phreshphish import ByteCapExceededError, build_subset


def _records(n: int = 50, html: str = "<html>hi</html>") -> Iterator[dict]:
    for i in range(n):
        yield {
            "sha256": f"h{i}",
            "url": f"http://d{i}.example.com/x",
            "html": html,
            "label": "phish" if i % 2 else "benign",
        }


def test_caps_per_class(tmp_path: Path) -> None:
    meta = build_subset(_records(), tmp_path, per_class=5, byte_cap=10**9, seed=42)
    df = pd.read_parquet(tmp_path / "subset.parquet")
    assert (df["is_phish"] == 1).sum() == 5
    assert (df["is_phish"] == 0).sum() == 5
    assert set(df.columns) >= {"html_text", "is_phish", "url"}
    assert meta["n_phish"] == 5 and meta["n_benign"] == 5 and meta["seed"] == 42


def test_hard_byte_cap_stops_before_exceeding(tmp_path: Path) -> None:
    big = "<html>" + "a" * 1000 + "</html>"
    meta = build_subset(_records(100, big), tmp_path, per_class=50, byte_cap=5000, seed=1)
    total = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert meta["html_bytes"] <= 5000
    assert total <= 5000 + 20000  # metadata overhead only
    assert meta["stopped_by"] == "byte_cap"


def test_single_record_over_cap_raises(tmp_path: Path) -> None:
    with pytest.raises(ByteCapExceededError):
        build_subset(_records(2, "x" * 100), tmp_path, per_class=1, byte_cap=10, seed=1)


def test_metadata_has_no_urls_or_html(tmp_path: Path) -> None:
    build_subset(_records(), tmp_path, per_class=3, byte_cap=10**9, seed=42)
    meta_text = (tmp_path / "meta.json").read_text()
    assert "example.com" not in meta_text and "<html>" not in meta_text
    assert json.loads(meta_text)["seed"] == 42


def test_does_not_touch_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    build_subset(_records(), tmp_path, per_class=2, byte_cap=10**9, seed=42)


def test_cache_dir_is_gitignored() -> None:
    from phishguard.data.phreshphish import DEFAULT_CACHE_DIR

    root = Path(__file__).resolve().parents[1]
    import subprocess

    r = subprocess.run(
        ["git", "check-ignore", "-q", str(DEFAULT_CACHE_DIR / "subset.parquet")],
        cwd=root,
    )
    assert r.returncode == 0
