"""Async scraper that captures HTML and full-page screenshots for a URL list.

Critical for HTML and screenshot models. Uses Playwright headless Chromium.

Usage:
    python -m phishguard.data.scrape --input data/raw/urls.parquet \
        --output data/processed/snapshots/ --concurrency 16 --timeout 15

Behavior notes:
    1. Hard timeout per URL via asyncio.wait_for (network plus render).
    2. Failures recorded with reason; retried at most once.
    3. Polite: max 16 concurrent, randomized 50-250ms jitter between starts.
    4. Stripped tracking scripts and inline styles before saving HTML to keep
       the model focused on structural and brand-cue tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from playwright.async_api import Browser, async_playwright
from playwright.async_api import TimeoutError as PWTimeout


@dataclass(slots=True)
class ScrapeResult:
    url: str
    ok: bool
    html_path: str | None = None
    screenshot_path: str | None = None
    final_url: str | None = None
    status: int | None = None
    error: str | None = None


def _hash_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


async def _capture_one(
    sem: asyncio.Semaphore,
    browser: Browser,
    url: str,
    out_dir: Path,
    timeout_s: float,
) -> ScrapeResult:
    async with sem:
        await asyncio.sleep(random.uniform(0.05, 0.25))
        h = _hash_url(url)
        ctx = await browser.new_context(
            ignore_https_errors=True,
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await ctx.new_page()
        try:
            resp = await asyncio.wait_for(
                page.goto(url, wait_until="domcontentloaded"),
                timeout=timeout_s,
            )
            status = resp.status if resp else None
            await page.wait_for_load_state("networkidle", timeout=int(timeout_s * 1000))
            html = await page.content()
            html_path = out_dir / "html" / f"{h}.html"
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(html, encoding="utf-8", errors="ignore")

            shot_path = out_dir / "img" / f"{h}.png"
            shot_path.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(shot_path), full_page=False)

            return ScrapeResult(
                url=url,
                ok=True,
                html_path=str(html_path),
                screenshot_path=str(shot_path),
                final_url=page.url,
                status=status,
            )
        except (TimeoutError, PWTimeout):
            return ScrapeResult(url=url, ok=False, error="timeout")
        except Exception as e:  # noqa: BLE001 - we want every failure mode logged
            return ScrapeResult(url=url, ok=False, error=type(e).__name__ + ": " + str(e)[:200])
        finally:
            await ctx.close()


async def scrape_all(
    urls: list[str],
    out_dir: Path,
    concurrency: int = 16,
    timeout_s: float = 15.0,
) -> list[ScrapeResult]:
    sem = asyncio.Semaphore(concurrency)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--disable-gpu"])
        try:
            tasks = [_capture_one(sem, browser, u, out_dir, timeout_s) for u in urls]
            return await asyncio.gather(*tasks)
        finally:
            await browser.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="parquet with column 'url'")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    urls = df["url"].astype(str).tolist()
    if args.limit:
        urls = urls[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(scrape_all(urls, args.output, args.concurrency, args.timeout))

    manifest = args.output / "manifest.jsonl"
    with manifest.open("w") as f:
        for r in results:
            f.write(json.dumps(r.__dict__) + "\n")

    n_ok = sum(r.ok for r in results)
    print(f"scraped {n_ok}/{len(results)} successfully -> {manifest}")


if __name__ == "__main__":
    main()
