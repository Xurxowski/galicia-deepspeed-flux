#!/usr/bin/env python3
"""Collect Google Images for horreos/cruceiros using SerpAPI.

Usage:
  export SERPAPI_API_KEY=...
  python scripts/collect_google_images_serpapi.py \
    --out_dir /tmp/galicia_raw \
    --queries "horreo galicia" "cruceiro galicia" \
    --per_query 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image
from tqdm import tqdm

SERPAPI_URL = "https://serpapi.com/search.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect images from Google Images via SerpAPI")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["horreo galicia", "cruceiro galicia"],
        help="Search queries",
    )
    parser.add_argument("--out_dir", required=True, help="Output root directory")
    parser.add_argument("--per_query", type=int, default=120, help="Target images per query")
    parser.add_argument("--max_pages", type=int, default=10, help="Max Google image result pages per query")
    parser.add_argument("--min_side", type=int, default=512, help="Minimum width/height")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout")
    parser.add_argument("--sleep_sec", type=float, default=0.7, help="Pause between API calls")
    return parser.parse_args()


def label_from_query(query: str) -> str:
    q = query.lower()
    if "cruce" in q:
        return "cruceiro"
    return "horreo"


def ext_from_content_type(content_type: str | None) -> str:
    if not content_type:
        return ".jpg"
    ct = content_type.lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    return ".jpg"


def serpapi_search(api_key: str, query: str, page_idx: int, timeout: float) -> list[dict]:
    params = {
        "engine": "google_images",
        "q": query,
        "ijn": page_idx,
        "hl": "es",
        "gl": "es",
        "safe": "active",
        "api_key": api_key,
    }
    response = requests.get(SERPAPI_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload.get("images_results", [])


def iter_image_urls(items: Iterable[dict]) -> Iterable[str]:
    for item in items:
        for key in ("original", "link", "thumbnail"):
            url = item.get(key)
            if isinstance(url, str) and url.startswith("http"):
                yield url
                break


def download_and_validate_image(url: str, timeout: float, min_side: int) -> tuple[Image.Image, str] | None:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.load()
        if min(image.width, image.height) < min_side:
            return None
        if image.mode != "RGB":
            image = image.convert("RGB")
        ext = ext_from_content_type(response.headers.get("Content-Type"))
        return image, ext
    except Exception:
        return None


def safe_stem(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:20]
    return digest


def main() -> None:
    args = parse_args()
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise SystemExit("Missing SERPAPI_API_KEY environment variable")

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}

    for query in args.queries:
        label = label_from_query(query)
        label_dir = root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        seen: set[str] = set()

        for page_idx in range(args.max_pages):
            if saved >= args.per_query:
                break

            items = serpapi_search(api_key, query, page_idx, args.timeout)
            if not items:
                break

            for url in tqdm(list(iter_image_urls(items)), desc=f"{query} page {page_idx}", leave=False):
                if saved >= args.per_query:
                    break
                if url in seen:
                    continue
                seen.add(url)

                result = download_and_validate_image(url, timeout=args.timeout, min_side=args.min_side)
                if result is None:
                    continue

                image, ext = result
                filename = f"{safe_stem(url)}{ext}"
                out_path = label_dir / filename
                if out_path.exists():
                    continue

                image.save(out_path, quality=95)
                saved += 1

            time.sleep(args.sleep_sec)

        summary[query] = saved
        print(f"[done] query='{query}' label='{label}' saved={saved}")

    print(json.dumps({"out_dir": str(root), "saved_per_query": summary}, indent=2))


if __name__ == "__main__":
    main()
