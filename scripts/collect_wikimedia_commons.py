#!/usr/bin/env python3
"""Collect images from Wikimedia Commons search results.

Usage:
  python scripts/collect_wikimedia_commons.py \
    --out_dir /tmp/galicia_raw \
    --queries "horreo galicia" "cruceiro galicia" \
    --per_query 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tqdm import tqdm

COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Wikimedia Commons images")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["horreo galicia", "cruceiro galicia"],
        help="Commons search queries",
    )
    parser.add_argument("--out_dir", required=True, help="Output root directory")
    parser.add_argument("--per_query", type=int, default=120, help="Target images per query")
    parser.add_argument("--max_requests", type=int, default=20, help="Max API calls per query")
    parser.add_argument("--min_side", type=int, default=512, help="Minimum width/height")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout")
    parser.add_argument("--sleep_sec", type=float, default=0.4, help="Pause between API calls")
    return parser.parse_args()


def label_from_query(query: str) -> str:
    q = query.lower()
    if "cruce" in q:
        return "cruceiro"
    return "horreo"


def ext_from_url(url: str) -> str:
    lower = url.lower()
    if lower.endswith(".png"):
        return ".png"
    if lower.endswith(".webp"):
        return ".webp"
    return ".jpg"


def commons_search(query: str, timeout: float, continuation: dict[str, Any] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": 50,
        "prop": "imageinfo",
        "iiprop": "url",
    }
    if continuation:
        params.update(continuation)

    response = requests.get(COMMONS_API, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def iter_image_urls(payload: dict[str, Any]):
    pages = payload.get("query", {}).get("pages", [])
    for page in pages:
        for item in page.get("imageinfo", []):
            url = item.get("url")
            if isinstance(url, str) and url.startswith("http"):
                yield url


def safe_stem(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:20]


def download_and_validate(url: str, timeout: float, min_side: int) -> Image.Image | None:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.load()
        if min(image.width, image.height) < min_side:
            return None
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}

    for query in args.queries:
        label = label_from_query(query)
        label_dir = root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        seen: set[str] = set()
        continuation: dict[str, Any] | None = None

        for _ in range(args.max_requests):
            if saved >= args.per_query:
                break

            payload = commons_search(query=query, timeout=args.timeout, continuation=continuation)
            urls = list(iter_image_urls(payload))
            if not urls:
                break

            for url in tqdm(urls, desc=f"{query}", leave=False):
                if saved >= args.per_query:
                    break
                if url in seen:
                    continue
                seen.add(url)

                image = download_and_validate(url=url, timeout=args.timeout, min_side=args.min_side)
                if image is None:
                    continue

                filename = f"{safe_stem(url)}{ext_from_url(url)}"
                out_path = label_dir / filename
                if out_path.exists():
                    continue

                image.save(out_path, quality=95)
                saved += 1

            continuation = payload.get("continue")
            if not continuation:
                break

            time.sleep(args.sleep_sec)

        summary[query] = saved
        print(f"[done] query='{query}' label='{label}' saved={saved}")

    print(json.dumps({"out_dir": str(root), "saved_per_query": summary}, indent=2))


if __name__ == "__main__":
    main()
